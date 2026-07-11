#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple, Optional
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
import imageio
from glob import glob
import tqdm
import h5py
import yaml
import cv2 as cv
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils.camera_utils import camera_nerfies_from_JSON

from utils.bspline_utils import build_bspline

from scipy.spatial.transform import RotationSpline
from scipy.interpolate import BSpline

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    # image: np.array
    width: int
    height: int
    fid: float
    depth: Optional[np.array] = None
    events: Optional[np.array] = None
    closest_event_index: Optional[int] = None
    image_timestamp: Optional[float] = None
    image_path: Optional[str] = None
    image_name: Optional[str] = None
    image: Optional[np.array] = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    all_events: Optional[np.array] = None
    
    Rspline: Optional[RotationSpline] = None
    Tspline: Optional[BSpline] = None
    cam_intri: Optional[list] = None
    K: Optional[np.array] = None
    rectify_mask: Optional[np.array] = None
    scale: Optional[float] = None

def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]]
                 for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return K, pose


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    num_frames = len(cam_extrinsics)
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write(
            "Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        fid = int(image_name) / (num_frames - 1)
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, fid=fid)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'],
                       vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                           images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(
            cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(
            cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            frame_time = frame['time']

            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3, :3])
            R[:, 0] = -R[:, 0]
            T = -matrix[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array(
                [1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            mask = norm_data[..., 3:4]

            arr = norm_data[:, :, :3] * norm_data[:, :,
                                                  3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(
                np.array(arr * 255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovx
            FovX = fovy

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                        image_path=image_path, image_name=image_name, width=image.size[
                                            0],
                                        height=image.size[1], fid=frame_time))

    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(
        path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(
        path, "transforms_test.json", white_background, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readDTUCameras(path, render_camera, object_camera):
    camera_dict = np.load(os.path.join(path, render_camera))
    images_lis = sorted(glob(os.path.join(path, 'image/*.png')))
    masks_lis = sorted(glob(os.path.join(path, 'mask/*.png')))
    n_images = len(images_lis)
    cam_infos = []
    cam_idx = 0
    for idx in range(0, n_images):
        image_path = images_lis[idx]
        image = np.array(Image.open(image_path))
        mask = np.array(imageio.imread(masks_lis[idx])) / 255.0
        image = Image.fromarray((image * mask).astype(np.uint8))
        world_mat = camera_dict['world_mat_%d' % idx].astype(np.float32)
        fid = camera_dict['fid_%d' % idx] / (n_images / 12 - 1)
        image_name = Path(image_path).stem
        scale_mat = camera_dict['scale_mat_%d' % idx].astype(np.float32)
        P = world_mat @ scale_mat
        P = P[:3, :4]

        K, pose = load_K_Rt_from_P(None, P)
        a = pose[0:1, :]
        b = pose[1:2, :]
        c = pose[2:3, :]

        pose = np.concatenate([a, -c, -b, pose[3:, :]], 0)

        S = np.eye(3)
        S[1, 1] = -1
        S[2, 2] = -1
        pose[1, 3] = -pose[1, 3]
        pose[2, 3] = -pose[2, 3]
        pose[:3, :3] = S @ pose[:3, :3] @ S

        a = pose[0:1, :]
        b = pose[1:2, :]
        c = pose[2:3, :]

        pose = np.concatenate([a, c, b, pose[3:, :]], 0)

        pose[:, 3] *= 0.5

        matrix = np.linalg.inv(pose)
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        FovY = focal2fov(K[0, 0], image.size[1])
        FovX = focal2fov(K[0, 0], image.size[0])
        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=image.size[
                                  0], height=image.size[1],
                              fid=fid)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readNeuSDTUInfo(path, render_camera, object_camera):
    print("Reading DTU Info")
    train_cam_infos = readDTUCameras(path, render_camera, object_camera)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=[],
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readNerfiesCameras(path):
    with open(f'{path}/scene.json', 'r') as f:
        scene_json = json.load(f)
    with open(f'{path}/metadata.json', 'r') as f:
        meta_json = json.load(f)
    with open(f'{path}/dataset.json', 'r') as f:
        dataset_json = json.load(f)

    coord_scale = scene_json['scale']
    scene_center = scene_json['center']

    name = path.split('/')[-2]
    if name.startswith('vrig'):
        train_img = dataset_json['train_ids']
        val_img = dataset_json['val_ids']
        all_img = train_img + val_img
        ratio = 0.25
    elif name.startswith('NeRF'):
        train_img = dataset_json['train_ids']
        val_img = dataset_json['val_ids']
        all_img = train_img + val_img
        ratio = 1.0
    elif name.startswith('interp'):
        all_id = dataset_json['ids']
        train_img = all_id[::4]
        val_img = all_id[2::4]
        all_img = train_img + val_img
        ratio = 0.5
    else:  # for hypernerf
        train_img = dataset_json['ids']# dataset_json['ids']# [::2]# dataset_json['ids'][::4]
        all_img = train_img
        num_img = len(all_img)
        ratio = 0.25 #0.5

    train_num = len(train_img)

    percent = 1.0
    used_for_train_num = int(num_img * percent)
    print(used_for_train_num)

    all_cam = [meta_json[i]['camera_id'] for i in all_img]
    all_time = [meta_json[i]['time_id'] for i in all_img]
    max_time = max(all_time)
    all_time = [meta_json[i]['time_id'] / max_time for i in all_img]
    selected_time = set(all_time)

    # all poses
    all_cam_params = []
    for im in all_img:
        camera = camera_nerfies_from_JSON(f'{path}/camera/{im}.json', ratio)
        camera['position'] = camera['position'] - scene_center
        camera['position'] = camera['position'] * coord_scale
        all_cam_params.append(camera)
    
    all_event = sorted(glob(os.path.join(path, f"events/{int(1 / ratio)}x/*.npz")))
    used_for_train_num_event = int(len(all_event) * percent)
    print(used_for_train_num_event)
    
    # load all events
    all_events = []
    for event in all_event[:used_for_train_num_event]:
        event_npz = np.load(event)
        t,x,y,pol = event_npz['t'], event_npz['x'], event_npz['y'], event_npz['p']
        event = np.stack([t,x,y,pol], axis=-1)
        all_events.append(event)
    all_events = np.concatenate(all_events, axis=0)

    print("Loading events...the number of events is ", "{:.5e}".format(all_events.shape[0]), "and the number of images is ", num_img)
    
    timestamp_img = np.arange(num_img, dtype=np.float32) / 25.0 # correct 25 is in setting means the video in 25 fps
    all_img = [f'{path}/rgb/{int(1 / ratio)}x/{i}.png' for i in all_img]

    cam_infos = []
    print("Reading images and events...")
    # for idx in tqdm.tqdm(range(len(all_img))):
    for idx in range(used_for_train_num):
        image_path = all_img[idx]
        image = np.array(Image.open(image_path))

        # diff = (image.shape[0] - 256) // 2 # keyboard
        # image = image[diff:-diff,:,:]

        diff = (image.shape[1] - 256) // 2
        image = image[:,diff:-diff,:]

        image = Image.fromarray((image).astype(np.uint8))
        image_name = Path(image_path).stem
        
        # ts = timestamp_img[idx] * 1e9

        # event_index = np.searchsorted(all_events[:,0], ts)
        # print(ts, all_events[event_index,0], all_events[event_index,0]-ts)
        orientation = all_cam_params[idx]['orientation'].T
        position = -all_cam_params[idx]['position'] @ orientation
        focal = all_cam_params[idx]['focal_length']
        # fid = all_time[idx]
        fid = timestamp_img[idx]
        T = position
        R = orientation

        FovY = focal2fov(focal, image.size[1])
        FovX = focal2fov(focal, image.size[0])
        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=image.size[0], height= image.size[1],
                              fid=fid, image_timestamp=timestamp_img[idx])
        
        cam_infos.append(cam_info)
    # print("last event index is ", last_event_index)
    Rspline, tbspline, cam_intri = build_bspline(cam_infos)
    sys.stdout.write('\n')
    return cam_infos, train_num, scene_center, coord_scale, all_events, Rspline, tbspline, cam_intri


def readNerfiesInfo(path, eval):
    print("Reading Nerfies Info")
    cam_infos, train_num, scene_center, scene_scale, all_events, Rspline, tbspline, cam_intri = readNerfiesCameras(path)

    if eval:
        train_cam_infos = cam_infos[:train_num]
        test_cam_infos = cam_infos[train_num:]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        print(f"Generating point cloud from nerfies...")

        xyz = np.load(os.path.join(path, "points.npy"))
        xyz = (xyz - scene_center) * scene_scale
        num_pts = xyz.shape[0]
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           all_events=all_events,
                           Rspline=Rspline,
                           Tspline=tbspline,
                           cam_intri=cam_intri)
    return scene_info


def readCamerasFromNpy(path, npy_file, split, hold_id, num_images):
    cam_infos = []
    video_paths = sorted(glob(os.path.join(path, 'frames/*')))
    poses_bounds = np.load(os.path.join(path, npy_file))

    poses = poses_bounds[:, :15].reshape(-1, 3, 5)
    H, W, focal = poses[0, :, -1]

    n_cameras = poses.shape[0]
    poses = np.concatenate(
        [poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
    bottoms = np.array([0, 0, 0, 1]).reshape(
        1, -1, 4).repeat(poses.shape[0], axis=0)
    poses = np.concatenate([poses, bottoms], axis=1)
    poses = poses @ np.diag([1, -1, -1, 1])

    i_test = np.array(hold_id)
    video_list = i_test if split != 'train' else list(
        set(np.arange(n_cameras)) - set(i_test))

    for i in video_list:
        video_path = video_paths[i]
        c2w = poses[i]
        images_names = sorted(os.listdir(video_path))
        n_frames = num_images

        matrix = np.linalg.inv(np.array(c2w))
        R = np.transpose(matrix[:3, :3])
        T = matrix[:3, 3]

        for idx, image_name in enumerate(images_names[:num_images]):
            image_path = os.path.join(video_path, image_name)
            image = Image.open(image_path)
            frame_time = idx / (n_frames - 1)

            FovX = focal2fov(focal, image.size[0])
            FovY = focal2fov(focal, image.size[1])

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovX=FovX, FovY=FovY,
                                        image=image,
                                        image_path=image_path, image_name=image_name,
                                        width=image.size[0], height=image.size[1], fid=frame_time))

            idx += 1
    return cam_infos


def readPlenopticVideoDataset(path, eval, num_images, hold_id=[0]):
    print("Reading Training Camera")
    train_cam_infos = readCamerasFromNpy(path, 'poses_bounds.npy', split="train", hold_id=hold_id,
                                         num_images=num_images)

    print("Reading Training Camera")
    test_cam_infos = readCamerasFromNpy(
        path, 'poses_bounds.npy', split="test", hold_id=hold_id, num_images=num_images)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = os.path.join(path, 'points3D.ply')
    if not os.path.exists(ply_path):
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readRealEventInfo(path):
    # the path is the sequence path
    cam_infos, all_events, Rspline, Tspline, cam_intri, K, rectify_mask, scale = readRealEventDataset(path)
    nerf_normalization = getNerfppNorm(cam_infos)
    ply_path = os.path.join(path, "points3d.ply")
    try:
        pcd = fetchPly(ply_path)
    except:
        print("!!!No point cloud found!!!")
        pcd = None
    scene_info = SceneInfo(point_cloud=pcd,
                    train_cameras=cam_infos,
                    test_cameras=[],
                    nerf_normalization=nerf_normalization,
                    ply_path=ply_path,
                    all_events=all_events,
                    Rspline=Rspline,
                    Tspline=Tspline,
                    cam_intri=cam_intri, K=K, rectify_mask=rectify_mask, scale=scale)
    return scene_info

def readRealEventDataset(path, eval=True):

    all_events = np.load(os.path.join(path,'events_rectified.npy'))

    all_events[:,0] = all_events[:,0] * 1e3
    scale = 1.0 # 1.0

    time_range = [4.420162400000000000e+10, 4.590170700000000000e+10]
    start_index = np.searchsorted(all_events[:, 0], time_range[0], side='left')
    end_index = np.searchsorted(all_events[:, 0], time_range[1], side='right')
    if start_index!=end_index:
        all_events = all_events[start_index:end_index]

    all_events[:,-1] = 2 * all_events[:,-1] - 1 # from 0, 1 to -1, 1
    print(max(all_events[:,-1]), min(all_events[:,-1]), all_events.shape)

    cam_infos = []
    poses = np.load(os.path.join(path,'llff', 'all_poses_bounds.npy')) # can be repalce as poses_bounds.npy
    poses = poses[:, :15].reshape(-1, 3, 5)
    poses = np.concatenate(
        [poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
    bottoms = np.array([0, 0, 0, 1]).reshape(
        1, -1, 4).repeat(poses.shape[0], axis=0)
    poses = np.concatenate([poses, bottoms], axis=1)
    poses = poses @ np.diag([1, -1, -1, 1])

    with open(os.path.join(path, "K.yaml"), "r") as f:
        K_info = yaml.safe_load(f)
    
    intrinsics = K_info['cam0']['intrinsics']
    resolution = K_info['cam0']['resolution']
    
    fovx = focal2fov(intrinsics[0], resolution[0] * scale) # fx, W
    fovy = focal2fov(intrinsics[1], resolution[1] * scale)

    K = np.array([[intrinsics[0], 0, resolution[0] * scale/2, 0],
                  [0, intrinsics[1], resolution[1] * scale/2, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]],dtype=np.float32)
    
    all_timestamp = np.load(os.path.join(path,'llff', 'all_timestamps.npy'))

    rectify_mask = np.load(os.path.join(path,'rectify_mask.npy'))

    for idx in range(poses.shape[0]):
        c2w = poses[idx]
        matrix = np.linalg.inv(np.array(c2w))
        R = np.transpose(matrix[:3, :3])
        T = matrix[:3, 3]

        timestamp = all_timestamp[idx] / 1e6 # in seconds

        FovY = fovy
        FovX = fovx

        image = None

        cam_infos.append(CameraInfo(uid = idx, R = R, T = T, FovY = FovY, FovX = FovX,
                                    image = image, image_path = None, image_name = None, 
                                    width = int(resolution[0]*scale), height = int(resolution[1]*scale), fid = timestamp))
        
    Rspline, Tspline, cam_intri = build_bspline(cam_infos)

    return cam_infos, all_events, Rspline, Tspline, cam_intri, K, rectify_mask, scale


def readBlenderEventInfo(path, is_color=False):

    cam_infos, all_events, Rspline, Tspline, cam_intri = readBlenderEventCameras(path, is_color)
    nerf_normalization = getNerfppNorm(cam_infos)
    ply_path = os.path.join(path, "points3d.ply")
    try:
        pcd = fetchPly(ply_path)
    except:
        print("!!!No point cloud found!!!")
        pcd = None
    scene_info = SceneInfo(point_cloud=pcd,
                        train_cameras=cam_infos,
                        test_cameras=[],
                        nerf_normalization=nerf_normalization,
                        ply_path=ply_path,
                        all_events=all_events,
                        Rspline=Rspline,
                        Tspline=Tspline,
                        cam_intri=cam_intri)
    return scene_info
    

def readBlenderEventCameras(path, is_color=False):
    print("Reading Blender Event Cameras data")
    if is_color:
        event_path = os.path.join(path, "color_events")
    else: 
        event_path = os.path.join(path, "events")
    
    percent = 1.0

    event_files = sorted(glob(os.path.join(event_path, "*.npz")))
    all_events = []
    for event_file in event_files[:int(len(event_files) * percent)]:
        event_npz = np.load(event_file)
        t,x,y,pol = event_npz['t'], event_npz['x'], event_npz['y'], event_npz['p']
        event = np.stack([t,x,y,pol], axis=-1)
        all_events.append(event)
    all_events = np.concatenate(all_events, axis=0)
    event_timestamps = all_events[:, 0]

    cam_infos = []
    with open(os.path.join(path, "cam_transforms.json"), "r") as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]
        frames = contents["frames"]
        for idx, frame in tqdm.tqdm(enumerate(frames[:int(len(frames) * percent)])):
            
            frame_time = frame['timestamp']
            # c2w in frame["transform_matrix"]
            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3, :3]) # R is stored transposed due to 'glm' in CUDA code
            R[:, 0] = -R[:, 0] 
            T = -matrix[:3, 3]

            image_path = os.path.join(path, "images", frame["name"])
            image_name = Path(image_path).stem
            image = Image.open(image_path)
            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovx
            FovX = fovy
            if is_color:
                event_idx = frame["events_index"] # frame["color_events_index"]
            else:
                event_idx = frame["events_index"]

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                        image_path=image_path, image_name=image_name, width=image.size[0], 
                                        height=image.size[1], fid=frame_time, closest_event_index=event_idx))
    Rspline, tbspline, cam_intri = build_bspline(cam_infos)

    return cam_infos, all_events, Rspline, tbspline, cam_intri
    

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,  # colmap dataset reader from official 3D Gaussian [https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/]
    "Blender": readNerfSyntheticInfo,  # D-NeRF dataset [https://drive.google.com/file/d/1uHVyApwqugXTFuIRRlE4abTW8_rrVeIK/view?usp=sharing]
    "DTU": readNeuSDTUInfo,  # DTU dataset used in Tensor4D [https://github.com/DSaurus/Tensor4D]
    "nerfies": readNerfiesInfo,  # NeRFies & HyperNeRF dataset proposed by [https://github.com/google/hypernerf/releases/tag/v0.1]
    "plenopticVideo": readPlenopticVideoDataset,  # Neural 3D dataset in [https://github.com/facebookresearch/Neural_3D_Video]
    "BlenderEvent": readBlenderEventInfo,  # Blender Event dataset created by myself
    "RealEvent": readRealEventInfo,  # Real Event dataset created by myself
}
