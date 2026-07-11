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

import torch
from scene import Scene, DeformModel
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.pose_utils import pose_spherical, render_wander_path
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import imageio
import numpy as np
import time

from utils.bspline_utils import get_viewcams_from_spline
from utils.graphics_utils import getIntrinsicMatrix

def recover_pose(R,T):
    pose = np.eye(4)
    # R = -R.T
    # T = -T # same
    R = R.T
    T = T
    pose[:3, :3] = R
    pose[:3, 3] = T
    tmp = pose[:3, :3].T @ pose[:3, :3]
    scale = np.sqrt(np.trace(tmp) / 3)
    return torch.from_numpy(pose).to(device='cuda'), scale

def visualize_gs_traj(image, gs_traj_2d, colors, path = None, events = None):
    import matplotlib.pyplot as plt
    import numpy as np

    data = gs_traj_2d.transpose(1, 0, 2)
    print(data.shape)
    trajs = [data[i, :, :] for i in range(data.shape[0])]

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    w, h = image.shape[1], image.shape[0]
    ax.imshow(image)

    for i, trajectory_points in enumerate(trajs):
        x_coords = trajectory_points[:, 0]
        y_coords = trajectory_points[:, 1]
        ax.plot(x_coords, y_coords, 
                marker='o',          # Style of the points
                linestyle='-',       # Style of the line connecting points
                color=colors[i],     # Assign a unique color
                markersize=0.2,
                alpha=0.1
                )
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    plt.grid(False)
    ax.set_aspect('equal',adjustable='box')
    ax.invert_yaxis()
    # ax.invert_xaxis()

    if events is not None:
        x = events[:, 1]
        y = events[:, 2]
        p = events[:, 3]
        idx_pos = np.asarray(p) > 0
        idx_neg = np.logical_not(idx_pos)
        # ax.scatter(w - x[idx_pos], y[idx_pos], c='b', s=0.1, alpha=0.1)
        # ax.scatter(w - x[idx_neg], y[idx_neg], c='r', s=0.1, alpha=0.1)

        ax.scatter(x[idx_pos], y[idx_pos], c='b', s=0.1, alpha=0.1)
        ax.scatter(x[idx_neg], y[idx_neg], c='r', s=0.1, alpha=0.1)

    if path is not None:
        fig.savefig(path)
    else:
        fig.savefig(os.path.join("gs_traj.png"))
    plt.close()

def gaussian_trajectory(dataset: ModelParams, iteration: int, num_frames: int):
    # export the motion of gaussians in [x, y, t] to compare with events
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        deform = DeformModel(dataset.is_blender, dataset.is_6dof)
        deform.load_weights(dataset.model_path)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # spline part
        Rspline = scene.Rspline
        Tspline = scene.Tspline
        cam_intri = scene.cam_intri

        views = scene.getTrainCameras()
        start_fid = views[0].fid
        end_fid = views[-1].fid

        # compute the time index of gaussians
        time_series = np.linspace(start_fid.cpu().numpy(), end_fid.cpu().numpy(), num_frames).reshape(-1)

        # print(time_series.shape)
        # from spline create new view_cams
        view_cams = get_viewcams_from_spline(time_series * 1e9, Rspline, Tspline, cam_intri, device='cuda')
        
        start_infer_time = time.time()

        gaussian_trajectory = []
        for i in range(time_series.shape[0]):
            fid = view_cams[i].fid
            # print(fid)
            xyz = gaussians.get_xyz
            time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
            d_xyz, d_rotation, d_scaling, d_color = deform.step(xyz.detach(), time_input)
            # d_xyz, d_rotation, d_scaling = 0.05 * d_xyz, 0.05 * d_rotation, 0.05 * d_scaling
            # project the gaussians to the image plane
            mean_3d = gaussians.get_xyz + d_xyz # [n,3]
            mean_3d_homo = torch.cat((mean_3d, torch.ones_like(mean_3d[:, 0:1])), dim=1)
            # projection matrix

            intri = getIntrinsicMatrix(view_cams[i].image_width, view_cams[i].image_height, view_cams[i].FoVx, view_cams[i].FoVy)
            intri = intri.to(mean_3d_homo.device)

            pose, scale = recover_pose(view_cams[i].R, view_cams[i].T)
            pose = pose.type(mean_3d_homo.dtype)
            # P_camera = pose @ mean_3d_homo.T
            P_camera = pose @ mean_3d_homo.T
            P_camera[:3, :] = P_camera[:3, :] / P_camera[2, :]
            mean2d = (intri @ P_camera).T

            mean2d = mean2d[:, :2] # / mean2d[:, 2:3]
            index_gs = torch.arange(mean_3d.shape[0], device=xyz.device)
            time_value = torch.ones_like(index_gs) * time_series[i]
            # print(mean2d.shape, torch.max(mean2d), torch.min(mean2d),torch.mean(mean2d))
            # print(index_gs.shape)
            # print(time_value.shape)
            # return
            mean2d_idx_time = torch.cat((mean2d, index_gs.unsqueeze(1), time_value.unsqueeze(1)), dim=1)
            gaussian_trajectory.append(mean2d_idx_time)

        gaussian_trajectory = torch.cat(gaussian_trajectory, dim=0)

        end_infer_time = time.time()
        print(f"Inference time: {end_infer_time - start_infer_time} seconds")

    np.save(os.path.join("gaussian_trajectory.npy"), gaussian_trajectory.cpu().numpy())
    
def image_plane_traj(dataset: ModelParams, iteration: int, fid: float, num_frames: int):
    
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        deform = DeformModel(dataset.is_blender, dataset.is_6dof)
        deform.load_weights(dataset.model_path)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        Rspline = scene.Rspline
        Tspline = scene.Tspline
        cam_intri = scene.cam_intri

        view_cam = get_viewcams_from_spline([fid * 1e9], Rspline, Tspline, cam_intri, device='cuda')[0]

        view_gts = scene.getTrainCameras()
        rot_gt = view_gts[0].R
        trans_gt = view_gts[0].T

        xyz = gaussians.get_xyz
        fid_torch = torch.tensor([fid],device='cuda')
        time_input = fid_torch.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling, d_color = deform.step(xyz.detach(), time_input)
        results = render(view_cam, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, d_color, is_6dof=False)
        # results = render(view_gts[0], gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, d_color, is_6dof=False)
        
        num_points = xyz.shape[0]
        rng = np.random.default_rng()
        colors = rng.uniform(0, 1, size=(num_points, 3))

        image = results["render"]
        traj = []
        
        rot_hist = []
        for i in range(num_frames):
            fid_next = fid + i * 0.001
            view_cam_next = get_viewcams_from_spline([fid_next * 1e9], Rspline, Tspline, cam_intri, device='cuda')[0]
            rot_hist.append(view_cam_next.R)
            # pose test
            # if i == 0:
            # rot_sp = view_cam_next.R
            # trans_sp = view_cam_next.T
            # print(rot_gt)
            # print(rot_sp.T @ rot_gt)
            # print(trans_sp - trans_gt)
            # view_cam_next = view_gts[0]

            fid_next_torch = torch.tensor([fid_next],device='cuda')
            time_input_next = fid_next_torch.unsqueeze(0).expand(xyz.shape[0], -1)
            d_xyz_next, d_rotation_next, d_scaling_next, d_color_next = deform.step(xyz.detach(), time_input_next)
            mean_3d = gaussians.get_xyz + d_xyz_next # [n,3]
            mean_3d_homo = torch.cat((mean_3d, torch.ones_like(mean_3d[:, 0:1])), dim=1)
            # projection matrix

            intri = getIntrinsicMatrix(view_cam_next.image_width, view_cam_next.image_height, view_cam_next.FoVx, view_cam_next.FoVy)
            intri = intri.to(mean_3d_homo.device)

            pose, _ = recover_pose(view_cam_next.R, view_cam_next.T)

            pose = pose.type(mean_3d_homo.dtype)
            P_camera = pose @ mean_3d_homo.T
            P_camera[:3, :] = P_camera[:3, :] / P_camera[2, :]
            mean2d = torch.matmul(intri, P_camera).T
            mean2d = mean2d[:, :2]
            traj.append(mean2d)

        traj = torch.cat(traj, dim=0)
        traj = traj.view(num_frames, -1, 2)
        # to numpy
        image = image.permute(1, 2, 0).detach().cpu().numpy()
        traj = traj.detach().cpu().numpy()
        traj = traj[:,::3,0:2]
        visualize_gs_traj(image, traj, colors)

        rot_hist = np.stack(rot_hist, axis=0)
        np.save("rot_hist.npy", rot_hist)

def image_plane_traj_batch(dataset: ModelParams, iteration: int, time_interval: float, num_frames: int):
    
    base_dir = '../test/traj'
    # Check if base_dir exists and has files, if so, clean it
    import os
    import shutil

    if os.path.exists(base_dir):
        # Remove all files in the directory
        for filename in os.listdir(base_dir):
            file_path = os.path.join(base_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        os.makedirs(base_dir, exist_ok=True)

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        deform = DeformModel(dataset.is_blender, dataset.is_6dof)
        deform.load_weights(dataset.model_path)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        Rspline = scene.Rspline
        Tspline = scene.Tspline
        cam_intri = scene.cam_intri

        view_cams = scene.getTrainCameras()
        xyz = gaussians.get_xyz

        num_points = xyz.shape[0]
        rng = np.random.default_rng()
        colors = rng.uniform(0, 1, size=(num_points, 3))
        all_events = scene.all_events

        # opacity = gaussians.get_opacity
        # opacity_mask = opacity > 0.8
        if len(view_cams) > 300:
        # Uniformly sample 300 views if there are more than 300
            step = len(view_cams) // 300
        if step == 0:
            step = 1
        view_cams = [view_cams[i] for i in range(0, len(view_cams), step)][:300]

        for j, view_cam in enumerate(view_cams):
            traj = []
            fid = view_cam.fid
            time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
            d_xyz, d_rotation, d_scaling, d_color = deform.step(xyz.detach(), time_input)
            d_xyz, d_rotation, d_scaling = 0.05 * d_xyz, 0.05 * d_rotation, 0.05 * d_scaling
            results = render(view_cam, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, d_color, is_6dof=False)
            image = results["render"]
            image = image[0, :, :].unsqueeze(0)
            # image = view_cam.original_image
            events = None
            if j != len(view_cams) - 1:
                event_id = view_cam.closest_event_index
                event_id_next = view_cams[j+1].closest_event_index
                events = all_events[event_id:event_id_next]
            
            # TODO: paralize this part
            for i in range(num_frames):
                fid_next = fid + i * time_interval
                fid_next_numpy = fid_next.cpu().numpy()
                # print(fid_next)
                view_cam_next = get_viewcams_from_spline(fid_next_numpy * 1e9, Rspline, Tspline, cam_intri, device='cuda')[0]
                time_input_next = fid_next.unsqueeze(0).expand(xyz.shape[0], -1)
                d_xyz_next, d_rotation_next, d_scaling_next, d_color_next = deform.step(xyz.detach(), time_input_next)
                d_xyz_next, d_rotation_next, d_scaling_next = 0.05 * d_xyz_next, 0.05 * d_rotation_next, 0.05 * d_scaling_next
                mean_3d = gaussians.get_xyz + d_xyz_next # [n,3]
                mean_3d_homo = torch.cat((mean_3d, torch.ones_like(mean_3d[:, 0:1])), dim=1)
                # projection matrix

                intri = getIntrinsicMatrix(view_cam_next.image_width, view_cam_next.image_height, view_cam_next.FoVx, view_cam_next.FoVy)
                intri = intri.to(mean_3d_homo.device)
                # print(intri)
                pose, _ = recover_pose(view_cam_next.R, view_cam_next.T)
                pose = pose.type(mean_3d_homo.dtype)
                P_camera = pose @ mean_3d_homo.T
                P_camera[:3, :] = P_camera[:3, :] / P_camera[2, :]
                mean2d = (intri @ P_camera).T
                mean2d = mean2d[:, :2]
                traj.append(mean2d)

            traj = torch.cat(traj, dim=0)
            traj = traj.view(num_frames, -1, 2)
            # to numpy
            image = image.permute(1, 2, 0).detach().cpu().numpy()
            # traj = traj[:,opacity_mask.squeeze(-1),:]
            traj = traj.detach().cpu().numpy()
            traj = traj[:,::10,:]
            visualize_gs_traj(image, traj, colors, os.path.join(base_dir, "{:04d}.png".format(j)), events=None)
            
def render_set(model_path, load2gpu_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, deform):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    # makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    t_list = []

    # print(len(views))
    if len(views) > 300:
        # Uniformly sample 300 views if there are more than 300
        step = len(views) // 300
        if step == 0:
            step = 1
        views = [views[i] for i in range(0, len(views), step)][:300]
    # print(len(views))

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if load2gpu_on_the_fly:
            view.load2device()
        fid = view.fid
        # print(fid)
        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling, d_color = deform.step(xyz.detach(), time_input)
        d_xyz, d_rotation, d_scaling = 0.05 * d_xyz, 0.05 * d_rotation, 0.05 * d_scaling
        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, d_color, is_6dof)
        rendering = results["render"][0, :, :].unsqueeze(0) # no color
        # rendering = results["render"]

        depth = results["depth"]
        depth_normalized = depth / (depth.max() + 1e-5)

        # gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(depth_normalized, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
        depth_map = depth.cpu().numpy()
        np.save(os.path.join(depth_path, '{0:05d}'.format(idx) + ".npy"), depth_map)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        fid = view.fid
        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)

        torch.cuda.synchronize()
        t_start = time.time()

        d_xyz, d_rotation, d_scaling, d_color = deform.step(xyz.detach(), time_input)
        d_xyz, d_rotation, d_scaling = 0.05 * d_xyz, 0.05 * d_rotation, 0.05 * d_scaling
        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, d_color, is_6dof)

        torch.cuda.synchronize()
        t_end = time.time()
        t_list.append(t_end - t_start)

    t = np.array(t_list[5:])
    fps = 1.0 / t.mean()
    print(f'Test FPS: \033[1;35m{fps:.5f}\033[0m, Num. of GS: {xyz.shape[0]}')


def interpolate_time(model_path, load2gpt_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, deform):
    render_path = os.path.join(model_path, name, "interpolate_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    frame = 150
    idx = torch.randint(0, len(views), (1,)).item()
    view = views[idx]
    renderings = []
    for t in tqdm(range(0, frame, 1), desc="Rendering progress"):
        fid = torch.Tensor([t / (frame - 1)]).cuda()
        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(t) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(t) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)


def interpolate_view(model_path, load2gpt_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, timer):
    render_path = os.path.join(model_path, name, "interpolate_view_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_view_{}".format(iteration), "depth")
    # acc_path = os.path.join(model_path, name, "interpolate_view_{}".format(iteration), "acc")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    # makedirs(acc_path, exist_ok=True)

    frame = 150
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    idx = torch.randint(0, len(views), (1,)).item()
    view = views[idx]  # Choose a specific time for rendering

    render_poses = torch.stack(render_wander_path(view), 0)
    # render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, frame + 1)[:-1]],
    #                            0)

    renderings = []
    for i, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        fid = view.fid

        matrix = np.linalg.inv(np.array(pose))
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        view.reset_extrinsic(R, T)

        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = timer.step(xyz.detach(), time_input)
        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)
        # acc = results["acc"]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(i) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(i) + ".png"))
        # torchvision.utils.save_image(acc, os.path.join(acc_path, '{0:05d}'.format(i) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)


def interpolate_all(model_path, load2gpt_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, deform):
    render_path = os.path.join(model_path, name, "interpolate_all_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_all_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    frame = 150
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, frame + 1)[:-1]],
                               0)
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    idx = torch.randint(0, len(views), (1,)).item()
    view = views[idx]  # Choose a specific time for rendering

    renderings = []
    for i, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        fid = torch.Tensor([i / (frame - 1)]).cuda()

        matrix = np.linalg.inv(np.array(pose))
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        view.reset_extrinsic(R, T)

        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(i) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(i) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)


def interpolate_poses(model_path, load2gpt_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, timer):
    render_path = os.path.join(model_path, name, "interpolate_pose_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_pose_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    # makedirs(acc_path, exist_ok=True)
    frame = 520
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    idx = torch.randint(0, len(views), (1,)).item()
    view_begin = views[0]  # Choose a specific time for rendering
    view_end = views[-1]
    view = views[idx]

    R_begin = view_begin.R
    R_end = view_end.R
    t_begin = view_begin.T
    t_end = view_end.T

    renderings = []
    for i in tqdm(range(frame), desc="Rendering progress"):
        fid = view.fid

        ratio = i / (frame - 1)

        R_cur = (1 - ratio) * R_begin + ratio * R_end
        T_cur = (1 - ratio) * t_begin + ratio * t_end

        view.reset_extrinsic(R_cur, T_cur)

        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = timer.step(xyz.detach(), time_input)

        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=60, quality=8)


def interpolate_view_original(model_path, load2gpt_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background,
                              timer):
    render_path = os.path.join(model_path, name, "interpolate_hyper_view_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_hyper_view_{}".format(iteration), "depth")
    # acc_path = os.path.join(model_path, name, "interpolate_all_{}".format(iteration), "acc")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    frame = 1000
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    R = []
    T = []
    for view in views:
        R.append(view.R)
        T.append(view.T)

    view = views[0]
    renderings = []
    for i in tqdm(range(frame), desc="Rendering progress"):
        fid = torch.Tensor([i / (frame - 1)]).cuda()

        query_idx = i / frame * len(views)
        begin_idx = int(np.floor(query_idx))
        end_idx = int(np.ceil(query_idx))
        if end_idx == len(views):
            break
        view_begin = views[begin_idx]
        view_end = views[end_idx]
        R_begin = view_begin.R
        R_end = view_end.R
        t_begin = view_begin.T
        t_end = view_end.T

        ratio = query_idx - begin_idx

        R_cur = (1 - ratio) * R_begin + ratio * R_end
        T_cur = (1 - ratio) * t_begin + ratio * t_end

        view.reset_extrinsic(R_cur, T_cur)

        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = timer.step(xyz.detach(), time_input)

        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=60, quality=8)


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool,
                mode: str):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        deform = DeformModel(dataset.is_blender, dataset.is_6dof)
        deform.load_weights(dataset.model_path)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if mode == "render":
            render_func = render_set
        elif mode == "time":
            render_func = interpolate_time
        elif mode == "view":
            render_func = interpolate_view
        elif mode == "pose":
            render_func = interpolate_poses
        elif mode == "original":
            render_func = interpolate_view_original
        else:
            render_func = interpolate_all

        if not skip_train:
            render_func(dataset.model_path, dataset.load2gpu_on_the_fly, dataset.is_6dof, "train", scene.loaded_iter,
                        scene.getTrainCameras(), gaussians, pipeline,
                        background, deform)

        if not skip_test:
            render_func(dataset.model_path, dataset.load2gpu_on_the_fly, dataset.is_6dof, "test", scene.loaded_iter,
                        scene.getTestCameras(), gaussians, pipeline,
                        background, deform)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mode", default='render', choices=['render', 'time', 'view', 'all', 'pose', 'original'])
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.mode)

    # gaussian_trajectory(model.extract(args), args.iteration, 1000)

    # image_plane_traj(model.extract(args), args.iteration, 0.0, 200)
    # image_plane_traj_batch(model.extract(args), args.iteration, 0.002, 50)