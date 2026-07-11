from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation, RotationSpline
from scene.cameras import Camera
import numpy as np
import torch

def build_bspline(camera_infos):
    """
    Input:
    camera_infos
    Output:
    RotationSpline, TranslationSpline
    """
    ts = []
    Rs = []
    Ts = []
    for camera_info in camera_infos:
        ts.append(camera_info.fid)
        Rs.append(camera_info.R)
        Ts.append(camera_info.T)
    ts = np.array(ts) # in seconds

    Rs_arr = np.array(Rs)
    # tmp = Rs_arr.transpose(0,2,1) @ Rs_arr
    # tmp = np.sum(tmp, axis=0) / Rs_arr.shape[0]
    # scale = np.sqrt(tmp[0,0])
    # # print(scale)

    # Rs_scaled = Rs_arr / scale
    scale = 1.0
    Rs_scaled = Rs_arr
    Rs_scaled = Rotation.from_matrix(Rs_scaled)
    Ts = np.array(Ts)
    Rspline = RotationSpline(ts, Rs_scaled)
    Tspline = BSpline(t=ts, c=Ts, k=5, axis=0, extrapolate=True)

    # viz_spline(Rspline, Tspline, camera_infos)
    # eval_poses(Rspline, Tspline, camera_infos) # eval all poses
    # return

    cam = camera_infos[0]
    cam_info = [cam.FovY, cam.FovX, cam.width, cam.height, cam.image, cam.image_path, # other pseudo except intrinsics
                 cam.image_name, cam.fid, cam.closest_event_index, cam.image_timestamp, scale]

    return Rspline, Tspline, cam_info

def get_viewcams_from_spline(timestamps, Rspline, Tspline, cam_intri, device='cuda'):
    """
    input: timestamp array
    output: viewcam list (CameraInfo class)
    """
    view_cams = []
    ts_s = np.array(timestamps) / 1e9 # in seconds
    trans = Tspline(ts_s)
    rotation = Rspline(ts_s)
    scale = cam_intri[-1]
    Rot = rotation.as_matrix() * scale
    image = torch.zeros((1, cam_intri[3], cam_intri[2]), device=device)
    for i in range(len(timestamps)):
        view_cam = Camera(colmap_id=-1, R=Rot[i], T=trans[i],
                        FoVx=cam_intri[1], FoVy=cam_intri[0],
                        image=image, gt_alpha_mask=None, 
                        image_name=cam_intri[6], uid=-1,
                        data_device=device, fid = ts_s[i], 
                        depth=None, events=None, image_timestamp=ts_s[i],closest_event_index=None)
        view_cams.append(view_cam)

    return view_cams

def compute_ape(ground_truth, estimated):
    """
    Input: transformation matrix
    Output: error
    """
    assert len(ground_truth) == len(estimated), "Ground truth and estimated arrays must have the same length."
    
    ape_translation = []
    ape_rotation = []
    
    for gt, est in zip(ground_truth, estimated):
        # Translation error
        translation_error = np.linalg.norm(gt[:3, 3] - est[:3, 3])
        ape_translation.append(translation_error)
        
        # Rotation error (angle between two rotations)
        gt_rot = Rotation.from_matrix(gt[:3,:3])  
        est_rot = Rotation.from_matrix(est[:3,:3])
        rotation_error = gt_rot.inv() * est_rot
        rotation_error_angle = rotation_error.magnitude()  # Angle between rotations
        ape_rotation.append(rotation_error_angle)
    
    return np.array(ape_translation), np.array(ape_rotation)

def compute_rpe(ground_truth, estimated):
    assert len(ground_truth) == len(estimated), "Ground truth and estimated arrays must have the same length."
    
    rpe_translation = []
    rpe_rotation = []
    
    for i in range(1, len(ground_truth)):
        # Relative translation error
        gt_diff = ground_truth[i][:3, 3] - ground_truth[i-1][:3, 3]
        est_diff = estimated[i][:3, 3] - estimated[i-1][:3, 3]
        translation_error = np.linalg.norm(gt_diff - est_diff)
        rpe_translation.append(translation_error)
        
        # Relative rotation error (angle between two relative rotations)
        gt_rot_1 = Rotation.from_matrix(ground_truth[i-1][:3,:3])
        gt_rot_2 = Rotation.from_matrix(ground_truth[i][:3,:3])
        est_rot_1 = Rotation.from_matrix(estimated[i-1][:3,:3])
        est_rot_2 = Rotation.from_matrix(estimated[i][:3,:3])
        
        gt_rel_rot = gt_rot_2.inv() * gt_rot_1
        est_rel_rot = est_rot_2.inv() * est_rot_1
        
        rotation_error = gt_rel_rot.inv() * est_rel_rot
        rotation_error_angle = rotation_error.magnitude()  # Angle between rotations
        rpe_rotation.append(rotation_error_angle)
    
    return np.array(rpe_translation), np.array(rpe_rotation)

def eval_poses(Rspline, Tspline, cam_infos=None):
    import matplotlib.pyplot as plt

    if cam_infos:
        ts = np.array([cam.fid for cam in cam_infos])
        Ts = np.array([cam.T for cam in cam_infos])
        Rs = np.array([cam.R for cam in cam_infos])
    else:
        path1 = '/home/daizj/data/dataset/Blender_event/gtduck/raw/transforms.json'
        path2 = '/home/daizj/data/dataset/Blender_event/gtduck/timestamps.txt'
        Rs, Ts, ts = read_all_transformation(path1, path2)
    interpolated_positions = Tspline(ts)
    interpolated_rotations = Rspline(ts).as_matrix()

    ground_truth = []
    estimated = []
    for i in range(ts.shape[0]):
        gt = np.eye(4)
        est = np.eye(4)
        gt[:3,:3] = Rs[i]
        gt[3,:3] = Ts[i]
        est[:3,:3] = interpolated_rotations[i]
        est[3,:3] = interpolated_positions[i]
        ground_truth.append(gt)
        estimated.append(est)
    ground_truth = np.asarray(ground_truth)
    est = np.asarray(estimated)

    rpe_t, rpe_r = compute_rpe(ground_truth, estimated)
    ape_t, ape_r = compute_ape(ground_truth, estimated)
    
    # Create a figure and axes
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Plotting the first graph
    axs[0, 0].plot(np.linspace(0,1,rpe_t.shape[0]), rpe_t, label='rt', color='b')
    axs[0, 0].set_title('rpe_t')
    axs[0, 0].grid(True)

    # Plotting the second graph
    axs[0, 1].plot(np.linspace(0,1,rpe_r.shape[0]), rpe_r, label='rr', color='r')
    axs[0, 1].set_title('rpe_r')
    axs[0, 1].grid(True)

    # Plotting the third graph
    axs[1, 0].plot(np.linspace(0,1,ape_t.shape[0]), ape_t, label='at', color='g')
    axs[1, 0].set_title('ape_t')
    axs[1, 0].grid(True)

    # Plotting the fourth graph
    axs[1, 1].plot(np.linspace(0,1,ape_r.shape[0]), ape_r, label='ar', color='purple')
    axs[1, 1].set_title('ape_r')
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig("viz_ape_rpe")

def read_all_transformation(path1,path2):
    import os
    import json
    timestamps = np.loadtxt(path2)
    with open(path1, "r") as json_file:
        contents = json.load(json_file)
        # fovx = contents["camera_angle_x"]
        frames = contents["frames"]
        Rs = []
        Ts = []
        ts = []
        for frame in frames:    
            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3, :3])
            R[:, 0] = -R[:, 0] 
            T = -matrix[:3, 3]
            Rs.append(R)
            Ts.append(T)
            # ts.append(frame_time)
    return np.array(Rs), np.array(Ts), timestamps
    
def viz_spline(Rspline, Tspline, cam_infos):
    import matplotlib.pyplot as plt
    
    ts = np.array([cam.fid for cam in cam_infos])
    Ts = np.array([cam.T for cam in cam_infos])
    
    len_seq = ts.shape[0]
    ts_fine = np.linspace(ts.min(), ts.max(), len_seq)
    interpolated_positions = Tspline(ts_fine)
    interpolated_rotations = Rspline(ts_fine).as_matrix()

    print(interpolated_rotations[0])

    fig = plt.figure(figsize=(12, 6))

    # Plot translation spline trajectory
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(interpolated_positions[:, 0], interpolated_positions[:, 1], interpolated_positions[:, 2],
             label='Spline Interpolation', color='blue')
    ax1.scatter(Ts[:, 0], Ts[:, 1], Ts[:, 2], color='red', label='Original Positions')

    # Visualize rotation axes (Rspline)
    length = 0.1  # length of axes for visualization
    for pos, rot in zip(interpolated_positions, interpolated_rotations):
        ax1.quiver(pos[0], pos[1], pos[2], rot[0, 0], rot[1, 0], rot[2, 0], color='r', length=length)
        ax1.quiver(pos[0], pos[1], pos[2], rot[0, 1], rot[1, 1], rot[2, 1], color='g', length=length)
        ax1.quiver(pos[0], pos[1], pos[2], rot[0, 2], rot[1, 2], rot[2, 2], color='b', length=length)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    ax1.set_title("Trajectory and Rotation Spline Visualization")

    # Optionally, a separate simplified plot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(interpolated_positions[:, 0], interpolated_positions[:, 1], interpolated_positions[:, 2], color='blue')
    ax2.scatter(Ts[:, 0], Ts[:, 1], Ts[:, 2], color='red')

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title("Spline Trajectory Only")

    plt.tight_layout()
    plt.savefig("spline_viz.png")
    plt.close()