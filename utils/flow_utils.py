import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from submodules.searaft.core.raft import RAFT
from submodules.searaft.config.parser import parse_args
import argparse

class FlowEstimator(nn.Module):
    # SEA-RAFT
    def __init__(self, device, parser):
        super(FlowEstimator, self).__init__()
        # model_path = "/home/daizj/data/event-gaussian/codes/De3GS/pretrained_models/Tartan480x640-M.pth"
        # config_path = "/home/daizj/data/event-gaussian/codes/De3GS/submodules/searaft/config/train/Tartan480x640-M.json"
        self.device = device
        self.args = parse_args(parser)
        state_dict = torch.load(self.args.path, map_location=self.device)
        self.model = RAFT(self.args)
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()

    def forward_flow(self, image1, image2):
        output = self.model(image1, image2, iters=self.args.iters, test_mode=True)
        flow_final = output['flow'][-1]
        info_final = output['info'][-1]
        return flow_final, info_final

    def calc_flow(self, image1, image2):
        img1 = F.interpolate(image1, scale_factor=2 ** self.args.scale, mode='bilinear', align_corners=False)
        img2 = F.interpolate(image2, scale_factor=2 ** self.args.scale, mode='bilinear', align_corners=False)
        H, W = img1.shape[2:]
        flow, info = self.forward_flow(img1, img2)
        flow_down = F.interpolate(flow, scale_factor=0.5 ** self.args.scale, mode='bilinear', align_corners=False) * (0.5 ** self.args.scale)
        info_down = F.interpolate(info, scale_factor=0.5 ** self.args.scale, mode='area')
        return flow_down, info_down
    
    def get_flow(self, image1, image2):
        """
        image1 and image2 all tensors with the shape of [1,h,w]
        """
        # H, W = image1.shape[1:]
        image1 = image1[None] # to [b,c,h,w]
        image2 = image2[None] 
        image1 = image1.repeat(1, 3, 1, 1)
        image2 = image2.repeat(1, 3, 1, 1)
        flow, info = self.calc_flow(image1, image2)

        return flow


###Motion-gs###
class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)
        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        # Points: B 4 HW 
        # K：B 4 4
        # T: B 4 4
        P = torch.matmul(K, T)[:, :3, :]  # B 3 4
        cam_points = torch.matmul(P, points)  # B 4 HW 
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2:3, :] + self.eps)  # B 2 HW
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width) # B 2 H W
        pix_coords = pix_coords.permute(0, 2, 3, 1) # B H W 2
        # normalize
        _pix_coords_ = torch.clone(pix_coords)
        _pix_coords_[..., 0] /= self.width - 1
        _pix_coords_[..., 1] /= self.height - 1
        _pix_coords_ = (_pix_coords_ - 0.5) * 2
        return _pix_coords_, pix_coords
    

def reproject(depth, cam, coords_3d, ones):
    inv_cam = torch.pinverse(cam)
    cam_points = depth * coords_3d
    cam_points = torch.cat([cam_points, ones], 0)
    cam_points = torch.einsum("ab,bhw->ahw", [inv_cam, cam_points])
    return cam_points

def project(points, cam): 
    cam_points = torch.einsum("ab,bhw->ahw", [cam, points])
    # cam_points = torch.matmul(cam, points)
    pix_coords = cam_points[:2] / (cam_points[2:3] + 1e-7)
    return pix_coords

def calculate_camera_flow(depth1, cam1, cam2):
    H, W = depth1.shape[-2:] # depth1: (B) (1) H W
    backprojdepth = BackprojectDepth(1, H, W).cuda()
    project3d = Project3D(1, H, W).cuda()
    inv_K1 = torch.linalg.inv(cam1.intrinsic.cuda())[None] # B 4 4
    K2 = cam2.intrinsic.cuda()[None] # B 4 4
    T12 = torch.matmul(torch.linalg.inv(cam2.extrinsic.cuda()), 
                     cam1.extrinsic.cuda())[None] # B 4 4
    points_3d = backprojdepth(depth1, inv_K1) # B 4 HW
    _, pixel_coords = project3d(points_3d, K2, T12) # B H W 2
    pixel_coords = pixel_coords.permute(0, 3, 1, 2) # B 2 H W
    ori_coords = backprojdepth.pix_coords.view(1, 3, H, W)[:, :2] # B 2 H W
    camere_flow = pixel_coords - ori_coords # B 2 H W
    return camere_flow[0] # 2 H W

def warping_gs_flow(depth_gt, gs_flow, camera_pose, next_camera_pose):
    H, W = depth_gt.shape[-2:] # depth1: (B) (1) H W
    backprojdepth = BackprojectDepth(1, H, W).cuda()
    project3d = Project3D(1, H, W).cuda()
    inv_K1 = torch.linalg.inv(camera_pose.intrinsic.cuda())[None] # B 4 4
    K2 = next_camera_pose.intrinsic.cuda()[None] # B 4 4
    T12 = torch.matmul(torch.linalg.inv(next_camera_pose.extrinsic.cuda()), 
                     camera_pose.extrinsic.cuda())[None] # B 4 4
    points_3d = backprojdepth(depth_gt, inv_K1) # B 4 HW
    pixel_coords_norm, _ = project3d(points_3d, K2, T12) # B H W 2
    gs_flow= F.grid_sample(gs_flow.unsqueeze(0), pixel_coords_norm, padding_mode="border", align_corners=True) # B 3 H W
    return gs_flow.squeeze(0)
###Motion-gs###

def compute_camera_flow(depth_before, view_cam_before, view_cam, K = None):
        
    device = depth_before.device
    H = depth_before.shape[1]
    W = depth_before.shape[2]
    if K is None:
        H = view_cam.image_height
        W = view_cam.image_width
        fovy = view_cam.FoVy
        fx = W / (2 * torch.tan(torch.tensor(fovy / 2, device=device)))
        cx = W / 2
        cy = H / 2
        K = torch.tensor([[fx, 0, cx],
                            [0, fx, cy],
                            [0, 0, 1]], device=device)
    def recover_extrinsic(R,T):
        pose = np.eye(4)
        # R[:, 0] = -R[:, 0] #TODO: check pose
        R = R.T
        T = T
        pose[:3, :3] = R
        pose[:3, 3] = T
        pose = np.linalg.inv(pose)
        tmp = pose[:3, :3].T @ pose[:3, :3]
        scale = np.sqrt(np.trace(tmp) / 3)
        return pose, scale
    
    pose1_np, scale1 = recover_extrinsic(view_cam_before.R, view_cam_before.T)
    pose2_np, scale2 = recover_extrinsic(view_cam.R, view_cam.T)

    # print(pose1, pose2)
    pose1 = torch.from_numpy(pose1_np).to(device).to(depth_before.dtype)
    pose2 = torch.from_numpy(pose2_np).to(device).to(depth_before.dtype)

    depth_before = depth_before[0]
    i, j = torch.meshgrid(torch.arange(W, device=device), 
                            torch.arange(H, device=device), 
                            indexing='xy')
    ones = torch.ones_like(i)
    pixels_hom = torch.stack([i, j, ones], dim=-1).to(depth_before.dtype)  # (N, 3)
    K_inv = torch.inverse(K)

    pixels_3d = pixels_hom @ K_inv.T * depth_before.unsqueeze(2) # * scale1  # (N, 3)
    
    points_3d = pixels_3d.reshape(-1, 3)
    points_3d_h = torch.cat([points_3d, torch.ones(points_3d.shape[0], 1, device=device)], dim=-1).T  # (4, N)

    # Transform points from camera1 to world, then to camera2
    pose1_inv = torch.inverse(pose1)
    cam2_points = pose1_inv @ pose2 @ points_3d_h  # (4, N)

    # Project back to 2D
    cam2_points = cam2_points[:3, :] / cam2_points[2, :]  # (3, N)
    pixels2 = torch.matmul(K, cam2_points).T  # (N, 3)

    # Reshape and calculate flow
    pixels1 = torch.stack([i, j], dim=-1).reshape(-1, 2)  # (N, 2)
    pixels2_xy = pixels2[:, :2]
    flow = pixels2_xy - pixels1
    flow = flow.reshape(H, W, 2)

    return flow, pose1_np, pose2_np

def get_optical_flow_batch(depth_maps, view_cams, K = None):
    
    device = depth_maps[0].device
    if K is None:
        H = view_cams[0].image_height
        W = view_cams[0].image_width
        fovy = view_cams[0].FoVy
        fx = W / (2 * torch.tan(torch.tensor(fovy / 2, device=device)))
        cx = W / 2
        cy = H / 2
        K = torch.tensor([[fx, 0, cx],
                            [0, fx, cy],
                            [0, 0, 1]], device=device)
    flow_batch = []
    poses = []
    # print(len(view_cams))
    for i in range(len(depth_maps)-1):
        # print(i)
        # flow = self.compute_optical_flow(depth_maps[i], view_cams[i], view_cams[i+1], K)
        flow, pose1, pose2 = compute_optical_flow(depth_maps[i], view_cams[i], view_cams[i+1], K)
        flow_batch.append(flow)
        poses.append(pose1)
    poses.append(pose2)

    return flow_batch, poses

def warp_events(events, flow):
    """
    TODO: non-linear warp
    flow : h w 2 
    """
    # backup
    flow_height, flow_width = flow.shape[:2]
    valid_mask = torch.logical_and(
        torch.logical_and(events[:, 1] >= 0, events[:, 1] < flow_width),
        torch.logical_and(events[:, 2] >= 0, events[:, 2] < flow_height)
    )
    events = events[valid_mask]
    warped_events = torch.zeros_like(events, device=events.device)
    params = flow[events[:, 2].long(), events[:, 1].long(), :]

    # Compute time differences (convert ns to s)
    dt = (events[:, 0] - events[0, 0]) * 1e-9
    
    # Warp coordinates (broadcast dt)
    x_prime = events[:, 1] - dt * params[:, 0]
    y_prime = events[:, 2] - dt * params[:, 1]
    
    # Create warped events tensor
    warped_events = torch.empty_like(events)
    warped_events[:, 0] = events[:, 0]  # timestamps unchanged
    warped_events[:, 1] = x_prime       # warped x coordinates
    warped_events[:, 2] = y_prime       # warped y coordinates
    warped_events[:, 3] = events[:, 3]  # polarities unchanged

    return warped_events

def warp_events_batch(event_batch, flow_batch):
    
    past_events = torch.zeros((1, 4), device=flow_batch[0].device)
    length = len(event_batch)
    for i in range(length):
        events = torch.cat((event_batch[-i-1], past_events), dim=0)
        new_warp = warp_events(events, flow_batch[-i-1])
        past_events = new_warp
    warped = past_events[1:, :]

    return warped


# def warp_gs_flow(depth, gs_flow, camera_pose, next_camera_pose, K):
#     device = depth.device
#     H = depth.shape[1]
#     W = depth.shape[2]
#     def recover_extrinsic(R,T):
#         pose = np.eye(4)
#         # R = -R.T
#         # T = -T
#         R = R.T
#         T = T
#         pose[:3, :3] = R
#         pose[:3, 3] = T
#         pose = np.linalg.inv(pose)
#         return torch.from_numpy(pose).to(device='cuda')

#     inv_K = torch.linalg.inv(K)
#     extrinsic1 = recover_extrinsic(camera_pose.R, camera_pose.T)
#     extrinsic2 = recover_extrinsic(next_camera_pose.R, next_camera_pose.T)
    
#     T12 = torch.matmul(torch.linalg.inv(extrinsic2), extrinsic1)
#     ones = torch.ones_like(i)
#     i, j = torch.meshgrid(torch.arange(W, device=device), 
#                         torch.arange(H, device=device), 
#                         indexing='xy')
#     pixels_hom = torch.stack([i, j, ones], dim=-1).to(depth.dtype)  # (N, 3)
#     pixels_3d = pixels_hom @ inv_K.T * depth.unsqueeze(2) # * scale1  # (N, 3)
    

#     pass

def calculate_gs_flow(gs_per_pixel, weight_per_gs_pixel, next_conic_2D, conic_2D_inv, proj_2D, next_proj_2D, x_mu):
    # from motion-gs
    conic_2D_inv = conic_2D_inv.detach() # K 3

    gs_per_pixel = gs_per_pixel.long() # K H W
    # valid_mask = ~(gs_per_pixel < 0).any(dim=0) # H W
    conv_conv = torch.zeros([conic_2D_inv.shape[0], 2, 2], device=conic_2D_inv.device) # K 2 2
    conv_conv[:, 0, 0] = next_conic_2D[:, 0] * conic_2D_inv[:, 0] + next_conic_2D[:, 1] * conic_2D_inv[:, 1]
    conv_conv[:, 0, 1] = next_conic_2D[:, 0] * conic_2D_inv[:, 1] + next_conic_2D[:, 1] * conic_2D_inv[:, 2]
    conv_conv[:, 1, 0] = next_conic_2D[:, 1] * conic_2D_inv[:, 0] + next_conic_2D[:, 2] * conic_2D_inv[:, 1]
    conv_conv[:, 1, 1] = next_conic_2D[:, 1] * conic_2D_inv[:, 1] + next_conic_2D[:, 2] * conic_2D_inv[:, 2]

    # isotropic gs flow
    # flow_per_pixel = next_proj_2D[gs_per_pixel] - proj_2D[gs_per_pixel].detach() # K H W 3

    # anisotropic gs flow
    conv_multi = (conv_conv[gs_per_pixel] @ x_mu.permute(0,2,3,1).unsqueeze(-1).detach()).squeeze() # K H W 2
    flow_per_pixel = (conv_multi + next_proj_2D[gs_per_pixel] - proj_2D[gs_per_pixel].detach() - x_mu.permute(0,2,3,1).detach()) # K H W 2

    weight_per_gs_pixel = weight_per_gs_pixel / (weight_per_gs_pixel.sum(dim=0, keepdim=True) + 1e-7) # K H W
    flow_gs = torch.einsum("khw, khwa -> ahw", [weight_per_gs_pixel.detach(), flow_per_pixel]) # 2 H W
    return flow_gs

def get_deltaL(grad_x, grad_y, flow):
    """
    grad_x, grad_y: [1, h, w]
    flow: [2, h, w]
    """
    # create warping matrix
    h, w = grad_x.shape[1:]
    y_coords = torch.arange(h, dtype=torch.float32, device=grad_x.device)
    x_coords = torch.arange(w, dtype=torch.float32, device=grad_x.device)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
    indices = torch.stack([y_grid, x_grid], dim=0)

    flow = flow.permute(2, 0, 1)

    warp_y = indices[0:1, :, :] - flow[1:2, :, :]
    warp_x = indices[1:2, :, :] - flow[0:1, :, :]

    warp_y = 2 * warp_y / (h - 1) - 1
    warp_x = 2 * warp_x / (w - 1) - 1

    grid_pos = torch.cat([warp_x, warp_y], dim=0).permute(1, 2, 0)
    
    # grad_x = grad_x.to(grid_pos.dtype)
    # grad_y = grad_y.to(grid_pos.dtype)
    # warp gradients
    warp_grad_x = F.grid_sample(grad_x.unsqueeze(0), grid_pos.unsqueeze(0), 
                                mode="bilinear", padding_mode="zeros",align_corners=True)
    warp_grad_y = F.grid_sample(grad_y.unsqueeze(0), grid_pos.unsqueeze(0), 
                                mode="bilinear", padding_mode="zeros",align_corners=True)

    pred_deltaL = warp_grad_x * flow[0:1, :, :] + warp_grad_y * flow[1:2, :, :]
    
    return pred_deltaL

def project_points_to_image(points, view_cam, intri):

    """
    points: [n, 3]
    extrinsic: [4, 4]
    intrinsic: [3, 3]
    """
    def recover_pose(R,T):
        pose = np.eye(4)
        # R = -R.T
        # T = -T
        R = R.T
        T = T
        pose[:3, :3] = R
        pose[:3, 3] = T
        return torch.from_numpy(pose).to(device='cuda')

    points_homo = torch.cat((points, torch.ones_like(points[:, 0:1])), dim=1)
    
    pose = recover_pose(view_cam.R, view_cam.T)
    pose = pose.type(points_homo.dtype)
    P_camera = pose @ points_homo.T

    depth = P_camera[2, :].clone()
    P_camera[:3, :] = P_camera[:3, :] / P_camera[2, :]
    mean2d = (intri @ P_camera).T
    mean2d = mean2d[:, :2]

    # only keep the points in the image
    valid_mask = torch.logical_and(
        torch.logical_and(mean2d[:, 0] >= 0, mean2d[:, 0] < view_cam.image_width),
        torch.logical_and(mean2d[:, 1] >= 0, mean2d[:, 1] < view_cam.image_height)
    )
    # mean2d = mean2d[valid_mask]
    # depth = depth[valid_mask]
    
    return mean2d, depth, valid_mask


