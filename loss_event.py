import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import SSIM
import numpy as np
from utils.flow_utils import get_optical_flow_batch, warp_events_batch, warp_events, get_deltaL, project_points_to_image
from utils.image_utils import bayer_filter, gamma_correction, calculate_spatial_gradients, rgb2gray, inv_bayer_filter

class EventLoss(nn.Module):

    def __init__(self, device):
        super(EventLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.device = device

    def event_component(self, events:torch.Tensor, rendered:torch.Tensor, rendered_before:torch.Tensor,
                           events_threshold_negpos:torch.Tensor, map_num = 5):
        h, w = rendered.shape[1], rendered.shape[2]
        esi_map, event_mask, esi_maps_tuple = self.events_single_integral_new(events, h, w, events_threshold_negpos, map_num)
        E = esi_map
        if rendered.shape[0] == 1:
            E_hat = torch.log(rendered + 1e-6) - torch.log(rendered_before + 1e-6) #[1,H,W]
        else: # rendered is rgb image
            bayer_img = bayer_filter(rendered)
            bayer_img_before = bayer_filter(rendered_before)
            E_hat = torch.log(bayer_img + 1e-6) - torch.log(bayer_img_before + 1e-6) #[1,H,W]
        
        event_mask = event_mask.unsqueeze(0)
        
        return E, E_hat, event_mask, esi_maps_tuple


    def l2_normalize(self, img):
        norm = torch.norm(img + 1e-6, p=2)
        return img / norm

    def event_textureless_loss(self, E, E_hat, event_mask, events_threshold_negpos):
        res = torch.abs(E - E_hat)
        res = res * ~ event_mask
        loss = torch.relu(res - events_threshold_negpos[0]) 
        return loss.mean()
    
    def white_balance_loss(self, rendered_mean, alpha = 20):
        # rendered_mean: value range [0, 1]
        # enforce gray world
        # ref inc-eventgs
        smooth_abs = torch.sqrt((rendered_mean - 0.5) ** 2 + 1e-6)
        return torch.sigmoid(alpha * (smooth_abs - 0.25))
    
    def events_single_integral_new(self, events:torch.Tensor, h:int, w:int, events_threshold_negpos:torch.Tensor, map_num = 5):

        esi_maps = []
        ts = []

        x = events[:, 1].long()
        y = events[:, 2].long()
        p = events[:, 3]
        t = events[:, 0]

        step = x.shape[0] // map_num
        for i in range(map_num):
            esi_map = torch.zeros((h+1, w+1, 2), dtype=torch.float32, device=events.device)
            x_start = x[i*step:(i+1)*step]
            y_start = y[i*step:(i+1)*step]
            p_start = p[i*step:(i+1)*step]
            ts.append(t[i*step])
            neg_mask = p_start < 0
            pos_mask = ~neg_mask 

            p_scaled = torch.zeros_like(p_start)
            p_scaled[neg_mask] = p_start[neg_mask] * events_threshold_negpos[0]
            p_scaled[pos_mask] = p_start[pos_mask] * events_threshold_negpos[1]

            esi_map.index_put_((y_start, x_start, neg_mask.long()), p_scaled, accumulate=True)
            esi_map = esi_map[:h, :w]
            esi_map = esi_map.permute(2,0,1)
            esi_map = esi_map.sum(dim=0, keepdim=True)
            esi_maps.append(esi_map)
        esi_maps = torch.stack(esi_maps, dim=0) # [b, 1, H, W]
        esi_map = torch.sum(esi_maps, dim=0) # [1, H, W]
        event_mask = torch.zeros((h+1, w+1), dtype=torch.bool, device=events.device)
        event_mask[y, x] = True 
        event_mask = event_mask[:h, :w]

        return esi_map, event_mask, (esi_maps, ts)

    def events_single_integral(self, events:torch.Tensor, h:int, w:int, events_threshold_negpos:torch.Tensor, map_num = 5):

        esi_maps = []
        esi_map = torch.zeros((h, w, 2), dtype=torch.float32, device=events.device)
        x = events[:, 1].long()
        y = events[:, 2].long()
        p = events[:, 3]

        neg_mask = p < 0
        pos_mask = ~neg_mask 

        p_scaled = torch.zeros_like(p)
        p_scaled[neg_mask] = p[neg_mask] * events_threshold_negpos[0]
        p_scaled[pos_mask] = p[pos_mask] * events_threshold_negpos[1]

        esi_map.index_put_((y, x, neg_mask.long()), p_scaled, accumulate=True)
        esi_map = esi_map.permute(2,0,1)
        esi_map = esi_map.sum(dim=0, keepdim=True)

        event_mask = torch.zeros((h, w), dtype=torch.bool, device=events.device)
        event_mask[y, x] = True 

        return esi_map, event_mask, esi_maps

    def event_ssim_loss(self, E, E_hat,data_range=1.0, size_average=True, channel=1):

        X = E.unsqueeze(0) # (1,H,W) to (1,1,H,W)
        Y = E_hat.unsqueeze(0) # (1,H,W) to (1,1,H,W)
        
        ssim_module = SSIM(data_range=data_range, size_average=size_average, channel=channel)
        ssim_loss = 1 - ssim_module(X, Y)
        return ssim_loss
    
    def event_spatial_loss(self, rendered_intensity, E_hat, weight = 0.05):

        log_intensity_pred = torch.log(rendered_intensity + 1e-6) # [1, h, w]
        x_grad = log_intensity_pred[:, 1:, :] - log_intensity_pred[:, 0:-1, :]
        y_grad = log_intensity_pred[:, :, 1:] - log_intensity_pred[:, :, 0:-1]
        spatial_loss = weight * (
            x_grad.abs().mean() + y_grad.abs().mean() + E_hat.abs().mean()
        )
        return spatial_loss

    def event_count_map(self, events, h, w):

        # check the validation of the events
        # easy to have cuda out of memory, must remove illegal values
        # Remove events with negative coordinates
        valid_mask = (events[:, 1] >= 0) & (events[:, 2] >= 0) & (events[:, 1] < w) & (events[:, 2] < h)
        events = events[valid_mask]
        count_map = torch.zeros(h+1, w+1, 1).to(events.device)

        x = events[:, 1].long()
        y = events[:, 2].long()
        p = events[:, 3]
        ones = torch.ones_like(p) 

        count_map.index_put_((y, x), ones, accumulate=True)

        return count_map

    def compute_score(self, warp_count_map, ori_count_map):

        mean_warp = torch.mean(warp_count_map)
        mean_ori = torch.mean(ori_count_map) # for normalization

        return (warp_count_map - mean_warp) / (ori_count_map - mean_ori), - (mean_warp / mean_ori)
    
    def event_flow_loss(self, img_before, events, flow, rectify_mask, iteration):
        """
        img_before: [1, H, W] / [3, H, W] lastest reconstructed image
        optical_flows: a list of [H, W, 2] flows
        """
        if img_before.shape[0] == 3:
            # img_before = rgb2gray(img_before)
            grad_x_r, grad_y_r, grad_magnitude_r = calculate_spatial_gradients(img_before[0].unsqueeze(0))
            grad_x_b, grad_y_b, grad_magnitude_b = calculate_spatial_gradients(img_before[1].unsqueeze(0))
            grad_x_g, grad_y_g, grad_magnitude_g = calculate_spatial_gradients(img_before[2].unsqueeze(0))
            pred_deltaL_r = get_deltaL(grad_x_r, grad_y_r, flow)
            pred_deltaL_b = get_deltaL(grad_x_b, grad_y_b, flow)
            pred_deltaL_g = get_deltaL(grad_x_g, grad_y_g, flow)
            
            pred_deltaL = torch.cat([pred_deltaL_r, pred_deltaL_b, pred_deltaL_g], dim=1)
            pred_deltaL = bayer_filter(pred_deltaL[0])
            pred_deltaL = pred_deltaL.unsqueeze(0)
            # print(pred_deltaL.shape)
        else:
            grad_x, grad_y, grad_magnitude = calculate_spatial_gradients(img_before) # then bayer filter?
            pred_deltaL = get_deltaL(grad_x, grad_y, flow)
        h, w = img_before.shape[1], img_before.shape[2]
        # warped_events = warp_events(events, flow)
        # warped_esi_map, event_mask = self.events_single_integral(warped_events, h, w, torch.tensor([0.25, 0.25]))
        # event_deltaL = warped_esi_map
        event_deltaL, event_mask, _ = self.events_single_integral(events, h, w, torch.tensor([0.1, 0.1]))

        if rectify_mask is not None:
            event_mask = event_mask * rectify_mask

        # + is 2e-4 - is 2.63e-4
        Loss = event_mask * torch.abs(event_deltaL[0]/(torch.norm(event_deltaL[0],p=2)+1e-6) + pred_deltaL[0,0]/(torch.norm(pred_deltaL[0,0],p=2)+1e-6))
        # Loss = torch.abs(event_deltaL[0]/(torch.norm(event_deltaL[0],p=2)+1e-6) - pred_deltaL[0,0]/(torch.norm(pred_deltaL[0,0],p=2)+1e-6))
        # test visualize event_deltaL[0] and pred_deltaL[0,0]
        if iteration == 10000 or iteration == 15000 or iteration == 5000 or iteration == 12000:
            tmp1 = event_deltaL[0]/(torch.norm(event_deltaL[0],p=2)+1e-6)
            tmp2 = pred_deltaL[0,0]/(torch.norm(pred_deltaL[0,0],p=2)+1e-6)
            vis_test(tmp1.unsqueeze(0), tmp2.unsqueeze(0))
            flow_viz = viz_flow(flow)
            draw_flow_arrows(flow, step=16, background=flow_viz)
            # vis_test(event_deltaL[0].unsqueeze(0), pred_deltaL[0,0].unsqueeze(0))
        # return
        return Loss.mean()

    def get_weight(self, event_list, depth_maps, view_cams, K = None): # start from this func

        def save_event_list_depth_map_pose(event_list, depth_maps, poses):
            poses = np.array(poses)
            import os
            import shutil
            
            # Create directory if it doesn't exist
            mid_dir = '../test/mid/'
            if os.path.exists(mid_dir):
                # Clean the directory by removing all files
                for file in os.listdir(mid_dir):
                    file_path = os.path.join(mid_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
            else:
                # Create the directory if it doesn't exist
                os.makedirs(mid_dir)
            np.save('../test/mid/poses.npy', poses)
            for i, depth_map in enumerate(depth_maps):
                depth_map = depth_map.squeeze(0)
                depth_map_np = depth_map.detach().cpu().numpy()
                np.save(f'../test/mid/depth_map_{i}.npy', depth_map_np)
            for i, event in enumerate(event_list):
                event = event.detach().cpu().numpy()
                np.save(f'../test/mid/event_{i}.npy', event)
        
        h, w = view_cams[0].image_height, view_cams[0].image_width

        flow_batch, poses = get_optical_flow_batch(depth_maps, view_cams, K)
        warped_events = warp_events_batch(event_list, flow_batch)
        warped_count_map = self.event_count_map(warped_events, h, w) 
        ori_count_map = self.event_count_map(torch.cat(event_list, dim=0), h, w)
        score, mean_count = self.compute_score(warped_count_map, ori_count_map) # could be loss
        
        score_mask = torch.zeros_like(score, device=score.device)
        score_mask[score!=-mean_count] = True
        
        # test part
        # flow_viz = viz_flow(flow_batch[0])
        # draw_flow_arrows(flow_batch[0], step=16, background=flow_viz)
        # viz_weight(weight_map, score)
        # save_event_list_depth_map_pose(event_list, depth_maps, poses)
        # return

        return score, score_mask, warped_count_map, flow_batch

    def event_motion_loss(self, gs_centers, depth_map_before, depth_map, patch_type, esi_maps, view_cams, K, is_color = False):
        """
        gs_center:  [b, n, 3]
        esi_map: [b, 1, H, W]  TODO: using time image representation
        view_cams: list for each esi_map  
        for xyz in each patch compute it's patch density(corvariance, sift feature and so on ), 
        then let them almost similar (should invariant to rotation translation scale)
        """
        # sample gs_center
        gs_index = self.sample_gs_center(gs_centers[0], type = 'random')
        gs_downsample = gs_centers[:, gs_index, :]
        h,w = esi_maps[0].shape[1:]
        patches = []
        valid_masks = []
        mean_2ds = []
        for i, view_cam in enumerate(view_cams):
            mean_2d, depth, valid_mask = project_points_to_image(gs_downsample[i], view_cam, K)
            if i == 0:
                depth_mask = self.filter_valid_center(mean_2d, depth, depth_map_before, depth_map)
            valid_masks.append(valid_mask)
            mean_2ds.append(mean_2d)
        valid_mask = torch.all(torch.stack(valid_masks, dim=0), dim=0)
        final_mask = torch.logical_and(valid_mask, depth_mask)
        # print(torch.sum(final_mask))
        mean_2ds = torch.stack(mean_2ds, dim=0)
        mean_2ds = mean_2ds[:,final_mask,:]
        patch_size = self.get_patch_size(mean_2ds[0], esi_maps[0], patch_type)

        for i in range(mean_2ds.shape[0]):
            patch_size = self.get_patch_size(mean_2ds[i], esi_maps[i], patch_type)
            if is_color:
                esi_map = inv_bayer_filter(esi_maps[i]) # upsample to orginal 
                esi_map = F.interpolate(esi_map.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False)
                esi_map = esi_map.squeeze(0)
            else:
                esi_map = esi_maps[i] 
            patch = self.split_batch(mean_2ds[i], esi_map, patch_size) # [n,c,h,w]
            patches.append(patch)

        patches = torch.stack(patches, dim=0) # [b, n, c, h, w]
        # print(patches.shape)
        # min_p = patches.min()
        # max_p = patches.max()
        # patches = (patches - min_p) / (max_p - min_p)
        view1 = patches[:-1]
        view2 = patches[1:]
        loss = self.mse_loss(view1, view2)

        # test
        random_idx = torch.randperm(mean_2ds.shape[0])[0]
        test = patches[:,random_idx,:,:,:] # bchw
        # print(test.shape)
        # return
        b, c, h, w = test.shape
        test = test.permute(1, 2, 0, 3).reshape(c, h, -1)
        
        return loss, test
    
    def sample_gs_center(self, gs_center, num_gs=5000, type = 'random', opacity_mask = None):
        """
        random sample
        according to opacity
        """
        if type == 'random':
            if num_gs > gs_center.shape[0]:
                index = torch.arange(gs_center.shape[0])
            else:
                random_idx = torch.randperm(gs_center.shape[0])[:num_gs]
                index = random_idx
        elif type == 'opacity':
            index = opacity_mask
        elif type == 'all':
            index = torch.arange(gs_center.shape[0])
        else:
            raise ValueError(f"Invalid type: {type}")
        
        return index

    def get_patch_size(self, gs_2d_center, esi_maps, type='constant'):
        """
        according to esi_maps
        """
        if type == 'constant':
            patch_size = torch.ones(gs_2d_center.shape[0],device=gs_2d_center.device) * 40
        elif type == 'adaptive':
            pass
        else:
            raise ValueError(f"Invalid type: {type}")
        
        return patch_size

    def split_batch(self, gs_2d_center, esi_map, patch_size):
        """
        Args:
            gs_center (torch.Tensor): Batch of center coordinates (x, y), shape [n, 2].
            esi_map (torch.Tensor): The source map, shape [c, H, W].
            patch_size (torch.Tensor): Batch of patch sizes, shape [n].

        Returns:
            torch.Tensor: A batch of extracted patches, shape [n, c, max_patch_size, max_patch_size].
        """
        c, h, w = esi_map.shape[:]
        n = gs_2d_center.shape[0]
        device = esi_map.device
        dtype = esi_map.dtype
        max_size = torch.max(patch_size).int().item()

        # Calculate patch boundaries for all patches at once (vectorized)
        y_min = gs_2d_center[:, 1] - patch_size / 2
        x_min = gs_2d_center[:, 0] - patch_size / 2
        y_max = y_min + patch_size
        x_max = x_min + patch_size

        # pad the esi_map
        max_pad_top = torch.max(torch.tensor(0, device=device), -torch.min(y_min)).int().item()
        max_pad_left = torch.max(torch.tensor(0, device=device), -torch.min(x_min)).int().item()
        max_pad_bottom = torch.max(torch.tensor(0, device=device), torch.max(y_max) - h).int().item()
        max_pad_right = torch.max(torch.tensor(0, device=device), torch.max(x_max) - w).int().item()

        padded_map = F.pad(esi_map, (max_pad_left, max_pad_right, max_pad_top, max_pad_bottom))
        padded_h, padded_w = padded_map.shape[-2:]

        yy, xx = torch.meshgrid(
            torch.arange(max_size, device=device, dtype=dtype),
            torch.arange(max_size, device=device, dtype=dtype),
            indexing='ij'
        )
        base_grid = torch.stack([xx, yy], dim=-1)  # Shape: [max_size, max_size, 2]

        top_left_x = gs_2d_center[:, 0] + max_pad_left - patch_size / 2
        top_left_y = gs_2d_center[:, 1] + max_pad_top - patch_size / 2
        
        top_left = torch.stack([top_left_x, top_left_y], dim=-1).view(n, 1, 1, 2)

        sampling_grid_pixels = base_grid + top_left # Shape: [n, max_size, max_size, 2]

        sampling_grid_norm = sampling_grid_pixels.clone()
        sampling_grid_norm[..., 0] = 2 * sampling_grid_pixels[..., 0] / (padded_w - 1) - 1
        sampling_grid_norm[..., 1] = 2 * sampling_grid_pixels[..., 1] / (padded_h - 1) - 1

        input_tensor = padded_map.unsqueeze(0).expand(n, -1, -1, -1)
    
        extracted_patches = F.grid_sample(
            input_tensor,
            sampling_grid_norm,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )
        return extracted_patches

    def filter_valid_center(self, gs_2d_center, depth, depth_map_before, depth_map):
        """
        1. filter out the center out of image(while projecting, this part already done)
        2. filter out the center with obviously lower depth in depth_map
        TODO: check the mask
        """
        # filter out the center with obviously lower depth in depth_map
        x_center = gs_2d_center[:, 0].long()
        y_center = gs_2d_center[:, 1].long()
        final_mask = torch.zeros(x_center.shape[0], dtype=torch.bool, device=x_center.device)
        valid_mask = torch.logical_and(
            torch.logical_and(x_center >= 0, x_center < depth_map.shape[2]),
            torch.logical_and(y_center >= 0, y_center < depth_map.shape[1])
        ).to(x_center.device)
        mean_depth_diff = torch.mean(depth_map[0, y_center[valid_mask], x_center[valid_mask]]-depth[valid_mask], dim=0)
        depth_mask1 = (depth[valid_mask] + mean_depth_diff) > depth_map_before[0, y_center[valid_mask], x_center[valid_mask]]
        depth_mask2 = (depth[valid_mask] + mean_depth_diff) > depth_map[0, y_center[valid_mask], x_center[valid_mask]]
        depth_mask = torch.logical_and(depth_mask1, depth_mask2)
        final_mask[valid_mask] = depth_mask
        return final_mask
    
    
def vis_test(E, E_hat): # TODO: align them at same image
    import matplotlib.pyplot as plt

    # Convert tensors to numpy arrays
    E_np = E.detach().cpu().numpy().transpose(1,2,0)
    E_hat_np = E_hat.detach().cpu().numpy().transpose(1,2,0)
    # Create a figure with two subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    # Plot E
    im1 = ax1.imshow(E_np, cmap='viridis')
    ax1.set_title('E')
    # Plot E_hat
    im2 = ax2.imshow(E_hat_np, cmap='viridis')
    ax2.set_title('E_hat')  
    # Plot E-E_hat
    im3 = ax3.imshow(E_np+E_hat_np, cmap='viridis')
    ax3.set_title('E-E_hat')     
    # Add a single color bar for all plots
    fig.colorbar(im3, ax=[ax1, ax2, ax3], orientation='vertical', fraction=0.02, pad=0.04)
    # Save the figure
    plt.savefig('event_comparison.png')
    plt.close()

def viz_event_gt(events, render_intensity, gt_intensity_before):
    import matplotlib.pyplot as plt
    import numpy as np
    events_np = events.detach().cpu().numpy()
    render_intensity = render_intensity.squeeze(0)
    render_intensity_np = render_intensity.detach().cpu().numpy()
    gt_intensity_before = gt_intensity_before.squeeze(0)
    gt_intensity_before_np = gt_intensity_before.detach().cpu().numpy()
    
    xnp = events_np[:,1]
    ynp = events_np[:,2]
    pnp = events_np[:,3]

    idx_pos = np.asarray(pnp) > 0
    idx_neg = np.logical_not(idx_pos)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 15))
    ax1.imshow(render_intensity_np, cmap='gray')
    ax1.scatter(xnp[idx_pos], ynp[idx_pos], c='b', s=0.1, alpha=0.1, label='Positive Events')
    ax1.scatter(xnp[idx_neg], ynp[idx_neg], c='r', s=0.1, alpha=0.1, label='Negative Events')

    ax2.imshow(gt_intensity_before_np, cmap='gray')
    ax2.scatter(xnp[idx_pos], ynp[idx_pos], c='b', s=0.1, alpha=0.1, label='Positive Events')
    ax2.scatter(xnp[idx_neg], ynp[idx_neg], c='r', s=0.1, alpha=0.1, label='Negative Events')

    plt.savefig('event_on_intensity.png')
    plt.close()

def viz_flow(flow):
    import cv2
    flow_np = flow.detach().cpu().numpy()
    h, w = flow_np.shape[:2]
    fx, fy = flow_np[..., 0], flow_np[..., 1]

    magnitude = np.sqrt(fx**2 + fy**2)
    angle = np.arctan2(fy, fx)  # angle in radians

    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = ((angle + np.pi) / (2 * np.pi) * 180).astype(np.uint8)  # Hue
    hsv[..., 1] = 255  # Full saturation
    hsv[..., 2] = np.clip((magnitude / np.max(magnitude)) * 255, 0, 255).astype(np.uint8)  # Value

    flow_viz = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return flow_viz

def viz_weight(weight_map, score_map):
    import matplotlib.pyplot as plt
    import numpy as np

    weight_map_np = weight_map.detach().cpu().numpy()
    score_map_np = score_map.detach().cpu().numpy()
    
    background_mask1 = weight_map_np > 0
    background_mask2 = score_map_np > 0

    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    im1 = axes[0, 0].imshow(weight_map_np, cmap='viridis')
    im2 = axes[0, 1].imshow(background_mask1)
    im3 = axes[1, 0].imshow(score_map_np, cmap='viridis')
    im4 = axes[1, 1].imshow(background_mask2)
    fig.colorbar(im1, ax=[axes[0, 0]], orientation='vertical', fraction=0.02, pad=0.04)
    fig.colorbar(im3, ax=[axes[1, 0]], orientation='vertical', fraction=0.02, pad=0.04)
    plt.savefig('weight_map.png')
    plt.close()

def draw_flow_arrows(flow, step=16, background=None, scale=1, title='Optical Flow (Arrows)'):
    import matplotlib.pyplot as plt
    import numpy as np
    """
    Draws optical flow as arrows on a quiver plot.

    Parameters:
    - flow: (H, W, 2) optical flow array
    - step: spacing between arrows (pixels)
    - background: (H, W, 3) image to show behind arrows (optional)
    - scale: scale factor for arrow size
    """
    flow_np = flow.detach().cpu().numpy()
    h, w = flow_np.shape[:2]
    fx, fy = flow_np[..., 0], flow_np[..., 1]

    y, x = np.mgrid[step//2:h:step, step//2:w:step]
    u = fx[y, x] * scale
    v = fy[y, x] * scale

    plt.figure(figsize=(5, 3.5))
    if background is not None:
        plt.imshow(background, cmap='gray')
    else:
        plt.imshow(np.zeros((h, w)), cmap='gray')

    plt.quiver(x, y, u, v, color='red', angles='xy', scale_units='xy', width=0.0025)
    plt.title(title)
    plt.axis('off')
    plt.savefig('flow_arrows.png')
    plt.close()
    

