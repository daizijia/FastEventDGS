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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

# origin code including RGB loss
def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def variance_loss(render_img, alpha = 1.0):
    # to deal with local minimal case, when only output gray image
    render_img = render_img.unsqueeze(0)
    var = torch.var(render_img, dim=(1, 2, 3), keepdim=True)
    loss = alpha * torch.exp(-var+1e-6)
    return loss.mean()

def patch_similarity_loss(batch_tensor, num_angles = 60):
    """
    patches: [b, n, c, h, w]
    """
    b, n, c, h, w = batch_tensor.shape
    if n < 2:
        return torch.tensor(0.0, device=batch_tensor.device)

    # 2. Reshape to process all patches at once
    # Merge 'b' and 'n' dimensions. Shape becomes [b * n, c, h, w]
    reshaped_patches = batch_tensor.view(b * n, c, h, w)

    # 3. Get rotation-invariant signatures for all patches
    num_radii = h // 2
    signatures_flat = get_rotation_invariant_signature(reshaped_patches, num_radii, num_angles)
    
    # Infer signature dimensions for reshaping
    sig_dims = signatures_flat.shape[1:] # (c, num_radii, num_freqs)

    # 4. Reshape signatures back to their group structure
    # Shape becomes [b, n, c, num_radii, num_freqs]
    signatures = signatures_flat.view(b, n, *sig_dims)

    # 5. Compute the mean signature for each group
    # We calculate the mean across the 'n' dimension (dim=1).
    # Shape becomes [b, 1, c, num_radii, num_freqs]
    mean_signatures = signatures.mean(dim=1, keepdim=True).detach()

    # 6. Compute the final MSE loss
    # Broadcasting expands mean_signatures to [b, n, ...] for comparison
    loss = F.mse_loss(signatures, mean_signatures.expand_as(signatures))
    
    return loss

def get_rotation_invariant_signature(patch_batch: torch.Tensor, num_radii: int, num_angles: int) -> torch.Tensor:
    """
    Computes the rotation-invariant signature for a batch of patches.
    
    Args:
        patch_batch (torch.Tensor): A batch of patches, shape (B, C, H, W).
    
    Returns:
        torch.Tensor: The rotation-invariant signature, shape (B, C, num_radii, num_freqs).
    """
    polar_patches = polar_transform(patch_batch, num_radii, num_angles)

    fft_coeffs = torch.fft.rfft(polar_patches, dim=-1) # dim=-1 is the angular dimension
    
    signature = fft_coeffs.abs()**2
    
    return signature

def polar_transform(image_batch: torch.Tensor, num_radii: int, num_angles: int) -> torch.Tensor:
    """ Differentiable polar transform using F.grid_sample """
    B, C, H, W = image_batch.shape
    device = image_batch.device
    radii = torch.linspace(0, 1, num_radii, device=device)
    angles = torch.linspace(0, 2 * torch.pi, num_angles + 1, device=device)[:-1]
    r_grid, theta_grid = torch.meshgrid(radii, angles, indexing='ij')
    x_cart = r_grid * torch.cos(theta_grid)
    y_cart = r_grid * torch.sin(theta_grid)
    grid = torch.stack([x_cart, y_cart], dim=-1)
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)
    polar_batch = F.grid_sample(image_batch, grid, mode='bilinear', padding_mode='border', align_corners=True)
    return polar_batch

def kl_divergence(rho, rho_hat):
    rho_hat = torch.mean(torch.sigmoid(rho_hat), 0)
    rho = torch.tensor([rho] * len(rho_hat)).cuda()
    return torch.mean(
        rho * torch.log(rho / (rho_hat + 1e-5)) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat + 1e-5)))


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def iso_loss(gaussians, weight = 0.05):
    # limit the scale
    scale = gaussians.get_scaling
    assert scale.shape[1] == 3
    loss = torch.std(scale, dim=1)

    return weight * loss.mean()