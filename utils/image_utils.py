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
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def bgr2gray(img):

    weights = torch.tensor([0.1140, 0.5870, 0.2989], device=img.device).view(3, 1, 1)
    return torch.sum(img * weights, dim=0, keepdim=True)

def rgb2gray(img):
    """
    Convert an RGB image tensor [3, H, W] to grayscale [1, H, W].
    """
    weights = torch.tensor([0.2989, 0.5870, 0.1140], device=img.device).view(3, 1, 1)
    return torch.sum(img * weights, dim=0, keepdim=True) 

def bayer_filter(image):
    """
    Input: image [3, H, W]
    Output: Bayer image [1, H, W]
    """
    R_c, G_c, B_c = image[0], image[1], image[2]

    # pattern: RGGB
    bayer_image = torch.zeros_like(R_c, device=image.device)
    bayer_image[0::2, 0::2] = R_c[0::2, 0::2]
    bayer_image[0::2, 1::2] = G_c[0::2, 1::2]
    bayer_image[1::2, 0::2] = G_c[1::2, 0::2]
    bayer_image[1::2, 1::2] = B_c[1::2, 1::2]

    return bayer_image.unsqueeze(0)

def inv_bayer_filter(bayer_image):
    """
    Input: Bayer image [1, H, W]
    Output: image [3, H, W]
    """
    R_c = bayer_image[:,0::2, 0::2]
    G_c1 = bayer_image[:,0::2, 1::2]
    G_c2 = bayer_image[:,1::2, 0::2]
    B_c = bayer_image[:,1::2, 1::2]
    image = torch.cat([R_c, G_c1, G_c2, B_c], dim=0)
    
    return image

def gamma_correction(image, gamma=2.2):
    return image ** gamma

def calculate_spatial_gradients(gray_image_tensor):
    """
    Calculates spatial gradients (Gx, Gy) and magnitude for a grayscale image tensor.
    """
    if not isinstance(gray_image_tensor, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor.")
    if gray_image_tensor.ndim != 3 or gray_image_tensor.shape[0] != 1:
        raise ValueError(f"Input tensor must have shape [1, H, W], but got {gray_image_tensor.shape}")

    # Ensure the tensor is float
    if not gray_image_tensor.is_floating_point():
        image_tensor = gray_image_tensor.float()
    else:
        image_tensor = gray_image_tensor

    # Reshape to [batch_size=1, channels=1, H, W] for conv2d
    image_tensor_4d = image_tensor.unsqueeze(1) # Adds the channel dimension

    # Sobel kernels
    sobel_kernel_x_values = [[-1., 0., 1.],
                             [-2., 0., 2.],
                             [-1., 0., 1.]]
    sobel_kernel_y_values = [[-1., -2., -1.],
                             [ 0.,  0.,  0.],
                             [ 1.,  2.,  1.]]

    # Create Sobel kernels as tensors and move to the same device as the input tensor
    device = image_tensor.device
    sobel_kernel_x = torch.tensor(sobel_kernel_x_values, dtype=image_tensor.dtype, device=device)
    sobel_kernel_y = torch.tensor(sobel_kernel_y_values, dtype=image_tensor.dtype, device=device)

    # Reshape kernels to [out_channels=1, in_channels=1, H_kernel, W_kernel]
    sobel_kernel_x = sobel_kernel_x.view(1, 1, 3, 3)
    sobel_kernel_y = sobel_kernel_y.view(1, 1, 3, 3)

    # Calculate Gx (gradient in x-direction)
    # padding='same' ensures the output has the same H, W as the input for stride=1
    grad_x_4d = F.conv2d(image_tensor_4d, sobel_kernel_x, padding='same')

    # Calculate Gy (gradient in y-direction)
    grad_y_4d = F.conv2d(image_tensor_4d, sobel_kernel_y, padding='same')

    # Calculate gradient magnitude
    grad_magnitude_4d = torch.sqrt(grad_x_4d**2 + grad_y_4d**2)

    # Remove the channel dimension to return [1, H, W]
    grad_x = grad_x_4d.squeeze(1)
    grad_y = grad_y_4d.squeeze(1)
    grad_magnitude = grad_magnitude_4d.squeeze(1)

    return grad_x, grad_y, grad_magnitude

def white_balance_correction(image, background = [128, 128, 128]):

    pass