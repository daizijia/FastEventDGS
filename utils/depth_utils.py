import torch
import torch.nn as nn
# import os
from vggt.models.vggt import VGGT
# from vggt.utils.load_fn import load_and_preprocess_images

class DepthEstimator:

    def __init__(self, model_path = None):
        self.model_path = model_path

    def initialize(self):
        self.model = VGGT()
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 10 else torch.float16
        _URL = "https://hf-mirror.com/facebook/VGGT-1B/resolve/main/model.pt"
        if self.model_path is None:
            self.model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
        else:
            self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        model_size = get_model_size_mb(self.model)
        print(f"Model size: {model_size:.2f} MB")
        
        device_num = find_available_gpu(model_size) # return integer of gpu id
        if device_num is not None:
            self.device = f"cuda:{device_num}"
        self.model = self.model.to(self.device)

    def estimate_depth(self, render_image):
        # [c, h, w] -> [1, c, h, w]
        if render_image.shape[0] == 1:
            render_tmp = torch.zeros(3, render_image.shape[1], render_image.shape[2])
            render_tmp[0,:,:] = render_image[0,:,:] * 0.2989
            render_tmp[1,:,:] = render_image[0,:,:] * 0.5870
            render_tmp[2,:,:] = render_image[0,:,:] * 0.1140
            render_image = render_tmp
            # render_image.repeat(3,1,1)
            # print("render_image shape", render_image.shape)
        render_image = single_image_preprocess(render_image, mode="crop")
        render_image = render_image.unsqueeze(0)

        render_image = render_image.to(self.device)
        # print("render_image shape", render_image.shape)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                predictions = self.model(render_image)
                depth = predictions['depth']
        return depth.squeeze(-1)

    def depth_silog_loss(self, depth_gaussian, render_image, mask=None, lambd=1.0): # mask can direct use the event mask
        # depth_gaussian: [1, h, w]
        # render_image: [c, h, w]
        depth_vggt = self.estimate_depth(render_image).squeeze(0)
        depth_vggt = depth_vggt.to(depth_gaussian.device)
        # print("depth_vggt shape", depth_vggt.shape)
        # print("depth_gaussian shape", depth_gaussian.shape)
        depth_gaussian = single_image_preprocess(depth_gaussian, mode="crop")
        pred = torch.clamp(depth_gaussian, min=1e-8)
        gt = torch.clamp(depth_vggt, min=1e-8)
        if mask is None:
            mask = torch.ones_like(pred, dtype=torch.bool)
        # Calculate log differences for valid pixels
        log_diff = torch.log(pred[mask]) - torch.log(gt[mask])
        # Calculate variance
        num_pixels = torch.sum(mask).float()

        term1 = torch.sum(log_diff**2) / num_pixels
        term2 = (torch.sum(log_diff) / num_pixels)**2
        
        return term1 - lambd * term2

def single_image_preprocess(tensor_image, mode="crop"):
    # modify from vggt.utils.load_fn.py
    # image: [c, h, w]
    target_size = 518
    h, w = tensor_image.shape[1], tensor_image.shape[2]

    if mode == "pad":
        if w >= h:
            new_width = target_size
            new_height = round(h * (new_width / w) / 14) * 14
        else:
            new_height = target_size
            new_width = round(w * (new_height / h) / 14) * 14
    else:
        new_width = target_size
        new_height = round(h * (new_width / w) / 14) * 14

    # print("tensor image shape", tensor_image.shape)

    resize_img = torch.nn.functional.interpolate(tensor_image.unsqueeze(0), size=(new_height, new_width), 
                                                 mode="bilinear", align_corners=False).squeeze(0)
    
    if mode == "crop" and new_height > target_size:
        start_y = (new_height - target_size) // 2
        resize_img = resize_img[:, start_y : start_y + target_size, :]
    if mode == "pad":
        h_padding = target_size - resize_img.shape[1]
        w_padding = target_size - resize_img.shape[2]
        if h_padding > 0 or w_padding > 0:
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left
            resize_img = torch.nn.functional.pad(
                resize_img, (pad_left, pad_right, pad_top, pad_bottom), 
                mode="constant", value=1.0)
            
    return resize_img

def get_model_size_mb(model: nn.Module):
    """
    The estimated size of the model's parameters in MB.
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    # Return size in megabytes
    return param_size / 1024**2

def find_available_gpu(required_mb: float, headroom_mb: int = 1024 * 3):
    """
    Finds the first available GPU that has enough memory.

    Args:
        required_mb: The amount of memory the user wants to allocate in MB.
        headroom_mb: A safety margin in MB to leave free.

    Returns:
        An integer representing the GPU index (e.g., 0, 1) if found, otherwise None.
    """
    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available on this system.")
        return None

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("[ERROR] No CUDA GPUs were found.")
        return None

    print(f"[INFO] Checking {num_gpus} GPU(s) for {required_mb + headroom_mb:.2f} MB of free space...")
    
    for gpu_id in range(num_gpus):
        device = f"cuda:{gpu_id}"
        free_mem_bytes, _ = torch.cuda.mem_get_info(device)
        free_mem_mb = free_mem_bytes / 1024**2
        
        if free_mem_mb > (required_mb + headroom_mb):
            print(f"[SUCCESS] Found suitable device: {device} ({free_mem_mb:.2f} MB free).")
            return gpu_id
        else:
            print(f"[INFO] Skipping {device}: Not enough memory ({free_mem_mb:.2f} MB free).")
            
    print(f"[FAIL] No GPU found with at least {required_mb + headroom_mb:.2f} MB free.")
    return None