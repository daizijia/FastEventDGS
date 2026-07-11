import torch
import numpy as np
from skimage.filters import threshold_otsu

def get_event_list(events, timestamps, dynamic_mask = None):
    event_list = []
    event_ts = events[:, 0].detach().cpu().numpy()
    former_index = 0
    for i in range(timestamps.shape[0]-1):
        index = np.searchsorted(event_ts, timestamps[i+1])
        current_events = events[former_index:index,:]
        if dynamic_mask is not None:
            assert dynamic_mask.shape[0] == events.shape[0]
            current_dynamic_mask = dynamic_mask[former_index:index]
            valid_mask = ~current_dynamic_mask
            current_events = current_events[valid_mask]
        event_list.append(current_events)
        former_index = index
        
    return event_list

# no use
def dynamic_mask_generation(d_xyz, d_rotation, d_scaling, d_xyz_before, d_rotation_before, d_scaling_before, time_interval, threshold=0.4):
    # dynamic_mask = torch.zeros((d_xyz.shape[0],1))
    displacement = d_xyz - d_xyz_before
    displacement_norm = torch.norm(displacement, dim=1)
    dynamic_mask = (displacement_norm / time_interval) > threshold

    return dynamic_mask, displacement

def update_loss_histogram(events, loss_map):
    """
    events: event sequence
    loss_map: loss map tensor
    """
    num_events = events.shape[0]
    row_indices = events[:, 2].long()
    col_indices = events[:, 1].long()
    loss_map = loss_map.detach().squeeze(0)
    selected_losses = loss_map[row_indices, col_indices]
    counts_column = torch.ones(num_events, device=selected_losses.device, dtype=selected_losses.dtype)
    histogram = torch.stack((selected_losses, counts_column), dim=1)
    
    return histogram

class LossHistogram:

    def __init__(self, num_events):
        eps = 1e-8
        self.histogram = torch.zeros((num_events, 3), device="cuda")
        self.histogram[:, 0] = eps
        # count, mean, M2(std)

    def update(self, events, loss_map, index1, index2): # index1 < index2

        num_events = events.shape[0]
        row_indices = events[:, 2].long()
        col_indices = events[:, 1].long()
        loss_map = loss_map.detach().squeeze(0)
        selected_losses = loss_map[row_indices, col_indices]
        counts_column = torch.ones(num_events, device=selected_losses.device, dtype=selected_losses.dtype)
        
        self.histogram[index1:index2, 0] = self.histogram[index1:index2, 0] + counts_column
        delta1 = selected_losses - self.histogram[index1:index2, 1]
        self.histogram[index1:index2, 1] = self.histogram[index1:index2, 1] + delta1 / self.histogram[index1:index2, 0] # update mean
        delta2 = selected_losses - self.histogram[index1:index2, 1]
        self.histogram[index1:index2, 2] = self.histogram[index1:index2, 2] + delta1 * delta2 # update M2

    def query_dynamic_mask(self, index1, index2):

        thresh_otsu = threshold_otsu(self.histogram[index1:index2, 1].detach().cpu().numpy())
        dynamic_mask = self.histogram[index1:index2, 1] > thresh_otsu

        return dynamic_mask
    
    def get_entire_dynamic_mask(self, all_timestamps, interval = 26855*20):
        """
        return mask_ts: [n, 2] first is mask index, second is timestamp
        """
        num_histogram = self.histogram.shape[0]
        dynamic_mask = torch.zeros((num_histogram, 2), device="cuda")
        for i in range(num_histogram):
            thresh_otsu = threshold_otsu(self.histogram[i, 1].detach().cpu().numpy())
            dynamic_mask = self.histogram[i, 1] > thresh_otsu

        pass

    

    