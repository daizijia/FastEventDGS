import numpy as np
import random

def generate_paris(num_pairs, interval_range, length, keep_uni = True):
    pairs = []
    if not keep_uni:
        for _ in range(num_pairs):
            start = np.random.randint(0, length-1)  # Random start value
            interval = np.random.randint(interval_range[0], interval_range[1])  # Random interval between 10 and 40
            # Ensure start + interval < n
            while start + interval >= length:
                start = np.random.randint(0, length-1)  # Re-generate start if condition is violated
                interval = np.random.randint(interval_range[0], interval_range[1])  # Re-generate interval
            pairs.append([start, interval])
    else:
        step = length / (num_pairs)
        tmp = np.random.randint(interval_range[0], interval_range[1])
        pairs=[[0, np.random.randint(interval_range[0], interval_range[1])],
               [length-tmp-1, tmp]] # keep first and last frame
        for i in range(num_pairs-1):
            start = np.random.randint(i * step, (i+1) * step)  # start in sequence
            interval = np.random.randint(interval_range[0], interval_range[1])  # Random interval between 10 and 40
            # Ensure start + interval < n
            while start + interval >= length-1:
                start = np.random.randint(0, length-1)
                interval = np.random.randint(interval_range[0], interval_range[1])
            pairs.append([start, interval])
        random.shuffle(pairs)
    return pairs

def adaptive_sample(num_pairs, range_time, range_event, event_ts):
    """Sample consider both time and event number"""
    max_ts = event_ts[-1]
    min_ts = event_ts[0]
    # print(max_ts)
    step_time = max_ts / num_pairs
    step_event = len(event_ts) / num_pairs
    pairs = []
    for i in range(num_pairs):
        # start_time_index = np.random.randint(0, len(event_ts)-1) # make it to index
        start_time = np.random.randint(min_ts, max_ts-1)
        start_time_index = np.searchsorted(event_ts, start_time)

        interval_time = np.random.randint(range_time[0], range_time[1]) # make it to index
        # print(max_ts, start_time, event_ts[start_time_index], interval_time)

        while event_ts[start_time_index] + interval_time >= max_ts-1:
            start_time_index = np.random.randint(0, len(event_ts)-1)
            interval_time = np.random.randint(range_time[0], range_time[1])
        end_time_index = np.searchsorted(event_ts[start_time_index:], event_ts[start_time_index] + interval_time)
        interval_time_index = end_time_index
        pairs.append([start_time_index, interval_time_index])
        # print(start_time_index, interval_time_index)
        # start_event_index = np.random.randint(i * step_event, (i+1) * step_event)
        # interval_event_index = np.random.randint(range_event[0], range_event[1])
        # while start_event_index + interval_event_index >= len(event_ts)-1:
        #     start_event_index = np.random.randint(0, len(event_ts)-1)
        #     interval_event_index = np.random.randint(range_event[0], range_event[1])
        # pairs.append([start_event_index, interval_event_index])
        
    random.shuffle(pairs)
    return pairs

def mask_sample(mask_ts):
    """
    mask_ts: [n, 2] first is mask index, second is timestamp
    """
    

    pass

def sample_test(num_pairs, interval_range, viewpoint_stack):
    """Sample focus on front events"""
    event_list = []
    for view_cam in viewpoint_stack:
        event_index = view_cam.closest_event_index
        event_list.append(event_index)
    event_list = np.array(event_list)
    event_list = np.sort(event_list)
    
    pairs = []
    for i in range(num_pairs):
        idx1 = np.random.randint(0, len(event_list)-1)
        idx_interval = np.random.randint(interval_range[0], interval_range[1])
        while idx1 + idx_interval >= len(event_list)-1:
            idx1 = np.random.randint(0, len(event_list)-1)
            idx_interval = np.random.randint(interval_range[0], interval_range[1])
        idx2 = event_list[idx1 + idx_interval] 
        interval = idx2 - idx1
        pairs.append([idx1, interval])
    random.shuffle(pairs)
    return pairs

def long_short_sample(num_pairs, interval_range, length, iteration, total_iter):
    
    pairs = []

    if iteration <= 6000:
        interval = interval_range[1]
    else:
        step = (interval_range[1] - interval_range[0]) / (total_iter - 6000)
        #print(step)
        interval = interval_range[1] - step * (iteration - 6000)
        interval = int(interval)
        #print(interval)
    for i in range(num_pairs):
        start = np.random.randint(0, length-1)
        while start + interval >= length-1:
            start = np.random.randint(0, length-1)
        pairs.append([start, interval])
    
    random.shuffle(pairs)
    return pairs

def greedy_sample(pairs_loss_accumilation, num_pairs, range_time, range_event, event_ts, percent = 0.1):
    """
    return pairs according to the result_map distribution
    # TODO: according to the loss difference to sample more events
    """
    pairs = []
    max_ts = event_ts[-1]
    length = len(pairs_loss_accumilation)
    loss_map_before = get_result_map(pairs_loss_accumilation[:length//2])
    loss_map_after = get_result_map(pairs_loss_accumilation[length//2:])
    result_map = loss_map_before - loss_map_after
    step = event_ts.shape[0] / result_map.shape[0]
    num_pairs_extra = int(num_pairs * percent)
    flatten_result = result_map.flatten()
    flatten_array = np.argsort(flatten_result)[-int(len(pairs_loss_accumilation) * 3 * percent):][::-1]
    yid, xid = np.unravel_index(flatten_array, result_map.shape)

    for i in range(num_pairs_extra):

        idx = np.random.randint(0, len(flatten_array))
        sid = int(yid[idx] * step)
        eid = int(xid[idx] * step)
        
        interval_time = eid - sid
        rand = 4e5
        start_index = np.random.randint(sid - rand, sid + rand) if sid - rand > 0 and sid + rand < len(event_ts) else sid
        # interval = np.random.randint(interval_time - rand, interval_time + rand) if start_index + interval_time + rand< len(event_ts) else (len(event_ts)-start_index-1)
        interval = np.random.randint(range_event[0], range_event[1]) 
        if start_index + interval > len(event_ts) - 1:
            interval = len(event_ts) - start_index - 1
        pairs.append([start_index, interval])

    for i in range(num_pairs - num_pairs_extra):
        start_time = np.random.randint(0, max_ts-1)
        start_time_index = np.searchsorted(event_ts, start_time)
        interval_time = np.random.randint(range_time[0], range_time[1]) # make it to index
        while event_ts[start_time_index] + interval_time >= max_ts-1:
            start_time_index = np.random.randint(0, len(event_ts)-1)
            interval_time = np.random.randint(range_time[0], range_time[1])
        end_time_index = np.searchsorted(event_ts[start_time_index:], event_ts[start_time_index] + interval_time)
        interval_time_index = end_time_index
        pairs.append([start_time_index, interval_time_index])
    random.shuffle(pairs)

    return pairs, result_map

def get_result_map(pairs_loss_accumilation):

    result = np.asarray(pairs_loss_accumilation)
    size = 50
    img_sum = np.zeros((size + 1, size + 1), dtype=result.dtype)
    img_count = np.zeros((size + 1, size + 1), dtype=int)
    img_count = img_count + 1e-6

    max_coords = np.max(result[:, 0:2])
    step = size / max_coords

    # Calculate the integer coordinates directly using vectorized operations
    x_coords = np.floor(result[:, 0] * step).astype(int)
    y_coords = np.floor(result[:, 1] * step).astype(int)
    losses = result[:, 2]

    # Use advanced indexing to accumulate the losses and counts
    np.add.at(img_sum, (y_coords, x_coords), losses)
    np.add.at(img_count, (y_coords, x_coords), 1)

    ave = img_sum / img_count
    return ave

def pairs_loss_vis(result, result_map = None):
    """
    result_list: 3xn list, [event_index, event_index_before, loss]
    """
    import matplotlib.pyplot as plt
    #print(result)
    result = np.asarray(result)
    # result = result[5000:,:]
    size = 50
    img = np.zeros((size+1, size+1, 2))
    img[:,:,1] = 1e-6
    fig, (ax1, ax2) = plt.subplots(1,2)
    step = size / np.max(result[:, 0:2])
    # print(step)
    for i in range(len(result)):
        x = int(result[i][0] * step)
        y = int(result[i][1] * step)
        img[y, x, 0] = img[y, x, 0] + result[i][-1]
        img[y, x, 1] = img[y, x, 1] + 1
    ave = img[:, :, 0] / img[:, :, 1]
    # ax1.xlabel('end time')
    # ax1.ylabel('start time')
    if result_map is not None:
        im1 = ax1.imshow(result_map, cmap='gray')
    else:
        im1 = ax1.imshow(ave, cmap='gray')
    fig.colorbar(im1, ax=ax1)
    im2 = ax2.imshow(img[:,:,1], cmap='gray')
    fig.colorbar(im2, ax=ax2)
    plt.savefig('loss_vis.png')


# backup
# def greedy_sample(pairs_loss_accumilation, num_pairs, range_time, range_event, event_ts, percent = 0.1):
#     """
#     return pairs according to the result_map distribution
#     """
#     pairs = []
#     max_ts = event_ts[-1]
#     result_map = get_result_map(pairs_loss_accumilation)
#     step = event_ts.shape[0] / result_map.shape[0]
#     num_pairs_extra = int(num_pairs * percent)
#     flatten_result = result_map.flatten()
#     flatten_array = np.argsort(flatten_result)[-int(len(pairs_loss_accumilation) * 3 * percent):][::-1]
#     yid, xid = np.unravel_index(flatten_array, result_map.shape)

#     for i in range(num_pairs_extra):

#         idx = np.random.randint(0, len(flatten_array))
#         sid = int(yid[idx] * step)
#         eid = int(xid[idx] * step)
        
#         interval_time = eid - sid
#         rand = 4e5
#         start_index = np.random.randint(sid - rand, sid + rand) if sid - rand > 0 and sid + rand < len(event_ts) else sid
#         # interval = np.random.randint(interval_time - rand, interval_time + rand) if start_index + interval_time + rand< len(event_ts) else (len(event_ts)-start_index-1)
#         interval = np.random.randint(range_event[0], range_event[1]) 
#         if start_index + interval > len(event_ts) - 1:
#             interval = len(event_ts) - start_index - 1
#         pairs.append([start_index, interval])

#         # if interval < 0:
#         #     print("error")
#         #     break

#     for i in range(num_pairs - num_pairs_extra):
#         start_time = np.random.randint(0, max_ts-1)
#         start_time_index = np.searchsorted(event_ts, start_time)
#         interval_time = np.random.randint(range_time[0], range_time[1]) # make it to index
#         while event_ts[start_time_index] + interval_time >= max_ts-1:
#             start_time_index = np.random.randint(0, len(event_ts)-1)
#             interval_time = np.random.randint(range_time[0], range_time[1])
#         end_time_index = np.searchsorted(event_ts[start_time_index:], event_ts[start_time_index] + interval_time)
#         interval_time_index = end_time_index
#         pairs.append([start_time_index, interval_time_index])
#     random.shuffle(pairs)

#     return pairs