import os
import torch
import random

from utils.loss_utils import l1_loss, ssim, kl_divergence, iso_loss
from gaussian_renderer import render, network_gui, render_batch_depth, render_grads_histogram, render_with_flow
import sys
from scene import Scene, GaussianModel, DeformModel
from utils.general_utils import safe_state, get_linear_noise_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, rgb2gray
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

# import cv2
from utils.depth_utils import DepthEstimator
from loss_event import EventLoss
from utils.bspline_utils import get_viewcams_from_spline
import numpy as np
from utils.mask_utils import get_event_list, LossHistogram
from utils.sample_utils import adaptive_sample, generate_paris
from utils.flow_utils import FlowEstimator, compute_camera_flow, calculate_gs_flow, calculate_camera_flow, warping_gs_flow
# from utils.loss_utils import variance_loss
import torch.nn.functional as F

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, args):
    device = dataset.data_device#torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    deform = DeformModel(dataset.is_blender, dataset.is_6dof)
    deform.train_setting(opt)

    scene = Scene(dataset, gaussians, shuffle=False)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000) # for 
    
    # new added part
    event_loss = EventLoss(device=device)
    depth_estimator = DepthEstimator()
    flow_estimator = FlowEstimator(device=device, parser=args)
    events_threshold_negpos = torch.tensor([opt.C_neg, opt.C_pos], dtype=torch.float32, device=device)
    all_events = scene.all_events # [N, 4] t,x,y,pol t increase

    # spline part
    Rspline = scene.Rspline
    Tspline = scene.Tspline
    cam_intri = scene.cam_intri
    # mask for real world event
    rectify_mask = scene.rectify_mask
    if rectify_mask is not None:
        rectify_mask = torch.from_numpy(rectify_mask).to(device)
    # get K here
    if isinstance(scene.K, torch.Tensor):
        K = scene.K.to(device)
    else:
        K = torch.from_numpy(scene.K).to(device)
    if scene.scale is not None:
        scale = scene.scale
    else:
        scale = 1.0

    i = 0
    pairs = None
    loss_histogram = LossHistogram(all_events.shape[0])
    for iteration in range(1, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.do_shs_python, pipe.do_cov_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree (max is 3 now)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack or i >= len(pairs)-1:
            viewpoint_stack = scene.getTrainCameras().copy() 
            if not opt.use_spline:
                pairs = generate_paris(200, opt.interval_range, len(viewpoint_stack), keep_uni = False)
            else:
                # Linearly decay time_scale from 1 to 0.1 over the course of training iterations
                time_scale = 1.0 - 0.1 * (iteration / float(opt.iterations))
                time_scale = max(time_scale, 0.1)
                time_interval_new =  (int(time_scale * opt.interval_time[0]), int(time_scale * opt.interval_time[1]))
                pairs = adaptive_sample(200, time_interval_new, opt.interval_range_spline, all_events[:,0])  
            num_frame = len(pairs)
            i = 0

        total_frame = num_frame - i 
        time_interval = 1 / total_frame
        
        if not opt.use_spline:
            # get viewpoint cam from different interval
            viewpoint_cam = viewpoint_stack[pairs[i][0] + pairs[i][1]]
            viewpoint_cam_before = viewpoint_stack[pairs[i][0]]
            event_index = viewpoint_cam.closest_event_index 
            event_index_before = viewpoint_cam_before.closest_event_index 
        else:
            event_index = pairs[i][0] + pairs[i][1]
            event_index_before = pairs[i][0]
            timestamp = all_events[event_index, 0]
            timestamp_before = all_events[event_index_before, 0]
            view_cams = get_viewcams_from_spline([timestamp, timestamp_before], Rspline, Tspline, cam_intri, device=device)
            viewpoint_cam, viewpoint_cam_before = view_cams[0], view_cams[1]
        i = i+1

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
            viewpoint_cam_before.load2device()
        fid = viewpoint_cam.fid
        fid_before = viewpoint_cam_before.fid
        
        # print(fid, fid_before)

        if iteration < opt.warm_up:
            d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
            # d_color = torch.zeros((gaussians.get_xyz.shape[0], 16, 3), device=device)
            d_color = None
            d_xyz_before, d_rotation_before, d_scaling_before = 0.0, 0.0, 0.0
            # d_color_before = torch.zeros((gaussians.get_xyz.shape[0], 16, 3), device=device)
            d_color_before = None
        else:
            N = gaussians.get_xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(N, -1)
            time_input_before = fid_before.unsqueeze(0).expand(N, -1)
            ast_noise = 0 if dataset.is_blender else torch.randn(1, 1, device=device).expand(N, -1) * time_interval * smooth_term(iteration)
            d_xyz, d_rotation, d_scaling, d_color = deform.step(gaussians.get_xyz.detach(), time_input + ast_noise)
            d_xyz_before, d_rotation_before, d_scaling_before, d_color_before = deform.step(gaussians.get_xyz.detach(), time_input_before + ast_noise)
            if iteration == opt.warm_up:
                np.save(os.path.join("d_xyz.npy"), d_xyz.detach().cpu().numpy())
            if iteration == 15000:
                np.save(os.path.join("d_xyz_end.npy"), d_xyz.detach().cpu().numpy())

            # TODO: scale increase
            d_xyz, d_rotation, d_scaling = 0.05 * d_xyz, 0.05 * d_rotation, 0.05 * d_scaling
            d_xyz_before, d_rotation_before, d_scaling_before = 0.05 * d_xyz_before, 0.05 * d_rotation_before, 0.05 * d_scaling_before
        
        # Render
        # TODO: only use one channel
        render_pkg_re_before = render_with_flow(viewpoint_cam_before, gaussians, pipe, background, 
                                      d_xyz_before, d_rotation_before, d_scaling_before, d_color_before, dataset.is_6dof)
        image_before = render_pkg_re_before["render"]
        depth_before = render_pkg_re_before["depth"].detach()

        render_pkg_re = render_with_flow(viewpoint_cam, gaussians, pipe, background, 
                               d_xyz, d_rotation, d_scaling, d_color, dataset.is_6dof)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re[
            "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]
        depth = render_pkg_re["depth"].detach()

        if dataset.is_color is False:
            image = image[0, :, :].unsqueeze(0)
            image_before = image_before[0, :, :].unsqueeze(0)

        # print(event_index_before, event_index)
        events = all_events[event_index_before:event_index]
        events = torch.tensor(events, dtype=torch.float32, device=image.device)

        # image = F.interpolate(image.unsqueeze(0), size=(int(image.shape[1]/scale), int(image.shape[2]/scale)), mode='bilinear', align_corners=False).squeeze(0)
        # image_before = F.interpolate(image_before.unsqueeze(0), size=(int(image_before.shape[1]/scale), int(image_before.shape[2]/scale)), mode='bilinear', align_corners=False).squeeze(0)

        E, E_hat, event_mask, esi_maps_tuple = event_loss.event_component(events, image, image_before, events_threshold_negpos)

        # Get 1st and 99th percentiles for normalization
        # percentile_1 = torch.quantile(E.flatten(), 0.01)
        # percentile_99 = torch.quantile(E.flatten(), 0.99)
        # E_normalized = (E - percentile_1) / (percentile_99 - percentile_1)
        # E_hat_normalized = (E_hat - percentile_1) / (percentile_99 - percentile_1)
        
        num_events = events.shape[0]

        min_e = E.min() 
        max_e = E.max()
        E_normalized = (E - min_e) / (max_e - min_e)
        E_hat_normalized = (E_hat - min_e) / (max_e - min_e)

        if iteration > 10000: # and iteration < 12000: 
            loss_histogram.update(events, torch.abs(E_normalized - E_hat_normalized * event_mask), event_index_before, event_index)

        score_mask = torch.zeros((256, 256, 1))
        event_warp_loss = torch.tensor(0.0, device=events.device)
        event_flow_loss = torch.tensor(0.0, device=events.device)
        event_spatial_loss = torch.tensor(0.0, device=events.device)
        event_textureless_loss = torch.tensor(0.0, device=events.device)
        event_depth_loss = torch.tensor(0.0, device=events.device)
        event_motion_loss = torch.tensor(0.0, device=events.device)
        # event_white_balance_loss = torch.tensor(0.0, device=events.device)

        if iteration > opt.warm_up:# and iteration < 10000:
            if opt.use_contrast:
                timestamp_wp = np.array([timestamp_before + opt.warp_time * i for i in range(opt.warp_batch+1)])
                view_cameras = get_viewcams_from_spline(timestamp_wp, Rspline, Tspline, cam_intri, device=device)
                # depth_maps, _ = render_batch_depth(view_cameras, gaussians, deform, pipe, background, dataset)

                dynamic_mask = loss_histogram.query_dynamic_mask(event_index_before, event_index)
                event_list = get_event_list(events, timestamp_wp, dynamic_mask) # where dynamic object removed
                # score_map, score_mask, warped_count_map, flow_batch = event_loss.get_weight(event_list, depth_maps, view_cameras, K[:3,:3])
                # event_warp_loss = - torch.abs(score_map).mean()

                event_tmp = torch.cat(event_list, dim=0) # event_list[0]

                render_pkg_re_tmp_t1 = render_with_flow(view_cameras[-1], gaussians, pipe, background, d_xyz_before, d_rotation_before, d_scaling_before, d_color_before, dataset.is_6dof)
                alpha, proj_2D, conic_2D, conic_2D_inv, gs_per_pixel, weight_per_gs_pixel, x_mu = render_pkg_re_tmp_t1[
                    "alpha"], render_pkg_re_tmp_t1["proj_2D"], render_pkg_re_tmp_t1["conic_2D"], render_pkg_re_tmp_t1["conic_2D_inv"
                    ], render_pkg_re_tmp_t1["gs_per_pixel"], render_pkg_re_tmp_t1["weight_per_gs_pixel"], render_pkg_re_tmp_t1["x_mu"]         

                N = gaussians.get_xyz.shape[0]
                time_input_tmp = view_cameras[-1].fid.unsqueeze(0).expand(N, -1)
                d_xyz_2, d_rotation_2, d_scaling_2, d_color_2 = deform.step(gaussians.get_xyz.detach(), time_input_tmp + ast_noise)
                d_xyz_2, d_rotation_2, d_scaling_2 = 0.05 * d_xyz_2, 0.05 * d_rotation_2, 0.05 * d_scaling_2
                render_pkg_re_tmp_t2 = render_with_flow(view_cameras[-1], gaussians, pipe, background, d_xyz_2, d_rotation_2, d_scaling_2, d_color_2, dataset.is_6dof)
                next_proj_2D, next_conic_2D = render_pkg_re_tmp_t2["proj_2D"], render_pkg_re_tmp_t2["conic_2D"] 

                # check shape
                # warp gs_flow to match motion flow
                gs_flow = calculate_gs_flow(gs_per_pixel, weight_per_gs_pixel, next_conic_2D, conic_2D_inv, proj_2D, next_proj_2D, x_mu)
                gs_flow = warping_gs_flow(depth_before, gs_flow, viewpoint_cam_before, view_cameras[-1]) 

                # camera_flow, _, _ = compute_camera_flow(depth_before, viewpoint_cam_before, view_cameras[-1], K[:3,:3]) # flow_batch[0]
                camera_flow = calculate_camera_flow(depth_before, viewpoint_cam_before, view_cameras[-1])
                optical_flow = gs_flow.detach() + camera_flow
                optical_flow = optical_flow.permute(1, 2, 0) 

                event_flow_loss = event_loss.event_flow_loss(image_before, event_tmp, optical_flow, rectify_mask, iteration)
                # event_flow_loss = event_loss.event_flow_loss(image_before, event_tmp, camera_flow.permute(1, 2, 0).detach(), rectify_mask, iteration)

        batch_test = None
        if opt.use_motion and iteration > opt.warm_up:
            gs_centers = []
            N = gaussians.get_xyz.shape[0]
            esi_maps, ts_small_window = esi_maps_tuple
            ts_small_window = [ts_small_window[i].detach().cpu().numpy() for i in range(len(ts_small_window))]

            view_cams_motion = get_viewcams_from_spline(ts_small_window, Rspline, Tspline, cam_intri, device=device)
            # TODO: parallel and give weight to different time interval
            for view in view_cams_motion:
                fid_motion = view.fid
                time_input_motion = fid_motion.unsqueeze(0).expand(N, -1)
                d_xyz_motion, _, _, _ = deform.step(gaussians.get_xyz.detach(), time_input_motion)
                mean_3d = gaussians.get_xyz.detach() + d_xyz_motion.detach()
                gs_centers.append(mean_3d)
            event_motion_loss, batch_test = event_loss.event_motion_loss(torch.stack(gs_centers, dim=0), depth_before.detach(), depth.detach(), 
                                                            'constant', esi_maps, view_cams_motion, K, dataset.is_color)            
            # batch test is [c,b*h,w], visualize it
                        
        # depth
        depth_iteration = 20000
        if iteration == (depth_iteration-1) and opt.use_depth:
            print("Initializing depth estimator")
            depth_estimator.initialize()
        
        # event_spatial_loss = event_loss.event_spatial_loss(image, E_hat, weight = 0.010) # 0.025 for syn
        if iteration > depth_iteration:
            event_spatial_loss = event_loss.event_spatial_loss(image, E_hat, weight = 0.010) # 0.025 for syn
            event_textureless_loss = event_loss.event_textureless_loss(E, E_hat, event_mask, events_threshold_negpos)
            if opt.use_depth:
                event_depth_loss = depth_estimator.depth_silog_loss(depth_before, image_before)

        # var_loss = variance_loss(render_img=image_before, alpha=0.05)

        if rectify_mask is not None: # different normalization
            # print(E.shape, E_hat.shape, event_mask.shape, rectify_mask.shape)
            rectify_mask = rectify_mask[:E.shape[1], :E.shape[2]]
            DSSIM_event = event_loss.event_ssim_loss(E * rectify_mask, E_hat * rectify_mask)
            Ll1_event = l1_loss(E * rectify_mask, E_hat * rectify_mask * event_mask)
            event_render_loss = ((1.0 - opt.lambda_event_dssim) * Ll1_event + opt.lambda_event_dssim * DSSIM_event) / torch.tensor(num_events/1e6, device=device)
            # consider ignore highest 1% percent loss
            
            loss =  event_render_loss + (opt.lambda_event_warp * event_warp_loss + opt.lambda_event_flow * event_flow_loss + \
                    opt.lambda_event_motion * event_motion_loss) / torch.tensor(num_events/1e6, device=device) + \
                    opt.lambda_event_depth * event_depth_loss + \
                    opt.lambda_event_textureless * event_textureless_loss + event_spatial_loss # + var_loss
        else:
            # Ll1_event = l1_loss(E_normalized, E_hat_normalized * event_mask)
            Ll1_event = l1_loss(E_normalized, E_hat_normalized)
            DSSIM_event = event_loss.event_ssim_loss(E_normalized, E_hat_normalized)
            event_render_loss = (1.0 - opt.lambda_event_dssim) * Ll1_event + opt.lambda_event_dssim * DSSIM_event
            loss =  event_render_loss + opt.lambda_event_warp * event_warp_loss + opt.lambda_event_flow * event_flow_loss + \
                    opt.lambda_event_motion * event_motion_loss + \
                    opt.lambda_event_depth * event_depth_loss + \
                    opt.lambda_event_textureless * event_textureless_loss + event_spatial_loss
        
        if not opt.use_spline:
            gt_image = viewpoint_cam.original_image.cuda()
            intensity_gt_img  = rgb2gray(viewpoint_cam.original_image)
            gt_image = intensity_gt_img
            Ll1 = l1_loss(image, gt_image)
            loss_rgb = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss = event_render_loss

        loss.backward()
        iter_end.record()

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                 radii[visibility_filter])

            # Log and save
            if not opt.use_spline:
                cur_psnr = training_report(tb_writer, iteration, loss_rgb, event_render_loss, event_textureless_loss, event_warp_loss, loss, iter_start.elapsed_time(iter_end),
                                        testing_iterations, E, E_hat, image, viewpoint_cam_before.original_image, scene, render, (pipe, background), deform,
                                        dataset.load2gpu_on_the_fly, dataset.is_6dof)
            else:
                cur_psnr = training_report_spline(tb_writer, iteration, Ll1_event, DSSIM_event, event_textureless_loss, event_warp_loss, event_flow_loss, 
                                                  event_spatial_loss, event_depth_loss, event_motion_loss, loss, iter_start.elapsed_time(iter_end),
                                        testing_iterations, E, E_hat, image, score_mask, image_before, scene, render, (pipe, background), deform,
                                        dataset.load2gpu_on_the_fly, dataset.is_6dof, dataset.is_color, batch_test)
            
            if iteration in testing_iterations:
                if cur_psnr.item() > best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                deform.save_weights(args.model_path, iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # viewspace_point_tensor_densify = render_pkg_re["viewspace_points_densify"]
                viewspace_point_tensor_densify = render_pkg_re["viewspace_points"]
                gaussians.add_densification_stats(viewspace_point_tensor_densify, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = None if iteration > opt.opacity_reset_interval else None # 20
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold) # 0.005 -> 0.01 -> 0.0075

                if (iteration % opt.opacity_reset_interval == 0 and iteration > 10000) or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    # gaussians.reset_opacity_smooth()
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                deform.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                deform.optimizer.zero_grad()
                deform.update_learning_rate(iteration)
            
            # if iteration < opt.iterations and iteration < opt.warm_up:
            #     gaussians.optimizer.step()
            #     gaussians.update_learning_rate(iteration)
            #     gaussians.optimizer.zero_grad(set_to_none=True)

            # if iteration > opt.warm_up and iteration < opt.iterations:
            #     deform.optimizer.step()
            #     deform.optimizer.zero_grad()
            #     deform.update_learning_rate(iteration)

    print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))
    

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, loss_rgb, event_render_loss, event_textureless_loss, event_warp_loss, loss, elapsed, testing_iterations, 
                    E, E_hat, render_img, gt_img, scene: Scene, renderFunc,
                    renderArgs, deform, load2gpu_on_the_fly, is_6dof=False):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/gray_loss', loss_rgb.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/event_render_loss', event_render_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/event_textureless_loss', event_textureless_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/event_warp_loss', event_warp_loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_psnr = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(0, 200, 50)]}) # modify

        tb_writer.add_images("/intensity_log_train_set", E_hat[None], global_step=iteration)
        tb_writer.add_images("/esi_train_set", E[None], global_step=iteration)
        tb_writer.add_images("/diff_train_set", E[None]-E_hat[None], global_step=iteration)
        tb_writer.add_images("/render_train_set", render_img[None], global_step=iteration)
        tb_writer.add_images("/ground_truth_train_set", gt_img[None], global_step=iteration)
        
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                images = torch.tensor([], device=render_img.device)
                gts = torch.tensor([], device=gt_img.device)
                for idx, viewpoint in enumerate(config['cameras']):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                    fid = viewpoint.fid
                    xyz = scene.gaussians.get_xyz
                    time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                    d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz, d_rotation, d_scaling, is_6dof)["render"],
                        0.0, 1.0)
                    if viewpoint.original_image is not None:
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    else:
                        gt_image = torch.zeros_like(image)
                    images = torch.cat((images, image.unsqueeze(0)), dim=0)
                    gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render_test_set".format(viewpoint.image_name),
                                             image[None], global_step=iteration)

                l1_test = l1_loss(images, gts)
                psnr_test = psnr(images, gts).mean()
                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_test
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return test_psnr

def training_report_spline(tb_writer, iteration, Ll1_event, DSSIM_event, event_textureless_loss, event_warp_loss, event_flow_loss,
                            event_spatial_loss, event_depth_loss, event_motion_loss, loss, elapsed, testing_iterations, 
                    E, E_hat, render_img, static_mask, image_before, scene: Scene, renderFunc,
                    renderArgs, deform, load2gpu_on_the_fly, is_6dof=False, is_color = False, batch_test = None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/Ll1_event', Ll1_event.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/DSSIM_event', DSSIM_event.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/event_textureless_loss', event_textureless_loss.item(), iteration)
        # tb_writer.add_scalar('train_loss_patches/event_warp_loss', event_warp_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/event_flow_loss', event_flow_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/event_depth_loss', event_depth_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/event_motion_loss', event_motion_loss.item(), iteration)
        # tb_writer.add_scalar('train_loss_patches/event_white_balance_loss', event_white_balance_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/event_spatial_loss', event_spatial_loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_psnr = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(0, 200, 50)]}) # modify

        tb_writer.add_images("/intensity_log_train_set", E_hat[None], global_step=iteration)
        tb_writer.add_images("/esi_train_set", E[None], global_step=iteration)
        tb_writer.add_images("/diff_train_set", E[None]-E_hat[None], global_step=iteration)
        tb_writer.add_images("/render_train_set", render_img[None], global_step=iteration)
        tb_writer.add_images("/render_train_set_before", image_before[None], global_step=iteration)
        if batch_test is not None:
            batch_test = torch.sum(batch_test, dim=0, keepdim=True)
            min_batch = torch.min(batch_test)
            max_batch = torch.max(batch_test)
            nomalize_batch = (batch_test-min_batch)/(max_batch-min_batch)
            tb_writer.add_images("/batch_test", nomalize_batch[None], global_step=iteration)

        static_mask = static_mask.squeeze(2).unsqueeze(0)
        tb_writer.add_images("/static_mask_train_set", static_mask[None], global_step=iteration)

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                images = torch.tensor([], device=render_img.device)
                gts = torch.tensor([], device=render_img.device)
                for idx, viewpoint in enumerate(config['cameras']):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                    fid = viewpoint.fid
                    xyz = scene.gaussians.get_xyz
                    time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                    d_xyz, d_rotation, d_scaling, d_color = deform.step(xyz.detach(), time_input)
                    d_xyz, d_rotation, d_scaling = 0.05 * d_xyz, 0.05 * d_rotation, 0.05 * d_scaling
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz, d_rotation, d_scaling, d_color, is_6dof)["render"],
                        0.0, 1.0)
                    if viewpoint.original_image is not None:
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    else:
                        gt_image = torch.zeros_like(image)
                    if is_color:
                        gt_image = viewpoint.original_image.to("cuda")
                        # image = gamma_correction(image)
                    else:
                        image = image[0, :, :].unsqueeze(0)
                        gt_image = rgb2gray(gt_image)
                    gt_image = torch.clamp(gt_image, 0.0, 1.0)
                    images = torch.cat((images, image.unsqueeze(0)), dim=0)
                    gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render_test_set".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render_test_set_gt".format(viewpoint.image_name),
                                             gt_image[None], global_step=iteration) 

                l1_test = l1_loss(images, gts)
                psnr_test = psnr(images, gts).mean()
                # ssim_test = ssim(images, gts)
                # lpips_test = lpips(images, gts).mean()

                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_test
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return test_psnr


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[5000, 6000, 7_000] + list(range(10000, 40001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 15_000, 20_000, 30_000, 40000])
    parser.add_argument("--quiet", action="store_true")

    model_path = "/home/daizj/data/event-gaussian/codes/De3GS/pretrained_models/Tartan480x640-M.pth"
    config_path = "/home/daizj/data/event-gaussian/codes/De3GS/submodules/searaft/config/train/Tartan480x640-M.json"
    parser.add_argument('--cfg', default=config_path, type=str)
    parser.add_argument('--path', default=model_path, type=str)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args)

    # All done
    print("\nTraining complete.")
