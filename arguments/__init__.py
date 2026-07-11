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

from argparse import ArgumentParser, Namespace
import sys
import os


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda:0"
        self.eval = False
        self.load2gpu_on_the_fly = False
        self.is_blender = False
        self.is_6dof = False
        self.is_color = True 
        self.render_gray = True
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 40_000
        self.warm_up = 3_000 # 3_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.deform_lr_max_steps = 40_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.001
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        # 
        self.lambda_dssim = 0.2
        self.lambda_event = 0.9
        self.lambda_event_dssim = 0.2 # 0.05 # 0.2 # 0.02 # 0.5 for duck # 0.20 for butterfly
        self.lambda_event_textureless = 0.1 # 0.2 for syn
        self.lambda_event_warp = 0.015
        self.lambda_event_flow = 10
        self.lambda_event_depth = 0.1
        self.lambda_event_motion = 0.1 # 1 for real world
        self.warp_time = 31914894 * 1 # 0.5 & 0.25
        self.warp_batch = 3 # 5
        self.C_neg = 0.1 #0.20 #!!!!!! 0.25 realworld
        self.C_pos = 0.1 #0.20 real world
        self.interval_range = (10, 40)
        self.interval_time = (31914894*1, 31914894*50) # (31914894*10, 31914894*40) (5,50)
        self.interval_range_event = (30000, 150000)
        self.interval_range_spline = (26855*15, 26855*80) # around 26855 events for each frame interval(15,50)
        self.use_spline = True # False
        self.use_contrast = True # False
        self.use_motion = True # False
        self.use_depth = True # False
        #
        self.densification_interval = 100
        self.opacity_reset_interval = 3000 # 3000 # 20000 for strawberry
        self.densify_from_iter = 500 # for some 1000
        self.densify_until_iter = 20_000 # 15000
        self.densify_grad_threshold = 0.0007 # 0.0008?
        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
