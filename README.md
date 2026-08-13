# FastEventDGS
The implementation of the paper "FastEventDGS: Deformable Gaussian Splatting for Fast Dynamic Scenes from a Single Event Camera"(CVPR2026).

## Dataset
Download link: https://drive.google.com/file/d/1xn_E52JvO3oH8Sgy-GO1KP6UHjV75TsO/view?usp=sharing

Note that we added two additional datasets (fan and strawberry) from Blender Event. While our algorithm currently performs suboptimally on them, they offer intriguing challenges that we encourage the community to tackle.

### Dataset Structure (Blender Event)

```
<scene>/
├── cam_transforms.json     # Camera transforms with timestamps
├── events.npy              # (or events/ dir with .npz files)
├── imgs/                   # Ground truth images
├── points3d.ply            # Initial point cloud
└── K.yaml                  # Camera intrinsics (for real event)
```

### Dataset Structure (Real Event)

```
<scene>/
├── events_rectified.npy    # Events [t, x, y, p]
├── llff/
│   ├── all_poses_bounds.npy
│   └── all_timestamps.npy
├── rectify_mask.npy        # Lens rectification mask
├── K.yaml                  # Camera intrinsics
└── points3d.ply
```
## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+
- NVIDIA GPU with sufficient VRAM (recommended >= 12GB)

### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd FastEventDGS
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install the CUDA submodules:
   ```bash
   # Install extended diff-gaussian-rasterization (with flow support)
   cd submodules/diff-gaussian-rasterization-extentions
   pip install -e .
   cd ../..

   # Install simple-knn
   cd submodules/simple-knn
   pip install -e .
   cd ../..
   ```

4. (Optional) Install [esim-torch](https://github.com/uzh-rpg/esim_torch) and [rpg_vid2e](https://github.com/uzh-rpg/rpg_vid2e) for synthetic event generation from videos.

## Usage

### Training

```bash
# Train on a Blender Event dataset (e.g., gtduck)
python train.py -s /path/to/dataset/gtduck -m output/gtduck_event --eval --iterations 20000

# Train on a real-world event dataset
python train.py -s /path/to/real_event/segment0_0-266 -m output/real_scene --eval --iterations 30000
```

**Key arguments:**

| Flag | Description |
|------|-------------|
| `-s / --source_path` | Path to the dataset directory |
| `-m / --model_path` | Output directory for the trained model |
| `--eval` | Split training/test cameras for evaluation |
| `--iterations` | Total number of training iterations |
| `--test_iterations` | Iterations at which to run evaluation |
| `--save_iterations` | Iterations at which to save checkpoints |
| `--is_blender` | Flag for Blender synthetic data |
| `--is_6dof` | Use 6-DoF deformation (instead of canonical-space) |
| `--is_color` | Use color events instead of grayscale |
| `--white_background` | Use white background |

### Event Contrast Thresholds

The event camera contrast thresholds can be tuned:

```bash
--C_neg 0.1 --C_pos 0.1    # Synthetic data (default)
# For real-world data:
--C_neg 0.25 --C_pos 0.25
```

### Optional Regularization Modules

| Flag | Description |
|------|-------------|
| `--use_spline` | Enable B-spline continuous-time camera trajectory (default: True) |
| `--use_contrast` | Enable event flow/warp contrast loss |
| `--use_motion` | Enable event motion consistency loss |
| `--use_depth` | Enable VGGT depth supervision (Note: If running into OOM errors, you can swap VGGT for a more memory-friendly depth estimator like DA3.)| 

## Monitoring

Training progress can be monitored with TensorBoard:

```bash
tensorboard --logdir=output/<scene>/
```

## Acknowledgments

This project builds upon the following works:

- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) — Kerbl et al., SIGGRAPH 2023
- [Deformable 3D Gaussians](https://github.com/ingra14m/Deformable-3DGS) — Yang et al.
- [VGGT](https://github.com/facebookresearch/vggt) — Wang et al. 
- [esim_torch](https://github.com/uzh-rpg/esim_torch) — RPG Group, University of Zurich

## License

This software is free for non-commercial, research and evaluation use.


