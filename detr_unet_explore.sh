#!/bin/bash -l
#SBATCH --job-name=vit_explore
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --mem-per-cpu=32g
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --gpus=rtx_3090:1
#SBATCH --account=ls_polle

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /cluster/project/cvg/students/shangwu/ftnet_inference_env_2
module load stack/2025-06
module load ffmpeg

export EGL_PLATFORM=surfaceless
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility

MESH=/cluster/project/cvg/students/shangwu/FrontierNet/eval_data/mesh/000040.glb
CONFIG=/cluster/project/cvg/students/shangwu/FrontierNet/config/hm3d_exploration_Actmap_MH3D_00040_0001_0.yaml
NAME=detr_shard3_5sample

# echo "Exploring..."
# python demo_exploration_headless.py \
# --write_path output/detr_shard3_5sample/exploration_state.json \
# --unet_weight /cluster/project/cvg/students/shangwu/Pytorch-UNet/checkpoints/shard3/CP_epoch500.pth \
# --detr_num_queries 10 \
# --model_type detr_unet \
# --mesh /cluster/project/cvg/students/shangwu/FrontierNet/eval_data/mesh/000040.glb \
# --config ${CONFIG}

echo "Visualizing..."
python eval/replay_visualization_headless.py \
--mesh /cluster/project/cvg/students/shangwu/FrontierNet/eval_data/mesh/000040.glb \
--json_file output/detr_shard3_5sample/exploration_state.json \
--config ${CONFIG} \
--output_dir output/detr_shard3_5sample/frames \
--make_video

echo "Re-encoding side-by-side video with ffmpeg (mpeg4)..."
ffmpeg -y \
  -framerate 5 \
  -i output/detr_shard3_5sample/frames/obs_%05d.png \
  -framerate 5 \
  -i output/detr_shard3_5sample/frames/robot_%05d.png \
  -filter_complex "[1:v]scale=-2:960[rob];[0:v][rob]hstack=inputs=2" \
  -c:v mpeg4 \
  -q:v 2 \
  -pix_fmt yuv420p \
  output/detr_shard3_5sample/frames/replay.mp4
echo "Done: output/detr_shard3_5sample/frames/replay.mp4"

# echo "Replaying..."
# python eval/replay_headless.py \
# --mesh ${MESH}  \
# --json_file output/${NAME}/exploration_state.json \
# --config ${CONFIG}  \
# --output output/${NAME}/exploration_state_with_volume.json \
# -ll 10

# echo "Computing statistics..."
# python eval/stat.py \
# --json_file output/${NAME}/exploration_state_with_volume.json \
# --voxel_grid /cluster/project/cvg/students/shangwu/FrontierNet/eval_data/voxel_grid/000876-voxel_grid.ply \
# --output output/${NAME}/exploration_stat.png \
# --no_show
