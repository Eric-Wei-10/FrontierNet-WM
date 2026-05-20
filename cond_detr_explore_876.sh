#!/bin/bash -l
#SBATCH --job-name=cdetr_explore
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --mem-per-cpu=32g
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --gpus=rtx_3090:1
#SBATCH --account=ls_polle

MESH=/cluster/project/cvg/students/shangwu/FrontierNet/eval_data/mesh/000876-mv2HUxq3B53.glb
VOXEL_GRID=/cluster/project/cvg/students/shangwu/FrontierNet/eval_data/voxel_grid/000876-voxel_grid.ply
CONFIG=/cluster/project/cvg/students/shangwu/FrontierNet/config/hm3d_exploration_0.yaml
MODEL_NAME=cond_detr
CKPT_NAME=frontier_dummy_cond_detr
SCENE=876
NAME=${MODEL_NAME}_${CKPT_NAME}_${SCENE}



source ~/miniconda3/etc/profile.d/conda.sh
conda activate /cluster/project/cvg/students/shangwu/ftnet_inference_env_2

module load stack/2025-06
module load ffmpeg

export EGL_PLATFORM=surfaceless
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility

# echo "Exploring..."
# python demo_exploration_headless.py \
# --write_path output/${NAME}/exploration_state.json \
# --unet_weight /cluster/project/cvg/students/shangwu/DETR-Factory-PyTorch/checkpoints/${CKPT_NAME}/CP_epoch500.pth \
# --detr_num_queries 20 \
# --model_type cond_detr \
# --mesh ${MESH} \
# --voxel_grid ${VOXEL_GRID} \
# --config ${CONFIG}

echo "Visualizing and computing volume..."
python eval/replay_visualization_headless.py \
--mesh ${MESH} \
--json_file output/${NAME}/exploration_state.json \
--config ${CONFIG} \
--output_dir output/${NAME}/frames \
--make_video \
--compute_volume \
--volume_output output/${NAME}/exploration_state_with_volume.json
echo "Done: output/${NAME}/frames/replay_${NAME}.mp4"

echo "Computing statistics..."
python eval/stat.py \
--json_file output/${NAME}/exploration_state_with_volume.json \
--voxel_grid /cluster/project/cvg/students/shangwu/FrontierNet/eval_data/voxel_grid/000876-voxel_grid.ply \
--output output/${NAME}/exploration_stat.png \
--no_show
