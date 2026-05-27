#!/bin/bash -l
#SBATCH --job-name=detr_explore
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --mem-per-cpu=32g
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --gpus=rtx_4090:1
#SBATCH --account=ls_polle

# Usage (interactive):  bash detr_explore.sh <scene> <pt> <ckpt_name>
# Usage (sbatch):       sbatch detr_explore.sh <scene> <pt> <ckpt_name>
#
# <scene>     : scene number, e.g. 876, 804, 807 ...
# <pt>        : pose index (1-based, matches config/<scene>/pt_<pt>/detr.yaml)
# <ckpt_name> : folder name under DETR-Factory-PyTorch/checkpoints/
#               e.g.  multi_10000   frontier_detr_shard3
#
# Example:
#   sbatch detr_explore.sh 876 1 multi_10000
#   sbatch detr_explore.sh 804 2 frontier_detr_shard3

SCENE=${1:?"Usage: $0 <scene> <pt> <ckpt_name>"}
PT=${2:?"Usage: $0 <scene> <pt> <ckpt_name>"}
CKPT=${3:?"Usage: $0 <scene> <pt> <ckpt_name>"}

ROOT=/cluster/project/cvg/students/shangwu/FrontierNet
CKPT_BASE=/cluster/project/cvg/students/shangwu/DETR-Factory-PyTorch/checkpoints
CKPT_PATH=${CKPT_BASE}/${CKPT}/best_model.pth
CONFIG=${ROOT}/config/${SCENE}/pt_${PT}/detr.yaml
SCENE_PAD=$(printf "%06d" "${SCENE}")
MESH=${ROOT}/eval_data/mesh/${SCENE_PAD}.glb
VOXEL_GRID=${ROOT}/eval_data/voxel_grid/${SCENE_PAD}-voxel_grid.ply
NAME=detr_${CKPT}_${SCENE}_pt${PT}

# ---- validate inputs ----
if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "ERROR: checkpoint not found: ${CKPT_PATH}"
    exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: config not found: ${CONFIG}"
    echo "  Available poses for scene ${SCENE}:"
    ls "${ROOT}/config/${SCENE}/pt_"*/detr.yaml 2>/dev/null | sed 's|.*/pt_||;s|/.*||' | xargs echo "  pt:"
    exit 1
fi
if [[ ! -f "${MESH}" ]]; then
    echo "ERROR: mesh not found: ${MESH}"
    exit 1
fi
if [[ ! -f "${VOXEL_GRID}" ]]; then
    echo "ERROR: voxel grid not found: ${VOXEL_GRID}"
    exit 1
fi

# ---- environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /cluster/project/cvg/students/shangwu/ftnet_inference_env_2

module load stack/2025-06
module load ffmpeg

export EGL_PLATFORM=surfaceless
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility

# Logs live beside output/${NAME}/ not inside it — demo_exploration_headless.py
# calls shutil.rmtree(output/${NAME}/) at startup and would delete them otherwise.
mkdir -p output
EXPLORE_LOG=output/${NAME}.explore.log
REPLAY_LOG=output/${NAME}.replay.log

# ---- explore ----
echo "Exploring  scene=${SCENE}  pt=${PT}  ckpt=${CKPT}  name=${NAME}"
python -u demo_exploration_headless.py \
    --write_path output/${NAME}/exploration_state.json \
    --unet_weight "${CKPT_PATH}" \
    --detr_num_queries 20 \
    --model_type detr \
    --mesh "${MESH}" \
    --voxel_grid "${VOXEL_GRID}" \
    --config "${CONFIG}" \
    --detr_conf_thresh 0.15 \
    --detr_visible_gain_discount 0.45 \
    --detr_dedup_radius 0.3 \
    --max_steps 1000 \
    --log_level 10 \
    > "${EXPLORE_LOG}" 2>&1
echo "Exploration log: ${EXPLORE_LOG}"

# ---- replay + volume ----
echo "Visualizing and computing volume..."
python -u eval/replay_visualization_headless.py \
    --mesh "${MESH}" \
    --json_file output/${NAME}/exploration_state.json \
    --config "${CONFIG}" \
    --output_dir output/${NAME}/frames \
    --make_video \
    --compute_volume \
    --volume_output output/${NAME}/exploration_state_with_volume.json \
    > "${REPLAY_LOG}" 2>&1
echo "Replay log: ${REPLAY_LOG}"
echo "Video: output/${NAME}/frames/replay_${NAME}.mp4"

# ---- statistics ----
echo "Computing statistics..."
python eval/stat.py \
    --json_file output/${NAME}/exploration_state_with_volume.json \
    --voxel_grid "${VOXEL_GRID}" \
    --output output/${NAME}/exploration_stat.png \
    --no_show
