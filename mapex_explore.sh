#!/bin/bash -l
#SBATCH --job-name=mapex_explore
#SBATCH --output=logs/mapex_%j.out
#SBATCH --error=logs/mapex_%j.err
#SBATCH --mem-per-cpu=32g
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --gpus=rtx_4090:1
#SBATCH --account=ls_polle

# MapEx baseline exploration using FrontierNet's planning pipeline.
# Frontier prediction: LaMa inpainting ensemble (MapEx, ICRA 2025).
# Planning / navigation: identical to detr_explore.sh.
#
# Usage (interactive):  bash mapex_explore.sh <scene> <pt>
# Usage (sbatch):       sbatch mapex_explore.sh <scene> <pt>
#
# <scene>  : scene number, e.g. 876, 804, 807 …
# <pt>     : pose index (1-based, matches config/<scene>/pt_<pt>/mapex.yaml)
#
# Example:
#   sbatch mapex_explore.sh 876 1
#   sbatch mapex_explore.sh 804 2

SCENE=${1:?"Usage: $0 <scene> <pt>"}
PT=${2:?"Usage: $0 <scene> <pt>"}

ROOT=/cluster/project/cvg/students/shangwu/FrontierNet_mapex
MAPEX_DIR=/cluster/project/cvg/students/shangwu/MapEx
MAPEX_ENV=/cluster/project/cvg/students/shangwu/mapex_env

CONFIG=${ROOT}/config/${SCENE}/pt_${PT}/mapex.yaml
SCENE_PAD=$(printf "%06d" "${SCENE}")
MESH=${ROOT}/eval_data/mesh/${SCENE_PAD}.glb
VOXEL_GRID=${ROOT}/eval_data/voxel_grid/${SCENE_PAD}-voxel_grid.ply
NAME=mapex_${SCENE}_pt${PT}

# ---- validate inputs ----
if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: config not found: ${CONFIG}"
    echo "  Available poses for scene ${SCENE}:"
    ls "${ROOT}/config/${SCENE}/pt_"*/mapex.yaml 2>/dev/null | sed 's|.*/pt_||;s|/.*||' | xargs echo "  pt:"
    exit 1
fi
if [[ ! -f "${MESH}" ]]; then
    echo "ERROR: mesh not found: ${MESH}"; exit 1
fi
if [[ ! -f "${VOXEL_GRID}" ]]; then
    echo "ERROR: voxel grid not found: ${VOXEL_GRID}"; exit 1
fi
if [[ ! -d "${MAPEX_DIR}/pretrained_models/weights/big_lama" ]]; then
    echo "ERROR: MapEx weights not found in ${MAPEX_DIR}/pretrained_models/weights/"
    echo "  Run setup_mapex_env.sh and download the pretrained models first."
    exit 1
fi

# ---- environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${MAPEX_ENV}"

module load stack/2025-06
module load ffmpeg

export EGL_PLATFORM=surfaceless
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility
export NO_ALBUMENTATIONS_UPDATE=1

mkdir -p logs output

EXPLORE_LOG=logs/${NAME}.explore.log
REPLAY_LOG=logs/${NAME}.replay.log

# ---- explore ----
echo "Exploring  scene=${SCENE}  pt=${PT}  name=${NAME}"
python -u demo_exploration_headless.py \
    --write_path  output/${NAME}/exploration_state.json \
    --model_type  mapex \
    --mapex_dir   "${MAPEX_DIR}" \
    --mesh        "${MESH}" \
    --voxel_grid  "${VOXEL_GRID}" \
    --config      "${CONFIG}" \
    --mapex_map_size          512 \
    --mapex_map_margin        3.0 \
    --mapex_min_frontier_size 3 \
    --mapex_gain_scale        10000 \
    --log_level   10 \
    > "${EXPLORE_LOG}" 2>&1
echo "Exploration log: ${EXPLORE_LOG}"

# ---- replay + volume ----
echo "Visualizing and computing volume..."
python -u eval/replay_visualization_headless.py \
    --mesh         "${MESH}" \
    --json_file    output/${NAME}/exploration_state.json \
    --config       "${CONFIG}" \
    --output_dir   output/${NAME}/frames \
    --make_video \
    --compute_volume \
    --volume_output output/${NAME}/exploration_state_with_volume.json \
    > "${REPLAY_LOG}" 2>&1
echo "Replay log: ${REPLAY_LOG}"
echo "Video: output/${NAME}/frames/replay_${NAME}.mp4"

# ---- statistics ----
echo "Computing statistics..."
python eval/stat.py \
    --json_file  output/${NAME}/exploration_state_with_volume.json \
    --voxel_grid "${VOXEL_GRID}" \
    --output     output/${NAME}/exploration_stat.png \
    --no_show
