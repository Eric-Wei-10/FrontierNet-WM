#!/bin/bash -l
#SBATCH --job-name=mapex_eval_all
#SBATCH --output=logs/mapex_eval_all_%j.out
#SBATCH --error=logs/mapex_eval_all_%j.err
#SBATCH --mem-per-cpu=32g
#SBATCH --ntasks=1
#SBATCH --time=120:00:00
#SBATCH --gpus=rtx_4090:1
#SBATCH --account=ls_polle

# Full MapEx evaluation across all 10 scenes, all poses, N_RUNS times each.
# Reports per-scene average coverage via eval/summarize.py.
#
# Usage (interactive):  bash mapex_eval_all.sh [<n_runs>]
# Usage (sbatch):       sbatch mapex_eval_all.sh [<n_runs>]
#
# <n_runs>  repetitions per (scene, pose); default 5
#           RRTConnect path planning is stochastic, so multiple runs are needed.
#
# Visualisation is disabled (no frames, no video).
# Volume replay matches detr_eval_all.sh (no --voxel_grid, bounds-filtered wavemap).

N_RUNS=${1:-5}

ROOT=/cluster/project/cvg/students/shangwu/FrontierNet_mapex
MAPEX_DIR=/cluster/project/cvg/students/shangwu/MapEx
MAPEX_ENV=/cluster/project/cvg/students/shangwu/mapex_env

# scene -> number of pt_N configs available
declare -A N_POSES
N_POSES[804]=3; N_POSES[807]=4; N_POSES[812]=2; N_POSES[824]=3; N_POSES[827]=3
N_POSES[834]=3; N_POSES[854]=1; N_POSES[876]=5; N_POSES[879]=4; N_POSES[880]=2
# SCENES=(804 807 812 824 827 834 854 876 879 880)
SCENES=876

# ---- validate ----
if [[ ! -d "${MAPEX_DIR}/pretrained_models/weights/big_lama" ]]; then
    echo "ERROR: MapEx weights not found in ${MAPEX_DIR}/pretrained_models/weights/"
    echo "  Download pretrained models first."
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

TOTAL_RUNS=0
for s in "${SCENES[@]}"; do TOTAL_RUNS=$(( TOTAL_RUNS + N_POSES[$s] * N_RUNS )); done

echo "================================================================"
echo " mapex_eval_all:  n_runs=${N_RUNS}  (RRT stochastic → multiple runs needed)"
echo " Scenes: ${SCENES[*]}"
echo " Total runs: ${TOTAL_RUNS}"
echo " Started: $(date)"
echo "================================================================"

RUN_IDX=0

for SCENE in "${SCENES[@]}"; do
    N_PT=${N_POSES[$SCENE]}
    SCENE_PAD=$(printf "%06d" "${SCENE}")
    MESH=${ROOT}/eval_data/mesh/${SCENE_PAD}.glb
    VOXEL_GRID=${ROOT}/eval_data/voxel_grid/${SCENE_PAD}-voxel_grid.ply

    echo ""
    echo "----------------------------------------------------------------"
    echo " Scene ${SCENE}  (${N_PT} poses × ${N_RUNS} runs)"
    echo "----------------------------------------------------------------"

    for PT in $(seq 1 "${N_PT}"); do
        CONFIG=${ROOT}/config/${SCENE}/pt_${PT}/mapex.yaml

        if [[ ! -f "${CONFIG}" ]]; then
            echo "  WARN: config not found, skipping: ${CONFIG}"
            continue
        fi

        for RUN in $(seq 1 "${N_RUNS}"); do
            RUN_IDX=$(( RUN_IDX + 1 ))
            NAME=mapex_${SCENE}_pt${PT}_run${RUN}
            EXPLORE_LOG=logs/${NAME}.explore.log
            REPLAY_LOG=logs/${NAME}.replay.log

            echo "[$(date '+%H:%M:%S')] (${RUN_IDX}/${TOTAL_RUNS}) scene=${SCENE} pt=${PT} run=${RUN}"

            # ---- explore ----
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
                --log_level   30 \
                > "${EXPLORE_LOG}" 2>&1

            EC=$?
            OUTPUT_JSON=output/${NAME}/exploration_state.json
            OUTPUT_SIZE=$(stat -c%s "${OUTPUT_JSON}" 2>/dev/null || echo 0)
            if [[ $EC -ne 0 ]]; then
                if [[ $EC -eq 134 && $OUTPUT_SIZE -gt 1000000 ]]; then
                    # SIGABRT (exit 134) from glibc heap-corruption-at-cleanup in LaMa/Open3D.
                    # The exploration completed and wrote a valid output file before crashing.
                    echo "  WARN: exit ${EC} (heap corruption at cleanup, output ${OUTPUT_SIZE} bytes — proceeding)"
                else
                    echo "  ERROR: exploration failed (exit ${EC}, output ${OUTPUT_SIZE} bytes) — see ${EXPLORE_LOG}"
                    continue
                fi
            fi

            # ---- compute volume (no video, no frames) ----
            python -u eval/replay_visualization_headless.py \
                --mesh          "${MESH}" \
                --json_file     output/${NAME}/exploration_state.json \
                --config        "${CONFIG}" \
                --compute_volume \
                --volume_output output/${NAME}/exploration_state_with_volume.json \
                > "${REPLAY_LOG}" 2>&1

            if [[ $? -ne 0 ]]; then
                echo "  ERROR: volume replay failed — see ${REPLAY_LOG}"
            fi
        done
    done

    # ---- per-scene summary ----
    GLOB="output/mapex_${SCENE}_pt*_run*/exploration_state_with_volume.json"
    METRICS_FILE="output/metrics/mapex_scene${SCENE}.json"
    echo ""
    python eval/summarize.py \
        --json_glob "${GLOB}" \
        --voxel_grid "${VOXEL_GRID}" \
        --scene "${SCENE}" \
        --save "${METRICS_FILE}" \
        --verbose

done

echo "================================================================"
echo " All done.  Finished: $(date)"
echo "================================================================"
