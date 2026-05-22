#!/bin/bash -l
#SBATCH --job-name=detr_804_807
#SBATCH --output=logs/eval_all_%j.out
#SBATCH --error=logs/eval_all_%j.err
#SBATCH --mem-per-cpu=24g
#SBATCH --ntasks=1
#SBATCH --time=120:00:00
#SBATCH --gpus=rtx_4090:1
#SBATCH --account=es_schin

# Run a full evaluation of one checkpoint across all 10 scenes, all poses, N_RUNS
# times each, then report per-scene average coverage.
#
# Usage (interactive):  bash detr_eval_all.sh <ckpt_name> [<n_runs>]
# Usage (sbatch):       sbatch detr_eval_all.sh <ckpt_name> [<n_runs>]
#
# <ckpt_name>  folder under DETR-Factory-PyTorch/checkpoints/, e.g. multi_10000
# <n_runs>     repetitions per (scene, pose); default 5
#
# Visualisation is disabled (no frames, no video).
# Volume replay is kept (needed for coverage stats).

CKPT=${1:?"Usage: $0 <ckpt_name> [n_runs]"}
N_RUNS=${2:-5}

ROOT=/cluster/project/cvg/students/shangwu/FrontierNet_mapex
CKPT_PATH=/cluster/project/cvg/students/shangwu/DETR-Factory-PyTorch/checkpoints/${CKPT}/best_model.pth

# scene -> number of pt_N configs available
declare -A N_POSES
N_POSES[804]=3; N_POSES[807]=4; N_POSES[812]=2; N_POSES[824]=3; N_POSES[827]=3
N_POSES[834]=3; N_POSES[854]=1; N_POSES[876]=5; N_POSES[879]=4; N_POSES[880]=2
SCENES=(876)

# ---- validate ----
if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "ERROR: checkpoint not found: ${CKPT_PATH}"; exit 1
fi

# ---- environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /cluster/project/cvg/students/shangwu/ftnet_inference_env_2

module load stack/2025-06

export EGL_PLATFORM=surfaceless
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility

mkdir -p logs output

TOTAL_RUNS=0
for s in "${SCENES[@]}"; do TOTAL_RUNS=$(( TOTAL_RUNS + N_POSES[$s] * N_RUNS )); done

echo "================================================================"
echo " detr_eval_all: ckpt=${CKPT}  n_runs=${N_RUNS}"
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
        CONFIG=${ROOT}/config/${SCENE}/pt_${PT}/detr.yaml

        if [[ ! -f "${CONFIG}" ]]; then
            echo "  WARN: config not found, skipping: ${CONFIG}"
            continue
        fi

        for RUN in $(seq 1 "${N_RUNS}"); do
            RUN_IDX=$(( RUN_IDX + 1 ))
            NAME=detr_${CKPT}_${SCENE}_pt${PT}_run${RUN}
            EXPLORE_LOG=logs/${NAME}.explore.log
            REPLAY_LOG=logs/${NAME}.replay.log

            echo "[$(date '+%H:%M:%S')] (${RUN_IDX}/${TOTAL_RUNS}) scene=${SCENE} pt=${PT} run=${RUN}"

            OUTPUT_JSON=output/${NAME}/exploration_state.json
            OUTPUT_SIZE=$(stat -c%s "${OUTPUT_JSON}" 2>/dev/null || echo 0)

            # ---- explore (skip if valid output already exists) ----
            if [[ $OUTPUT_SIZE -gt 10000 ]]; then
                echo "  SKIP: exploration output already exists (${OUTPUT_SIZE} bytes)"
            else
                python -u demo_exploration_headless.py \
                    --write_path   "${OUTPUT_JSON}" \
                    --unet_weight  "${CKPT_PATH}" \
                    --detr_num_queries 20 \
                    --model_type   detr \
                    --mesh         "${MESH}" \
                    --voxel_grid   "${VOXEL_GRID}" \
                    --config       "${CONFIG}" \
                    --detr_conf_thresh          0.4 \
                    --detr_visible_gain_discount 0.45 \
                    --log_level    30 \
                    > "${EXPLORE_LOG}" 2>&1

                EC=$?
                OUTPUT_SIZE=$(stat -c%s "${OUTPUT_JSON}" 2>/dev/null || echo 0)
                if [[ $EC -ne 0 ]]; then
                    if [[ $EC -eq 134 && $OUTPUT_SIZE -gt 10000 ]]; then
                        # SIGABRT (exit 134) from glibc heap-corruption-at-cleanup in Open3D.
                        # The exploration completed and wrote a valid output file before crashing.
                        echo "  WARN: exit ${EC} (heap corruption at cleanup, output ${OUTPUT_SIZE} bytes — proceeding)"
                    else
                        echo "  ERROR: exploration failed (exit ${EC}, output ${OUTPUT_SIZE} bytes) — see ${EXPLORE_LOG}"
                        continue
                    fi
                fi
            fi

            # ---- compute volume (no video, no frames) ----
            python -u eval/replay_visualization_headless.py \
                --mesh        "${MESH}" \
                --json_file   output/${NAME}/exploration_state.json \
                --config      "${CONFIG}" \
                --compute_volume \
                --volume_output output/${NAME}/exploration_state_with_volume.json \
                > "${REPLAY_LOG}" 2>&1

            if [[ $? -ne 0 ]]; then
                echo "  ERROR: volume replay failed — see ${REPLAY_LOG}"
            fi
        done
    done

    # ---- per-scene summary ----
    GLOB="output/detr_${CKPT}_${SCENE}_pt*_run*/exploration_state_with_volume.json"
    METRICS_FILE="output/metrics/detr_${CKPT}_scene${SCENE}.json"
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
