#!/bin/bash -l
#SBATCH --job-name=detr_876_gm_all
#SBATCH --output=/cluster/scratch/hshang/detr_eval/logs/detr_876_gm_all_%j.out
#SBATCH --error=/cluster/scratch/hshang/detr_eval/logs/detr_876_gm_all_%j.err
#SBATCH --mem-per-cpu=24g
#SBATCH --ntasks=1
#SBATCH --time=6:00:00
#SBATCH --gpus=rtx_4090:1
#SBATCH --account=ls_polle

# Run DETR globalmap exploration for all 5 pts of scene 876, N_RUNS each.
#
# Usage (interactive):  bash detr_eval_876_globalmap_all.sh <ckpt_name> [<n_runs>] [--resume | --replay_only | --metrics_only]
# Usage (sbatch):       sbatch detr_eval_876_globalmap_all.sh <ckpt_name> [<n_runs>]
#
# <ckpt_name>: folder under DETR-Factory-PyTorch/checkpoints/, e.g. multi_32000

CKPT=${1:?"Usage: $0 <ckpt_name> [n_runs] [--resume|--replay_only|--metrics_only]"}
N_RUNS=${2:-3}
RESUME=0
REPLAY_ONLY=0
METRICS_ONLY=0
for arg in "$@"; do
    [[ "${arg}" == "--resume"       ]] && RESUME=1
    [[ "${arg}" == "--replay_only"  ]] && REPLAY_ONLY=1
    [[ "${arg}" == "--metrics_only" ]] && METRICS_ONLY=1
done

ROOT=/cluster/project/cvg/students/shangwu/FrontierNet_mapex
CKPT_PATH=/cluster/project/cvg/students/shangwu/DETR-Factory-PyTorch/checkpoints/${CKPT}/best_model.pth
SCRATCH=/cluster/scratch/hshang/detr_eval
LOG_DIR=${SCRATCH}/logs
OUTPUT_DIR=${SCRATCH}/output

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /cluster/project/cvg/students/shangwu/ftnet_inference_env_2
module load stack/2025-06

export EGL_PLATFORM=surfaceless
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
cd "${ROOT}"

if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "ERROR: checkpoint not found: ${CKPT_PATH}"
    exit 1
fi

SCENE=876
N_PT=5
SCENE_PAD=000876
MESH=${ROOT}/eval_data/mesh/${SCENE_PAD}.glb
VOXEL_GRID=${ROOT}/eval_data/voxel_grid/${SCENE_PAD}-voxel_grid.ply

TOTAL_RUNS=$(( N_PT * N_RUNS ))
echo "================================================================"
echo " DETR globalmap — scene 876, pts 1-${N_PT}, n_runs=${N_RUNS}"
echo " ckpt     : ${CKPT}"
echo " Total runs: ${TOTAL_RUNS}"
echo " Started: $(date)"
echo "================================================================"

RUN_IDX=0

for PT in $(seq 1 "${N_PT}"); do
    CONFIG=${ROOT}/config/${SCENE}/pt_${PT}/detr.yaml

    if [[ ! -f "${CONFIG}" ]]; then
        echo "  WARN: config not found, skipping: ${CONFIG}"
        continue
    fi

    for RUN in $(seq 1 "${N_RUNS}"); do
        RUN_IDX=$(( RUN_IDX + 1 ))
        NAME=detr_globalmap_${CKPT}_${SCENE}_pt${PT}_run${RUN}
        EXPLORE_LOG=${LOG_DIR}/${NAME}.explore.log
        REPLAY_LOG=${LOG_DIR}/${NAME}.replay.log
        OUTPUT_JSON=${OUTPUT_DIR}/${NAME}/exploration_state.json
        VOLUME_JSON=${OUTPUT_DIR}/${NAME}/exploration_state_with_volume.json

        echo ""
        echo "[$(date '+%H:%M:%S')] (${RUN_IDX}/${TOTAL_RUNS}) scene=${SCENE} pt=${PT} run=${RUN}"

        if [[ $METRICS_ONLY -eq 1 ]]; then
            [[ ! -f "${VOLUME_JSON}" ]] && echo "  SKIP: no volume JSON at ${VOLUME_JSON}"
            continue
        fi

        # ---- explore ----
        if [[ $REPLAY_ONLY -eq 0 ]]; then
            if [[ $RESUME -eq 1 && -f "${OUTPUT_JSON}" ]]; then
                echo "  SKIP explore: json exists (${OUTPUT_JSON})"
            else
                python -u demo_exploration_headless.py \
                    --write_path      "${OUTPUT_JSON}" \
                    --unet_weight     "${CKPT_PATH}" \
                    --model_type      detr \
                    --detr_num_queries 20 \
                    --detr_conf_thresh 0.3 \
                    --detr_dedup_radius 0.3 \
                    --mesh            "${MESH}" \
                    --voxel_grid      "${VOXEL_GRID}" \
                    --config          "${CONFIG}" \
                    --max_steps       1000 \
                    --log_level       30 \
                    > "${EXPLORE_LOG}" 2>&1
                EC=$?
                SIZE=$(stat -c%s "${OUTPUT_JSON}" 2>/dev/null || echo 0)
                if [[ $EC -ne 0 && ! ( $EC -eq 134 && $SIZE -gt 10000 ) ]]; then
                    echo "  ERROR: explore exit=${EC}, json=${SIZE} bytes — see ${EXPLORE_LOG}"
                    continue
                fi
                echo "  explore exit=${EC}, json=${SIZE} bytes"
            fi
        fi

        # ---- skip replay if volume already exists and resuming ----
        if [[ $RESUME -eq 1 && -f "${VOLUME_JSON}" ]]; then
            echo "  SKIP replay: volume JSON exists (${VOLUME_JSON})"
            continue
        fi

        if [[ ! -f "${OUTPUT_JSON}" ]]; then
            echo "  SKIP replay: no exploration JSON at ${OUTPUT_JSON}"
            continue
        fi

        # ---- replay + volume ----
        python -u eval/replay_visualization_headless.py \
            --mesh          "${MESH}" \
            --json_file     "${OUTPUT_JSON}" \
            --config        "${CONFIG}" \
            --compute_volume \
            --volume_output "${VOLUME_JSON}" \
            > "${REPLAY_LOG}" 2>&1
        REPLAY_EC=$?
        VSIZE=$(stat -c%s "${VOLUME_JSON}" 2>/dev/null || echo 0)
        if [[ $REPLAY_EC -ne 0 && ! ( $REPLAY_EC -eq 134 && $VSIZE -gt 100000 ) ]]; then
            echo "  ERROR: replay exit=${REPLAY_EC}, volume=${VSIZE} bytes — see ${REPLAY_LOG}"
            continue
        fi
        echo "  replay exit=${REPLAY_EC}, volume=${VSIZE} bytes"
    done
done

# ---- per-scene summary ----
echo ""
echo "================================================================"
GLOB="${OUTPUT_DIR}/detr_globalmap_${CKPT}_${SCENE}_pt*_run*/exploration_state_with_volume.json"
METRICS_FILE="${OUTPUT_DIR}/metrics/detr_globalmap_${CKPT}_scene${SCENE}.json"
mkdir -p "${OUTPUT_DIR}/metrics"
python eval/summarize.py \
    --json_glob  "${GLOB}" \
    --voxel_grid "${VOXEL_GRID}" \
    --scene      "${SCENE}" \
    --save       "${METRICS_FILE}" \
    --verbose

echo "================================================================"
echo " All done.  Finished: $(date)"
echo "================================================================"
