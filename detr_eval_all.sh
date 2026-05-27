#!/bin/bash -l
#SBATCH --job-name=detr_876
#SBATCH --output=logs/eval_all_%j.out
#SBATCH --error=logs/eval_all_%j.err
#SBATCH --mem-per-cpu=24g
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --gpus=rtx_4090:1
#SBATCH --account=ls_polle

# Run a full evaluation of one checkpoint across all 10 scenes, all poses, N_RUNS
# times each, then report per-scene average coverage.
#
# Usage (interactive):  bash detr_eval_all.sh <ckpt_name> [<n_runs>] [--metrics_only | --replay_only | --resume]
# Usage (sbatch):       sbatch detr_eval_all.sh <ckpt_name> [<n_runs>] [--metrics_only | --replay_only | --resume]
#
# <ckpt_name>     folder under DETR-Factory-PyTorch/checkpoints/, e.g. multi_10000
# <n_runs>        repetitions per (scene, pose); default 5
# --metrics_only  skip exploration and replay; just run summarize.py on existing output files.
# --replay_only   skip exploration; run replay only for runs that have exploration_state.json
#                 but no exploration_state_with_volume.json.
# --resume        skip runs that already have exploration_state_with_volume.json (fully done);
#                 for runs with only exploration_state.json run replay only;
#                 for runs with neither run the full explore+replay pipeline.
#
# Visualisation is disabled (no frames, no video).
# Volume replay is kept (needed for coverage stats).

CKPT=${1:?"Usage: $0 <ckpt_name> [n_runs] [--metrics_only|--replay_only|--resume] [--shard=N] [--n_shards=N]"}
N_RUNS=${2:-5}
METRICS_ONLY=0
REPLAY_ONLY=0
RESUME=0
SHARD=-1
N_SHARDS=1
for arg in "$@"; do
    [[ "${arg}" == "--metrics_only"  ]] && METRICS_ONLY=1
    [[ "${arg}" == "--replay_only"   ]] && REPLAY_ONLY=1
    [[ "${arg}" == "--resume"        ]] && RESUME=1
    [[ "${arg}" == "--shard="*       ]] && SHARD="${arg#--shard=}"
    [[ "${arg}" == "--n_shards="*    ]] && N_SHARDS="${arg#--n_shards=}"
done

ROOT=/cluster/project/cvg/students/shangwu/FrontierNet_mapex
CKPT_PATH=/cluster/project/cvg/students/shangwu/DETR-Factory-PyTorch/checkpoints/${CKPT}/best_model.pth

# scene -> number of pt_N configs available
declare -A N_POSES
N_POSES[804]=3; N_POSES[807]=4; N_POSES[812]=2; N_POSES[824]=3; N_POSES[827]=3
N_POSES[834]=3; N_POSES[854]=1; N_POSES[876]=5; N_POSES[879]=4; N_POSES[880]=2
SCENES=(876 804 807 812 824 827 834 854 879 880)

# Explicit shard assignments — balanced by total pose count (3 shards):
#   shard 0: 876 812 834      →  5+2+3 = 10 poses
#   shard 1: 804 824 854 880  →  3+3+1+2 =  9 poses
#   shard 2: 807 827 879      →  4+3+4 = 11 poses
declare -A _SCENE_SHARD
_SCENE_SHARD[876]=0; _SCENE_SHARD[812]=0; _SCENE_SHARD[834]=0
_SCENE_SHARD[804]=1; _SCENE_SHARD[824]=1; _SCENE_SHARD[854]=1; _SCENE_SHARD[880]=1
_SCENE_SHARD[807]=2; _SCENE_SHARD[827]=2; _SCENE_SHARD[879]=2

if [[ ${SHARD} -ge 0 ]]; then
    SHARDED=()
    for s in "${SCENES[@]}"; do
        [[ "${_SCENE_SHARD[$s]}" -eq ${SHARD} ]] && SHARDED+=("$s")
    done
    SCENES=("${SHARDED[@]}")
fi

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

SHARD_TAG=""
[[ ${SHARD} -ge 0 ]] && SHARD_TAG="  shard=${SHARD}/${N_SHARDS}"
echo "================================================================"
echo " detr_eval_all: ckpt=${CKPT}  n_runs=${N_RUNS}${SHARD_TAG}"
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

            if [[ $METRICS_ONLY -eq 1 ]]; then
                VOLUME_JSON=output/${NAME}/exploration_state_with_volume.json
                if [[ ! -f "${VOLUME_JSON}" ]]; then
                    echo "  SKIP: no volume JSON found at ${VOLUME_JSON}"
                fi
                continue
            fi

            if [[ $REPLAY_ONLY -eq 1 ]]; then
                OUTPUT_JSON=output/${NAME}/exploration_state.json
                VOLUME_JSON=output/${NAME}/exploration_state_with_volume.json
                if [[ ! -f "${OUTPUT_JSON}" ]]; then
                    echo "  SKIP: no exploration JSON at ${OUTPUT_JSON}"
                    continue
                fi
                if [[ -f "${VOLUME_JSON}" ]]; then
                    echo "  SKIP: volume JSON already exists — ${VOLUME_JSON}"
                    continue
                fi
                echo "  Replaying ${NAME} ..."
                # fall through to the replay block below
            elif [[ $RESUME -eq 1 ]]; then
                OUTPUT_JSON=output/${NAME}/exploration_state.json
                VOLUME_JSON=output/${NAME}/exploration_state_with_volume.json
                if [[ -f "${VOLUME_JSON}" ]]; then
                    echo "  SKIP: already complete — ${VOLUME_JSON}"
                    continue
                fi
                if [[ -f "${OUTPUT_JSON}" ]]; then
                    echo "  Resume: exploration exists, running replay only for ${NAME} ..."
                    # fall through to replay block, skip explore
                else
                    echo "  Resume: running full explore+replay for ${NAME} ..."
                    # fall through to explore block below
                fi
            fi

            if [[ $REPLAY_ONLY -eq 0 && ! ( $RESUME -eq 1 && -f "output/${NAME}/exploration_state.json" ) ]]; then
            # ---- explore ----
            python -u demo_exploration_headless.py \
                --write_path   output/${NAME}/exploration_state.json \
                --unet_weight  "${CKPT_PATH}" \
                --detr_num_queries 20 \
                --model_type   detr \
                --mesh         "${MESH}" \
                --voxel_grid   "${VOXEL_GRID}" \
                --config       "${CONFIG}" \
                --detr_conf_thresh          0.3 \
                --detr_visible_gain_discount 0.45 \
                --log_level    30 \
                > "${EXPLORE_LOG}" 2>&1

            EC=$?
            OUTPUT_JSON=output/${NAME}/exploration_state.json
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
            fi  # end of [[ $REPLAY_ONLY -eq 0 ]] else block

            # ---- compute volume (no video, no frames) ----
            python -u eval/replay_visualization_headless.py \
                --mesh        "${MESH}" \
                --json_file   output/${NAME}/exploration_state.json \
                --config      "${CONFIG}" \
                --compute_volume \
                --volume_output output/${NAME}/exploration_state_with_volume.json \
                > "${REPLAY_LOG}" 2>&1

            REPLAY_EC=$?
            VOLUME_JSON=output/${NAME}/exploration_state_with_volume.json
            VOLUME_SIZE=$(stat -c%s "${VOLUME_JSON}" 2>/dev/null || echo 0)
            if [[ $REPLAY_EC -ne 0 ]]; then
                if [[ $REPLAY_EC -eq 134 && $VOLUME_SIZE -gt 100000 ]]; then
                    # SIGABRT at cleanup (same glibc heap issue as exploration).
                    # Volume JSON was written before the crash — check it's complete.
                    LAST_LINE=$(tail -c 2 "${VOLUME_JSON}" 2>/dev/null || echo "")
                    if [[ "${LAST_LINE}" == *"}"* ]]; then
                        echo "  WARN: replay exit ${REPLAY_EC} (heap corruption at cleanup, volume ${VOLUME_SIZE} bytes — proceeding)"
                    else
                        echo "  ERROR: replay exit ${REPLAY_EC} — volume JSON truncated (${VOLUME_SIZE} bytes) — see ${REPLAY_LOG}"
                        continue
                    fi
                else
                    echo "  ERROR: volume replay failed (exit ${REPLAY_EC}, volume ${VOLUME_SIZE} bytes) — see ${REPLAY_LOG}"
                    continue
                fi
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
