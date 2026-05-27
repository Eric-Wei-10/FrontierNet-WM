#!/bin/bash
# Split 10 scenes into N_SHARDS SLURM jobs and submit them in parallel.
#
# Usage: bash detr_eval_launch.sh <ckpt_name> [n_runs] [--metrics_only|--replay_only|--resume]
#
# All arguments are forwarded to detr_eval_all.sh; --shard and --n_shards are
# appended automatically.  Each job gets a distinct name (detr_eval_s0 / s1 / s2)
# and writes its own log via the %j SLURM job-ID token.

N_SHARDS=3
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ -z "${1}" ]]; then
    echo "Usage: $0 <ckpt_name> [n_runs] [--metrics_only|--replay_only|--resume]"
    exit 1
fi

echo "Submitting ${N_SHARDS} shards for ckpt=${1} ..."
for SHARD in $(seq 0 $(( N_SHARDS - 1 ))); do
    JID=$(sbatch \
            --job-name="detr_eval_s${SHARD}" \
            "${SCRIPT_DIR}/detr_eval_all.sh" \
            "$@" --shard=${SHARD} --n_shards=${N_SHARDS} \
        | awk '{print $NF}')
    echo "  shard ${SHARD}: job ${JID}  →  logs/eval_all_${JID}.out"
done
