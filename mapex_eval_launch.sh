#!/bin/bash
# Split 10 scenes into N_SHARDS SLURM jobs and submit them in parallel.
#
# Usage: bash mapex_eval_launch.sh [n_runs] [--metrics_only|--replay_only|--resume]
#
# All arguments are forwarded to mapex_eval_all.sh; --shard and --n_shards are
# appended automatically.  Each job gets a distinct name (mapex_eval_s0 / s1 / s2)
# and writes its own log via the %j SLURM job-ID token.

N_SHARDS=3
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Submitting ${N_SHARDS} MapEx evaluation shards ..."
for SHARD in $(seq 0 $(( N_SHARDS - 1 ))); do
    JID=$(sbatch \
            --job-name="mapex_eval_s${SHARD}" \
            "${SCRIPT_DIR}/mapex_eval_all.sh" \
            "$@" --shard=${SHARD} --n_shards=${N_SHARDS} \
        | awk '{print $NF}')
    echo "  shard ${SHARD}: job ${JID}  →  logs/mapex_eval_all_${JID}.out"
done
