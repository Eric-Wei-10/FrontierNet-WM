#!/bin/bash -l
#SBATCH --job-name=vox@k
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --mem-per-cpu=48g
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --gpus=rtx_4090:1

# ==============================================================================
# usage and information helper
# ==============================================================================
usage() {
    echo "Usage: $0 <MODEL> <SCENE_ID> <ATTEMPT_TIMES>"
    echo ""
    echo "Arguments:"
    echo "  <MODEL>    : Specify the model type (see list below)."
    echo "  <SCENE_ID> : The scene ID."
    echo "               Format: Must be 6 digits (e.g., 000876, 000999)."
    echo "  <ATTEMPT_TIMES>  : The attempt number."
    echo ""
    echo "Available Options for MODEL:"
    echo "  - SINGLE_TRAJ"
    echo "  - MULTI_TRAJ"
    echo "  - SINGLE_MONO"
    echo "  - MULTI_MONO"
    echo "  - GT"
    echo "  - GT_MONO"
    echo "  - FULL"
    echo ""
    echo "Example:"
    echo "  $0 GT 000876 10"
    exit 1
}

if [ "$#" -ne 3 ]; then
    usage
fi

MODEL=$1
SCENE_ID=$2
ATTEMPT_TIMES=$3
SCRIPT_TO_RUN="./vox_at_k.sh"

# check validity of MODEL parameter
case "$MODEL" in
    "SINGLE_TRAJ" | \
    "MULTI_TRAJ" | \
    "SINGLE_MONO" | \
    "MULTI_MONO" | \
    "GT" | \
    "GT_MONO" | \
    "FULL")
        ;;
    *)
        echo "Error: Invalid MODEL parameter: '$MODEL'"
        echo "Please allow one of the following:"
        echo "SINGLE_TRAJ, MULTI_TRAJ, SINGLE_MONO, MULTI_MONO, GT, GT_MONO, FULL"
        exit 1
        ;;
esac

# check validity of SCENE_ID parameter
if [[ ! "$SCENE_ID" =~ ^[0-9]{6}$ ]]; then
    echo "Error: Invalid SCENE_ID format: '$SCENE_ID'"
    echo "Requirement: SCENE_ID must be exactly 6 digits (e.g., 000876)."
    exit 1
fi

echo "========================================"
echo "Starting Batch Processing"
echo "MODEL: $MODEL"
echo "SCENE: $SCENE_ID"
echo "========================================"

# Loop over START_POSE from 0 to 4
for START_POSE in {0..4}
do
    # Loop over ATTEMPT from 1 to 10
    for (( ATTEMPT=1; ATTEMPT<=ATTEMPT_TIMES; ATTEMPT++ ))
    do
        echo "--------------------------------------------------"
        echo "Running: Pose $START_POSE | Attempt $ATTEMPT"
        echo "--------------------------------------------------"
        
        bash $SCRIPT_TO_RUN "$MODEL" "$SCENE_ID" "$ATTEMPT" "$START_POSE"
        
    done
done

echo "========================================"
echo "All tasks finished."
echo "========================================"
