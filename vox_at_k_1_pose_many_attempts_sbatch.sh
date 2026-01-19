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
    echo "Usage: $0 <MODEL> <SCENE_ID> <ATTEMPT_TIMES> <START_POSE>"
    echo ""
    echo "Arguments:"
    echo "  <MODEL>    : Specify the model type (see list below)."
    echo "  <SCENE_ID> : The scene ID."
    echo "               Format: Must be 6 digits (e.g., 000876, 000999)."
    echo "  <ATTEMPT_TIMES>  : The attempt number."
    echo "  <START_POSE>: The start pose."
    echo "               Format: Must be an integer between 0 and 4."
    echo ""
    echo "Available Options for MODEL:"
    echo "  - SINGLE_TRAJ"
    echo "  - MULTI_TRAJ"
    echo "  - SINGLE_MONO"
    echo "  - MULTI_MONO"
    echo "  - GT"
    echo "  - FULL"
    echo ""
    echo "Example:"
    echo "  $0 GT 000876 10 0"
    exit 1
}

if [ "$#" -ne 4 ]; then
    usage
fi

MODEL=$1
SCENE_ID=$2
ATTEMPT_TIMES=$3
START_POSE=$4
SCRIPT_TO_RUN="./vox_at_k.sh"

# check validity of MODEL parameter
case "$MODEL" in
    "SINGLE_TRAJ" | \
    "MULTI_TRAJ" | \
    "SINGLE_MONO" | \
    "MULTI_MONO" | \
    "GT" | \
    "FULL")
        ;;
    *)
        echo "Error: Invalid MODEL parameter: '$MODEL'"
        echo "Please allow one of the following:"
        echo "SINGLE_TRAJ, MULTI_TRAJ, SINGLE_MONO, MULTI_MONO, GT, FULL"
        exit 1
        ;;
esac

# check validity of SCENE_ID parameter
if [[ ! "$SCENE_ID" =~ ^[0-9]{6}$ ]]; then
    echo "Error: Invalid SCENE_ID format: '$SCENE_ID'"
    echo "Requirement: SCENE_ID must be exactly 6 digits (e.g., 000876)."
    exit 1
fi

# check validity of ATTEMP_TIMES parameter
if [[ ! "$ATTEMPT_TIMES" =~ ^[0-9]+$ ]] || [ "$ATTEMPT_TIMES" -lt 1 ]; then
    echo "Error: Invalid ATTEMPT_TIMES: '$ATTEMPT_TIMES'"
    echo "Requirement: Must be a positive integer (e.g., 1, 5, 10)."
    exit 1
fi

# check validity of START_POSE parameter
if [[ ! "$START_POSE" =~ ^[0-4]$ ]]; then
    echo "Error: Invalid START_POSE: '$START_POSE'"
    echo "Requirement: Must be an integer between 0 and 4."
    exit 1
fi

echo "========================================"
echo "Starting Batch Processing"
echo "MODEL: $MODEL"
echo "SCENE: $SCENE_ID"
echo "START_POSE: $START_POSE"
echo "========================================"

# Loop over ATTEMPT from 1 to $ATTEMPT_TIMES
for (( ATTEMPT=1; ATTEMPT<=ATTEMPT_TIMES; ATTEMPT++ ))
do
    echo "--------------------------------------------------"
    echo "Running: Pose $START_POSE | Attempt $ATTEMPT"
    echo "--------------------------------------------------"
    
    bash $SCRIPT_TO_RUN "$MODEL" "$SCENE_ID" "$ATTEMPT" "$START_POSE"
    
done

echo "========================================"
echo "All tasks finished."
echo "========================================"
