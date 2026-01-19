#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /cluster/project/cvg/students/shangwu/ftnet_inference_env_2

export EGL_PLATFORM=surfaceless
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility

usage() {
    echo "Usage: $0 <MODEL> <SCENE_ID> <ATTEMPT> <START_POSE>"
    echo ""
    echo "Arguments:"
    echo "  <MODEL>    : Specify the model type (see list below)."
    echo "  <SCENE_ID> : The scene ID."
    echo "               Format: Must be 6 digits (e.g., 000876, 000999)."
    echo "  <ATTEMPT>  : The attempt number."
    echo "               Format: Must be an integer between 1 and 10."
    echo "  <START_POSE>: The start pose."
    echo "               Format: Must be an integer between 0 and 4."
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
    echo "  $0 GT 000876 0 0"
    exit 1
}

# Check if all arguments are provided
if [ "$#" -ne 4 ]; then
    usage
fi

## Input Parameters
MODEL=$1
SCENE_ID=$2
ATTEMPT=$3
START_POSE=$4
# if model name ends with "_MONO", change depth_source parameter for exploration
DEPTH_SOURCE="GT"
if [[ "$MODEL" == *"_MONO" ]]; then
    DEPTH_SOURCE="Metric3D"
fi

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

# check validity of ATTEMP parameter
if [[ ! "$ATTEMPT" =~ ^[0-9]+$ ]] || [ "$ATTEMPT" -lt 1 ]; then
    echo "Error: Invalid ATTEMPT: '$ATTEMPT'"
    echo "Requirement: Must be a positive integer (e.g., 1, 5, 10)."
    exit 1
fi

# check validity of START_POSE parameter
if [[ ! "$START_POSE" =~ ^[0-4]$ ]]; then
    echo "Error: Invalid START_POSE: '$START_POSE'"
    echo "Requirement: Must be an integer between 0 and 4."
    exit 1
fi

echo "MODEL: ${MODEL}"
echo "SCENE_ID: ${SCENE_ID}"
echo "ATTEMPT: ${ATTEMPT}"
echo "START_POSE: ${START_POSE}"
echo "DEPTH_SOURCE: ${DEPTH_SOURCE}"


CONFIG="config/hm3d_exploration_${START_POSE}.yaml" ## if full, we need to change the model size from 7 to 11
NAME="${MODEL}_${SCENE_ID}_attempt_${ATTEMPT}_startpose_${START_POSE}"
EVAL_DATA_DIR="/cluster/project/cvg/students/shangwu/FrontierNet/eval_data"
MESH="${EVAL_DATA_DIR}/mesh/${SCENE_ID}-*.glb"
VOXEL_GRID="${EVAL_DATA_DIR}/voxel_grid/${SCENE_ID}-*.ply"

SINGLE_TRAJ_CKPT="/cluster/project/cvg/students/shangwu/Pytorch-UNet/Actmap_single_1000/checkpoints/single_traj_rgbd_with_depth/CP_epoch500_depth_False_2.0.pth"
MULTI_TRAJ_CKPT="/cluster/project/cvg/students/shangwu/Pytorch-UNet/Actmap_multi_1000/checkpoints/multi_traj_rgbd_with_depth/CP_epoch500_depth_False_2.0.pth"
GT_CKPT="/cluster/project/cvg/students/shangwu/Pytorch-UNet/Actmap_gt_1000/checkpoints/trained_with_depth/CP_epoch1000_depth_False_2.0.pth"
GT_MONO_CKPT="/cluster/project/cvg/students/shangwu/Pytorch-UNet/Actmap_gt_1000/checkpoints/trained_with_mono_depth/CP_epoch1000_depth_False_2.0.pth"
FULL_CKPT="model_weights/rgbd_11cls.pth"
MULTI_MONO_CKPT="/cluster/project/cvg/students/shangwu/Pytorch-UNet/Actmap_multi_1000/checkpoints/multi_traj_rgbd_with_mono_depth/CP_epoch500_depth_False_2.0.pth"
SINGLE_MONO_CKPT="/cluster/project/cvg/students/shangwu/Pytorch-UNet/Actmap_single_1000/checkpoints/single_traj_rgbd_with_mono_depth/CP_epoch500_depth_False_2.0.pth"


CKPT=""
if [ "$MODEL" == "SINGLE_TRAJ" ]; then
    CKPT=${SINGLE_TRAJ_CKPT}
elif [ "$MODEL" == "MULTI_TRAJ" ]; then
    CKPT=${MULTI_TRAJ_CKPT}
elif [ "$MODEL" == "GT" ]; then
    CKPT=${GT_CKPT}
elif [ "$MODEL" == "GT_MONO" ]; then
    CKPT=${GT_MONO_CKPT}
elif [ "$MODEL" == "FULL" ]; then
    CKPT=${FULL_CKPT}
elif [ "$MODEL" == "SINGLE_MONO" ]; then
    CKPT=${SINGLE_MONO_CKPT}
elif [ "$MODEL" == "MULTI_MONO" ]; then
    CKPT=${MULTI_MONO_CKPT}
else
    echo "Unknown MODEL: ${MODEL}"
    exit 1
fi

echo "Model: ${MODEL}, START_POSE: ${START_POSE}, ATTEMPT: ${ATTEMPT} starts exploring..."

python demo_exploration_headless.py \
--mesh ${MESH}  \
--config ${CONFIG}  \
--write_path output/${NAME}.json \
--unet_weight ${CKPT} \
--depth_source ${DEPTH_SOURCE}

echo "Model: ${MODEL}, START_POSE: ${START_POSE}, ATTEMPT: ${ATTEMPT} starts replaying..."

python eval/replay_headless.py \
--mesh ${MESH}  \
--json_file output/${NAME}.json \
--config ${CONFIG}  \
--output output/${NAME}_with_volume.json \
-ll 10

echo "Model: ${MODEL}, START_POSE: ${START_POSE}, ATTEMPT: ${ATTEMPT} starts generating result..."

python eval/stat.py \
--json_file output/${NAME}_with_volume.json \
--voxel_grid ${VOXEL_GRID} \
--no_show \
--out output/${NAME}.png