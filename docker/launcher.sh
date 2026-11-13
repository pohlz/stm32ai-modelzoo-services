# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025-2026 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
#!/usr/bin/env bash
set -e

# Resolve paths relative to this script (works from any cwd)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Set image name
IMAGE_NAME="modelzoo_docker"

# GPU toggle (default: true). Override with env or CLI flag.
# Usage examples:
#   USE_GPU=false ./docker/launcher.sh
#   ./docker/launcher.sh --cpu
#   ./docker/launcher.sh --gpu

USE_GPU="${USE_GPU:-true}"

## Optional container-side clone of stm32ai-modelzoo
# Set these env vars before running to trigger a clone INSIDE the container:
#   CLONE_STM32AI_MODELZOO=true
#   STM32AI_MODELZOO_BRANCH=main   # optional (default main)
#   STM32AI_MODELZOO_DIRNAME=stm32ai-modelzoo  # optional

BUILD_ARGS=""
RUN_ENV=""

# Proxy variables (optional): forwarded to build and run if set
PROXY_VARS=(HTTP_PROXY http_proxy HTTPS_PROXY https_proxy NO_PROXY no_proxy)
for VAR in "${PROXY_VARS[@]}"; do
    VAL="$(eval echo \${$VAR})"
    if [ ! -z "$VAL" ]; then
        BUILD_ARGS+=" --build-arg $VAR=$VAL"
        RUN_ENV+=" -e $VAR=$VAL"
    fi
done

# Clone controls (optional): forwarded to build and run if set
CLONE_VARS=(CLONE_STM32AI_MODELZOO STM32AI_MODELZOO_BRANCH STM32AI_MODELZOO_DIRNAME)
for VAR in "${CLONE_VARS[@]}"; do
    VAL="$(eval echo \${$VAR})"
    if [ ! -z "$VAL" ]; then
        BUILD_ARGS+=" --build-arg $VAR=$VAL"
        RUN_ENV+=" -e $VAR=$VAL"
    fi
done

MODELS=("pose_estimation" "object_detection" "image_classification" "human_activity_recognition" "audio_event_detection" "hand_posture" "semantic_segmentation" "depth_estimation" "instance_segmentation" "face_detection" "neural_style_transfer" "speech_enhancement" "re_identification" "hand_posture")

declare -A DATASETS
#Put the paths to your datasets here
#EXAMPLE: 
#DATASETS["model_name"]="/path/to/dataset.zip"
#DATASETS["object_detection"]="/home/user/environment/od_validation_dataset_coco_person.zip"


# Create host_tools directory if it doesn't exist, we recommend to put any ST tools(stm32cubeide for example) you want to use inside the docker container in this directory
mkdir -p "$SCRIPT_DIR/host_tools"

echo "Building modelzoo docker image"
DOCKERFILE_PATH="$SCRIPT_DIR/Dockerfile"
CONTEXT_PATH="$REPO_ROOT"

# Build image
docker build $BUILD_ARGS -t "$IMAGE_NAME" -f "$DOCKERFILE_PATH" "$CONTEXT_PATH"

VOLUME_FLAGS=""

for model in "${MODELS[@]}"; do
    MODEL_ORIG_PATH="$REPO_ROOT/$model"
    if [[ -d "$MODEL_ORIG_PATH" ]]; then
        # Always mount TF experiments_outputs
        LOCAL_EXP_DIR_TF="$MODEL_ORIG_PATH/tf/src/experiments_outputs"
        CONTAINER_EXP_DIR_TF="/workspace/stm32ai-modelzoo-services/$model/tf/src/experiments_outputs"
        mkdir -p "$LOCAL_EXP_DIR_TF"
        VOLUME_FLAGS+=" -v $LOCAL_EXP_DIR_TF:$CONTAINER_EXP_DIR_TF"

        # Mount pytorch experiments_outputs only for image_classification and object_detection
        if [[ "$model" == "image_classification" || "$model" == "object_detection" ]]; then
            LOCAL_EXP_DIR_PT="$MODEL_ORIG_PATH/pt/src/experiments_outputs"
            CONTAINER_EXP_DIR_PT="/workspace/stm32ai-modelzoo-services/$model/pt/src/experiments_outputs"
            mkdir -p "$LOCAL_EXP_DIR_PT"
            VOLUME_FLAGS+=" -v $LOCAL_EXP_DIR_PT:$CONTAINER_EXP_DIR_PT"
        fi
    else
        echo "Warning: $MODEL_ORIG_PATH does not exist, skipping creation of experiments_outputs."
    fi

    dataset_path="${DATASETS[$model]}"
    if [[ -e "$dataset_path" ]]; then
        dataset_name=$(basename "$dataset_path")
        VOLUME_FLAGS+=" -v $dataset_path:/workspace/stm32ai-modelzoo-services/$model/datasets/$dataset_name"
    else
        echo "Could not find dataset for $model : $dataset_path"
    fi
done

echo
echo "Launching container's shell"
# Mount host_tools directory
HOST_TOOLS_ABS="$SCRIPT_DIR/host_tools"
VOLUME_FLAGS+=" -v $HOST_TOOLS_ABS:/workspace/stm32ai-modelzoo-services/docker/host_tools"

# Optional CLI override for USE_GPU
if [[ "${1:-}" == "--cpu" ]]; then
    USE_GPU="false"
elif [[ "${1:-}" == "--gpu" ]]; then
    USE_GPU="true"
fi

echo "USE_GPU=$USE_GPU"

# Shared memory size for DataLoader and CUDA ops
# Override by setting SHM_SIZE (e.g., SHM_SIZE=8g)
SHM_SIZE="${SHM_SIZE:-8g}"

# Optional: use host IPC namespace to maximize shared memory handling
# Enable by setting USE_IPC_HOST=true
USE_IPC_HOST="${USE_IPC_HOST:-false}"
IPC_FLAG=""
if [[ "$USE_IPC_HOST" == "true" ]]; then
    IPC_FLAG=" --ipc=host"
    SHM_FLAG="" # --shm-size is ignored with --ipc=host
    echo "IPC=host enabled; ignoring SHM_SIZE (using host /dev/shm)"
else
    SHM_FLAG=" --shm-size=$SHM_SIZE"
fi

if [[ "$USE_GPU" == "true" ]]; then
    # GPU run (default): mount volumes and use local user permissions
    docker run -it --rm --gpus all$IPC_FLAG$SHM_FLAG $RUN_ENV $VOLUME_FLAGS "$IMAGE_NAME"
else
    # CPU run
    docker run -it --rm$IPC_FLAG$SHM_FLAG $RUN_ENV $VOLUME_FLAGS "$IMAGE_NAME"
fi
