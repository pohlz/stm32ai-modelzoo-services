#!/bin/bash
#
# Copyright (c) 2025 STMicroelectronics.
# All rights reserved.
#
# This software is licensed under terms that can be found in the LICENSE file
# in the root directory of this software component.
# If no LICENSE file comes with this software, it is provided AS-IS.

COMPATIBLE=$(cat /proc/device-tree/compatible)
SOFTWARE="AI_NPU"
STM32MP257="stm32mp257"
STM32MP255="stm32mp255"
STM32MP253="stm32mp253"
STM32MP251="stm32mp251"
STM32MP235="stm32mp235"
STM32MP233="stm32mp233"
STM32MP231="stm32mp231"
STM32MP2_NPU="stm32mp2x with GPU/NPU"

if [[ "$COMPATIBLE" == *"$STM32MP253"* ]] || [[ "$COMPATIBLE" == *"$STM32MP251"* ]] || [[ "$COMPATIBLE" == *"$STM32MP233"* ]] || [[ "$COMPATIBLE" == *"$STM32MP231"* ]]; then
  if [[ "$SOFTWARE" == "AI_NPU" ]]; then
    echo "Software X-LINUX-AI installed is not compatible with the board, please install X-LINUX-AI CPU version for plateform without hardware accelerator"
    exit 1
  fi
fi

if [[ "$FRAMEWORK" == "nbg" ]]; then
  NN_EXT=".nb"
elif [[ "$FRAMEWORK" == "tflite" ]]; then
  NN_EXT=".tflite"
elif [[ "$FRAMEWORK" == "onnx" ]]; then
  NN_EXT=".onnx"
else
  #define a default value if no framework is specified
  if [[ "$COMPATIBLE" == *"$STM32MP257"* ]] || [[ "$COMPATIBLE" == *"$STM32MP255"* ]] || [[ "$COMPATIBLE" == *"$STM32MP235"* ]]; then
    NN_EXT=".nb"
  else
    NN_EXT=".tflite"
  fi
fi

if [[ "$COMPATIBLE" == *"$STM32MP257"* ]] || [[ "$COMPATIBLE" == *"$STM32MP255"* ]] || [[ "$COMPATIBLE" == *"$STM32MP235"* ]]; then
  CAMERA_SRC="LIBCAMERA"
  SEMANTIC_SEGMENTATION_MODEL="deeplabv3_257_int8_per_tensor$NN_EXT"
  SEMANTIC_SEGMENTATION_LABEL="labels_pascalvoc"
  POSE_ESTIMATION_MODEL="yolov8n_256_quant_pt_uf_pose_coco-st$NN_EXT"
  MACHINE=$STM32MP2_NPU
  DWIDTH=760
  DHEIGHT=568
  DFPS=30
  COMPUTE_ENGINE="--npu"
  OPTIONS="--dual_camera_pipeline"
  IMAGE_CLASSIFICATION_MODEL="mobilenetv2_a100_224_int8_per_tensor$NN_EXT"
  IMAGE_CLASSIFICATION_LABEL="labels_imagenet_2012"
  if [[ "$NN_EXT" == ".nb" ]]; then
    OBJ_DETEC_MODEL="ssd_mobilenet_v2_fpnlite_10_256_int8_per_tensor$NN_EXT"
    OBJ_DETEC_MODEL_LABEL="class_names"
  else
    OBJ_DETEC_MODEL="ssd_mobilenet_v2_fpnlite_10_256_int8$NN_EXT"
    OBJ_DETEC_MODEL_LABEL="class_names"
  fi
else
  echo "Software X-LINUX-AI installed is not compatible with the board, please install X-LINUX-AI CPU version for plateform without hardware accelerator"
  exit 1
fi

echo "machine used = "$MACHINE