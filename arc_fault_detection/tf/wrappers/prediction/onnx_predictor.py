# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

from common.registries.predictor_registry import PREDICTOR_WRAPPER_REGISTRY

from arc_fault_detection.tf.src.prediction import AFDONNXPredictor

__all__ = ["AFDONNXPredictor"]

PREDICTOR_WRAPPER_REGISTRY.register(
    predictor_name="onnx_predictor",
    framework="tf",
    use_case="arc_fault_detection"
)(AFDONNXPredictor)
