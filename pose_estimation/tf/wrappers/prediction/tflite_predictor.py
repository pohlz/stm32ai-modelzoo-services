# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
from common.registries.predictor_registry import PREDICTOR_WRAPPER_REGISTRY

from pose_estimation.tf.src.prediction.tflite_predictor import TFLiteQuantizedModelPredictor

__all__ = ['TFLiteQuantizedModelPredictor']

# Register the TFLite Predictor class from another folder
PREDICTOR_WRAPPER_REGISTRY.register(
    predictor_name="tflite_predictor",
    framework="tf",
    use_case="pose_estimation"
)(TFLiteQuantizedModelPredictor)

