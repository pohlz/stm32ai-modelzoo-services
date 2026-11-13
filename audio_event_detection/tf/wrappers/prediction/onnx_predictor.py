# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
from common.registries.predictor_registry import PREDICTOR_WRAPPER_REGISTRY

from audio_event_detection.tf.src.prediction import AEDONNXPredictor

__all__ = ['AEDONNXPredictor']


PREDICTOR_WRAPPER_REGISTRY.register(
    predictor_name="onnx_predictor",
    framework="tf",
    use_case="audio_event_detection"
)(AEDONNXPredictor)