# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
from common.registries.evaluator_registry import EVALUATOR_WRAPPER_REGISTRY

from human_activity_recognition.tf.src.evaluation import KerasModelEvaluator

__all__ = ['KerasModelEvaluator']


EVALUATOR_WRAPPER_REGISTRY.register(
    evaluator_name="keras_evaluator",
    framework="tf",
    use_case="human_activity_recognition"
)(KerasModelEvaluator)

