# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
from common.registries.evaluator_registry import EVALUATOR_WRAPPER_REGISTRY

from speech_enhancement.pt.src.evaluators import SEONNXEvaluatorWrapper

__all__ = ['SEONNXEvaluatorWrapper']

# Register the ONNX Evaluator class from another folder
EVALUATOR_WRAPPER_REGISTRY.register(
    evaluator_name="onnx_evaluator",
    framework="torch",
    use_case="speech_enhancement"
)(SEONNXEvaluatorWrapper)

