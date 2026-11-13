# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
from common.registries.quantizer_registry import QUANTIZER_WRAPPER_REGISTRY

from speech_enhancement.pt.src.quantization import SEONNXPTQQuantizer
__all__ = ['SEONNXPTQQuantizer']

# Register the ONNX PTQ Quantizer class from another folder
QUANTIZER_WRAPPER_REGISTRY.register(
    quantizer_name="onnx_quantizer",
    framework="torch",
    use_case="speech_enhancement"
)(SEONNXPTQQuantizer)

