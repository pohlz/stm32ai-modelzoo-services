# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

from .base import BaseTorchEvaluator, BaseONNXEvaluator
from .spec import MagSpecONNXEvaluator, MagSpecTorchEvaluator
from .evaluate import SEONNXEvaluatorWrapper, SETorchEvaluatorWrapper