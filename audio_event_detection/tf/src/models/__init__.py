# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

from .custom_model import get_custom_model
from .model_utils import add_head, prepare_kwargs_for_model
from .yamnet import get_yamnet
from .miniresnetv1 import get_miniresnetv1
from .miniresnetv2 import get_miniresnetv2