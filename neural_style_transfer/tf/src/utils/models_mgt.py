#  /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import sys
import os
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import numpy as np
import tensorflow as tf
from omegaconf import DictConfig

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from common.utils import check_attributes, transfer_pretrained_weights, check_model_support

def ai_runner_invoke(image_processed, ai_runner_interpreter):
    output_tensor = ai_runner_interpreter.invoke(image_processed)
    return output_tensor