# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

from common.registries.dataset_registry import DATASET_WRAPPER_REGISTRY
from .utils import get_dataloaders
from common.utils import log_to_file

@DATASET_WRAPPER_REGISTRY.register(framework='torch', dataset_name='valentini', use_case="speech_enhancement")
def get_valentini(cfg):
    log_to_file(cfg.output_dir, f"Dataset: Valentini")
    return get_dataloaders(cfg)
