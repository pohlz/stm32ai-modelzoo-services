# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

from common.registries.trainer_registry import TRAINER_WRAPPER_REGISTRY

from arc_fault_detection.tf.src.training import AFDTrainer

__all__ = ["AFDTrainer"]

TRAINER_WRAPPER_REGISTRY.register(
    trainer_name="afd_trainer",
    framework="tf",
    use_case="arc_fault_detection"
)(AFDTrainer)
