# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
from common.registries.trainer_registry import TRAINER_WRAPPER_REGISTRY

from human_activity_recognition.tf.src.training.har_trainer import HARTrainer

__all__ = ['HARTrainer']

# Register the trainer class from another folder
TRAINER_WRAPPER_REGISTRY.register(
    trainer_name="har_trainer",
    framework="tf",
    use_case="human_activity_recognition"
)(HARTrainer)
