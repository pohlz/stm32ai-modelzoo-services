# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
from common.registries.trainer_registry import TRAINER_WRAPPER_REGISTRY

#from image_classification.tf.src.training.ic_trainer import train
#__all__ = ['get_ic_trainer']

from image_classification.tf.src.training.ic_trainer import ICTrainer

__all__ = ['ICTrainer']

# Register the trainer class from another folder
TRAINER_WRAPPER_REGISTRY.register(
    trainer_name="ic_trainer",
    framework="tf",
    use_case="image_classification"
)(ICTrainer)

#def get_ic_trainer(cfg, model, dataloaders):
#    trained_model = train(cfg=cfg, 
#                          model=model, 
#                          train_ds=dataloaders['train'], 
#                          valid_ds=dataloaders['valid'], 
#                          test_ds=dataloaders['test'])
#    return trained_model

