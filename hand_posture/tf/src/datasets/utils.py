

# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
from omegaconf import DictConfig
import tensorflow as tf

def prepare_kwargs_for_dataloader(cfg: DictConfig):
        
    # Prepare kwargs
    batch_size = getattr(cfg.training, 'batch_size', 32) if cfg.training else 32
    dataloader_kwargs = {
        'dataset_name': getattr(cfg.dataset, 'dataset_name', None),
        'training_path': getattr(cfg.dataset, 'training_path', None),
        'validation_path': getattr(cfg.dataset, 'validation_path', None),
        'test_path': getattr(cfg.dataset, 'test_path', None),
        'validation_split': getattr(cfg.dataset, 'validation_split', 0.25),
        'test_split': getattr(cfg.dataset, 'test_split', 0.25),
        'class_names': getattr(cfg.dataset, 'class_names', None),
        'batch_size': batch_size, 
        'seed': getattr(cfg.dataset, 'seed', 127),
        'Max_distance': getattr(cfg.preprocessing, 'Max_distance', None),
        'Min_distance': getattr(cfg.preprocessing, 'Min_distance', None),
        'Background_distance': getattr(cfg.preprocessing, 'Background_distance', None)
    }

    return dataloader_kwargs