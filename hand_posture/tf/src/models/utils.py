# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/


import os
from omegaconf import OmegaConf, DictConfig


def prepare_kwargs_for_model(cfg: DictConfig):
    '''
    reads relevant fields from cfg.model and cfg.training and prepares keyword arguments for model instantiation
    args:
        cfg: DictConfig
    returns:
        model_kwargs: dict 
    '''
    dropout = cfg.training.dropout if cfg.training and 'dropout' in cfg.training else None

    model_kwargs = {
        'model_path': getattr(cfg.model, 'model_path', None),
        'alpha': getattr(cfg.model, 'alpha', None),
        'model_type': getattr(cfg.model, 'model_type', None),
        'depth': getattr(cfg.model, 'depth', None),
        'input_shape': getattr(cfg.model, 'input_shape', None),
        'dropout': dropout,
    }

    return model_kwargs
