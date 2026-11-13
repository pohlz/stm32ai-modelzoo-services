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
    ''' Prepare keyword arguments for model instantiation based on configuration. 
    Args:
        cfg (DictConfig): Configuration dictionary.
    Returns:
        dict: Keyword arguments for model instantiation.
    '''
    dropout = cfg.training.dropout if cfg.training and 'dropout' in cfg.training else None

    model_kwargs = {
        'input_shape': getattr(cfg.model, 'input_shape', None),
        'dropout': dropout
    }

    return model_kwargs