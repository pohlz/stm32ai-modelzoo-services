# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/


def prepare_kwargs_for_model(cfg):
    """
    Extract model-specific keyword arguments from the config for pose estimation models.
    Args:
        cfg: The configuration object (OmegaConf or dict).
    Returns:
        dict: Model-specific keyword arguments.
    """
    # Example extraction logic, adapt as needed for your config structure
    
    model_kwargs = {
        'depth': getattr(cfg.model, 'depth', None),
        'pretrained': getattr(cfg.model, 'pretrained', False),
        'num_classes': getattr(cfg.dataset, 'num_classes', None),
        'dropout': getattr(cfg.training, 'dropout', None),
        'input_shape': getattr(cfg.model, 'input_shape', None),
    }
    return model_kwargs