# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

def prepare_kwargs_for_model(cfg):
    in_chans = 3
    #if getattr(cfg, 'input_size', None) is not None:
    in_chans = tuple(cfg.model.input_shape)[0] # ST: HWC Torch: CHW

    model_kwargs = {
        'in_chans': in_chans,
        'drop_rate': getattr(getattr(cfg, 'data_augmentation', None), 'drop', None),
        'drop_path_rate': getattr(getattr(cfg, 'data_augmentation', None), 'drop_path', None),
        'drop_block_rate': getattr(getattr(cfg, 'data_augmentation', None), 'drop_block', None),

        'global_pool': getattr(getattr(cfg, 'model', None), 'gp', None),

        'bn_momentum': getattr(getattr(cfg, 'training', None), 'bn_momentum', None),
        'bn_eps': getattr(getattr(cfg, 'training', None), 'bn_eps', None),
        'scriptable': False,
        'checkpoint_path': getattr(cfg.model, "model_path", None),
        **(getattr(cfg, 'model_kwargs', {}) or {}),
    }
    return model_kwargs
