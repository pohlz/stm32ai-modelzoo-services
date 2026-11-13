#  /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
import tensorflow as tf
from omegaconf import DictConfig

def prepare_kwargs_for_model(cfg: DictConfig):
    model_kwargs = {
        'num_classes': getattr(cfg.dataset, 'num_classes', 80),
        'model_type': getattr(cfg.model, 'model_type', None),
        'input_shape': getattr(cfg.model, 'input_shape', None),
        'num_anchors': getattr(cfg.postprocessing, 'num_anchors', None),
    }
    return model_kwargs


def model_family(model_type: str) -> str:
    
    if model_type in ("facedetect_front"):
        return "facedetect_front"
    elif model_type in ("yunet"):
        return "yunet"
    else:
        raise ValueError(f"Internal error: unknown model type {model_type}")