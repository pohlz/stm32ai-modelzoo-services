# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
import os
import tensorflow as tf
from common.registries.dataset_registry import DATASET_WRAPPER_REGISTRY
from instance_segmentation.tf.src.datasets import load_coco


__all__ = ['get_coco']

def prepare_kwargs_for_dataloader(cfg):

    input_shape = None
    input_shape = cfg.model.input_shape
    image_size = (input_shape[1], input_shape[2])

    dataloader_kwargs = {
        'prediction_path': getattr(cfg.dataset, 'prediction_path', None),
        'class_names': getattr(cfg.dataset, 'class_names', None),
        'image_size': image_size,
        'interpolation': getattr(cfg.preprocessing.resizing, 'interpolation', None), 
        'aspect_ratio': getattr(cfg.preprocessing.resizing, 'aspect_ratio', None), 
        'color_mode': getattr(cfg.preprocessing, 'color_mode', None), 
        'seed': getattr(cfg.dataset, 'seed', 127),
        'rescaling_scale': getattr(cfg.preprocessing.rescaling, 'scale', 1.0/255.0), 
        'rescaling_offset': getattr(cfg.preprocessing.rescaling, 'offset', 0), 
        'normalization_mean': getattr(cfg.preprocessing.normalization, 'mean', 0.0), 
        'normalization_std': getattr(cfg.preprocessing.normalization, 'std', 1.0), 
    }

    return dataloader_kwargs

@DATASET_WRAPPER_REGISTRY.register(framework='tf', dataset_name='coco_is', use_case="instance_segmentation")
def get_coco(cfg):
    """
    Returns dataloaders for instance segmentation prediction, with preprocessing applied.
    """
    args = prepare_kwargs_for_dataloader(cfg)
                                     
    dataloaders = load_coco(prediction_path=args['prediction_path'],
                                  scale=args['rescaling_scale'],
                                  offset=args['rescaling_offset'],
                                  image_size=args['image_size'],
                                  aspect_ratio=args['aspect_ratio'],
                                  interpolation=args['interpolation'])
    return dataloaders
