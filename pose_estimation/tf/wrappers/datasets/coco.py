# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
from common.registries.dataset_registry import DATASET_WRAPPER_REGISTRY
from pose_estimation.tf.src.datasets import load_coco, prepare_kwargs_for_dataloader


__all__ = ["get_coco"]

@DATASET_WRAPPER_REGISTRY.register(framework='tf', dataset_name='coco', use_case='pose_estimation')
def get_coco(cfg):
    """
    Thin wrapper: prepare args then call load_coco (single build location).

    Args:
        cfg: configuration object.

    Returns:
        Dict with keys: train, val, test, predict, quantization.
    """
    args = prepare_kwargs_for_dataloader(cfg)
    dataloaders = load_coco(
        training_path=args['training_path'],
        validation_path=args['validation_path'],
        validation_split=args["validation_split"],
        test_path=args['test_path'],
        prediction_path=args['prediction_path'],
        quantization_path=args['quantization_path'],
        quantization_split=args['quantization_split'],
        image_size=args['image_size'],
        train_image_size=args['train_image_size'],
        color_mode=args['color_mode'],
        batch_size=args['batch_size'],
        seed=args['seed'],
        aspect_ratio=args['aspect_ratio'],
        interpolation=args['interpolation'],
        scale=args['scale'],
        offset=args['offset']
    )
    return dataloaders