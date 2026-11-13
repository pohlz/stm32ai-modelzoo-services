# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/


from common.registries.dataset_registry import DATASET_WRAPPER_REGISTRY
from neural_style_transfer.tf.src.datasets.coco import load_coco
from neural_style_transfer.tf.src.datasets import prepare_kwargs_for_dataloader

__all__ = ['get_coco']

@DATASET_WRAPPER_REGISTRY.register(framework='tf', dataset_name='coco', use_case="neural_style_transfer")
def get_coco(cfg):

    # Get dataloader kwargs
    args = prepare_kwargs_for_dataloader(cfg)

    # Creates datasets
    dataloaders = load_coco(prediction_path=args['prediction_path'],
                            image_size=args['image_size'],
                            interpolation=args['interpolation'],
                            aspect_ratio=args['aspect_ratio'],
                            color_mode=args['color_mode'],
                            seed=args['seed'])

    return dataloaders
