# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
import os

from common.registries.dataset_registry import DATASET_WRAPPER_REGISTRY
from depth_estimation.tf.src.datasets import depth_prediction_dataloader
from depth_estimation.tf.src.preprocessing.preprocess import preprocess_image


__all__ = ['get_nyu_depthv2']

def prepare_args_for_dataloader(cfg):
    """
    Extracts prediction config parameters from cfg for depth estimation and returns as a dict.
    """
    return {
        'prediction_path': cfg.dataset.prediction_path,
        'color_mode': getattr(cfg.dataset, 'color_mode', 'rgb'),
        'height': cfg.model.input_shape[1],
        'width': cfg.model.input_shape[2],
        'aspect_ratio': cfg.preprocessing.resizing.aspect_ratio,
        'interpolation': cfg.preprocessing.resizing.interpolation,
        'scale': cfg.preprocessing.rescaling.scale,
        'offset': cfg.preprocessing.rescaling.offset
    }

@DATASET_WRAPPER_REGISTRY.register(framework='tf', dataset_name='nyu_depthv2', use_case="depth_estimation")
def get_nyu_depthv2(cfg):
    """
    Returns dataloaders for depth estimation prediction, with preprocessing applied.
    """
    args = prepare_args_for_dataloader(cfg)

    def predict_loader():
        prediction_path = args['prediction_path']
        if not os.path.isdir(prediction_path):
            raise ValueError(f"Prediction path '{prediction_path}' does not exist or is not a directory.")
        image_files = [f for f in os.listdir(prediction_path) if os.path.isfile(os.path.join(prediction_path, f))]
        if not image_files:
            raise ValueError(f"No images found in prediction path '{prediction_path}'.")
        print(f"[INFO] : Found {len(image_files)} images for prediction in '{prediction_path}'.")
        count = 0
        for raw_img, img_path in depth_prediction_dataloader(prediction_path, color_mode=args['color_mode']):
            img = preprocess_image(
                img=raw_img,
                height=args['height'],
                width=args['width'],
                aspect_ratio=args['aspect_ratio'],
                interpolation=args['interpolation'],
                scale=args['scale'],
                offset=args['offset'],
                perform_scaling=True
            )
            count += 1
            yield img, img_path
        if count == 0:
            raise ValueError(f"No valid images yielded from dataloader in '{prediction_path}'.")

    dataloaders =  {'train': None, 'valid': None, 'quantization': None, 'test': None, 'predict': predict_loader()}
    return dataloaders
