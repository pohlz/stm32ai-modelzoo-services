# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/


import tensorflow as tf
from typing import Dict, Optional, List


def prepare_kwargs_for_dataloader(cfg):
    """
    Collect dataloader parameters.

    Args:
        cfg: configuration object.

    Returns:
        Dict of all required dataloader kwargs including image_size from model.input_shape.
    """
    input_shape = getattr(cfg.model, 'input_shape', None)
    model_path = cfg.model.model_path

    # If no model_path, use input_shape directly (for data augmentation demos)
    if model_path is None:
        image_size = tuple(input_shape)[:-1]
    else:
        file_extension = str(model_path).split('.')[-1]
        input_shape = cfg.model.input_shape
        if file_extension in ['h5', 'keras']:
            image_size = tuple(input_shape)[:-1]
        elif file_extension == 'tflite':
            image_size = tuple(input_shape)[-3:-1]
        elif file_extension == 'onnx':
            image_size = tuple(input_shape)[-2:]

    # if the random_periodic_resizing data-aug is set, the training image is imported at the maximum possible res to avoid losing information when resizing
    if cfg.data_augmentation is not None:
        if cfg.data_augmentation.config is not None:
            if cfg.data_augmentation.config.random_periodic_resizing is not None:
                random_sizes = tf.cast(cfg.data_augmentation.config.random_periodic_resizing.image_sizes,tf.int32).numpy()
                train_image_size = random_sizes[tf.argmax(tf.reduce_prod(random_sizes,-1))]
            else:
                train_image_size = image_size
        else:
            train_image_size = image_size
    else:
        train_image_size = image_size

    dataloader_kwargs = {
        'training_path': cfg.dataset.training_path,
        'validation_path': cfg.dataset.validation_path,
        'validation_split': cfg.dataset.validation_split,
        'test_path': cfg.dataset.test_path,
        'prediction_path': cfg.dataset.prediction_path,
        'quantization_path': cfg.dataset.quantization_path,
        'quantization_split': cfg.dataset.quantization_split,
        'image_size': tuple(image_size),
        'train_image_size': tuple(train_image_size),
        'color_mode': cfg.preprocessing.color_mode,
        'batch_size': getattr(cfg.training, 'batch_size', 32) if cfg.training else 32,
        'seed': cfg.general.global_seed,
        'aspect_ratio': cfg.preprocessing.resizing.aspect_ratio,
        'interpolation': cfg.preprocessing.resizing.interpolation,
        'scale': cfg.preprocessing.rescaling.scale,
        'offset': cfg.preprocessing.rescaling.offset,
    }
    return dataloader_kwargs