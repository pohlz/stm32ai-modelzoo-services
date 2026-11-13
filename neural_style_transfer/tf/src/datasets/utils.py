# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import tensorflow as tf
import numpy as np
import os
import onnxruntime
from pathlib import Path
from omegaconf import OmegaConf, DictConfig


def _get_path_dataset(path: str,
                      seed: int = None,
                      shuffle: bool = True) -> tf.data.Dataset:
    """
    Creates a tf.data.Dataset from a flat directory of images for COCO prediction mode.
    Ignores class subfolders and labels.

    Args:
        path (str): Path of the dataset folder containing images.
        seed (int, optional): Seed for shuffling.
        shuffle (bool, optional): Whether to shuffle the dataset.

    Returns:
        tf.data.Dataset: Dataset yielding image file paths only.
    """

    # List all image files with supported extensions
    supported_exts = [".jpg", ".jpeg", ".png"]
    image_files = [
        os.path.join(path, f)
        for f in sorted(os.listdir(path))
        if Path(f).suffix.lower() in supported_exts
    ]

    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(image_files)

    dataset = tf.data.Dataset.from_tensor_slices(image_files)

    return dataset

def get_prediction_ds(data_path: str = None,
                      label_mode: str = None,
                      image_size: tuple[int] = None,
                      interpolation: str = None,
                      aspect_ratio: str = None,
                      color_mode: str = None,
                      seed: int = None,
                      shuffle: bool = True,
                      to_cache: bool = False) -> tf.data.Dataset:
    """
    Loads images from a directory and returns a tf.data.Dataset for prediction.
    Supports flat directory (COCO style) or class-subfolder structure.

    Args:
        data_path (str): Path to directory containing images.
        label_mode (str): Mode for generating labels. Use 'none' or None for no labels.
        image_size (tuple[int]): Target image size.
        interpolation (str): Interpolation method for resizing.
        aspect_ratio (str): Whether to crop images to aspect ratio.
        color_mode (str): Color mode (e.g., 'rgb').
        seed (int): Seed for shuffling.
        shuffle (bool): Whether to shuffle dataset.
        to_cache (bool): Cache dataset.

    Returns:
        tf.data.Dataset: Dataset yielding images (and optionally paths).
    """

    label_mode = label_mode if label_mode else "none"
    interpolation = interpolation if interpolation else "bilinear"
    aspect_ratio = aspect_ratio if aspect_ratio else "fit"
    color_mode = color_mode if color_mode else "rgb"

    supported_exts = [".jpg", ".jpeg", ".png"]

    if label_mode == "none":
        # List all image files in the directory (flat structure) filtered by supported extensions
        all_files = tf.io.gfile.glob(os.path.join(data_path, '*'))
        image_files = [f for f in all_files if Path(f).suffix.lower() in supported_exts]
        dataset = tf.data.Dataset.from_tensor_slices(image_files)
    else:
        # Existing behavior for class-subfolder structure
        dataset = _get_path_dataset(data_path, seed=seed)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(list(dataset)), reshuffle_each_iteration=True, seed=seed)

    def preprocess_fn(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3 if color_mode == "rgb" else 1)
        image.set_shape([None, None, 3 if color_mode == "rgb" else 1])

        crop = False if aspect_ratio == "fit" else True
        image = tf.image.resize(image, image_size, method=interpolation)
        if crop:
            # Optional cropping logic here if needed
            pass

        image = tf.cast(image, tf.float32) / 1.0

        if label_mode == "none":
            return image, path
        else:
            # Implement label extraction if needed
            pass

    dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(1)

    if to_cache:
        dataset = dataset.cache()

    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


def prepare_kwargs_for_dataloader(cfg: DictConfig):
    # Extract image size from the model
    input_shape = cfg.model.input_shape
    model_path = cfg.model.model_path
    file_extension = str(model_path).split('.')[-1]
    if file_extension in ['h5', 'keras']:
        image_size = tuple(input_shape)[:-1]
    elif file_extension == 'tflite':
        image_size = tuple(input_shape)[-3:-1]
    elif file_extension == 'onnx':
        image_size = tuple(input_shape)[1:]
    print("input_shape=", input_shape)
    print("image_size=", image_size)

    # Prepare kwargs
    dataloader_kwargs = {
        'prediction_path': getattr(cfg.dataset, 'prediction_path', None),
        'image_size': image_size,
        'interpolation': getattr(cfg.preprocessing.resizing, 'interpolation', None),
        'aspect_ratio': getattr(cfg.preprocessing.resizing, 'aspect_ratio', None),
        'color_mode': getattr(cfg.preprocessing, 'color_mode', None),
        'seed': getattr(cfg.dataset, 'seed', 127),
        'rescaling_scale': getattr(cfg.preprocessing.rescaling, 'scale', 1.0/255.0),
        'rescaling_offset': getattr(cfg.preprocessing.rescaling, 'offset', 0),
        'normalization_mean': getattr(cfg.preprocessing.normalization, 'mean', 0.0),
        'normalization_std': getattr(cfg.preprocessing.normalization, 'std', 1.0),
        'data_dir':  getattr(cfg.dataset, 'data_dir', './datasets/'),
        'data_download': getattr(cfg.dataset, 'data_download', True),
    }

    return dataloader_kwargs