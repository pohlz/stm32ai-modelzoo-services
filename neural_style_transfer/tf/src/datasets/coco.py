# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
import numpy as np
import tensorflow as tf
from typing import Tuple, List
from .utils import get_prediction_ds

def load_coco(prediction_path: str = None,
              image_size: tuple[int] = None,
              interpolation: str = None,
              aspect_ratio: str = None,
              color_mode: str = None,
              seed: int = None,
              shuffle: bool = False,
              to_cache: bool = False) -> dict:
    """
    Loads COCO dataset images for prediction only as a tf.data.Dataset.

    Args:
        predict_path (str): Path to the directory containing COCO images for prediction.
        image_size (tuple[int], optional): Size to resize images.
        interpolation (str, optional): Interpolation method for resizing.
        aspect_ratio (str, optional): Whether to crop images to maintain aspect ratio.
        color_mode (str, optional): Color mode of images.
        seed (int, optional): Seed for shuffling.
        shuffle (bool, optional): Whether to shuffle dataset.
        to_cache (bool, optional): Whether to cache dataset.

    Returns:
        dict: Dictionary with keys 'train', 'valid', 'quantization', 'test' set to None, and 'predict' set to the prediction dataset.
    """

    if prediction_path is None:
        raise ValueError("predict_path must be provided for COCO prediction dataset.")

    predict_ds = get_prediction_ds(
        data_path=prediction_path,
        image_size=image_size,
        interpolation=interpolation,
        aspect_ratio=aspect_ratio,
        color_mode=color_mode,
        seed=seed,
        shuffle=shuffle,
        to_cache=to_cache
    )

    return {'train': None, 'valid': None, 'quantization': None, 'test': None, 'predict': predict_ds}