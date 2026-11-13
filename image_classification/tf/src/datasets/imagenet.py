# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
import pickle
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Tuple, List
from .utils import get_train_val_ds, get_ds, get_prediction_ds


def load_imagenet_like(training_path: str = None,
                       validation_path: str = None,
                       quantization_path: str = None,
                       test_path: str = None,
                       prediction_path: str = None,
                       validation_split: float = None,
                       quantization_split: float = None,
                       class_names: list[str] = None,
                       image_size: tuple[int] = None,
                       interpolation: str = None,
                       aspect_ratio: str = None,
                       color_mode: str = None,
                       batch_size: int = None,
                       seed: int = None
                    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Loads the images from the given dataset root directories and returns training,
    validation, and test tf.data.Datasets.
    The datasets have the following directory structure (checked in parse_config.py):
        dataset_root_dir:
            class_a:
                a_image_1.jpg
                a_image_2.jpg
            class_b:
                b_image_1.jpg
                b_image_2.jpg

    Args:
        training_path (str): Path to the directory containing the training images.
        validation_path (str): Path to the directory containing the validation images.
        quantization_path (str): Path to the directory containing the quantization images.
        test_path (str): Path to the directory containing the test images.
        validation_split (float): Fraction of the data to use for validation.
        quantization_split (float): Fraction of the data to use for quantization.
        class_names (list[str]): List of class names to use for the images.
        image_size (tuple[int]): resizing (height, width) of input images
        interpolation (str): Interpolation method to use when resizing the images.
        aspect_ratio (bool): Whether or not to crop the images to the specified aspect ratio.
        color_mode (str): Color mode to use for the images.
        batch_size (int): Batch size to use for the datasets.
        seed (int): Seed to use for shuffling the data.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: Training, validation, 
        quantization, test and prediction datasets.
    """

    # Get training and validation sets
    if training_path and not validation_path:
        # There is no validation. We split the
        # training set in two to create one.
        train_ds, val_ds = get_train_val_ds(
            training_path,
            class_names=class_names,
            image_size=image_size,
            interpolation=interpolation,
            aspect_ratio=aspect_ratio,
            color_mode=color_mode,
            validation_split=validation_split,
            batch_size=batch_size,
            shuffle=True,
            seed=seed)
    elif training_path and validation_path:
        train_ds = get_ds(
            training_path,
            class_names=class_names,
            image_size=image_size,
            interpolation=interpolation,
            aspect_ratio=aspect_ratio,
            color_mode=color_mode,
            batch_size=batch_size,
            shuffle=True,
            seed=seed)

        val_ds = get_ds(
            validation_path,
            class_names=class_names,
            image_size=image_size,
            interpolation=interpolation,
            aspect_ratio=aspect_ratio,
            color_mode=color_mode,
            batch_size=batch_size,
            shuffle=False,
            seed=seed)
    elif validation_path:
        val_ds = get_ds(
            validation_path,
            class_names=class_names,
            image_size=image_size,
            interpolation=interpolation,
            aspect_ratio=aspect_ratio,
            color_mode=color_mode,
            batch_size=batch_size,
            shuffle=False,
            seed=seed)
        train_ds = None
    else:
        train_ds = None
        val_ds = None

    # Get quantization set
    if quantization_path:
        quantization_ds = get_ds(
            quantization_path,
            class_names=class_names,
            image_size=image_size,
            interpolation=interpolation,
            aspect_ratio=aspect_ratio,
            color_mode=color_mode,
            batch_size=1,
            shuffle=False,
            seed=seed)
    elif train_ds is not None: 
        quantization_ds = get_ds(
            training_path,
            class_names=class_names,
            image_size=image_size,
            interpolation=interpolation,
            aspect_ratio=aspect_ratio,
            color_mode=color_mode,
            batch_size=1,
            shuffle=False,
            seed=seed)
    else:
        quantization_ds = None
    if quantization_ds:
        quant_split = quantization_split if quantization_split else 1.0
        print(f'[INFO] : Quantizing by using {quant_split * 100} % of the provided dataset...')
        quantization_ds = quantization_ds.take(int(len(quantization_ds) * float(quant_split)))

    # Get test set
    if test_path:
        test_ds = get_ds(
            test_path,
            class_names=class_names,
            image_size=image_size,
            interpolation=interpolation,
            aspect_ratio=aspect_ratio,
            color_mode=color_mode,
            batch_size=batch_size,
            shuffle=False,
            seed=seed)
    else:
        test_ds = None

    # Get prediction set
    if prediction_path:
        predict_ds = get_prediction_ds(
            prediction_path,
            class_names=class_names,
            image_size=image_size,
            interpolation=interpolation,
            aspect_ratio=aspect_ratio,
            color_mode=color_mode,
            batch_size=1,
            shuffle=False,
            seed=seed)
    else:
        predict_ds = None

    return {'train': train_ds, 'valid': val_ds, 'quantization': quantization_ds, 'test': test_ds, 'predict': predict_ds}

