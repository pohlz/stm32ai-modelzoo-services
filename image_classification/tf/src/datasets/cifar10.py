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
from .utils import load_cifar_batch, get_ds, get_prediction_ds


def _load_cifar_10(training_path: str, num_classes: int = None, input_size: list = None,
                  interpolation: str = None, aspect_ratio: str = None,
                  batch_size: int = None, seed: int = None, to_cache: bool = False) -> Tuple:
    """
    Loads the CIFAR-10 dataset and returns two TensorFlow datasets for training and validation.

    Args:
        training_path (str): The path to the CIFAR-10 training data.
        num_classes (int, optional): The number of classes in the dataset. Must be 10. Defaults to None.
        input_size (list, optional): The size of the input images. Defaults to None.
        interpolation (str, optional): The interpolation method to use when resizing images. Defaults to None.
        aspect_ratio (bool, optional): Whether to crop images to maintain the aspect ratio. Defaults to None.
        batch_size (int, optional): The batch size for the datasets. Defaults to None.
        seed (int): seed for random shuffler. Defaults to None.
        to_cache (bool, optional): Whether to cache the datasets in memory. Defaults to False.

    Returns:
        tuple: A tuple of two TensorFlow datasets for training and validation.
    """
    # When calling this function using the config file data, some of the arguments
    # may be used but equal to None (happens when an attribute is missing in the
    # config file or has no value). For this reason, all the arguments in the
    # definition of the function defaults to None and we set default values here
    # in case the function is called in another context with missing arguments.

    # Set default values for optional arguments
    interpolation = interpolation if interpolation else "bilinear"
    aspect_ratio = aspect_ratio if aspect_ratio else "fit"
    batch_size = batch_size if batch_size else 32

    input_size = list(input_size)

    if num_classes != 10:
        raise ValueError('Number of classes must be 10.' f"Received: number of classes={num_classes}.")

    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype="uint8")
    y_train = np.empty((num_train_samples,), dtype="uint8")

    for i in range(1, 6):
        fpath = os.path.join(training_path, "data_batch_" + str(i))
        (
            x_train[(i - 1) * 10000: i * 10000, :, :, :],
            y_train[(i - 1) * 10000: i * 10000],
        ) = load_cifar_batch(fpath)

    fpath = os.path.join(training_path, "test_batch")
    x_test, y_test = load_cifar_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train),))
    y_test = np.reshape(y_test, (len(y_test),))

    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)

    x_test = x_test.astype(np.uint8)
    y_test = y_test.astype(np.uint8)
    print("Found {} files belonging to {} classes.".format(len(x_train) + len(x_test), num_classes))
    print("Using {} files for training.".format(len(x_train)))
    print("Using {} files for validation.".format(len(x_test)))

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(len(x_train), reshuffle_each_iteration=True, seed=seed).batch(batch_size)

    valid_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    valid_ds = valid_ds.batch(batch_size)

    if to_cache:
        train_ds = train_ds.cache()
        valid_ds = valid_ds.cache()

    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    valid_ds = valid_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    if input_size != [32, 32]:
        crop_to_aspect_ratio = False if aspect_ratio == "fit" else True
        train_ds = train_ds.map(
            lambda x, y: (tf.keras.layers.Resizing(
                input_size[0], input_size[1],
                interpolation=interpolation,
                crop_to_aspect_ratio=crop_to_aspect_ratio
            )(x), y))

        valid_ds = valid_ds.map(
            lambda x, y: (tf.keras.layers.Resizing(
                input_size[0], input_size[1],
                interpolation=interpolation,
                crop_to_aspect_ratio=crop_to_aspect_ratio
            )(x), y))

    return train_ds, valid_ds


def load_cifar10(training_path: str = None,
                 quantization_path: str = None,
                 test_path: str = None,
                 prediction_path: str = None,
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
        quantization_path (str): Path to the directory containing the quantization images.
        test_path (str): Path to the directory containing the test images.
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

    if class_names:
        num_classes = len(class_names)
    elif quantization_path:
        class_names = []
        num_classes = 0
    else:
        return None, None, None, None
    
    # Get training and validation sets
    train_ds, val_ds = _load_cifar_10(training_path,
                                      num_classes=num_classes,
                                      input_size=image_size,
                                      interpolation=interpolation,
                                      aspect_ratio=aspect_ratio,
                                      batch_size=batch_size,
                                      seed=seed,
                                      to_cache=False)

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
        quantization_ds = train_ds
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
