# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
import scipy.io
import numpy as np
import tensorflow as tf
from typing import Tuple, List
from .utils import get_ds, get_prediction_ds


def _load_emnist_by_class(training_path: str,
                         num_classes: int = None,
                         input_size: list[int] = None,
                         interpolation: str = None,
                         aspect_ratio: str = None,
                         batch_size: int = None,
                         seed: int = None,
                         to_cache: bool = False) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Loads the EMNIST dataset by class and returns training and validation datasets.

    Args:
        training_path (str): Path to the EMNIST dataset file.
        num_classes (int): Number of classes to use from the dataset.
        input_size (list[int]): Size of the input images to resize them to.
        interpolation (str): Interpolation method to use when resizing the images.
        aspect_ratio (bool): Whether or not to crop the images to the specified aspect ratio.
        batch_size (int): Batch size to use for training and validation.
        seed (int): seed for random shuffler. Defaults to None.
        to_cache (bool): Whether or not to cache the datasets.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset]: Training and validation datasets.
    """
    # When calling this function using the config file data, some of the arguments
    # may be used but equal to None (happens when an attribute is missing in the
    # config file or has no value). For this reason, all the arguments in the
    # definition of the function defaults to None and we set default values here
    # in case the function is called in another context with missing arguments.

    interpolation = interpolation if interpolation else "bilinear"
    aspect_ratio = aspect_ratio if aspect_ratio else "fit"
    batch_size = batch_size if batch_size else 32

    input_size = list(input_size)

    first_matlab_file = list(filter(lambda x : x[-4:]==".mat",os.listdir(training_path)))[0]

    training_path = os.path.join(training_path,first_matlab_file)

    emnist_byclass = scipy.io.loadmat(training_path)
    x_train = emnist_byclass["dataset"][0][0][0][0][0][0]
    y_train = emnist_byclass["dataset"][0][0][0][0][0][1]
    x_test = emnist_byclass["dataset"][0][0][1][0][0][0]
    y_test = emnist_byclass["dataset"][0][0][1][0][0][1]

    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1, order='F')
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1, order='F')
    y_train = y_train.reshape(y_train.shape[0])
    y_test = y_test.reshape(y_test.shape[0])

    remove_item = []
    for i in range(y_train.shape[0]):
        if y_train[i] > 35:
            remove_item.append(i)

    x_train = np.delete(x_train, remove_item, 0)
    y_train = np.delete(y_train, remove_item, 0)

    remove_item = []
    for i in range(y_test.shape[0]):
        if y_test[i] > 35:
            remove_item.append(i)

    x_test = np.delete(x_test, remove_item, 0)
    y_test = np.delete(y_test, remove_item, 0)

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

    if input_size != [28, 28]:
        crop_to_aspect_ratio = False if aspect_ratio == "fit" else True
        train_ds = train_ds.map(lambda x, y: (tf.keras.layers.Resizing(
            input_size[0], input_size[1],
            interpolation=interpolation,
            crop_to_aspect_ratio=crop_to_aspect_ratio
        )(x), y))
        valid_ds = valid_ds.map(lambda x, y: (tf.keras.layers.Resizing(
            input_size[0], input_size[1],
            interpolation=interpolation,
            crop_to_aspect_ratio=crop_to_aspect_ratio
        )(x), y))

    return train_ds, valid_ds


def load_emnist(training_path: str = None,
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

    if class_names:
        num_classes = len(class_names)
    elif quantization_path:
        class_names = []
        num_classes = 0
    else:
        return None, None, None, None
    
    # Get training and validation sets
    train_ds, val_ds = _load_emnist_by_class(training_path,
                                             num_classes=num_classes,
                                             input_size=image_size,
                                             interpolation=interpolation,
                                             aspect_ratio=aspect_ratio,
                                             batch_size=batch_size,
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
