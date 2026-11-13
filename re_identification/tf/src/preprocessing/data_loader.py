# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
import string
import pickle
import scipy.io
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Tuple, List

import random
from collections import defaultdict
from functools import partial

# function retrun all information for reid_batch_generator outside for loop
def reid_batch_generator_info(dataset, batch_size, K):
    """
    PK sampler generator for re-identification.

    Args:
        dataset: tf.data.Dataset yielding (x, label) tensors.
        batch_size: total batch size (P * K).
        K: number of samples per identity in each batch.
    Returns:
        x_data: numpy array of all images in the dataset.
        y_data: numpy array of all labels in the dataset.
        unique_ids: list of unique identities in the dataset.
        id_to_indices: dictionary mapping each identity to a list of its sample indices.
        P: number of unique identities per batch.
        num_samples: total number of samples in the dataset.
    """
    P = batch_size // K
    if batch_size % K != 0:
        raise ValueError("batch_size must be divisible by K")

    # Convert dataset to numpy arrays
    x_data = []
    y_data = []
    for x, y in dataset:
        x_data.append(x.numpy())
        y_data.append(y.numpy())
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # Group indices by identity
    id_to_indices = defaultdict(list)
    for idx, label in enumerate(y_data):
        id_to_indices[label].append(idx)

    unique_ids = list(id_to_indices.keys())

    num_samples = len(y_data)

    return x_data, y_data, unique_ids, id_to_indices, num_samples

def reid_batch_generator_loop(x_data, y_data, unique_ids, id_to_indices, num_samples, batch_size, K):
    """
    PK sampler generator for re-identification.
    Args:
        x_data: numpy array of all images in the dataset.
        y_data: numpy array of all labels in the dataset.
        unique_ids: list of unique identities in the dataset.
        id_to_indices: dictionary mapping each identity to a list of its sample indices.
        P: number of unique identities per batch.
        num_samples: total number of samples in the dataset.
        batch_size: total batch size (P * K).
        K: number of samples per identity in each batch.
    Yields:
        Tuple of (x_batch, y_batch) numpy arrays.
    """
    for _ in range(0, num_samples, batch_size):
        # Select P unique identities
        # Shuffle identities
        np.random.shuffle(unique_ids)
        batch_indices = []
        for pid in unique_ids:
            indices = id_to_indices[pid]
            selected = np.random.choice(indices, size=min(K, len(indices)), replace=False)
            batch_indices.extend(selected)
            if len(batch_indices) >= batch_size:
                break
        batch_indices = batch_indices[:batch_size]
        yield x_data[batch_indices], y_data[batch_indices]

def reid_batch_generator(dataset, batch_size, K, image_size=None):
    """
    Returns a PK sampler generator for re-identification along with dataset info.

    Args:
        dataset: tf.data.Dataset yielding (x, label) tensors.
        batch_size: total batch size (P * K).
        K: number of samples per identity in each batch.

    Returns:
        generator: A generator yielding (x_batch, y_batch) tuples.
        info: A dictionary containing dataset information.
    """
    x_data, y_data, unique_ids, id_to_indices, num_samples = reid_batch_generator_info(dataset, batch_size, K)

    generator = partial(reid_batch_generator_loop, x_data, y_data, unique_ids, id_to_indices, num_samples, batch_size, K)

    output_signature = (tf.TensorSpec(shape=(None, image_size[0], image_size[1], 3), dtype=tf.float32),
                        tf.TensorSpec(shape=(None,), dtype=tf.int32))
    
    return tf.data.Dataset.from_generator(generator, output_signature=output_signature)

def check_dataset_integrity(dataset_root_dir: str, check_image_files: bool = False) -> None:
    """
    This function checks that a dataset has the following directory structure:
        dataset_root_dir:
            id1_1.jpg
            id1_2.jpg
            id2_1.jpg
            id2_2.jpg

    If the `check_images` argument is set to True, an attempt is made to load each 
    image file. If a file fails the test, it is reported together with the list of
    supported image formats.

    Args:
        dataset_root_dir (str): the root directory of the dataset.
        check_image_files (bool): if set to True, an attempt is made to load each image file.

    Returns:
        None

    Errors:
        - The root directory of the dataset provided in argument cannot be found.
        - A class directory contains a subdirectory (should be files only).
        - An image file cannot be loaded.
    """

    message = ["The directory structure should be:",
               "    dataset_root:",
               "       id1_1.jpg",
               "       id1_2.jpg",
               "       id2_1.jpg",
               "       id2_2.jpg"]
    message = ('\n').join(message)

    if not os.path.isdir(dataset_root_dir):
        raise ValueError(f"\nThe dataset root directory {dataset_root_dir} cannot be found.\n{message}")
    
    image_paths = [x for x in os.listdir(dataset_root_dir)
                   if os.path.isfile(os.path.join(dataset_root_dir, x))]

    # Try to load each image file if it was requested
    if check_image_files:
        for im_path in image_paths:
            try:
                data = tf.io.read_file(im_path)
            except:
                raise ValueError(f"\nUnable to read file {im_path}\nThe file may be corrupt.")
            try:
                tf.image.decode_image(data, channels=3)
            except:
                raise ValueError(f"\nUnable to read image file {im_path}\n"
                                 "Supported image file formats are JPEG, PNG, GIF and BMP.")
            


def _get_path_dataset(path: str,
                     class_names: list[str],
                     seed: int,
                     shuffle: bool = True) -> tf.data.Dataset:
    """
    Creates a tf.data.Dataset from a dataset root directory path.
    The dataset has the following directory structure (checked in parse_config.py):
        dataset_root_dir:
            id1_1.jpg
            id1_2.jpg
            id2_1.jpg
            id2_2.jpg

    Args:
        path (str): Path of the dataset folder.
        class_names (list(str)): List of the classes names.
        seed (int): seed when performing shuffle.
        shuffle (bool): Initial shuffling (or not) of input files names.

    Returns:
        dataset(tf.data.Dataset) -> dataset with a tuple (path, label) of each sample. 
    """
    image_names = sorted([x for x in os.listdir(path) if (x.endswith('.jpg') 
                                                          or x.endswith('.bmp')
                                                          or x.endswith('.gif')
                                                          or x.endswith('.jpeg') 
                                                          or x.endswith('.png'))])

    data_list = []
    for img_file in image_names:
        label_str = img_file.split('_')[0]
        if label_str in class_names:
            label_idx = class_names.index(label_str)
        else:
            label_idx = -1  # unknown class
        data_list.append((os.path.join(path, img_file), label_idx))

    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(data_list)
    
    imgs, labels = zip(*data_list)

    dataset = tf.data.Dataset.from_tensor_slices((list(imgs), list(labels)))

    return dataset


def _preprocess_function(data_x : tf.Tensor,
                         data_y : tf.Tensor,
                         image_size: tuple[int],
                         interpolation: str,
                         aspect_ratio: str,
                         color_mode: str,
                         label_mode: str,
                         num_classes: int) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Load images from path and apply necessary transformations.
    """
    height, width = image_size
    channels = 1 if color_mode == "grayscale" else 3

    image = tf.io.read_file(data_x)
    image = tf.image.decode_image(image, channels=channels, expand_animations=False)
    if aspect_ratio == "fit":
        image = tf.image.resize(image, [height, width], method=interpolation, preserve_aspect_ratio=False)
    else:
        image = tf.image.resize_with_crop_or_pad(image, height, width)
    return image, data_y


def _get_train_val_ds(training_path: str,
                     image_size: tuple[int] = None,
                     label_mode: str = None,
                     class_names: list[str] = None,
                     interpolation: str = None,
                     aspect_ratio: str = None,
                     color_mode: str = None,
                     validation_split: float = None,
                     batch_size: int = None,
                     seed: int = None,
                     shuffle: bool = True,
                     to_cache: bool = False) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Loads the images under a given dataset root directory and returns training 
    and validation tf.Data.datasets.
    The dataset has the following directory structure (checked in parse_config.py):
        dataset_root_dir:
            id1_1.jpg
            id1_2.jpg
            id2_1.jpg
            id2_2.jpg

    Args:
        training_path (str): Path to the directory containing the training images.
        image_size (tuple[int]): Size of the input images to resize them to.
        label_mode (str): Mode for generating the labels for the images.
        class_names (list[str]): List of class names to use for the images.
        interpolation (float): Interpolation method to use when resizing the images.
        aspect_ratio (bool): Whether or not to crop the images to the specified aspect ratio.
        color_mode (str): Color mode to use for the images.
        validation_split (float): Fraction of the data to use for validation.
        batch_size (int): Batch size to use for training and validation.
        seed (int): Seed to use for shuffling the data.
        shuffle (bool): Whether or not to reshuffle at each iteration the dataset.
        to_cache (bool): Whether or not to cache the datasets.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset]: Training and validation datasets.
    """
    # When calling this function using the config file data, some of the arguments
    # may be used but equal to None (happens when an attribute is missing in the
    # config file or has no value). For this reason, all the arguments in the
    # definition of the function defaults to None and we set default values here
    # in case the function is called in another context with missing arguments.

    label_mode = label_mode if label_mode else "int"
    interpolation = interpolation if interpolation else "bilinear"
    aspect_ratio = aspect_ratio if aspect_ratio else "fit"
    color_mode = color_mode if color_mode else "rgb"
    validation_split = validation_split if validation_split else 0.2
    batch_size = batch_size if batch_size else 32

    preprocess_params = (image_size, 
                         interpolation,
                         aspect_ratio,
                         color_mode,
                         label_mode,
                         len(class_names))

    dataset = _get_path_dataset(training_path, class_names, seed=seed)

    train_size = int(len(dataset)*(1-validation_split))
    train_ds = dataset.take(train_size)
    val_ds = dataset.skip(train_size)

    if shuffle:
        train_ds = train_ds.shuffle(len(train_ds), reshuffle_each_iteration=True, seed=seed)
    
    train_ds = train_ds.map(lambda *data : _preprocess_function(*data,*preprocess_params))
    val_ds = val_ds.map(lambda *data : _preprocess_function(*data,*preprocess_params))

    train_ds = reid_batch_generator(train_ds, batch_size, K=4, image_size=image_size)
    val_ds = reid_batch_generator(val_ds, batch_size, K=4, image_size=image_size)

    if to_cache:
        train_ds = train_ds.cache()
        val_ds = val_ds.cache()
    
    train_ds = train_ds.prefetch(buffer_size=4)
    val_ds = val_ds.prefetch(buffer_size=4)

    return train_ds, val_ds


def _get_ds(data_path: str = None,
           label_mode: str = None,
           class_names: list[str] = None,
           image_size: tuple[int] = None,
           interpolation: str = None,
           aspect_ratio: str = None,
           color_mode: str = None,
           batch_size: int = None,
           seed: int = None,
           shuffle: bool = True,
           to_cache: bool = False,
           reid_batch: bool = False) -> tf.data.Dataset:
    """
    Loads the images from the given dataset root directory and returns a tf.data.Dataset.
    The dataset has the following directory structure (checked in parse_config.py):
        dataset_root_dir:
            id1_1.jpg
            id1_2.jpg
            id2_1.jpg
            id2_2.jpg

    Args:
        data_path (str): Path to the directory containing the images.
        label_mode (str): Mode for generating the labels for the images.
        class_names (list[str]): List of class names to use for the images.
        image_size (tuple[int]): Size of the input images to resize them to.
        interpolation (str): Interpolation method to use when resizing the images.
        aspect_ratio (bool): Whether or not to crop the images to the specified aspect ratio.
        color_mode (str): Color mode to use for the images.
        batch_size (int): Batch size to use for the dataset.
        seed (int): Seed to use for shuffling the data.
        shuffle (bool): Whether or not to reshuffle the dataset at each iteration.
        to_cache (bool): Whether or not to cache the dataset.

    Returns:
        tf.data.Dataset: Dataset containing the images.
    """
    # When calling this function using the config file data, some of the arguments
    # may be used but equal to None (happens when an attribute is missing in the
    # config file or has no value). For this reason, all the arguments in the
    # definition of the function defaults to None and we set default values here
    # in case the function is called in another context with missing arguments.

    label_mode = label_mode if label_mode else "int"
    interpolation = interpolation if interpolation else "bilinear"
    aspect_ratio = aspect_ratio if aspect_ratio else "fit"
    color_mode = color_mode if color_mode else "rgb"
    batch_size = batch_size if batch_size else 32

    preprocess_params = (image_size, 
                         interpolation,
                         aspect_ratio,
                         color_mode,
                         label_mode,
                         len(class_names))
    
    dataset = _get_path_dataset(data_path, class_names, seed=seed)

    if shuffle:
        dataset = dataset.shuffle(len(dataset), reshuffle_each_iteration=True, seed=seed)
    
    dataset = dataset.map(lambda *data: _preprocess_function(*data, *preprocess_params))

    if reid_batch:
        dataset = reid_batch_generator(dataset, batch_size, K=4, image_size=image_size)
    else:
        dataset = dataset.batch(batch_size)

    if to_cache:
        dataset = dataset.cache()
    
    if reid_batch:
        dataset = dataset.prefetch(buffer_size=4)
    else:
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

# transform dataset from x,y to x,(y,y)
def _double_output(ds: tf.data.Dataset) -> tf.data.Dataset:
    return ds.map(lambda x, y: (x, (y, y)))


def load_dataset(dataset_name: str = None,
                 training_path: str = None,
                 validation_path: str = None,
                 quantization_path: str = None,
                 test_query_path: str = None,
                 test_gallery_path: str = None,
                 validation_split: float = None,
                 class_names: list[str] = None,
                 class_names_test: list[str] = None,
                 image_size: tuple[int] = None,
                 interpolation: str = None,
                 aspect_ratio: str = None,
                 color_mode: str = None,
                 batch_size: int = None,
                 seed: int = None) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Loads the images from the given dataset root directories and returns training,
    validation, and test tf.data.Datasets.
    The datasets have the following directory structure (checked in parse_config.py):
        dataset_root_dir:
            id1_1.jpg
            id1_2.jpg
            id2_1.jpg
            id2_2.jpg

    Args:
        dataset_name (str): Name of the dataset to load.
        training_path (str): Path to the directory containing the training images.
        validation_path (str): Path to the directory containing the validation images.
        quantization_path (str): Path to the directory containing the quantization images.
        test_path (str): Path to the directory containing the test images.
        validation_split (float): Fraction of the data to use for validation.
        class_names (list[str]): List of class names to use for the images.
        class_names_test (list[str]): List of class names to use for the test images.
        image_size (tuple[int]): resizing (height, width) of input images
        interpolation (str): Interpolation method to use when resizing the images.
        aspect_ratio (bool): Whether or not to crop the images to the specified aspect ratio.
        color_mode (str): Color mode to use for the images.
        batch_size (int): Batch size to use for the datasets.
        seed (int): Seed to use for shuffling the data.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: Training, validation, and test datasets.
    """

    if class_names:
        num_classes = len(class_names)
    elif class_names_test:
        num_classes = len(class_names_test)
        class_names = []
    elif quantization_path:
        class_names = []
        num_classes = 0
    else:
        return None, None, None, None, None
    
    if class_names and training_path and not validation_path:
        # There is no validation. We split the
        # training set in two to create one.
        train_ds, val_ds = _get_train_val_ds(
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
    elif class_names and training_path and validation_path:
        train_ds = _get_ds(
            training_path,
            class_names=class_names,
            image_size=image_size,
            interpolation=interpolation,
            aspect_ratio=aspect_ratio,
            color_mode=color_mode,
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
            to_cache=False,
            reid_batch=True)
        val_ds = _get_ds(
            validation_path,
            class_names=class_names,
            image_size=image_size,
            interpolation=interpolation,
            aspect_ratio=aspect_ratio,
            color_mode=color_mode,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            to_cache=False,
            reid_batch=True)
    elif class_names and validation_path:
        val_ds = _get_ds(
            validation_path,
            class_names=class_names,
            image_size=image_size,
            interpolation=interpolation,
            aspect_ratio=aspect_ratio,
            color_mode=color_mode,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            to_cache=False,
            reid_batch=True)
        train_ds = None
    else:
        train_ds = None
        val_ds = None

    # change the first y of (x,(y,y)) to one-hot encoding
    if train_ds is not None:
        train_ds = _double_output(train_ds)
        train_ds = train_ds.map(lambda x, y: (x, (tf.keras.utils.to_categorical(y[0], num_classes), y[1])))
    if val_ds is not None:
        val_ds = _double_output(val_ds)
        val_ds = val_ds.map(lambda x, y: (x, (tf.keras.utils.to_categorical(y[0], num_classes), y[1])))

    if quantization_path:
        quantization_ds = _get_ds(
            quantization_path,
            class_names=class_names,
            image_size=image_size,
            interpolation=interpolation,
            aspect_ratio=aspect_ratio,
            color_mode=color_mode,
            batch_size=1,
            shuffle=False,
            seed=seed)
    elif training_path:# is not None: 
        quantization_ds = _get_ds(
            training_path,
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

    if test_query_path and test_gallery_path:
        test_query_ds = _get_ds(test_query_path,
                                class_names=class_names_test,
                                image_size=image_size,
                                interpolation=interpolation,
                                aspect_ratio=aspect_ratio,
                                color_mode=color_mode,
                                batch_size=batch_size,
                                shuffle=False,
                                seed=seed)
        test_gallery_ds = _get_ds(test_gallery_path,
                                  class_names=class_names_test,
                                  image_size=image_size,
                                  interpolation=interpolation,
                                  aspect_ratio=aspect_ratio,
                                  color_mode=color_mode,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  seed=seed)
    else:
        test_query_ds = None
        test_gallery_ds = None

    return train_ds, val_ds, quantization_ds, test_query_ds, test_gallery_ds
