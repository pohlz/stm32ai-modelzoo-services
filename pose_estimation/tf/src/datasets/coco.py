# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
import string
import random
import pickle
import numpy as np
import tensorflow as tf
from typing import Tuple, List

def _parse_labels(label_path : str):
    """
    Parsing of the labels files
    Args:
        label_path (str): Path of the label file.

    Returns:
        ground_truths (np.array) : shape (ground_truths, 5+3*keypoints) ground truths present in the label file
    """
    if label_path == "None":
        ground_truths = np.array([17*3*[0.]],np.float32) # TODO: Modify this section to have the correct number of keypoints
    else:
        file = open(label_path)
        txt = file.read().split("\n")
        txt = [x.split(" ") for x in txt]
        if len(txt[-1])==1:txt=txt[:-1]
        ground_truths = np.array([[float(j) for j in i] for i in txt],np.float32)
    return ground_truths

def _normalize_labels(label, n : int, l : int):
    """
    Normalization of the labels -> same shape for every label regarding the number of ground truths
    Args:
        label (np.array): shape (ground_truths, 5+3*keypoints) ground truths present in the label file
        l     (int): shape (1, ) maximum number of ground truths present in a label file
        n     (int): shape (1, ) current number of ground truths present in this label file

    Returns:
        normalized_label (np.array) : shape (l, 5+3*keypoints) label with normalized shape
    """
    m = n - len(label)
    miss = np.zeros((m,l)) #-np.ones((m,l)) # create missing labels with 0 values
    normalized_label = np.concatenate([label,miss])
    return normalized_label

def _get_path_dataset(path : str, seed : int, shuffle : bool = True) -> tf.data.Dataset:
    """
    Creates a tf.data.Dataset from a dataset root directory path.
    The dataset has the following directory structure (checked in parse_config.py):
        dataset_root_dir:
            image_1.jpg
            image_1.txt
            ...
            image_2.jpg
            image_2.txt

    Args:
        path (str): Path of the dataset folder.
        seed (int): seed when performing shuffle.
        shuffle (bool): Shuffle the dataset.

    Returns:
        dataset(tf.data.Dataset) -> dataset with a tuple (path, label) of each sample. 
    """

    paths_imgs = sorted([os.path.join(path,file) for file in os.listdir(path) if (file.endswith((".jpg",".jpeg",".JPG",".JPEG",".png",".bmp",".gif")))])
    paths_labels = sorted([os.path.join(path,file) for file in os.listdir(path) if file.endswith(".txt")])

    if len(paths_imgs) != len(paths_labels):
        paths_labels = ["None"]*len(paths_imgs)

    paths_labels = list(map(_parse_labels,paths_labels))

    len_max = max([len(p) for p in paths_labels])
    len_label = len(paths_labels[0][0])

    paths_labels = list(map(lambda x : _normalize_labels(x,n=len_max,l=len_label), paths_labels))

    data_list = list(zip(paths_imgs,paths_labels))

    if shuffle:
        random.seed(seed)
        random.shuffle(data_list)
    
    imgs, labels = zip(*data_list)

    dataset = tf.data.Dataset.from_tensor_slices((list(imgs), list(labels)))

    return dataset

def _get_padded_labels(data : tf.Tensor, r : float, R : float, height : int, width : int) -> tf.Tensor:

    """
    Adjust bounding boxes and keypoints after padding an image to a new aspect ratio.

    This function assumes that the image has been padded (letterboxed) to change its
    aspect ratio and updates the associated labels accordingly. It handles both
    bounding boxes and keypoints, taking into account whether the padding was applied
    horizontally or vertically.

    The input `data` is expected to be organized per object as:
        [class_id, x_center, y_center, box_width, box_height, kpt_1_x, kpt_1_y, kpt_1_v, ..., kpt_K_x, kpt_K_y, kpt_K_v]
    where:
    - Bounding boxes are in normalized coordinates in the range [0, 1] and follow
      the (x_{center}, y_{center}, w, h) format.
    - Keypoints are in triplets (x, y, v), where v is the visibility/validity flag.
      Only keypoints with v > 0 are kept; others are zeroed out in the transformed
      output.

    The function computes how the padding changes the effective coordinates given
    the original and target aspect ratios, and applies the corresponding transformation
    to:
    - Box corners (then re-converted back to (x_{center}, y_{center}, w, h)).
    - Keypoints (x, y), preserving the visibility flag.

    Args:    
        data (tf.Tensor) :
            A 2D tensor of shape (P, 1 + 4 + 3K) where:
            - P is the number of objects.
            - The first column is the class ID.
            - The next 4 columns are the bounding box coordinates
              (x_{center}, y_{center}, w, h) in normalized coordinates.
            - The remaining columns are keypoints grouped by triplets
              (x, y, v) for each of the K keypoints.
        r (float) :
            Original aspect ratio of the image (e.g. r = height / width).
        R (float) :
            Target aspect ratio after padding.
        height (int) :
            Original image height in pixels.
        width (int) :
            Original image width in pixels.

    Returns
        padded_labels (tf.Tensor) :
            A tensor of shape (P, 1 + 4 + 3K) with the same structure as `data`,
            but with bounding boxes and keypoints transformed to match the padded
            (letterboxed) image coordinates.

            - Column 0: class ID (unchanged).
            - Columns 1–4: updated bounding boxes (x_{center}, y_{center}, w, h).
            - Remaining columns: updated keypoints triplets (x, y, v), with
              (x, y) transformed and v preserved.

    Notes
    -----
    - When r > R, padding is applied along the width dimension (horizontal padding),
      and when r <= R, padding is applied along the height dimension (vertical padding).
    - The computation handles both even and odd numbers of padding pixels to keep the
      objects centered correctly in the padded image.
    """

    sh = tf.shape(data)

    padded_boxes = data[:,1:5] # shape : (P,4)

    x1 = padded_boxes[:,0] - padded_boxes[:,2]/2 # shape : (P)
    y1 = padded_boxes[:,1] - padded_boxes[:,3]/2 # shape : (P)
    x2 = padded_boxes[:,0] + padded_boxes[:,2]/2 # shape : (P)
    y2 = padded_boxes[:,1] + padded_boxes[:,3]/2 # shape : (P)

    xboxes = tf.cast(tf.stack([x1,x2]),tf.float32) # shape : (2,P)
    yboxes = tf.cast(tf.stack([y1,y2]),tf.float32) # shape : (2,P)

    padded_keypoints = data[:,5:]  # shape : (P,nbr_keypoints*3)
    padded_keypoints = tf.cast(tf.transpose(tf.reshape(padded_keypoints,[sh[0],-1,3]),[2,0,1]),tf.float32) # shape : (3,P,nbr_keypoints)

    if r>R :
        ax = tf.cast(width,tf.float32)
        ra = tf.cast(R/r,tf.float32)
        kp = tf.cast(1,tf.float32)

    else:
        ax = tf.cast(height,tf.float32)
        ra = tf.cast(r/R,tf.float32)
        kp = tf.cast(0,tf.float32)

    nb_px_added = tf.cast(1 - ax*ra,tf.float32) # the number of pixels added to the original image to form the new aspect ratio
    odd = tf.cast(nb_px_added%2,tf.float32) # 1 : odd | 0 : even -> to know if the number of pixels added to the original image is odd or even
    vectorxy = (tf.cast(ra,tf.float32) * (xboxes*(1-kp)+yboxes*kp - 0.5) + 0.5 ) - odd*0.5/(ax-1)  # shape : (2,P)
    vector   = (tf.cast(ra,tf.float32) * (padded_keypoints[0]*(1-kp) + padded_keypoints[1]*kp - 0.5) + 0.5 ) - odd*0.5/(ax-1) # shape : (P,nbr_keypoints)

    pk0 = tf.cast(kp,tf.float32)*padded_keypoints[0] + (1-tf.cast(kp,tf.float32))*vector
    pk1 = (1-tf.cast(kp,tf.float32))*padded_keypoints[1] + tf.cast(kp,tf.float32)*vector

    pk0 = pk0*tf.cast(padded_keypoints[2]>0,tf.float32)
    pk1 = pk1*tf.cast(padded_keypoints[2]>0,tf.float32)

    xboxes = (1-tf.cast(kp,tf.float32))*vectorxy + tf.cast(kp,tf.float32)*xboxes # shape : (2,P)
    yboxes = tf.cast(kp,tf.float32)*vectorxy + (1-tf.cast(kp,tf.float32))*yboxes # shape : (2,P)

    x = (xboxes[0] + xboxes[1]) / 2 # shape : (P)
    y = (yboxes[0] + yboxes[1]) / 2 # shape : (P)
    w =  xboxes[1] - xboxes[0]      # shape : (P)
    h =  yboxes[1] - yboxes[0]      # shape : (P)

    padded_boxes = tf.stack([x,y,w,h],1) # shape : (P,4)

    padded_keypoints = tf.stack([pk0,pk1,padded_keypoints[2]]) # shape : (3,P,nbr_keypoints)
    padded_keypoints = tf.reshape(tf.transpose(padded_keypoints,[1,2,0]),[sh[0],-1]) # shape : (P,nbr_keypoints*3)

    data = tf.cast(data,tf.float32)

    padded_labels = tf.concat([data[:,:1],padded_boxes,padded_keypoints],-1) # shape : (P,1+4+nbr_keypoints*3)
    padded_labels = tf.cast(padded_labels,tf.float32)

    return padded_labels

def _preprocess_function(data_x : tf.Tensor,
                         data_y : tf.Tensor,
                         image_size: tuple[int],
                         interpolation: str,
                         aspect_ratio: str,
                         color_mode: str,
                         scale: float,
                         offset: int) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Load images from path and apply necessary transformations.
    """
    height, width = image_size
    channels = 1 if color_mode == "grayscale" else 3

    image = tf.io.read_file(data_x)
    image = tf.image.decode_image(image, channels=channels, expand_animations=False)

    s = tf.shape(image)
    r = s[1]/s[0]
    R = height/width

    if aspect_ratio == "fit":
        image = tf.image.resize(image, [height, width], method=interpolation, preserve_aspect_ratio=False)
        data_y = tf.cast(data_y,tf.float32)
    elif aspect_ratio == "padding":
        image = tf.image.resize_with_pad(image, height, width)
        data_y = _get_padded_labels(data_y,r,R,height,width)
    else:
        raise ValueError("In config file, at section preprocessing.aspect_ratio choose 'fit' or 'padding'")

    image = scale * tf.cast(image, tf.float32) + offset

    return image, data_y

def _pred_preprocess_function(data_x : tf.Tensor,
                              data_y : tf.Tensor,
                              image_size: tuple[int],
                              interpolation: str,
                              aspect_ratio: str,
                              color_mode: str,
                              scale: float,
                              offset: int) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Load images from path and apply necessary transformations.
    """
    height, width = image_size
    channels = 1 if color_mode == "grayscale" else 3

    image = tf.io.read_file(data_x)
    image = tf.image.decode_image(image, channels=channels, expand_animations=False)

    s = tf.shape(image)
    r = s[1]/s[0]
    R = height/width

    if aspect_ratio == "fit":
        image = tf.image.resize(image, [height, width], method=interpolation, preserve_aspect_ratio=False)
    elif aspect_ratio == "padding":
        image = tf.image.resize_with_pad(image, height, width)
    else:
        raise ValueError("In config file, at section preprocessing.aspect_ratio choose 'fit' or 'padding'")

    image = scale * tf.cast(image, tf.float32) + offset

    return image, data_x

def _get_train_val_ds(training_path: str,
                      image_size: tuple[int] = None,
                      train_image_size: tuple[int] = None,
                      interpolation: str = None,
                      aspect_ratio: str = None,
                      color_mode: str = None,
                      validation_split: float = None,
                      batch_size: int = None,
                      seed: int = None,
                      shuffle: bool = True,
                      to_cache: bool = False,
                      scale: float = None,
                      offset: int = None) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Loads the images under a given dataset root directory and returns training 
    and validation tf.Data.datasets.
    The dataset has the following directory structure (checked in parse_config.py):
        dataset_root_dir:
            image_1.jpg
            image_1.txt
            ...
            image_2.jpg
            image_2.txt

    Args:
        training_path (str): Path to the directory containing the training images.
        image_size (tuple[int]): Size of the input images to resize them to.
        interpolation (float): Interpolation method to use when resizing the images.
        aspect_ratio (bool): Whether or not to crop the images to the specified aspect ratio.
        color_mode (str): Color mode to use for the images.
        validation_split (float): Fraction of the data to use for validation.
        batch_size (int): Batch size to use for training and validation.
        seed (int): Seed to use for shuffling the data.
        shuffle (bool): Whether or not to shuffle the data.
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
    color_mode = color_mode if color_mode else "rgb"
    validation_split = validation_split if validation_split else 0.2
    batch_size = batch_size if batch_size else 32

    train_preprocess_params = (train_image_size if train_image_size is not None else image_size,
                               interpolation,
                               aspect_ratio,
                               color_mode,
                               scale,
                               offset)

    val_preprocess_params   = (image_size, 
                               interpolation,
                               aspect_ratio,
                               color_mode,
                               scale,
                               offset)

    dataset = _get_path_dataset(training_path, seed=seed)

    train_size = int(len(dataset)*(1-validation_split))
    train_ds = dataset.take(train_size)
    val_ds = dataset.skip(train_size)

    if shuffle:
        train_ds = train_ds.shuffle(len(train_ds), reshuffle_each_iteration=True, seed=seed)
    
    train_ds = train_ds.map(lambda *data : _preprocess_function(*data,*train_preprocess_params))
    val_ds   =   val_ds.map(lambda *data : _preprocess_function(*data,*val_preprocess_params))
    
    train_ds = train_ds.batch(batch_size, drop_remainder=True)
    val_ds   =   val_ds.batch(batch_size, drop_remainder=False)

    if to_cache:
        train_ds = train_ds.cache()
        val_ds   =   val_ds.cache()
    
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds   =   val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds


def _get_ds(data_path: str = None,
           image_size: tuple[int] = None,
           interpolation: str = None,
           aspect_ratio: str = None,
           color_mode: str = None,
           batch_size: int = None,
           seed: int = None,
           shuffle: bool = False,
           drop_remainder: bool = False,
           to_cache: bool = False,
           scale: float = None,
           offset: int = None) -> tf.data.Dataset:
    """
    Loads the images from the given dataset root directory and returns a tf.data.Dataset.
    The dataset has the following directory structure (checked in parse_config.py):
        dataset_root_dir:
            image_1.jpg
            image_1.txt
            ...
            image_2.jpg
            image_2.txt

    Args:
        data_path (str): Path to the directory containing the images.
        image_size (tuple[int]): Size of the input images to resize them to.
        interpolation (str): Interpolation method to use when resizing the images.
        aspect_ratio (bool): Whether or not to crop the images to the specified aspect ratio.
        color_mode (str): Color mode to use for the images.
        batch_size (int): Batch size to use for the dataset.
        seed (int): Seed to use for shuffling the data.
        shuffle (bool): Whether or not to shuffle the data.
        drop_remainder (bool): Whether or not to drop the last batch if not full.
        to_cache (bool): Whether or not to cache the dataset.

    Returns:
        tf.data.Dataset: Dataset containing the images.
    """
    # When calling this function using the config file data, some of the arguments
    # may be used but equal to None (happens when an attribute is missing in the
    # config file or has no value). For this reason, all the arguments in the
    # definition of the function defaults to None and we set default values here
    # in case the function is called in another context with missing arguments.

    interpolation = interpolation if interpolation else "bilinear"
    aspect_ratio = aspect_ratio if aspect_ratio else "fit"
    color_mode = color_mode if color_mode else "rgb"
    batch_size = batch_size if batch_size else 32

    preprocess_params = (image_size, 
                         interpolation,
                         aspect_ratio,
                         color_mode,
                         scale,
                         offset)

    dataset = _get_path_dataset(data_path, seed=seed)

    if shuffle:
        dataset = dataset.shuffle(len(dataset), reshuffle_each_iteration=True, seed=seed)
    
    dataset = dataset.map(lambda *data : _preprocess_function(*data, *preprocess_params))
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    if to_cache:
        dataset = dataset.cache()
    
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


def _get_pred_ds(prediction_path: str = None,
                image_size: Tuple[int, int] = None,
                interpolation: str = None,
                aspect_ratio: str = None,
                color_mode: str = None,
                seed: int = None,
                scale: float = None,
                offset: float = None) -> tf.data.Dataset:
    """
    Returns a tf.data.Dataset yielding (image_tensor, path_string).
    Keep batch_size=1 to simplify downstream usage.
    """

    interpolation = interpolation if interpolation else "bilinear"
    aspect_ratio = aspect_ratio if aspect_ratio else "fit"
    color_mode = color_mode if color_mode else "rgb"

    preprocess_params = (image_size, 
                         interpolation,
                         aspect_ratio,
                         color_mode,
                         scale,
                         offset)

    dataset = _get_path_dataset(prediction_path, seed=seed)
    
    dataset = dataset.map(lambda *data : _pred_preprocess_function(*data, *preprocess_params))
    dataset = dataset.batch(1)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


def load_coco(training_path: str = None,
              validation_path: str = None,
              validation_split: float = None,
              test_path: str = None,
              prediction_path: str = None,
              quantization_path: str = None,
              quantization_split: float = None,
              image_size: tuple[int] = None,
              train_image_size: tuple[int] = None,
              color_mode: str = None,
              batch_size: int = None,
              seed: int = None,
              aspect_ratio: str = None,
              interpolation: str = None,
              scale: float = None,
              offset: int = None,
              ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Loads the images from the given dataset root directories and returns training,
    validation, and test tf.data.Datasets.
    The datasets have the following directory structure (checked in parse_config.py):
        dataset_root_dir:
            image_1.jpg
            image_1.txt
            ...
            image_2.jpg
            image_2.txt

    Args:
        dataset_name (str): Name of the dataset to load.
        training_path (str): Path to the directory containing the training images.
        validation_path (str): Path to the directory containing the validation images.
        quantization_path (str): Path to the directory containing the quantization images.
        test_path (str): Path to the directory containing the test images.
        validation_split (float): Fraction of the data to use for validation.
        image_size (tuple[int]): resizing (width, height) of input images
        train_image_size (tuple[int]): if random_periodic_resizing then resize train_ds input images
        interpolation (str): Interpolation method to use when resizing the images.
        aspect_ratio (bool): Whether or not to crop the images to the specified aspect ratio.
        color_mode (str): Color mode to use for the images.
        batch_size (int): Batch size to use for the datasets.
        seed (int): Seed to use for shuffling the data.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: Training, validation, and test datasets.
    """

    # Get training and validation sets
    if training_path and not validation_path:
        # There is no validation. We split the
        # training set in two to create one.
        train_ds, val_ds = _get_train_val_ds(
            training_path,
            image_size=image_size,
            train_image_size=train_image_size,
            interpolation=interpolation,
            aspect_ratio=aspect_ratio,
            color_mode=color_mode,
            validation_split=validation_split,
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
            scale=scale,
            offset=offset)
    elif training_path and validation_path:
        train_ds = _get_ds(
            training_path,
            image_size=train_image_size,
            interpolation=interpolation,
            aspect_ratio=aspect_ratio,
            color_mode=color_mode,
            batch_size=batch_size,
            seed=seed,
            shuffle=True,
            drop_remainder=True,
            scale=scale,
            offset=offset)

        val_ds = _get_ds(
            validation_path,
            image_size=image_size,
            interpolation=interpolation,
            aspect_ratio=aspect_ratio,
            color_mode=color_mode,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            scale=scale,
            offset=offset)
    elif validation_path:
        train_ds = None
        val_ds = _get_ds(
            validation_path,
            image_size=image_size,
            interpolation=interpolation,
            aspect_ratio=aspect_ratio,
            color_mode=color_mode,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            scale=scale,
            offset=offset)
    else:
        train_ds = None
        val_ds = None

    # Get quantization set
    if quantization_path:
        quantization_ds = _get_ds(
            quantization_path,
            image_size=image_size,
            interpolation=interpolation,
            aspect_ratio=aspect_ratio,
            color_mode=color_mode,
            batch_size=1,
            shuffle=False,
            seed=seed,
            scale=scale,
            offset=offset)
        if quantization_split:
            dataset_size = int(len(quantization_ds) * quantization_split)
            quantization_ds = quantization_ds.take(dataset_size)
    elif train_ds is not None: 
        quantization_ds = _get_ds(
            training_path,
            image_size=image_size,
            interpolation=interpolation,
            aspect_ratio=aspect_ratio,
            color_mode=color_mode,
            batch_size=1,
            shuffle=False,
            seed=seed,
            scale=scale,
            offset=offset)
        if quantization_split:
            dataset_size = int(len(quantization_ds) * quantization_split)
            quantization_ds = quantization_ds.take(dataset_size)
    else:
        quantization_ds = None

    # Get test set
    if test_path:
        test_ds = _get_ds(
            test_path,
            image_size=image_size,
            interpolation=interpolation,
            aspect_ratio=aspect_ratio,
            color_mode=color_mode,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            scale=scale,
            offset=offset)
    else:
        test_ds = None

    if prediction_path:
        predict_ds = _get_pred_ds(prediction_path=prediction_path,
                                  image_size=image_size,
                                  interpolation=interpolation,
                                  aspect_ratio=aspect_ratio,
                                  color_mode=color_mode,
                                  seed=seed,
                                  scale=scale,
                                  offset=offset)
    else:
        predict_ds = None


    dataloaders={'train': train_ds, 'valid': val_ds, 'test': test_ds, 'quantization': quantization_ds, 'predict': predict_ds}

    return dataloaders
