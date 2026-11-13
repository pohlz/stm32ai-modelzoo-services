# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
from pathlib import Path
from glob import glob
from omegaconf import DictConfig
import math
import random
import numpy as np
import tensorflow as tf

from face_detection.tf.src.utils import bbox_center_to_corners_coords, bbox_abs_to_normalized_coords


def _get_example_paths(dataset_root: str = None, shuffle: bool = True, seed: int = None) -> list:
    """
    Gets all the paths to .jpg image files and corresponding .tfs labels
    files under a dataset root directory.
    
    Image and label file paths are grouped in pairs as follows:
        [ 
           [dataset_root/basename_1.jpg, dataset_root/basename_1.tfs],
           [dataset_root/basename_2.jpg, dataset_root/basename_2.tfs],
            ...
        ]
    If the .tfs file that corresponds to a given .jpg file is missing,
    the .jpg file is ignored.

    If the function is called with the `shuffle` argument set to True
    and without the `seed` argument, or with the `seed` argument set 
    to None, the file paths are shuffled but results are not reproducible.
    
    if the `shuffle` argument is set to False, paths are sorted
    in alphabetical order.
    
    Arguments:
        dataset_root:
            A string, the path to the directory that contains the image
            and labels files.
        shuffle:
            A boolean, specifies whether paths should be shuffled or not.
            Defaults to True.
        seed:
            An integer, the seed to use to make paths shuffling reproducible.
            Used only when `shuffle` is set to True.
    
    Returns:
        A list of [<image-file-path>, <labels-file-path>] pairs.
    """

    if not os.path.isdir(dataset_root):
        raise ValueError(f"Unable to find dataset directory {dataset_root}")
        
    jpg_file_paths = glob(os.path.join(Path(dataset_root), "*.jpg"))
    if not jpg_file_paths:
        raise ValueError(f"Could not find any .jpg image files in directory {dataset_root}")
        
    tfs_file_paths = glob(os.path.join(Path(dataset_root), "*.tfs"))
    if not tfs_file_paths:
        raise ValueError(f"Could not find any .tfs labels files in directory {dataset_root}")
 
    if shuffle:
        random.seed(seed)
        random.shuffle(jpg_file_paths)
    else:
        jpg_file_paths.sort()
        
    example_paths = []
    for jpg_path in jpg_file_paths:
        tfs_path = os.path.join(dataset_root, Path(jpg_path).stem + ".tfs")
        if os.path.isfile(tfs_path):
            example_paths.append([jpg_path, tfs_path])

    return example_paths


def _get_image_paths(dataset_root: str = None, shuffle: bool = True, seed: int = None) -> list:
    """
    Gets all the paths to .jpg image files under a dataset root directory.

    If the function is called with the `shuffle` argument set to True
    and without the `seed` argument, or with the `seed` argument set 
    to None, the file paths are shuffled but results are not reproducible.
    
    if the `shuffle` argument is set to False, paths are sorted
    in alphabetical order.

    Arguments:
        dataset_root:
            A string, the path to the directory that contains the image files.
        shuffle:
            A boolean, specifies whether file paths should be shuffled or not.
            Defaults to True.
        seed:
            An integer, the seed to use to make paths shuffling reproducible.
            Used only when `shuffle` is set to True.
    
    Returns:
        A list of image file paths.
    """

    if not os.path.isdir(dataset_root):
        raise ValueError(f"Unable to find dataset directory {dataset_root}")
        
    jpg_file_paths = glob(os.path.join(Path(dataset_root), "*.jpg"))
    if not jpg_file_paths:
        raise ValueError(f"Could not find any .jpg image files in directory {dataset_root}")

    if shuffle:
        random.seed(seed)
        random.shuffle(jpg_file_paths)
    else:
        jpg_file_paths.sort()

    return jpg_file_paths


def _split_file_paths(data_paths, split_ratio=None):
    """
    Splits a list in two according to a specified split ratio.

    Arguments:
        paths:
            A list, the list to split. Items can be either image file paths
            or (image, labels) pairs of file paths.
        split_ratio:
            A float greater than 0 and less than 1, specifies the ratio 
            to use to split the input list.

    Returns:
        Two sub-lists of the input list. The length of the first sublist is
        N*(1 - split_ratio) and the length of the second one is N*split_ratio.
    """

    num_examples = len(data_paths)
    size = num_examples - math.floor(split_ratio * num_examples)
    return data_paths[:size], data_paths[size:]
        
                    
def _create_image_and_labels_loader(
            example_paths: list = None,
            image_size: tuple = None,
            batch_size: int = None,
            rescaling: tuple = None,
            interpolation: str = None,
            aspect_ratio: str = None,
            color_mode: str = None,
            normalize: bool = None,
            clip_boxes: bool = True,
            shuffle_buffer_size: bool = False,
            prefetch: bool = False) -> tf.data.Dataset:
    """"
    Creates a tf.data.Dataset data loader for object detection.
    Supplies batches of images with their groundtruth labels.
    
    Labels in the dataset .tfs files must be in (class, x, y, w, h)
    format. The (x, y, w, h) bounding box coordinates must be 
    normalized. The data loader converts them to a pair of diagonally
    opposite corners coordinates (x1, y1, x2, y2), with either normalized
    or absolute values.
    As the coordinates of input bounding boxes are in normalized 
    (x, y, w, h) format, they don't need to be updated as the image
    gets resized. They are invariant.
    
    Arguments:
        example_paths:
            List of (<image-file-path>, <labels-file-path>) pairs,
            each pair being a dataset example.
        image_size:
            A tuple of 2 integers: (width, height).
            Size of the images supplied by the data loader.
        batch_size:
            An integer, the size of data batches supplied
            by the data loader.
        rescaling:
            A tuple of 2 floats: (scale, offset). Specifies
            the factors to use to rescale the input images.
        interpolation:
            A string, the interpolation method to use to resize
            the input images.
        aspect_ratio:
            A string, the aspect ratio method to use to resize
            the input images (fit, crop, pad).
        color_mode:
            A string, the color mode (rgb or grayscale).
        normalize:
            A boolean. If True, the coordinates values of the bounding
            boxes supplied by the generator are normalized. If False,
            they are absolute.
        clip_boxes:
            A boolean. If True, the coordinates of the bounding boxes
            supplied by the generator are clipped to [0, 1] if they are
            normalized and to the image dimensions if they are absolute.
            If False, they are left as is.
            Defaults to True.
        shuffle_buffer_size:
            An integer, specifies the size of the shuffle buffer.
            If not set or set to 0, no shuffle buffer is used.
        prefetch:
            A boolean, specifies whether prefetch should be used.
            Defaults to False.
        
    Returns:
        A tf.data.Dataset data loader.
    """

    def _load_with_fit(data_paths):
    
        image_path = data_paths[0]
        labels_path = data_paths[1]

        # Load the input image
        channels = 1 if color_mode == "grayscale" else 3
        data = tf.io.read_file(image_path)
        image_in = tf.io.decode_jpeg(data, channels=channels)
        if color_mode.lower()=='bgr':
            image_in = image_in[...,::-1]
        # Resize the input image
        width_out = image_size[0]
        height_out = image_size[1]
        image_out = tf.image.resize(image_in, (height_out, width_out), method=interpolation)
        
        # Rescale the output image
        image_out = tf.cast(image_out, tf.float32)
        image_out = rescaling[0] * image_out + rescaling[1]
        
        # Load the input labels
        data = tf.io.read_file(labels_path)
        labels_in = tf.io.parse_tensor(data, out_type=tf.float32)
        
        # Convert the input boxes coordinates from
        # normalized (x, y, w, h) to absolute opposite
        # corners coordinates (x1, y1, x2, y2)
        boxes_out = bbox_center_to_corners_coords(
                            tf.expand_dims(labels_in[..., 1:], axis=0),
                            image_size=(width_out, height_out),
                            normalize=normalize,
                            clip_boxes=clip_boxes)
        boxes_out = tf.squeeze(boxes_out)

        # Concatenate classes and output boxes
        labels_out = tf.concat([labels_in[..., 0:1], boxes_out], axis=-1)
        
        return image_out, labels_out
    
    
    def _load_with_crop_or_pad(data_paths):
    
        image_path = data_paths[0]
        labels_path = data_paths[1]

        # Load the input image
        channels = 1 if color_mode == "grayscale" else 3
        data = tf.io.read_file(image_path)
        image_in = tf.io.decode_jpeg(data, channels=channels)
        
        # Resize the input image with crop or pad
        width_out = image_size[0]
        height_out = image_size[1]
        image_out = tf.image.resize_with_crop_or_pad(image_in, height_out, width_out)
        
        # Rescale the output image
        image_out = tf.cast(image_out, tf.float32)
        image_out = rescaling[0] * image_out + rescaling[1]
        
        # Read the input labels
        data = tf.io.read_file(labels_path)
        labels_in = tf.io.parse_tensor(data, out_type=tf.float32)
        
        # Convert the input boxes coordinates from
        # normalized (x, y, w, h) to absolute opposite
        # corners coordinates (x1, y1, x2, y2)
        width_in = tf.shape(image_in)[1]
        height_in = tf.shape(image_in)[0]
        boxes_in = bbox_center_to_corners_coords(
                            tf.expand_dims(labels_in[..., 1:], axis=0),
                            image_size=(width_in, height_in),
                            normalize=False,
                            clip_boxes=True)
        boxes_in = tf.squeeze(boxes_in)

        # Convert input/output image dimensions to floats
        w_in = tf.cast(width_in, tf.float32)
        h_in = tf.cast(height_in, tf.float32)
        w_out = tf.cast(width_out, tf.float32)
        h_out = tf.cast(height_out, tf.float32)
        
        # Calculate the opposite corners coordinates of the output boxes
        x1 = tf.round(boxes_in[:, 0] - 0.5 * (w_in - w_out))
        y1 = tf.round(boxes_in[:, 1] - 0.5 * (h_in - h_out))
        x2 = tf.round(boxes_in[:, 2] - 0.5 * (w_in - w_out))
        y2 = tf.round(boxes_in[:, 3] - 0.5 * (h_in - h_out))
        
        # Keep track of boxes that are outside of the output
        # image (this may happen when cropping the input image)
        cond_x = tf.math.logical_or(x2 <= 0, x1 >= w_out)
        cond_y = tf.math.logical_or(y2 <= 0, y1 >= h_out)
        is_outside_image = tf.math.logical_or(cond_x, cond_y)
    
        # Clip the calculated coordinates of the output
        # boxes to the size of the output image
        x1 = tf.math.maximum(x1, 0)
        y1 = tf.math.maximum(y1, 0)
        x2 = tf.math.minimum(x2, w_out)
        y2 = tf.math.minimum(y2, h_out)
        
        boxes_out = tf.stack([x1, y1, x2, y2], axis=-1)

        if normalize:
            boxes_out = tf.expand_dims(boxes_out, axis=0)
            boxes_out = bbox_abs_to_normalized_coords(boxes_out, (w_out, h_out), clip_boxes=clip_boxes)
            boxes_out = tf.squeeze(boxes_out)

        # The output padding boxes include the boxes that
        # are outside of the output image and the boxes 
        # that correspond to input padding boxes.
        coords_sum = tf.math.reduce_sum(boxes_in, axis=-1)
        is_padding = tf.math.less_equal(coords_sum, 0)
        is_padding = tf.math.logical_or(is_outside_image, is_padding)

        # Gather the labels that are not padding labels
        classes = labels_in[:, 0:1]
        labels_out = tf.concat([classes, boxes_out], axis=-1)
        indices = tf.where(tf.math.logical_not(is_padding))
        true_labels = tf.gather_nd(labels_out, indices)
        
        # Create the padding labels
        pad_size = tf.math.reduce_sum(tf.cast(is_padding, dtype=tf.int32))
        padding_labels = tf.zeros([pad_size, 5], dtype=tf.float32)

        # Concatenate the true labels and padding labels
        labels_out = tf.concat([true_labels, padding_labels], axis=0)

        return image_out, labels_out

    
    ds = tf.data.Dataset.from_tensor_slices((example_paths))
    if shuffle_buffer_size:
        buffer_size = len(example_paths)
        ds = ds.shuffle(buffer_size, reshuffle_each_iteration=True)
    if aspect_ratio == "fit":
        ds = ds.map(_load_with_fit)
    else:
        ds = ds.map(_load_with_crop_or_pad)
    ds = ds.batch(batch_size)
    if prefetch:
        ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _create_image_loader(
            image_paths: list,
            image_size: tuple = None,
            batch_size: int = None,
            rescaling: tuple = None,
            interpolation: str = None,
            aspect_ratio: str = None,
            color_mode: str = None) -> tf.data.Dataset:
            
    """
    Creates a tf.data.Dataset data loader for images.
    
    Arguments:
        image_paths:
            List of paths to image files.
        image_size:
            A tuple of 2 integers: (width, height).
            Specifies the size of the images supplied by
            the data loader.
        batch_size:
            An integer, the size of data batches supplied
            by the data loader.
        rescaling:
            A tuple of 2 floats: (scale, offset). Specifies
            the factors to use to rescale the input images.
        interpolation:
            A string, the interpolation method to use to resize
            the input images.
        aspect_ratio:
            A string, the aspect ratio method to use to resize
            the input images (fit, crop, pad).
        color_mode:
            A string, the color mode (rgb or grayscale).

    Returns:
        A tf.data.Dataset data loader.
    """
    
    def _load_image(img_path):
    
        # Load the input image
        channels = 1 if color_mode == "grayscale" else 3
        data = tf.io.read_file(img_path)
        image_in = tf.io.decode_jpeg(data, channels=channels)
        if color_mode.lower() == 'bgr':
            image_in = image_in[...,::-1]
        
        # Resize the input image
        width_out = image_size[0]
        height_out = image_size[1]
        if aspect_ratio == "fit":
            image_out = tf.image.resize(image_in, (height_out, width_out), method=interpolation)
        else:
            image_out = tf.image.resize_with_crop_or_pad(image_in, height_out, width_out)
        
        # Rescale the output image
        image_out = tf.cast(image_out, tf.float32)
        # image_out = rescaling[0] * image_out + rescaling[1]
        if isinstance(rescaling[0], (list, tuple)) or isinstance(rescaling[1], (list, tuple)):
            # Assuming rescaling[0] and rescaling[1] are lists/tuples of length equal to number of channels
            variances = tf.constant(rescaling[0], dtype=tf.float32)
            means = tf.constant(rescaling[1], dtype=tf.float32)
            image_out = (image_out / variances ) + means
        else:
            # Default scalar rescaling
            image_out = rescaling[0] * image_out + rescaling[1]
        
        return image_out, tf.convert_to_tensor(img_path)
        
    ds = tf.data.Dataset.from_tensor_slices(image_paths)
    ds = ds.map(_load_image)
    ds = ds.batch(batch_size)
    return ds


def get_training_data_loaders(
                cfg: DictConfig,
                image_size: tuple = None,
                train_batch_size: int = None,
                val_batch_size: int = None,
                normalize: bool = True,
                clip_boxes: bool = True,
                seed: int = None,
                verbose: bool = True) -> tf.data.Dataset:
                
    """
    Creates two data loaders for training a model: one to get 
    batches of training set examples and another one to get 
    batches of validation set examples.

    If no validation set is provided, the training set is split
    in two to create one.
    
    The validation data loader is used during training to calculate
    the mAP metrics at the end of each epoch. The batch size is set
    to 128 by default as experience showed that it is large enough 
    to get reliable enough mAP results.

    Arguments:
        cfg:
            A dictionary, the entire configuration file dictionary.
        image_size:
            A tuple of 2 integers: (width, height).
            Specifies the size of the images supplied by
            the data loaders.
        train_batch_size:
            An integer, the size of training data batches supplied
            by the training data loader.
            Defaults to cfg.training.batch_size.
        val_batch_size:
            An integer, the size of validation data batches supplied
            by the validation data loader.
            Defaults to 128.
        normalize:
            A boolean. If True, the coordinates values of the bounding
            boxes supplied by the generators are normalized. If False,
            they are absolute.
            Defaults to True.
        clip_boxes:
            A boolean. If True, the coordinates of the bounding boxes
            supplied by the generators are clipped to [0, 1] if they are
            normalized and to the image dimensions if they are absolute.
            If False, they are left as is.
            Defaults to True.
        seed:
            An integer, the seed to use to make file paths shuffling and
            training set splitting reproducible.
            Defaults to cfg.dataset.seed
        verbose:
            A boolean. If True, the dataset path and size are displayed.
            If False, no message is displayed.
            Default to True.

    Returns:
        A tuple of two tf.data.Dataset data loaders.
    """

    if not image_size:
        val_image_size = cfg.training.model.input_shape[:2]
        train_image_size = cfg.training.model.input_shape[:2]
    else:
        train_image_size = image_size
        val_image_size = image_size

    # if the random_periodic_resizing data-aug is set, the training image is imported at the maximum possible res to avoid losing information when resizing
    if cfg.data_augmentation and cfg.data_augmentation.random_periodic_resizing is not None:
        random_sizes = tf.cast(cfg.data_augmentation.random_periodic_resizing.image_sizes,tf.int32)
        train_image_size = random_sizes[tf.argmax(tf.reduce_prod(random_sizes,-1))]

    if not train_batch_size:
        train_batch_size = cfg.training.batch_size

    if not val_batch_size:
        val_batch_size = cfg.training.batch_size
                
    cds = cfg.dataset
    if not seed:
        seed = cds.seed
        
    train_example_paths = _get_example_paths(cds.training_path, seed=seed)

    if cds.validation_path:
        val_example_paths = _get_example_paths(cds.validation_path, seed=seed)
    else:
        train_example_paths, val_example_paths = _split_file_paths(
                    train_example_paths, split_ratio=cds.validation_split)

    if verbose:
        print("Training set:")
        print(" path:", cds.training_path)
        print(" size:", len(train_example_paths))
        print("Validation set:")
        if cds.validation_path:
            print(" path:", cds.validation_path)
        else:
            print(" created using {:.1f}% of the training data {}".
                        format(100*cds.validation_split, cds.training_path))
        print(" size:", len(val_example_paths))
    
    cpp = cfg.preprocessing
    train_ds = _create_image_and_labels_loader(
                    train_example_paths,
                    image_size=train_image_size,
                    batch_size=train_batch_size,
                    rescaling=(cpp.rescaling.scale, cpp.rescaling.offset),
                    interpolation=cpp.resizing.interpolation,
                    aspect_ratio=cpp.resizing.aspect_ratio,
                    color_mode=cpp.color_mode,
                    normalize=normalize,
                    clip_boxes=clip_boxes,
                    shuffle_buffer_size=True,
                    prefetch=True)

    val_ds = _create_image_and_labels_loader(
                    val_example_paths,
                    image_size=val_image_size,
                    batch_size=val_batch_size,
                    rescaling=(cpp.rescaling.scale, cpp.rescaling.offset),
                    interpolation=cpp.resizing.interpolation,
                    aspect_ratio=cpp.resizing.aspect_ratio,
                    color_mode=cpp.color_mode,
                    normalize=normalize,
                    clip_boxes=clip_boxes)
 
    return train_ds, val_ds


def get_evaluation_data_loader(
                    cfg: DictConfig,
                    image_size: tuple = None,
                    batch_size: int = 64,
                    normalize: bool = None,
                    clip_boxes: bool = True,
                    seed: int = None,
                    verbose: bool =True) -> tf.data.Dataset:
                
    """
    Creates a data loader for evaluating a model.

    The evaluation dataset is chosen in the following precedence order:
      1. test set
      2. validation set
      3. validation set created by splitting the training set
    
    Arguments:
        cfg:
            A dictionary, the entire configuration file dictionary.
        image_size:
            A tuple of 2 integers: (width, height).
            Specifies the size of the images supplied by
            the data loaders.
        batch_size:
            An integer, the size of data batches supplied
            by the data loader.
            Defaults to 64.
        normalize:
            A boolean. If True, the coordinates values of the bounding
            boxes supplied by the generators are normalized. If False,
            they are absolute.
            Defaults to True.
        clip_boxes:
            A boolean. If True, the coordinates of the bounding boxes
            supplied by the generators are clipped to [0, 1] if they are
            normalized and to the image dimensions if they are absolute.
            If False, they are left as is.
            Defaults to True.
        seed:
            An integer, the seed to use to make file paths shuffling and
            training set splitting reproducible.
            Defaults to cfg.dataset.seed
        verbose:
            A boolean. If True, the dataset path and size are displayed.
            If False, no message is displayed.
            Default to True.

    Returns:
        A tf.data.Dataset data loader
    """

    cds = cfg.dataset
    if not seed:
        seed = cds.seed
        
    if cds.test_path:
        example_paths = _get_example_paths(cds.test_path, seed=seed)        
    elif cds.validation_path:
        example_paths = _get_example_paths(cds.validation_path, seed=seed)
    else:
        train_example_paths = _get_example_paths(cds.training_path, seed=seed)
        _, example_paths = _split_file_paths(train_example_paths, split_ratio=cds.validation_split)

    if verbose:
        print("Evaluation dataset:")
        if cds.test_path:
            print(" path:", cds.test_path)
        elif cds.validation_path:
            print(" path:", cds.validation_path)
        else:
            print(" created using {:.1f}% of training data {}".
                       format(100*cds.validation_split, cds.training_path))
        print(" size:", len(example_paths))

    cpp = cfg.preprocessing
    test_ds = _create_image_and_labels_loader(
                    example_paths,
                    image_size=image_size,
                    batch_size=batch_size,
                    rescaling=(cpp.rescaling.scale, cpp.rescaling.offset),
                    interpolation=cpp.resizing.interpolation,
                    aspect_ratio=cpp.resizing.aspect_ratio,
                    color_mode=cpp.color_mode,
                    normalize=normalize,
                    clip_boxes=clip_boxes)

    return test_ds


def get_quantization_data_loader(
                cfg: DictConfig,
                image_size: tuple = None,
                batch_size: int = 1,
                image_paths_only: bool = False,
                seed: int = None,
                verbose: bool = True) -> tf.data.Dataset:
    """
    Creates a data loader for quantizing a float model.

    The dataset is chosen in the following precedence order:
      1. quantization set
      2. test set
      2. training set

    If a quantization split ratio was set, the chosen dataset
    is split accordingly. Otherwise, it is used entirely.
    If no dataset is available, the function returns None.
    In this case, quantization will be done using fake data.

    Arguments:
        cfg:
            A dictionary, the entire configuration file dictionary.
        image_size:
            A tuple of 2 integers: (width, height).
            Specifies the size of the images supplied by
            the data loaders.
        batch_size:
            An integer, the size of data batches supplied by the data
            loader. Defaults to 1.
        seed:
            An integer, the seed to use to make file paths shuffling and
            dataset splitting reproducible.
            Defaults to cfg.dataset.seed
        verbose:
            A boolean. If True, the dataset path and size are displayed.
            If False, no message is displayed.
            Default to True.

    Returns:
        A tf.data.Dataset data loader
    """

    # Quantize with fake data if quantization_split is set to 0
    cds = cfg.dataset 
    if cds.quantization_split is not None and cds.quantization_split == 0:
        return None
     
    # Look for a dataset
    if cds.quantization_path:
        ds_path = cds.quantization_path
    elif cds.training_path:
        ds_path = cds.training_path
    else:
        # No dataset available, quantize with fake data
        return None 
    
    if not seed:
        seed = cds.seed        
    image_paths = _get_image_paths(ds_path, shuffle=True, seed=seed)
    
    if cds.quantization_split:
        num_images = int(len(image_paths) * cds.quantization_split)
        percent_used = "{:.3f}%".format(100 * cds.quantization_split)
    else:
        # quantization_split is not set, get the entire dataset.
        num_images = len(image_paths)
        percent_used = "100% (use quantization_split to choose a different percentage)"
        
    image_paths = image_paths[:num_images]

    if verbose:
        print("Quantization dataset:")
        print("  path:", ds_path)
        print(f"  percentage used: {percent_used}")
        print(f"  number of images: {num_images}")

    if not image_paths_only:
        cpp = cfg.preprocessing
        quantization_ds = _create_image_loader(
                        image_paths,
                        image_size=image_size,
                        batch_size=batch_size,
                        rescaling=(cpp.rescaling.scale, cpp.rescaling.offset),
                        interpolation=cpp.resizing.interpolation,
                        aspect_ratio=cpp.resizing.aspect_ratio,
                        color_mode=cpp.color_mode)
        return quantization_ds
    else:
        return image_paths


def get_prediction_data_loader(
            cfg: DictConfig,
            image_size: tuple = None,
            batch_size: int = 64,
            seed: int = None,
            verbose: bool = True) -> tf.data.Dataset:
    """
    Creates a data loader for making predictions.

    Arguments:
        cfg:
            A dictionary, the entire configuration file dictionary.
        image_size:
            A tuple of 2 integers: (width, height).
            Specifies the size of the images supplied by
            the data loaders.
        batch_size:
            An integer, the size of data batches supplied by the data
            loader. Defaults to 64.
        seed:
            An integer, the seed to use to make file paths shuffling
            reproducible.
            Defaults to cfg.prediction.seed
        verbose:
            A boolean. If True, the dataset path and size are displayed.
            If False, no message is displayed.
            Default to True.

    Returns:
        A tf.data.Dataset data loader
    """

    if not seed:
        seed = cfg.prediction.seed

    image_paths = _get_image_paths(cfg.dataset.prediction_path, seed=seed)           

    if verbose:
        print("Prediction dataset:")
        print("  path:", cfg.dataset.prediction_path)
        print("  size:", len(image_paths))
        print("  sampling seed:", cfg.prediction.seed)

    cpp = cfg.preprocessing
    predict_ds = _create_image_loader(
                            image_paths,
                            image_size=image_size,
                            batch_size=batch_size,      
                            rescaling=(cpp.rescaling.scale, cpp.rescaling.offset),
                            interpolation=cpp.resizing.interpolation,
                            aspect_ratio=cpp.resizing.aspect_ratio,
                            color_mode=cpp.color_mode)

    return predict_ds

