# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import numpy as np
import tensorflow as tf
from common.data_augmentation import check_dataaug_argument, remap_pixel_values_range, apply_change_rate


def objdet_random_blur(images, filter_size=None, padding="reflect",
                constant_values=0, pixels_range=(0.0, 1.0),
                change_rate=0.5):

    """
    This function randomly blurs input images using a mean filter 2D.
    The filter is square with sizes that are sampled from a specified
    range. The larger the filter size, the more pronounced the blur
    effect is.

    The same filter is used for all the images of a batch. By default,
    change_rate is set to 0.5, meaning that half of the input images
    will be blurred on average. The other half of the images will 
    remain unchanged.
    
    Arguments:
        images:
            Input RGB or grayscale images, a tensor with shape
            [batch_size, width, height, channels]. 
        filter_size:
            A tuple of 2 integers greater than or equal to 1, specifies
            the range of values the filter sizes are sampled from (one
            per image). The width and height of the filter are both equal to 
            `filter_size`. The larger the filter size, the more pronounced
            the blur effect is. If the filter size is equal to 1, the image
            is unchanged.    
        padding:
            A string one of "reflect", "constant", or "symmetric". The type
            of padding algorithm to use.
        constant_values:
            A float or integer, the pad value to use in "constant" padding mode.
        change_rate:
            A float in the interval [0, 1], the number of changed images
            versus the total number of input images average ratio.
            For example, if `change_rate` is set to 0.25, 25% of the input
            images will get changed on average (75% won't get changed).
            If it is set to 0.0, no images are changed. If it is set
            to 1.0, all the images are changed.
            
    Returns:
        The blurred images. The data type and range of pixel values 
        are the same as in the input images.

    Usage example:
        filter_size = (1, 4)
    """

    check_dataaug_argument(filter_size, "filter_size", function_name="random_blur", data_type=int, tuples=1)
   
    if isinstance(filter_size, (tuple, list)):
        if filter_size[0] < 1 or filter_size[1] < 1:
            raise ValueError("Argument `filter_size` of function `random_blur`: expecting a tuple "
                             "of 2 integers greater than or equal to 1. Received {}".format(filter_size))
        if padding not in {"reflect", "constant", "symmetric"}:
            raise ValueError('Argument `padding` of function `random_blur`: supported '
                             'values are "reflect", "constant" and "symmetric". '
                             'Received {}'.format(padding))
    else:
        filter_size = (filter_size, filter_size)
    
    pixels_dtype = images.dtype
    images = remap_pixel_values_range(images, pixels_range, (0.0, 1.0), dtype=tf.float32)
    
    # Sample a filter size
    random_filter_size = tf.random.uniform([], minval=filter_size[0], maxval=filter_size[1] + 1, dtype=tf.int32)

    # We use a square filter.
    fr_width = random_filter_size
    fr_height = random_filter_size

    # Pad the images
    pad_top = (fr_height - 1) // 2
    pad_bottom = fr_height - 1 - pad_top
    pad_left = (fr_width - 1) // 2
    pad_right = fr_width - 1 - pad_left
    paddings = [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
    
    padded_images = tf.pad(images, paddings, mode=padding.upper(), constant_values=constant_values)

    # Create the kernel
    channels = tf.shape(padded_images)[-1]    
    fr_shape = tf.stack([fr_width, fr_height, channels, 1])
    kernel = tf.ones(fr_shape, dtype=images.dtype)

    # Apply the filter to the input images
    output = tf.nn.depthwise_conv2d(padded_images, kernel, strides=(1, 1, 1, 1), padding="VALID")
    area = tf.cast(fr_width * fr_height, dtype=tf.float32)
    images_aug = output / area

    # Apply change rate and remap pixel values to input images range
    outputs = apply_change_rate(images, images_aug, change_rate)
    return remap_pixel_values_range(outputs, (0.0, 1.0), pixels_range, dtype=pixels_dtype)


def objdet_random_gaussian_noise(images, stddev=None, pixels_range=(0.0, 1.0), change_rate=0.5):
                          
    """
    This function adds gaussian noise to input images. The standard 
    deviations of the gaussian distribution are sampled from a specified
    range. The mean of the distribution is equal to 0.

    The same standard deviation is used for all the images of a batch.
    By default, change_rate is set to 0.5, meaning that noise will be
    added to half of the input images on average. The other half of 
    the images will remain unchanged.
   
    Arguments:
        images:
            Input RGB or grayscale images, a tensor with shape
            [batch_size, width, height, channels]. 
        stddev:
            A tuple of 2 floats greater than or equal to 0.0, specifies
            the range of values the standard deviations of the gaussian
            distribution are sampled from (one per image). The larger 
            the standard deviation, the larger the amount of noise added
            to the input image is. If the standard deviation is equal 
            to 0.0, the image is unchanged.
        pixels_range:
            A tuple of 2 integers or floats, specifies the range 
            of pixel values in the input images and output images.
            Any range is supported. It generally is either
            [0, 255], [0, 1] or [-1, 1].
        change_rate:
            A float in the interval [0, 1], the number of changed images
            versus the total number of input images average ratio.
            For example, if `change_rate` is set to 0.25, 25% of the input
            images will get changed on average (75% won't get changed).
            If it is set to 0.0, no images are changed. If it is set
            to 1.0, all the images are changed.
            
    Returns:
        The images with gaussian noise added. The data type and range
        of pixel values are the same as in the input images.

    Usage example:
        stddev = (0.02, 0.1)
    """
    
    check_dataaug_argument(stddev, "stddev", function_name="random_gaussian_noise", data_type=float, tuples=2)
    if stddev[0] < 0 or stddev[1] < 0:
        raise ValueError("\nArgument `stddev` of function `random_gaussian_noise`: expecting float "
                         "values greater than or equal to 0.0. Received {}".format(stddev))

    images_shape = tf.shape(images)
    batch_size = images_shape[0]
    width = images_shape[1]
    height = images_shape[2]
    channels = images_shape[3]

    pixels_dtype = images.dtype
    images = remap_pixel_values_range(images, pixels_range, (0.0, 1.0), dtype=tf.float32)

    # Sample an std value, generate gaussian noise and add it to the images
    random_stddev = tf.random.uniform([1], minval=stddev[0], maxval=stddev[1], dtype=tf.float32)
    noise = tf.random.normal([batch_size, width, height, channels], mean=0.0, stddev=random_stddev)
    images_aug = images + noise

    # Clip the images with noise added
    images_aug = tf.clip_by_value(images_aug, 0.0, 1.0)

    # Apply change rate and remap pixel values to input images range
    outputs = apply_change_rate(images, images_aug, change_rate)
    return remap_pixel_values_range(outputs, (0.0, 1.0), pixels_range, dtype=pixels_dtype)
    

def random_periodic_resizing(
                images,
                gt_labels,
                interpolation=None,
                new_image_size=None):
    """
    This function periodically resizes the input images. The size of
    the images is held constant for a specified number of batches,
    referred to as the "resizing period". Every time a period ends,
    a new size is sampled from a specified set of sizes. Then, the
    size is held constant for the next period, etc.
    
    This function is intended to be used with the 'data_augmentation.py'
    package as it needs the current batch number and the size of the
    images of the previous batch.
    
    Arguments:
        images:
            Input RGB or grayscale images, a tensor with shape
            [batch_size, width, height, channels]. 
        period:
            An integer, the resizing period.
        image_sizes:
            A tuple or list of integers, the set of sizes the image
            sizes are sampled from.
        interpolation:
            A string, the interpolation method used to resize the images.
            Supported values are "bilinear", "nearest", "area", "gaussian",
            "lanczos3", "lanczos5", "bicubic" and "mitchellcubic"
            (resizing is done using the Tensorflow tf.image.resize() function).
        batch:
            An integer, the current batch number starting from the beginning
            of the training.
        last_image_size:
            An tuple of 2 integers, the size of the images of the previous
            batch of images.

    Returns:
        The periodally resized images.
    """


    # Resize the images
    resized_images = tf.image.resize(images, new_image_size, method=interpolation)

    return resized_images, gt_labels
