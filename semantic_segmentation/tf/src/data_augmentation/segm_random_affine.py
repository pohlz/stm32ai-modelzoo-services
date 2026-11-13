# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

"""
References:
----------
Some of the code in this package is from or was inspired by:

    Keras Image Preprocessing Layers
    The Tensorflow Authors
    Copyright (c) 2019

Link to the source code:
    https://github.com/keras-team/keras/blob/v2.12.0/keras/layers/preprocessing/image_preprocessing.py#L394-L495

"""

import math
import tensorflow as tf
from common.data_augmentation import \
            check_dataaug_argument, check_fill_and_interpolation, transform_images, \
            get_flip_matrix, get_translation_matrix, get_rotation_matrix, \
            get_shear_matrix, get_zoom_matrix
from semantic_segmentation.tf.src.data_augmentation import segm_apply_change_rate



def _transform_images_and_labels(
            images,
            labels,
            transforms,
            fill_mode='reflect',
            fill_value=0.0,
            interpolation='bilinear'):
              
    trd_images = transform_images(
            images,
            transforms,
            fill_mode=fill_mode,
            fill_value=fill_value,
            interpolation=interpolation)

    trd_labels = transform_images(
            labels,
            transforms,
            fill_mode=fill_mode,
            fill_value=fill_value,
            interpolation=interpolation)
                    
    return trd_images, trd_labels


#------------------------- Random flip -------------------------
    
def segm_random_flip(images, labels, mode=None, change_rate=0.5):
    """
    This function randomly flips input images and the bounding boxes
    in the associated groundtruth labels.

    Setting `change_rate` to 0.5 usually gives good results (don't set
    it to 1.0, otherwise all the images will be flipped).
    
    Arguments:
        images:
            Input RGB or grayscale images
            Shape: [batch_size, width, height, channels]
        labels:
            Groundtruth labels associated to the images in 
            (class, x1, y1, x2, y2) format. Bounding box coordinates
            must be absolute, opposite corners coordinates.
            Shape: [batch_size, num_labels, 5] 
        mode:
            A string representing the flip axis. Either "horizontal",
            "vertical" or "horizontal_and_vertical".
        change_rate:
            A float in the interval [0, 1] representing the number of 
            changed images versus the total number of input images average
            ratio. For example, if `change_rate` is set to 0.25, 25% of
            the input images will get changed on average (75% won't get
            changed). If it is set to 0.0, no images are changed. If it is
            set to 1.0, all the images are changed.

    Returns:
        The flipped images and groundtruth labels with flipped bounding boxes.
    """

    if mode not in ("horizontal", "vertical", "horizontal_and_vertical"):
        raise ValueError(
            "Argument `mode` of function `random_flip`: supported values are 'horizontal', "
            "'vertical' and 'horizontal_and_vertical'. Received {}".format(mode))

    images_shape = tf.shape(images)
    batch_size = images_shape[0]
    image_width = images_shape[1]
    image_height = images_shape[2]
    
    matrix = get_flip_matrix(batch_size, image_width, image_height, mode)
    flipped_images, flipped_labels = _transform_images_and_labels(images, labels, matrix)

    # Apply the change rate to images and labels
    images_aug, labels_aug = segm_apply_change_rate(
            images, labels, flipped_images, flipped_labels, change_rate=change_rate)

    return images_aug, labels_aug


#------------------------- Random translation -------------------------

def segm_random_translation(
            images, labels,
            width_factor, height_factor,
            fill_mode='reflect', interpolation='bilinear', fill_value=0.0,
            change_rate=1.0):
    """
    This function randomly translates input images and the bounding boxes
    in the associated groundtruth labels.

    Arguments:
        images:
            Input RGB or grayscale images with shape
            Shape: [batch_size, width, height, channels]
        labels:
            Groundtruth labels associated to the images in 
            (class, x1, y1, x2, y2) format. Bounding box coordinates
            must be absolute, opposite corners coordinates.
            Shape: [batch_size, num_labels, 5]
        width_factor:
            A float or a tuple of 2 floats, specifies the range of values
            the horizontal shift factors are sampled from (one per image).
            If a scalar value v is used, it is equivalent to the tuple (-v, v).
            A negative factor means shifting the image left, while a positive 
            factor means shifting the image right.
            For example, `width_factor`=(-0.2, 0.3) results in an output shifted
            left by up to 20% or shifted right by up to 30%.
        height_factor:
            A float or a tuple of 2 floats, specifies the range of values
            the vertical shift factors are sampled from (one per image).
            If a scalar value v is used, it is equivalent to the tuple (-v, v).
            A negative factor means shifting the image up, while a positive
            factor means shifting the image down.
            For example, `height_factor`=(-0.2, 0.3) results in an output shifted
            up by up to 20% or shifted down by up to 30%.
        fill_mode:
            Points outside the boundaries of the input are filled according
            to the given mode. One of {'constant', 'reflect', 'wrap', 'nearest'}.
            See Tensorflow documentation at https://tensorflow.org
            for more details.
        interpolation:
            A string, the interpolation method. Supported values: 'nearest', 'bilinear'.
        change_rate:
            A float in the interval [0, 1] representing the number of 
            changed images versus the total number of input images average
            ratio. For example, if `change_rate` is set to 0.25, 25% of
            the input images will get changed on average (75% won't get
            changed). If it is set to 0.0, no images are changed. If it is
            set to 1.0, all the images are changed.

    Returns:
        The translated images and groundtruth labels with translated bounding boxes.
    """

    check_dataaug_argument(width_factor, "width_factor", function_name="random_translation", data_type=float)
    if isinstance(width_factor, (tuple, list)):
        width_lower = width_factor[0]
        width_upper = width_factor[1]
    else:
        width_lower = -width_factor
        width_upper = width_factor
        
    check_dataaug_argument(height_factor, "height_factor", function_name="random_translation", data_type=float)
    if isinstance(height_factor, (tuple, list)):
        height_lower = height_factor[0]
        height_upper = height_factor[1]
    else:
        height_lower = -height_factor
        height_upper = height_factor

    check_fill_and_interpolation(fill_mode, interpolation, fill_value, function_name="random_translation")

    images_shape = tf.shape(images)
    batch_size = images_shape[0]
    image_width = images_shape[1]
    image_height = images_shape[2]
        
    width_translate = tf.random.uniform(
            [batch_size, 1], minval=width_lower, maxval=width_upper, dtype=tf.float32)
    width_translate = width_translate * tf.cast(image_width, tf.float32)
    
    height_translate = tf.random.uniform(
            [batch_size, 1], minval=height_lower, maxval=height_upper, dtype=tf.float32)
    height_translate = height_translate * tf.cast(image_height, tf.float32)

    translations = tf.cast(
            tf.concat([width_translate, height_translate], axis=1),
            dtype=tf.float32)

    matrix = get_translation_matrix(translations)
    
    translated_images, translated_labels = _transform_images_and_labels(
                images,
                labels,
                matrix,
                interpolation=interpolation,
                fill_mode=fill_mode,
                fill_value=fill_value)

    # Apply the change rate to images and labels
    images_aug, labels_aug = segm_apply_change_rate(
            images, labels, translated_images, translated_labels, change_rate=change_rate)

    return images_aug, labels_aug


#------------------------- Random rotation -------------------------
    
def segm_random_rotation(
                images, labels, factor=None,
                fill_mode='reflect', interpolation='bilinear', fill_value=0.0,
                change_rate=1.0):
    """
    This function randomly rotates input images and the bounding boxes
    in the associated groundtruth labels.

    Arguments:
        images:
            Input RGB or grayscale images with shape
            Shape: [batch_size, width, height, channels]
        labels:
            Groundtruth labels associated to the images in 
            (class, x1, y1, x2, y2) format. Bounding box coordinates
            must be absolute, opposite corners coordinates.
            Shape: [batch_size, num_labels, 5]
        factor:
            A float or a tuple of 2 floats, specifies the range of values the
            rotation angles are sampled from (one per image). If a scalar 
            value v is used, it is equivalent to the tuple (-v, v).
            Rotation angles are in gradients (fractions of 2*pi). A positive 
            angle means rotating counter clock-wise, while a negative angle 
            means rotating clock-wise.
            For example, `factor`=(-0.2, 0.3) results in an output rotated by
            a random amount in the range [-20% * 2pi, 30% * 2pi].
        fill_mode:
            Points outside the boundaries of the input are filled according
            to the given mode. One of {'constant', 'reflect', 'wrap', 'nearest'}.
            See Tensorflow documentation at https://tensorflow.org
            for more details.
        interpolation:
            A string, the interpolation method. Supported values: 'nearest', 'bilinear'.
        change_rate:
            A float in the interval [0, 1] representing the number of 
            changed images versus the total number of input images average
            ratio. For example, if `change_rate` is set to 0.25, 25% of
            the input images will get changed on average (75% won't get
            changed). If it is set to 0.0, no images are changed. If it is
            set to 1.0, all the images are changed.

    Returns:
        The rotated images and groundtruth labels with rotated bounding boxes.
    """

    check_dataaug_argument(factor, "factor", function_name="random_rotation", data_type=float)
    if not isinstance(factor, (tuple, list)):
        factor = (-factor, factor)
        
    check_fill_and_interpolation(fill_mode, interpolation, fill_value, function_name="random_rotation")

    images_shape = tf.shape(images)
    batch_size = images_shape[0]
    image_width = images_shape[1]
    image_height = images_shape[2]

    min_angle = factor[0] * 2. * math.pi
    max_angle = factor[1] * 2. * math.pi
    angles = tf.random.uniform([batch_size], minval=min_angle, maxval=max_angle)

    matrix = get_rotation_matrix(angles, image_width, image_height)
    
    rotated_images, rotated_labels = _transform_images_and_labels(
                images,
                labels,
                matrix,
                interpolation=interpolation,
                fill_mode=fill_mode,
                fill_value=fill_value)

    # Apply the change rate to images and labels
    images_aug, labels_aug = segm_apply_change_rate(
            images, labels, rotated_images, rotated_labels, change_rate=change_rate)

    return images_aug, labels_aug


#------------------------- Random shear -------------------------

def segm_random_shear(
        images,
        labels,
        factor=None,
        axis='xy',
        fill_mode='reflect',
        interpolation='bilinear',
        fill_value=0.0,
        change_rate=1.0):
    """
    This function randomly shears input images.

    Arguments:
        images:
            Input RGB or grayscale images with shape
            [batch_size, width, height, channels]. 
        factor:
            A float or a tuple of 2 floats, specifies the range of values
            the shear angles are sampled from (one per image). If a scalar 
            value v is used, it is equivalent to the tuple (-v, v). Angles 
            are in radians (fractions of 2*pi). 
            For example, factor=(-0.349, 0.785) results in an output sheared
            by a random angle in the range [-20 degrees, +45 degrees].
        axis:
            The shear axis:
                'xy': shear along both axis
                'x': shear along the x axis only
                'y': shear along the y axis only  
        fill_mode:
            Points outside the boundaries of the input are filled according
            to the given mode. One of {'constant', 'reflect', 'wrap', 'nearest'}.
            See Tensorflow documentation at https://tensorflow.org
            for more details.
        interpolation:
            A string, the interpolation method. Supported values: 'nearest', 'bilinear'.
        change_rate:
            A float in the interval [0, 1] representing the number of 
            changed images versus the total number of input images average
            ratio. For example, if `change_rate` is set to 0.25, 25% of
            the input images will get changed on average (75% won't get
            changed). If it is set to 0.0, no images are changed. If it is
            set to 1.0, all the images are changed.
    Returns:
        The sheared images.
    """
    
    if axis == 'x':
        function_name = "random_shear_x"
    elif axis == 'y':
        function_name = "random_shear_y"
    else:
        function_name = "random_shear"

    check_dataaug_argument(factor, "factor", function_name=function_name, data_type=float)
    if not isinstance(factor, (tuple, list)):
        factor = (-factor, factor)
        
    check_fill_and_interpolation(fill_mode, interpolation, fill_value, function_name=function_name)

    batch_size = tf.shape(images)[0]
    min_angle = factor[0] * 2. * math.pi
    max_angle = factor[1] * 2. * math.pi
    angles = tf.random.uniform([batch_size], minval=min_angle, maxval=max_angle)

    matrix = get_shear_matrix(angles, axis=axis)
    
    sheared_images, sheared_labels = _transform_images_and_labels(
                images,
                labels,
                matrix,
                interpolation=interpolation,
                fill_mode=fill_mode,
                fill_value=fill_value)

    # Apply the change rate to images and labels
    images_aug, labels_aug = segm_apply_change_rate(
            images, labels, sheared_images, sheared_labels, change_rate=change_rate)

    return images_aug, labels_aug


#------------------------- Random zoom -------------------------

def segm_random_zoom(
            images, labels, width_factor=None, height_factor=None,
            fill_mode='reflect', interpolation='bilinear', fill_value=0.0,
            change_rate=1.0):
    """
    This function randomly zooms input images and the bounding boxes
    in the associated groundtruth labels.

    If `width_factor` and `height_factor` are both set, the images are zoomed
    in or out on each axis independently, which may result in noticeable distortion.
    If you want to avoid distortion, only set `width_factor` and the mages will be
    zoomed by the same amount in both directions.
 
    Arguments:
        images:
            Input RGB or grayscale images with shape
            Shape: [batch_size, width, height, channels] 
        labels:
            Groundtruth labels associated to the images in 
            (class, x1, y1, x2, y2) format. Bounding box coordinates
            must be absolute, opposite corners coordinates.
            Shape: [batch_size, num_labels, 5]
        width_factor:
            A float or a tuple of 2 floats, specifies the range of values horizontal
            zoom factors are sampled from (one per image). If a scalar value v is used,
            it is equivalent to the tuple (-v, v). Factors are fractions of the width
            of the image. A positive factor means zooming out, while a negative factor
            means zooming in.
            For example, width_factor=(0.2, 0.3) results in an output zoomed out by
            a random amount in the range [+20%, +30%]. width_factor=(-0.3, -0.2) results
            in an output zoomed in by a random amount in the range [+20%, +30%].
        height_factor:
            A float or a tuple of 2 floats, specifies the range of values vertical
            zoom factors are sampled from (one per image). If a scalar value v is used,
            it is equivalent to the tuple (-v, v). Factors are fractions of the height
            of the image. A positive value means zooming out, while a negative value
            means zooming in.
            For example, height_factor=(0.2, 0.3) results in an output zoomed out 
            between 20% to 30%. height_factor=(-0.3, -0.2) results in an output zoomed
            in between 20% to 30%.
            If `height_factor` is not set, it defaults to None. In this case, images
            images will be zoomed by the same amounts in both directions and no image
            distortion will occur.
        fill_mode:
            Points outside the boundaries of the input are filled according
            to the given mode. One of {'constant', 'reflect', 'wrap', 'nearest'}.
            See Tensorflow documentation at https://tensorflow.org
            for more details.
        interpolation:
            A string, the interpolation method. Supported values: 'nearest', 'bilinear'.
        change_rate:
            A float in the interval [0, 1] representing the number of 
            changed images versus the total number of input images average
            ratio. For example, if `change_rate` is set to 0.25, 25% of
            the input images will get changed on average (75% won't get
            changed). If it is set to 0.0, no images are changed. If it is
            set to 1.0, all the images are changed.

    Returns:
        The zoomed images and groundtruth labels with zoomed bounding boxes.
    """

    check_dataaug_argument(width_factor, "width_factor", function_name="random_zoom", data_type=float)
    if isinstance(width_factor, (tuple, list)):
        width_lower = width_factor[0]
        width_upper = width_factor[1]
    else:
        width_lower = -width_factor
        width_upper = width_factor
                
    if height_factor is not None:
        check_dataaug_argument(height_factor, "height_factor", function_name="random_zoom", data_type=float)
        if isinstance(height_factor, (tuple, list)):
            height_lower = height_factor[0]
            height_upper = height_factor[1]
        else:
            height_lower = -height_factor
            height_upper = height_factor
        if abs(height_lower) > 1.0 or abs(height_upper) > 1.0:
            raise ValueError(
                "Argument `height_factor` of function `random_zoom`: expecting float "
                "values in the interval [-1.0, 1.0]. Received: {}".format(height_factor))
    else:
        height_lower = width_lower
        height_upper = width_upper
    
    check_fill_and_interpolation(fill_mode, interpolation, fill_value, function_name="random_zoom")

    images_shape = tf.shape(images)
    batch_size = images_shape[0]
    image_width = images_shape[1]
    image_height = images_shape[2]

    height_zoom = tf.random.uniform(
            [batch_size, 1], minval=1. + height_lower, maxval=1. + height_upper, dtype=tf.float32)
    width_zoom = tf.random.uniform(
            [batch_size, 1], minval=1. + width_lower, maxval=1. + width_upper, dtype=tf.float32)
            
    zooms = tf.cast(tf.concat([width_zoom, height_zoom], axis=1), dtype=tf.float32)
      
    matrix = get_zoom_matrix(zooms, image_width, image_height)
    
    zoomed_images, zoomed_labels = _transform_images_and_labels(
                images,
                labels,
                matrix,
                interpolation=interpolation,
                fill_mode=fill_mode,
                fill_value=fill_value)

    # Apply the change rate to images and labels
    images_aug, labels_aug = segm_apply_change_rate(
            images, labels, zoomed_images, zoomed_labels, change_rate=change_rate)

    return images_aug, labels_aug
