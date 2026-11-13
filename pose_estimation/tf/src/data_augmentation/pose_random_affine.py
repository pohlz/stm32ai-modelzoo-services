# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
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
import numpy as np
import tensorflow as tf

from common.data_augmentation import \
            check_fill_and_interpolation, transform_images, check_dataaug_argument, \
            get_flip_matrix, get_translation_matrix, get_rotation_matrix, \
            get_shear_matrix, get_zoom_matrix
from .pose_random_utils import objdet_apply_change_rate, pose_apply_change_rate
from pose_estimation.tf.src.data_augmentation import swap_list_dict

def _xywh_to_xy1xy2(boxes):
    """
    This function convert xywh coordinates of boxes into xy1xy2
    
    Arguments:
        boxes:
            Boxes in xy(centered) wh coordinates
            Shape:[batch_size, num_boxes, 4]

    Returns:
        xy1xy2:
            Boxes in x1y1 x2y2 coordinates
            Shape:[batch_size, num_boxes, 4]
    """

    x1 = boxes[...,0] - boxes[...,2]/2
    y1 = boxes[...,1] - boxes[...,3]/2
    x2 = boxes[...,0] + boxes[...,2]/2
    y2 = boxes[...,1] + boxes[...,3]/2

    xy1xy2 = tf.stack([x1,y1,x2,y2],-1)

    return xy1xy2

def _xy1xy2_to_xywh(boxes):
    """
    
    This function convert xy1xy2 coordinates of boxes into xywh
    Arguments:
        boxes:
            Boxes in x1y1 x2y2 coordinates
            Shape:[batch_size, num_boxes, 4]

    Returns:
        xywh:
            Boxes in xy(centered) wh coordinates
            Shape:[batch_size, num_boxes, 4]
    """

    x = (boxes[...,0] + boxes[...,2])/2
    y = (boxes[...,1] + boxes[...,3])/2
    w = tf.abs(boxes[...,0] - boxes[...,2])
    h = tf.abs(boxes[...,1] - boxes[...,3])

    xywh = tf.stack([x,y,w,h],-1)

    return xywh

def _transform_boxes(boxes, transforms, image_width, image_height, scale=1.):
    """
    This function applies affine transformations to a batch of boxes.
    The transformation matrices are independent from each other
    and are generally different from one batch item to another.
    
    Arguments:
        boxes:
            Boxes the matrices are applied to
            Shape:[batch_size, num_boxes, 4]
        transforms:
            Matrices coefficients to apply to the boxes
            Shape:[batch_size, 8]

    Returns:
        Transformed boxes
        Shape:[batch_size, num_boxes, 4]
    """
    
    image_width = tf.cast(image_width, tf.float32)
    image_height = tf.cast(image_height, tf.float32)

    boxes = _xywh_to_xy1xy2(boxes)
    
    boxes_shape = tf.shape(boxes)

    batch_size = boxes_shape[0]
    num_boxes = boxes_shape[1]
    
    # Create a mask to keep track of padding boxes
    coords_sum = tf.math.reduce_sum(boxes, axis=-1)
    padding_mask = tf.where(coords_sum > 0, 1., 0.)
    padding_mask = tf.repeat(padding_mask, 4)
    padding_mask = tf.reshape(padding_mask, [batch_size, num_boxes, 4])
        
    # Create and invert the matrices (inversion is necessary
    # to align with the TF function that transforms the images)
    transforms = tf.concat([
            transforms,
            tf.ones([batch_size, 1], dtype=tf.float32)],
            axis=-1)
    matrices = tf.reshape(transforms, [batch_size, 3, 3])
    matrices = tf.linalg.inv(matrices)
    
    # The same transform has to be applied to all the boxes
    # of a batch item, so we replicate the matrices.
    matrices = tf.expand_dims(matrices, axis=1)    
    matrices = tf.tile(matrices, [1, num_boxes, 1, 1])

    x1 = boxes[..., 0]
    y1 = boxes[..., 1]
    x2 = boxes[..., 2]
    y2 = boxes[..., 3]

    # Reduce the size of the boxes before transforming them
    if scale < 1:
        dx = scale * (x2 - x1)
        dy = scale * (y2 - y1)
        boxes = tf.stack([x1 + dx, y1 + dx, x2 - dx, y2 - dy], axis=-1)

    # Stack box corner vectors to create 4x4 matrices
    # Then multiply by transformation matrices to get
    # the transformed corner vectors.
    corners = tf.concat([
            tf.stack([x1, x2, x2, x1], axis=-1),
            tf.stack([y1, y1, y2, y2], axis=-1),
            tf.ones([batch_size, num_boxes, 4], dtype=tf.float32)],
            axis=-1)
    corners = tf.reshape(corners, [batch_size, num_boxes, 3, 4])
    
    trd_corners = tf.linalg.matmul(matrices, corners)

    # Project transformed corner vectors onto x and y axis
    tx1 = tf.math.reduce_min(trd_corners[..., 0, :], axis=-1)
    tx2 = tf.math.reduce_max(trd_corners[..., 0, :], axis=-1)
    ty1 = tf.math.reduce_min(trd_corners[..., 1, :], axis=-1)
    ty2 = tf.math.reduce_max(trd_corners[..., 1, :], axis=-1)

    # Clip transformed coordinates
    tx1 = tf.math.maximum(tx1, 0)
    tx1 = tf.math.minimum(tx1, image_width)
    
    tx2 = tf.math.maximum(tx2, 0)
    tx2 = tf.math.minimum(tx2, image_width)
    
    ty1 = tf.math.maximum(ty1, 0)
    ty1 = tf.math.minimum(ty1, image_height)
    
    ty2 = tf.math.maximum(ty2, 0)
    ty2 = tf.math.minimum(ty2, image_height)
    
    trd_boxes = tf.stack([tx1, ty1, tx2, ty2], axis=-1)

    # Get rid of boxes that don't make sense
    valid_boxes = tf.math.logical_and(tx2 > tx1, ty2 > ty1)
    valid_boxes = tf.cast(valid_boxes, tf.float32)
    trd_boxes *= tf.expand_dims(valid_boxes, axis=-1)

    # Set to 0 the coordinates of padding boxes as transforms
    # may have resulted in some non-zeros coordinates.
    trd_boxes *= padding_mask

    trd_boxes = _xy1xy2_to_xywh(trd_boxes)
    
    return trd_boxes


def _transform_keypoints(kpts, transforms, image_width, image_height):
    """
    This function applies affine transformations to a batch of boxes.
    The transformation matrices are independent from each other
    and are generally different from one batch item to another.
    
    Arguments:
        kpts:
            Shape:[batch_size, num_boxes, 3*keypoints]
        transforms:
            Matrices coefficients to apply to the keypoints
            Shape:[batch_size, 8]

    Returns:
        Transformed keypoints
        Shape:[batch_size, num_boxes, 3, keypoints]
    """
    
    image_width = tf.cast(image_width, tf.float32)
    image_height = tf.cast(image_height, tf.float32)
    
    boxes_shape = tf.shape(kpts)
    batch_size = boxes_shape[0]
    num_boxes = boxes_shape[1]
    nb_kpts = boxes_shape[2]//3

    keypoints = tf.reshape(kpts,[batch_size,num_boxes,nb_kpts,3]) # shape (batch_size,num_boxes,keypoints,3)
    keypoints = tf.transpose(keypoints,[0,1,3,2])                 # shape (batch_size,num_boxes,3,keypoints)
    
    # Create a mask to keep track of padding boxes
    coords_sum = tf.math.reduce_sum(keypoints, axis=-2)
    padding_mask = tf.where(coords_sum > 0, 1., 0.)[...,None,:]
        
    # Create and invert the matrices (inversion is necessary
    # to align with the TF function that transforms the images)
    transforms = tf.concat([
            transforms,
            tf.ones([batch_size, 1], dtype=tf.float32)],
            axis=-1)
    matrices = tf.reshape(transforms, [batch_size, 3, 3])
    matrices = tf.linalg.inv(matrices)
    
    # The same transform has to be applied to all the boxes
    # of a batch item, so we replicate the matrices.
    matrices = tf.expand_dims(matrices, axis=1)
    matrices = tf.tile(matrices, [1, num_boxes, 1, 1])
    
    trd_corners = tf.linalg.matmul(matrices, keypoints)

    # Project transformed corner vectors onto x and y axis
    tx = trd_corners[..., 0, :]
    ty = trd_corners[..., 1, :]
    tz = trd_corners[..., 2, :]

    # Get rid of keypoints that don't make sense
    valid_x = tf.math.logical_and(tx>=0., 1.>=tx)
    valid_y = tf.math.logical_and(ty>=0., 1.>=ty)
    valid_keypoints = tf.math.logical_and(valid_x,valid_y)
    valid_keypoints = tf.cast(valid_keypoints, tf.float32)

    # Clip transformed coordinates
    tx = tf.math.maximum(tx, 0)
    tx = tf.math.minimum(tx, image_width)

    ty = tf.math.maximum(ty, 0)
    ty = tf.math.minimum(ty, image_height)
    
    tz = tf.math.maximum(tz, 0)
    tz = tf.math.minimum(tz, 1.)
    
    trd_kpts = tf.stack([tx, ty, tz], axis=-2)

    trd_kpts *= valid_keypoints[...,None,:]

    trd_kpts *= padding_mask

    trd_kpts = tf.transpose(trd_kpts,[0,1,3,2]) # shape (batch_size,num_boxes,keypoints,3)

    trd_kpts = tf.reshape(trd_kpts,[batch_size,num_boxes,-1])
    
    return trd_kpts

#------------------------- Random flip -------------------------

def _keypoints_rightleft_swap(kpts):
    """
    This function swaps right and left keypoints based on a dictionnary provided in swap_list_dict.py
    Right and left keypoints must be swapped if the image is flipped.

    Arguments:
        kpts:
            The keypoints
            Shape : [batch, num_labels, 3*keypoints]

    Returns:
        The swapped keypoints.
    """

#    from swap_list_dict import swap_list_dict

    sh = tf.shape(kpts)
    nb_kpts = sh[2]//3

    class Lambda:
        # This is made to bypass the python limitation of
        #   defining only one lambda function in a list
        def __init__(self,x):
            self.x = x
        def __call__(self):
            return self.x

    def default(): return tf.range(nb_kpts)

    case_functions = [(nb_kpts == key, Lambda(tf.constant(swap_list_dict[key]))) for key in swap_list_dict.keys()]

    swap_list = tf.case(case_functions, default=default, exclusive=True)

    keypoints = tf.reshape(kpts,[sh[0],sh[1],sh[2]//3,3])
    keypoints = tf.gather(keypoints, indices=swap_list, axis=-2)
    swapped_kpts = tf.reshape(keypoints,[sh[0],sh[1],sh[2]])

    return swapped_kpts

def pose_random_flip(images, labels, mode=None, change_rate=0.5):
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
            ([x_center, y_center, w, h]+3*keypoints) format.
            Shape: [batch_size, num_labels, 5+3*keypoints] 
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

    im_matrix = get_flip_matrix(batch_size, width=image_width, height=image_height, mode=mode)
    lb_matrix = get_flip_matrix(batch_size, width=2., height=2., mode=mode)

    boxes = labels[..., 1:5]
    keypoints = labels[...,5:]
    flipped_images = transform_images(images, im_matrix)
    flipped_boxes = _transform_boxes(boxes, lb_matrix, image_width=1., image_height=1.)
    flipped_keypoints = _transform_keypoints(keypoints, lb_matrix, image_width=1., image_height=1.)


    if mode in ["horizontal","vertical"]:
        # because the keypoints have been flipped, if the pose is symmetric then right/left must be swapped
        flipped_keypoints = _keypoints_rightleft_swap(flipped_keypoints)

    # Apply the change rate to images and labels
    images_aug, boxes_aug, kpts_aug = pose_apply_change_rate(
            images, boxes, keypoints, flipped_images, flipped_boxes, flipped_keypoints, change_rate=change_rate)
    classes = labels[..., 0:1]
    labels_aug = tf.concat([classes, boxes_aug, kpts_aug], axis=-1)

    return images_aug, labels_aug


#------------------------- Random translation -------------------------

def objdet_random_translation(
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
    
    classes = labels[..., 0]
    boxes = labels[..., 1:]
    
    width_translate = tf.random.uniform(
            [batch_size, 1], minval=width_lower, maxval=width_upper, dtype=tf.float32)
    width_translate = width_translate * tf.cast(image_width, tf.float32)
    
    height_translate = tf.random.uniform(
            [batch_size, 1], minval=height_lower, maxval=height_upper, dtype=tf.float32)
    height_translate = height_translate * tf.cast(image_height, tf.float32)

    translations = tf.cast(
            tf.concat([width_translate, height_translate], axis=1),
            dtype=tf.float32)

    translation_matrix = get_translation_matrix(translations)
    
    translated_images = transform_images(
            images,
            translation_matrix,
            interpolation=interpolation,
            fill_mode=fill_mode,
            fill_value=fill_value)

    translated_boxes = _transform_boxes(
            boxes,
            translation_matrix,
            image_width,
            image_height)

    # Apply the change rate to images and labels
    images_aug, boxes_aug = objdet_apply_change_rate(
            images, boxes, translated_images, translated_boxes, change_rate=change_rate)
    classes = tf.expand_dims(labels[..., 0], axis=-1)
    labels_aug = tf.concat([classes, boxes_aug], axis=-1)

    return images_aug, labels_aug


#------------------------- Random rotation -------------------------

def _get_3Drotation_matrix(phi, theta, psi, width, height):
    """
    This function creates a batch of rotation matrices given a batch of angles.
    Angles are independent from each other and may be different from
    one batch item to another.
    
    The rotation matrix is:
        [ cos(theta)*cos(psi)  -cos(phi)*sin(psi)+sin(phi)*sin(theta)*cos(psi)  x_offset ]
        [ cos(theta)*sin(psi)   cos(phi)*cos(psi)+sin(phi)*sin(theta)*sin(psi)  y_offset ]
        [ 0                    0                                                1        ]
    x_offset and y_offset are calculated from the angles and image dimensions.
    """

    width = tf.cast(width, tf.float32)
    height = tf.cast(height, tf.float32)
    
    num_angles = tf.shape(phi)[0]
    x_offset = ((width - 1) - (tf.cos(theta)*tf.cos(psi) * (width - 1) - tf.cos(phi)*tf.sin(psi)+tf.sin(phi)*tf.sin(theta)*tf.cos(psi) * (height - 1))) / 2.0
    y_offset = ((height - 1) - (tf.cos(theta)*tf.sin(psi) * (width - 1) + tf.cos(phi)*tf.cos(psi)+tf.sin(phi)*tf.sin(theta)*tf.sin(psi) * (height - 1))) / 2.0
    
    matrix = tf.concat([
                (tf.cos(theta)*tf.cos(psi))[:, None],
                -(tf.cos(phi)*tf.sin(psi)+tf.sin(phi)*tf.sin(theta)*tf.cos(psi))[:, None],
                x_offset[:, None],
                (tf.cos(theta)*tf.sin(psi))[:, None],
                (tf.cos(phi)*tf.cos(psi)+tf.sin(phi)*tf.sin(theta)*tf.sin(psi))[:, None],
                y_offset[:, None],
                tf.zeros((num_angles, 2), tf.float32)
                ],
                axis=1)

    return matrix

def pose_random_rotation(
                images, labels, factor=None,
                fill_mode='reflect', interpolation='bilinear', fill_value=0.0,
                change_rate=1.0, rotation_type='2D'):
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
    
    if rotation_type=='2D':
        angles = tf.random.uniform([batch_size], minval=min_angle, maxval=max_angle)
        im_rotation_matrix = get_rotation_matrix(angles, image_width, image_height)
        lb_rotation_matrix = get_rotation_matrix(angles, 2., 2.)
    elif rotation_type=='3D':
        phi   = tf.random.uniform([batch_size], minval=min_angle, maxval=max_angle)
        theta = tf.random.uniform([batch_size], minval=min_angle, maxval=max_angle)
        psi   = tf.random.uniform([batch_size], minval=min_angle, maxval=max_angle)
        im_rotation_matrix = _get_3Drotation_matrix(phi, theta, psi, image_width, image_height)
        lb_rotation_matrix = _get_3Drotation_matrix(phi, theta, psi, 2., 2.)


    classes = labels[..., 0:1]
    boxes     = labels[..., 1:5]
    keypoints = labels[...,5:]

    rotated_images = transform_images(images,im_rotation_matrix,fill_mode=fill_mode,fill_value=fill_value,interpolation=interpolation)
    rotated_boxes = _transform_boxes(boxes,lb_rotation_matrix,1.,1.) #,scale=0.1)
    rotated_keypoints = _transform_keypoints(keypoints, lb_rotation_matrix, image_width=1., image_height=1.)

    # Apply the change rate to images and labels
    images_aug, boxes_aug, kpts_aug = pose_apply_change_rate(
            images, boxes, keypoints, rotated_images, rotated_boxes, rotated_keypoints, change_rate=change_rate)
    labels_aug = tf.concat([classes, boxes_aug, kpts_aug], axis=-1)

    return images_aug, labels_aug


#------------------------- Random shear -------------------------

def objdet_random_shear(
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

    images_shape = tf.shape(images)
    batch_size = images_shape[0]
    image_width = images_shape[1]
    image_height = images_shape[2]

    min_angle = factor[0] * 2. * math.pi
    max_angle = factor[1] * 2. * math.pi
    angles = tf.random.uniform([batch_size], minval=min_angle, maxval=max_angle)

    classes = labels[..., 0]
    boxes = labels[..., 1:]

    shear_matrix = get_shear_matrix(angles, axis=axis)
    
    sheared_images = transform_images(
                        images,
                        shear_matrix,
                        fill_mode=fill_mode,
                        fill_value=fill_value,
                        interpolation=interpolation)
 
    sheared_boxes = _transform_boxes(
                        boxes,
                        shear_matrix,
                        image_width,
                        image_height,
                        scale=0.1)
                        
     # Apply the change rate to images and labels
    images_aug, boxes_aug = objdet_apply_change_rate(
            images, boxes, sheared_images, sheared_boxes, change_rate=change_rate)
    classes = tf.expand_dims(labels[..., 0], axis=-1)
    labels_aug = tf.concat([classes, boxes_aug], axis=-1)

    return images_aug, labels_aug


#------------------------- Random zoom -------------------------

def objdet_random_zoom(
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

    classes = labels[..., 0]
    boxes = labels[..., 1:]

    height_zoom = tf.random.uniform(
            [batch_size, 1], minval=1. + height_lower, maxval=1. + height_upper, dtype=tf.float32)
    width_zoom = tf.random.uniform(
            [batch_size, 1], minval=1. + width_lower, maxval=1. + width_upper, dtype=tf.float32)
            
    zooms = tf.cast(tf.concat([width_zoom, height_zoom], axis=1), dtype=tf.float32)
      
    zoom_matrix = get_zoom_matrix(zooms, image_width, image_height)
    
    zoomed_images = transform_images(
                images,
                zoom_matrix,
                fill_mode=fill_mode,
                fill_value=fill_value,
                interpolation=interpolation)

    zoomed_boxes = _transform_boxes(
                boxes,
                zoom_matrix,
                image_width,
                image_height)
    
    # Apply the change rate to images and labels
    images_aug, boxes_aug = objdet_apply_change_rate(
            images, boxes, zoomed_images, zoomed_boxes, change_rate=change_rate)
    classes = tf.expand_dims(labels[..., 0], axis=-1)
    labels_aug = tf.concat([classes, boxes_aug], axis=-1)

    return images_aug, labels_aug
