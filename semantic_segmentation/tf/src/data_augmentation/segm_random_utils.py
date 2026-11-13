# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import tensorflow as tf


def segm_apply_change_rate(images, labels, images_augmented, labels_augmented, change_rate=1.0): 
    """
    This function outputs a mix of augmented images and original
    images. The argument `change_rate` is a float in the interval 
    [0.0, 1.0] representing the number of changed images versus 
    the total number of input images average ratio. For example,
    if `change_rate` is set to 0.25, 25% of the input images will
    get changed on average (75% won't get changed). If it is set
    to 0.0, no images are changed. If it is set to 1.0, all the
    images are changed.
    """

    if change_rate == 1.0:
        return images_augmented, labels_augmented
        
    if change_rate < 0. or change_rate > 1.:
        raise ValueError("The value of `change_rate` must be in the interval [0, 1]. ",
                         "Received {}".format(change_rate))
    
    image_shape = tf.shape(images)
    batch_size = image_shape[0]
    width = image_shape[1]
    height = image_shape[2]
    channels = image_shape[3]
    
    # Randomy select the images and labels that will be changed
    batch_size = tf.shape(images)[0]
    probs = tf.random.uniform([batch_size], minval=0, maxval=1, dtype=tf.float32)
    change_mask = tf.where(probs < change_rate, True, False)

    # Create the mix of changed/unchanged images
    mask = tf.repeat(change_mask, width * height * channels)
    mask = tf.reshape(mask, [batch_size, width, height, channels])
    mask_not = tf.math.logical_not(mask)
    images_mix = tf.cast(mask_not, images.dtype) * images + tf.cast(mask, images.dtype) * images_augmented

    # Create the mix of changed/unchanged labels 
    mask = tf.repeat(change_mask, width * height)
    mask = tf.reshape(mask, [batch_size, width, height, 1])
    mask_not = tf.math.logical_not(mask)

    labels = tf.cast(labels, tf.uint8)
    labels_augmented = tf.cast(labels_augmented, tf.uint8)
    
    labels_mix = tf.cast(mask_not, tf.uint8) * labels + tf.cast(mask, tf.uint8) * labels_augmented

    return images_mix, labels_mix
