# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import tensorflow as tf


def segmentation_loss(y_true, y_pred, num_classes=None, loss_weights=None, ignore_label=255) -> tf.Tensor:
    """
    The custom loss function that will be used during model training.

    Args:
        y_true (tf.Tensor): The ground truth labels for each pixel.
        y_pred (tf.Tensor): The predicted labels for each pixel.

    Returns:
        tf.Tensor: The computed loss as a scalar tensor.
    """

    logits = y_pred
    labels = y_true
        
    if loss_weights is None:
        if num_classes == 21:
            loss_weights = [0.5] + [1.0] * (num_classes - 1)
        else:
            loss_weights = [1.0] * (num_classes)

    # Flatten logits and labels tensors for processing.
    logits = tf.reshape(logits, [-1, num_classes])
    labels = tf.reshape(labels, [-1])

    # Create a mask to exclude the ignored label from the loss computation.
    not_ignored_mask = tf.not_equal(labels, ignore_label)
    labels = tf.boolean_mask(labels, not_ignored_mask)
    logits = tf.boolean_mask(logits, not_ignored_mask)

    # Cast labels to an integer type for further processing.
    labels = tf.cast(labels, tf.int32)

    # Apply class weights if provided.
    class_weights = tf.constant(loss_weights, dtype=tf.float32)
    weights = tf.gather(class_weights, labels)
    # Convert labels to one-hot encoding for compatibility with softmax cross entropy.
    labels_one_hot = tf.one_hot(labels, depth=num_classes)

    # Compute the cross entropy loss for each pixel.

    pixel_losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels_one_hot, logits=logits)

    weighted_pixel_losses = pixel_losses * weights 
    total_loss = tf.reduce_sum(weighted_pixel_losses)

    # Calculate the number of pixels that contribute to the loss (excluding ignored ones).
    num_positive = tf.reduce_sum(tf.cast(not_ignored_mask, tf.float32))

    # Normalize the loss by the number of contributing pixels to get the final loss value.
    loss = total_loss / (num_positive + 1e-5)

    return loss
