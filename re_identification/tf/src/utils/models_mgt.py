#  /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import sys
import os
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import tensorflow as tf
from omegaconf import DictConfig

from common.utils import transfer_pretrained_weights, check_attribute_value, check_model_support, check_attributes
from .metrics import pairwise_distance


def ai_runner_invoke(image_processed,ai_runner_interpreter):
    preds, _ = ai_runner_interpreter.invoke(image_processed)
    nb_class = preds[0].shape[-1]
    return preds[0].reshape([-1, nb_class])


def change_model_number_of_classes(model,num_classes):

    output_shape = num_classes

    # If the model already has the correct number of classes -> dont do anything
    for outp in model.outputs:
        if outp.shape[-1] == output_shape:
            return model

    l = -1
    l_list = []

    while True:

        layer_type = type(model.layers[l])
        layer_config = model.layers[l].get_config()

        if layer_type in [tf.keras.layers.Conv2D,
                          tf.keras.layers.Conv2DTranspose,
                          tf.keras.layers.Conv1D,
                          tf.keras.layers.Conv1DTranspose,
                          tf.keras.layers.Dense]:
            if layer_type in [tf.keras.layers.Conv2D,tf.keras.layers.Conv2DTranspose,tf.keras.layers.Conv1D,tf.keras.layers.Conv1DTranspose]:
                layer_config['filters'] = output_shape
                new_layer = layer_type(**layer_config)
                outputs = new_layer(model.layers[l-1].output)
            else:
                layer_config['units'] = output_shape
                new_layer = layer_type(**layer_config)
                outputs = new_layer(model.layers[l-1].output)

            for i,new_l in enumerate(l_list[::-1]):
                outputs = new_l(outputs)

            return tf.keras.Model(inputs=model.input, outputs=outputs, name=model.name)

        else:
            l_list.append(layer_type(**layer_config))
            l-=1

    return None

def get_loss(num_classes: int, label_smoothing: float = 0.1) -> tf.keras.losses:
    """
    Returns the appropriate loss function based on the number of classes in the dataset.

    Args:
        num_classes (int): The number of classes in the dataset.

    Returns:
        tf.keras.losses: The appropriate loss function based on the number of classes in the dataset.
    """
    # We use the sparse version of the categorical crossentropy because
    # this is what we use to load the dataset.
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=label_smoothing)
    return loss

def triplet_loss(y_true,
                 embeddings,
                 margin: float = 0.2,
                 mining: str = 'hard',
                 distance_metric: str = 'cosine',
                 self_distance: bool = False):
                 
    """
    Triplet loss with dot product similarity and selectable mining strategy.

    Args:
        y_true: True labels, shape (batch_size,).
        embeddings: Embeddings from the model, shape (batch_size, embedding_dim).
        margin: Margin for the triplet loss.
        mining: Mining strategy to use ('simple', 'hard', 'semi_hard').
        distance_metric: Distance metric to use ('euclidean', 'squared_euclidean', 'cosine').
        self_distance: Whether to consider self-comparisons in the loss computation.

    Returns:
        loss: Computed triplet loss.
    """
    pairwise_dist = pairwise_distance(embeddings, distance_metric=distance_metric)  # Compute pairwise distances
    labels = tf.reshape(y_true, [-1, 1])
    label_equal = tf.equal(labels, tf.transpose(labels))
    # calculate how many positive pairs in the batch
    positive_mask = tf.cast(label_equal, tf.float32) - tf.eye(tf.shape(embeddings)[0], dtype=tf.float32)  # Exclude self-comparisons
    negative_mask = 1.0 - tf.cast(label_equal, tf.float32)

    if self_distance:
        self_mask = tf.eye(tf.shape(embeddings)[0], dtype=tf.float32)
        # # set the pairwise distance to mean_pairwise_dist for self-comparisons
        # pairwise_dist = pairwise_dist * (1.0 - self_mask) + mean_positive_pairwise_dist * self_mask
        max_dist = tf.reduce_max(pairwise_dist) + 1.0  # Maximum distance for masking
        min_dist = tf.reduce_min(pairwise_dist + self_mask * max_dist)  # Mask out self-comparisons
        pairwise_dist = pairwise_dist * (1.0 - self_mask) + min_dist * self_mask  # Set self-distances to min distance
        positive_mask = 1- negative_mask

    if mining == 'simple':
        # Compute loss for all triplets
        anchor_positive_dist = tf.expand_dims(pairwise_dist, axis=2)  # (batch, batch, 1)
        anchor_negative_dist = tf.expand_dims(pairwise_dist, axis=1)  # (batch, 1, batch)

        triplet_loss_tensor = anchor_positive_dist - anchor_negative_dist + margin
        mask = positive_mask[:, :, None] * negative_mask[:, None, :]
        triplet_loss_tensor = tf.maximum(triplet_loss_tensor, 0.0) * mask
        loss = tf.reduce_sum(triplet_loss_tensor) / (tf.reduce_sum(mask) + 1e-16)

    elif mining == 'hard':
        hardest_positive_dist = tf.reduce_max(pairwise_dist * positive_mask + (1-positive_mask) * 0.0, axis=1)
        # Hardest negative: min distance among negatives for each anchor
        max_dist = tf.reduce_max(pairwise_dist) + 1.0
        masked_neg_dist = pairwise_dist + max_dist * (1-negative_mask)  # Mask out positives
        hardest_negative_dist = tf.reduce_min(masked_neg_dist, axis=1)

        triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)
        loss = tf.reduce_mean(triplet_loss)

    elif mining == 'semi_hard':
        anchor_positive_dist = tf.expand_dims(pairwise_dist, axis=2)  # (batch, batch, 1)
        anchor_negative_dist = tf.expand_dims(pairwise_dist, axis=1)  # (batch, 1, batch)
        large_val = tf.fill(tf.shape(anchor_negative_dist), 1e9)
        semi_hard_negatives = tf.reduce_min(
            tf.where(
                (anchor_negative_dist > anchor_positive_dist) & (positive_mask[:, :, None] * negative_mask[:, None, :] > 0),
                anchor_negative_dist,
                large_val
            ),
            axis=2
        )
        triplet_loss_tensor = tf.maximum(pairwise_dist * positive_mask - semi_hard_negatives + margin, 0.0)
        valid_triplets = tf.reduce_sum(tf.cast(triplet_loss_tensor > 1e-16, tf.float32))
        loss = tf.reduce_sum(triplet_loss_tensor) / (valid_triplets + 1e-16)

    else:
        raise ValueError(f"Invalid mining option '{mining}'. Choose from ['simple', 'hard', 'semi_hard'].")
    return loss


# Fonction retournant un objet tf.keras.losses.Loss
def get_triplet_loss(margin=0.2, mining='hard', distance_metric='cosine', self_distance=False):
    """
    Returns a triplet loss function with specified parameters.
    Args:
        margin (float): Margin for the triplet loss.
        mining (str): Mining strategy to use ('simple', 'hard', 'semi_hard').
        distance_metric (str): Distance metric to use ('euclidean', 'squared_euclidean', 'cosine').
        self_distance (bool): Whether to consider self-comparisons in the loss computation.
    Returns:
        A callable loss function that takes (y_true, y_pred) as inputs.
    """
    from keras.saving import register_keras_serializable
    @register_keras_serializable()
    def triplet(y_true, y_pred):
        # tf.print("y_pred shape:", tf.shape(y_pred))
        # tf.print("y_true shape:", tf.shape(y_true))
        return triplet_loss(y_true, y_pred, margin=margin, mining=mining, distance_metric=distance_metric, self_distance=self_distance)
    return triplet

