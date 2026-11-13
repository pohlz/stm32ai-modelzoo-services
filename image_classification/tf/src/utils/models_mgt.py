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
from image_classification.tf.src.models import get_mobilenetv1, get_mobilenetv2, get_fdmobilenet, get_resnet, \
                                 get_resnet50v2, get_squeezenetv11, get_st_mnistv1, get_st_efficientnetlcv1, \
                                 get_st_fdmobilenetv1, get_efficientnetv2, get_custom_model


def ai_runner_invoke(image_processed, ai_runner_interpreter):
    """
    Docstring for ai_runner_invoke
    
    Args:
        image_processed (tf.Tensor): input images
        ai_runner_interpreter: ai_runner object to be invoked on input images
    Returns:
        prediction outputs
    """
    preds, _ = ai_runner_interpreter.invoke(image_processed)
    nb_class = preds[0].shape[-1]
    return preds[0].reshape([-1, nb_class])


def change_model_number_of_classes(model: tf.keras.Model, num_classes: int):
    """
    Docstring for change_model_number_of_classes

    Args:
        model (tf.keras.Model): Keras model
        num_classes (int): new number of classes as output
    Returns: 
        (tf.keras.Model): a new model with updated number of classes
    """

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

            for i, new_l in enumerate(l_list[::-1]):
                outputs = new_l(outputs)

            return tf.keras.Model(inputs=model.input, outputs=outputs, name=model.name)

        else:
            l_list.append(layer_type(**layer_config))
            l-=1

    return None


def change_model_input_shape(model: tf.keras.Model, new_inp_shape: Tuple):
    """
    Change model input shape
    
    Args
        model (tf.keras.Model): keras model
        new_inp_shape (Tuple): new input shape for model update

    Returns:
        (tf.keras.Model): updated model
    """
    
    conf = model.get_config()
    conf['layers'][0]['config']['batch_shape'] = new_inp_shape
    new_model = model.__class__.from_config(conf, custom_objects={})

    # iterate over all the layers that we want to get weights from
    weights = [layer.get_weights() for layer in model.layers[1:]]
    for layer, weight in zip(new_model.layers[1:], weights):
        layer.set_weights(weight)

    old_inp_shape = model.get_config()['layers'][0]['config']['batch_shape']

    return new_model, old_inp_shape


def get_loss(num_classes: int) -> tf.keras.losses:
    """
    Returns the appropriate loss function based on the number of classes in the dataset.

    Args:
        num_classes (int): The number of classes in the dataset.

    Returns:
        tf.keras.losses: The appropriate loss function based on the number of classes in the dataset.
    """
    # We use the sparse version of the categorical crossentropy because
    # this is what we use to load the dataset.
    if num_classes > 2:
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    else:
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    return loss
