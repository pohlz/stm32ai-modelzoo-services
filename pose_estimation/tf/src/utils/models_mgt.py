# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
import tensorflow as tf
from omegaconf import DictConfig
from common.utils import check_model_support, check_attributes


def ai_runner_invoke(image_processed, ai_runner_interpreter):
    """
    Run inference with AI runner interpreter and post-process predictions.

    Args:
        image_processed (np.array):
            NumPy array representing an image batch
        ai_runner_interpreter : Class
            Object exposing an `invoke` method
    Returns:
        predictions (list[np.array]):
            A list of NumPy arrays corresponding to each
            prediction, with shapes reduced by removing middle
            dimensions equal to 1. Each element is a copy of the processed
            prediction tensor.
    """
    def reduce_shape(x):  # reduce shape (request by legacy API)
        old_shape = x.shape
        n_shape = [old_shape[0]]
        for v in x.shape[1:len(x.shape) - 1]:
            if v != 1:
                n_shape.append(v)
        n_shape.append(old_shape[-1])
        return x.reshape(n_shape)

    preds, _ = ai_runner_interpreter.invoke(image_processed)
    predictions = []
    for x in preds:
        x = reduce_shape(x)
        predictions.append(x.copy())
    return predictions


def change_model_number_of_classes(model,num_classes):
    """
    Adapt a Keras model to a new number of output classes.

    This function inspects the output layers of a Keras model and, if needed,
    rebuilds the final part of the network so that the last convolutional or
    dense layer has `num_classes` filters/units. It walks backwards from the
    last layer until it finds a supported layer type, reconstructs that layer
    with the requested number of classes, then re-applies the subsequent
    layers on top of it.

    Supported final trainable layer types:
    - `tf.keras.layers.Conv2D`
    - `tf.keras.layers.Conv2DTranspose`
    - `tf.keras.layers.Conv1D`
    - `tf.keras.layers.Conv1DTranspose`
    - `tf.keras.layers.Dense`

    If the model already has an output dimension equal to `num_classes`, it is
    returned unchanged.

    Args:
        model (tf.keras.Model):
            The original Keras model whose output layer should be adapted.
        num_classes (int):
            Desired number of output classes (size of the last dimension of the
            model output).

    Returns:
        (tf.keras.Model or None)
            A new Keras model with the same architecture as `model` but with the
            final supported layer modified to output `num_classes` channels/units.
            If the model already has the correct number of classes, the original
            `model` is returned. Returns `None` only if no supported layer type
            could be found when walking backwards from the end of the model.
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

            for i,new_l in enumerate(l_list[::-1]):
                outputs = new_l(outputs)

            return tf.keras.Model(inputs=model.input, outputs=outputs, name=model.name)

        else:
            l_list.append(layer_type(**layer_config))
            l-=1

    return None

def change_model_input_shape(model,new_inp_shape):
    """
    Sets the input shape of a Keras model to (None) values while preserving its weights.

    This function rebuilds the model with a new input 
    shape, typically from 
    (None, height, width, channels)
    to 
    (None, None, None, channels)
    and copies the weights of all layers.

    Args:
        model (tf.keras.Model):
            The original Keras model whose input shape should be modified.
        new_inp_shape (tuple):
            New input shape to set in the first layer configuration,
            e.g. `(None, height, width, channels)`.

    Returns:
        (tuple)
            `(new_model, old_inp_shape)` where:
            - `new_model` is a Keras model instance with the updated input shape
              and the same weights.
            - `old_inp_shape` is the original batch input shape taken from the
              first layer's configuration of `model`.
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


