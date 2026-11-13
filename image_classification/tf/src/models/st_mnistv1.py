# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import keras
from keras import layers
from typing import Tuple


def get_st_mnistv1(num_classes: int = None, input_shape: Tuple[int, int, int] = None,
                dropout: float = None, pretrained: bool = False, **kwargs) -> keras.models.Model:
    """
    Returns a stmnist model for mnist type datasets.

    Args:
    - num_classes: integer, the number of output classes.
    - input_shape: tuple of integers, the shape of the input tensor (height, width, channels).

    Returns:
    - keras.models.Model object, the stmnist model.

    """
    
    if pretrained:
      print("WARNING: No pretrained weights are found for 'stmnist' model. Random weights are used instead.")

    # Define the input tensor
    inputs = keras.Input(shape=input_shape)

    # Define the number of filters
    num_filters = 16

    # Block 1
    x = layers.Conv2D(num_filters, kernel_size=3, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Block 2
    num_filters = 2 * num_filters
    x = layers.Conv2D(num_filters, kernel_size=3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.DepthwiseConv2D(kernel_size=3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Block 3
    num_filters = 2 * num_filters
    x = layers.Conv2D(num_filters, kernel_size=1, strides=1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Global average pooling layer
    x = layers.GlobalAveragePooling2D()(x)

    if dropout:
        x = layers.Dropout(rate=dropout, name="dropout")(x)

    # Output layer
    x = layers.Dense(num_classes, activation="softmax")(x)

    # Create the model
    model = keras.models.Model(inputs, x, name="st_mnistv1")

    return model
