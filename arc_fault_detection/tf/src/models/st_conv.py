# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

from tensorflow import keras
from tensorflow.keras import layers

def get_st_conv(input_shape: tuple[int] = (4,256, 1), num_classes: int = 2):
    """
    Build the convolutional Arc Fault Detection model.

    Args:
        input_shape (tuple[int]): Model input shape (channels, length, 1).
        num_classes (int): Number of output classes.

    Returns:
        keras.Model: The constructed Keras model.
    """

    inputs = keras.Input(shape=input_shape)  # (batch, n_channels, seq, 1)
    x = inputs
    conv_layers = ((6, 8), (6, 8), (6, 8), (6, 8))
    for filters, kernel_size in conv_layers:
        x = layers.Conv2D(filters, (1,kernel_size), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.1)(x)

    # x: (batch, n_channels, seq_len, filters)
    x = layers.Reshape((input_shape[0],-1))(x) # (batch, n_channels, seq_len * filters)

    x = layers.Dense(16, kernel_initializer='random_uniform')(x)  # (batch, n_channels, 16)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.05)(x)

    x = layers.Dense(8, kernel_initializer='random_uniform')(x)   # (batch, n_channels, 8)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.15)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)  # (batch, n_channels, num_classes)
    model = keras.Model(inputs=inputs, outputs=outputs, name="st_conv")
    return model