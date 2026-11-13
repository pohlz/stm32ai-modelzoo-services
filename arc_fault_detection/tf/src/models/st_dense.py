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

def get_st_dense_model(input_shape=(4,512,1), num_classes=2):
    """
    Build the dense Arc Fault Detection model.

    Args:
        input_shape (tuple): Model input shape (channels, length, 1).
        num_classes (int): Number of output classes.

    Returns:
        keras.Model: The constructed Keras model.
    """
    inputs = keras.Input(shape=input_shape)
    # reshape to (n_channels, seq_len)
    x = layers.Reshape((input_shape[0], input_shape[1]))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(16, kernel_initializer='random_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.05)(x)
    x = layers.Dense(8, kernel_initializer='random_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.15)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="st_dense")
    return model