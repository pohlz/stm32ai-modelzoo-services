# /*---------------------------------------------------------------------------------------------
#  * Copyright 2015 The TensorFlow Authors.
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/


from typing import Tuple
import keras
from keras import layers
from keras.src.applications import imagenet_utils
from keras.applications import MobileNet


def get_mobilenetv1(input_shape: tuple, alpha: float = None, num_classes: int = None, 
                    dropout: float = None, pretrained: bool = False, **kwargs) -> keras.Model:
    """
    Returns a MobileNetV1 model with a custom classifier.

    Args:
        input_shape (tuple): The shape of the input tensor.
        alpha (float, optional): The width multiplier for the MobileNetV1 backbone. Defaults to None.
        dropout (float, optional): The dropout rate for the custom classifier. Defaults to 1e-6.
        num_classes (int, optional): The number of output classes. Defaults to None.
        pretrained_weights (str, optional): The pre-trained weights to use. Either "imagenet"
        or None. Defaults to "imagenet".

    Returns:
        keras.Model: The MobileNetV1 model with a custom classifier.
    """
    
    # commented because no longer used later-on
#    if pretrained_weights:
#        training = False
#        # model is set in inference mode so that moving avg and var of any BN are kept untouched
#        # should help the convergence according to Keras tutorial
#    else:
#        training = True
    pretrained_weights = "imagenet" if pretrained else None
    # fetch the backbone pre-trained on imagenet or random
    base_model = MobileNet(
        include_top=False,
        weights=pretrained_weights,
        input_tensor=None,
        input_shape=input_shape,
        pooling="avg",
        alpha=alpha, 
        classes=num_classes,
    )

    if dropout:
        x = layers.Dropout(rate=dropout, name="dropout")(base_model.output)
        if num_classes > 2:
            outputs = layers.Dense(num_classes, activation="softmax")(x)
        else:
            outputs = layers.Dense(1, activation="sigmoid")(x)
    else:
        if num_classes > 2:
            outputs = layers.Dense(num_classes, activation="softmax")(base_model.output)
        else:
            outputs = layers.Dense(1, activation="sigmoid")(base_model.output)
    
    # Create the Keras model
    model = keras.Model(inputs=base_model.input, outputs=outputs, name="mobilenet_v1_alpha_{}".format(alpha))

    return model

