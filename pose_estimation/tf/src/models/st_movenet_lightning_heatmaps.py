# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, UpSampling2D, Activation, Add, BatchNormalization, ReLU, GlobalMaxPool2D
from typing import Any


def _mobileNetV2(shape: tuple = (192, 192, 3), alpha: float = 1.0,
                 pretrained: bool = True, trainable: bool = True) -> Any:
    """
    Creates a MobileNetV2 backbone with optional ImageNet weights and trainability.

    Args:
        shape (tuple): Input shape of the model, default (192, 192, 3).
        alpha (float): Width multiplier for MobileNetV2, default 1.0.
        pretrained (bool): If True, use ImageNet weights; otherwise, random initialization.
        trainable (bool): If False, freeze all layers in the backbone.

    Returns:
        tf.keras.Model: The MobileNetV2 backbone model.
    """
    weights = 'imagenet' if pretrained else None
    backBone = tf.keras.applications.MobileNetV2(weights=weights,
                                                 alpha=alpha, 
                                                 include_top=False, 
                                                 input_shape=shape)

    return backBone


def get_st_movenet_lightning_heatmaps(input_shape: tuple, nb_keypoints: int, alpha: float,
    pretrained: bool, final_activation: str = 'sigmoid', **kwargs) -> Any:
    """
    Builds the ST MoveNet Lightning model with heatmap output for keypoint detection.

    Args:
        input_shape (tuple): Shape of the input tensor.
        nb_keypoints (int): Number of keypoints to predict.
        alpha (float): Width multiplier for MobileNetV2 backbone.
        pretrained (bool): If True, use ImageNet weights for backbone.
        backbone_trainable (bool): If False, freeze backbone layers.
        final_activation (str): Activation function for the output layer ('sigmoid' or 'softmax').

    Returns:
        tf.keras.Model: The full pose estimation model.
    """
    backbone = _mobileNetV2(input_shape,alpha,pretrained)

    conv_0 = Conv2D(24, kernel_size=1, padding='SAME', use_bias=False)(backbone.get_layer(name='block_2_add').output) #index = 19).output) # block_2_add
    conv_1 = Conv2D(32, kernel_size=1, padding='SAME', use_bias=False)(backbone.get_layer(name='block_5_add').output) #index = 37).output) # block_5_add
    conv_2 = Conv2D(64, kernel_size=1, padding='SAME', use_bias=False)(backbone.get_layer(name='block_9_add').output) #index = 61).output) # block_9_add

    conv_0 = BatchNormalization()(conv_0)
    conv_1 = BatchNormalization()(conv_1)
    conv_2 = BatchNormalization()(conv_2)

    x = Conv2D(64, kernel_size=1, padding='SAME', use_bias=False)(backbone.output)
    x = BatchNormalization()(x)
    x = UpSampling2D(size=(2, 2),interpolation='bilinear')(x)
    x = Add()([x,conv_2])

    x = DepthwiseConv2D(kernel_size=3, padding='SAME', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, kernel_size=1, padding='SAME', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = UpSampling2D(size=(2, 2),interpolation='bilinear')(x)
    x = Add()([x,conv_1])

    x = DepthwiseConv2D(kernel_size=3, padding='SAME', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(24, kernel_size=1, padding='SAME', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = UpSampling2D(size=(2, 2),interpolation='bilinear')(x)
    x = Add()([x,conv_0])

    x = DepthwiseConv2D(kernel_size=3, padding='SAME', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(24, kernel_size=1, padding='SAME', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x_kptsHM = DepthwiseConv2D(kernel_size=3, padding='SAME', use_bias=False)(x)
    x_kptsHM = BatchNormalization()(x_kptsHM)
    x_kptsHM = Conv2D(96, kernel_size=1, padding='SAME', use_bias=False)(x_kptsHM)
    x_kptsHM = BatchNormalization()(x_kptsHM)
    x_kptsHM = ReLU()(x_kptsHM)

    x_kptsHM = Conv2D(nb_keypoints, kernel_size=1, padding='SAME', use_bias=False)(x_kptsHM)
    x_kptsHM = BatchNormalization()(x_kptsHM)

    if final_activation=='softmax':
        gmp = GlobalMaxPool2D(keepdims=True)(x_kptsHM)
        x_kptsHM = tf.keras.ops.exp(x_kptsHM-gmp)
        x_kptsHMs = tf.keras.ops.sum(x_kptsHM,axis=[1,2],keepdims=True)
        outputs = x_kptsHM / x_kptsHMs
    elif final_activation=='empty':
        outputs = x_kptsHM
    else:
        outputs  = Activation(final_activation)(x_kptsHM)

    model = Model(inputs=backbone.input, outputs=outputs, name= "st_movenet_lightning_heatmaps")

    return model