# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
from typing import Tuple, Optional
import keras
from keras import layers

def ConvLayer(x, out_channels, kernel_size, stride=1, padding='valid', groups=1):
    x = layers.Conv2D(out_channels, kernel_size, strides=stride, padding=padding,
                      use_bias=False, groups=groups)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def Conv1x1(x, out_channels, stride=1, groups=1):
    x = layers.Conv2D(out_channels, 1, strides=stride, padding='valid',
                      use_bias=False, groups=groups)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def Conv1x1Linear(x, out_channels, stride=1):
    x = layers.Conv2D(out_channels, 1, strides=stride, padding='valid', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    return x

def LightConv3x3(x, out_channels):
    x = layers.Conv2D(out_channels, 1, strides=1, padding='valid', use_bias=False)(x)
    x = layers.DepthwiseConv2D(3, strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def ChannelGate(xa, xb, xc, xd, in_channels, num_gates=None, return_gates=False,
                gate_activation='sigmoid', reduction=16, layer_norm=False):
    inputa, inputb, inputc, inputd = xa, xb, xc, xd
    if num_gates is None:
        num_gates = in_channels

    gb = layers.GlobalAveragePooling2D(keepdims=True)
    cnv1 = layers.Conv2D(in_channels // reduction, 1, padding='valid', use_bias=True)
    rl1 = layers.ReLU()
    cnv2 = layers.Conv2D(num_gates, 1, padding='valid', use_bias=True)
    rl2 = layers.ReLU()
    sg = layers.Activation('sigmoid')

    xa, xb, xc, xd = gb(xa), gb(xb), gb(xc), gb(xd)
    xa, xb, xc, xd = cnv1(xa), cnv1(xb), cnv1(xc), cnv1(xd)
    xa, xb, xc, xd = rl1(xa), rl1(xb), rl1(xc), rl1(xd)
    xa, xb, xc, xd = cnv2(xa), cnv2(xb), cnv2(xc), cnv2(xd)

    if gate_activation == 'sigmoid':
        xa, xb, xc, xd = sg(xa), sg(xb), sg(xc), sg(xd)
    elif gate_activation == 'relu':
        xa, xb, xc, xd = rl2(xa), rl2(xb), rl2(xc), rl2(xd)
    elif gate_activation == 'linear':
        pass
    else:
        raise RuntimeError(f"Unknown gate activation: {gate_activation}")

    if return_gates:
        return xa, xb, xc, xd
    return inputa * xa, inputb * xb, inputc * xc, inputd * xd

def OSBlock(x, in_channels, out_channels, layer_norm=False):
    residual = x
    mid_channels = out_channels // 4

    x = Conv1x1(x, mid_channels)
    xa = LightConv3x3(x, mid_channels)

    xb = LightConv3x3(x, mid_channels)
    xb = LightConv3x3(xb, mid_channels)

    xc = LightConv3x3(x, mid_channels)
    xc = LightConv3x3(xc, mid_channels)
    xc = LightConv3x3(xc, mid_channels)

    xd = LightConv3x3(x, mid_channels)
    xd = LightConv3x3(xd, mid_channels)
    xd = LightConv3x3(xd, mid_channels)
    xd = LightConv3x3(xd, mid_channels)

    xgate_a, xgate_b, xgate_c, xgate_d = ChannelGate(xa, xb, xc, xd, mid_channels)

    xgate = xgate_a + xgate_b + xgate_c + xgate_d
    xout = Conv1x1Linear(xgate, out_channels)

    if in_channels != out_channels:
        residual = Conv1x1Linear(residual, out_channels)

    out = xout + residual
    return layers.ReLU()(out)


def get_osnet(num_classes: int = None, input_shape: Tuple[int, int, int] = None,
                     dropout: Optional[float] = None, alpha: float = 1.0, **kwargs) -> keras.Model:
    """
    Creates a custom image classification model with the given number of classes and input shape.

    Args:
        num_classes (int): Number of classes in the classification task.
        input_shape (Tuple[int, int, int]): Shape of the input image.
        dropout (Optional[float]): Dropout rate to be applied to the model.

    Returns:
        keras.Model: Custom image classification model.
    """
    # Define the input layer
    blocks = [OSBlock, OSBlock, OSBlock]
    num_layers = [2, 2, 2]  # Number of layers in each block
    if alpha == 1.0:
        channels = [64, 256, 384, 512]
    elif alpha == 0.75:
        channels = [48, 192, 288, 384]
    elif alpha == 0.5:
        channels = [32, 128, 192, 256]
    elif alpha == 0.25:
        channels = [16, 64, 96, 128]
    else:
        raise ValueError(f"Unsupported alpha value: {alpha}")
    layer_norm = False

    inputs = keras.Input(shape=input_shape)

    # Stem
    x = ConvLayer(inputs, channels[0], 7, stride=2, padding='same')
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    count = 0
    for block in blocks:
        if block == OSBlock:
            x = OSBlock(x, channels[count], channels[count + 1], layer_norm=layer_norm)
            for _ in range(1, num_layers[count]):
                x = OSBlock(x, channels[count + 1], channels[count + 1], layer_norm=layer_norm)
            if count < 2:
                x = Conv1x1(x, channels[count + 1])
                x = layers.AveragePooling2D(pool_size=2, strides=2)(x)
        count += 1

    x = Conv1x1(x, channels[3])

    # Define the classification layers
    x = layers.GlobalAveragePooling2D()(x)
    if dropout:
        x = layers.Dropout(dropout)(x)
    if num_classes > 2:
        outputs = layers.Dense(num_classes, activation="softmax")(x)
    else:
        outputs = layers.Dense(1, activation="sigmoid")(x)

    # Define and return the model
    model = keras.Model(inputs=inputs, outputs=outputs, name="osnet")
    return model
