# /*---------------------------------------------------------------------------------------------
#  * Copyright 2018 The TensorFlow Authors.
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras import layers, models, backend
from common.training import set_all_layers_trainable_parameter


def _calculate_dilation_rates(height):
    """
    Calculate dilation rates based on the height of the input tensor.
    """
    dilation_rate1 = height // 5
    dilation_rate2 = dilation_rate1 * 2
    dilation_rate3 = dilation_rate1 * 3
    return [dilation_rate1, dilation_rate2, dilation_rate3]

def _dilated_conv2d(x, out_filters, dilation_rate):
    """
    Apply dilated Conv2D with BatchNormalization and ReLU activation.
    """
    x = layers.Conv2D(out_filters, (3, 3), padding='same', dilation_rate=(dilation_rate, dilation_rate), kernel_initializer='he_normal', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def _dilated_depthwise_conv2d(x, out_filters, dilation_rate):
    """
    Apply dilated DepthwiseConv2D with BatchNormalization and ReLU activation, followed by a pointwise Conv2D.
    """
    x = layers.DepthwiseConv2D((3, 3), padding='same', dilation_rate=(dilation_rate, dilation_rate), depthwise_initializer='he_normal', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(out_filters, (1, 1), padding='same', kernel_initializer='he_normal', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def _aspp(x, out_filters, aspp_size, version='v1'):
    """
    ASPP module with either standard or depthwise separable atrous convolutions.
    """
    # Extract input resolution
    _, height, width, channels = backend.int_shape(x)

    # Calculate dilation rates based on height
    dilation_rates = _calculate_dilation_rates(height)

    # Start with a 1x1 convolution
    x1 = layers.Conv2D(out_filters, (1, 1), padding='same', kernel_initializer='he_normal', use_bias=False)(x)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('relu')(x1)

    if version == 'v1':
        x2 = _dilated_conv2d(x, out_filters, dilation_rates[0])
        x3 = _dilated_conv2d(x, out_filters, dilation_rates[1])
        x4 = _dilated_conv2d(x, out_filters, dilation_rates[2])
    elif version == 'v2':
        x2 = _dilated_depthwise_conv2d(x, out_filters, dilation_rates[0])
        x3 = _dilated_depthwise_conv2d(x, out_filters, dilation_rates[1])
        x4 = _dilated_depthwise_conv2d(x, out_filters, dilation_rates[2])
    else:
        raise ValueError(f"Unsupported ASPP version {version}")

    # Apply global average pooling and upsample to match the feature map size
    img_pool = layers.GlobalAveragePooling2D()(x)
    img_pool = layers.Reshape((1, 1, channels))(img_pool)
    img_pool = layers.Conv2D(out_filters, (1, 1), padding='same', kernel_initializer='he_normal', use_bias=False)(img_pool)
    img_pool = layers.BatchNormalization()(img_pool)
    img_pool = layers.Activation('relu')(img_pool)
    # img_pool = layers.UpSampling2D(size=aspp_size, interpolation='bilinear')(img_pool)

    # Concatenate the atrous features
    x = layers.Concatenate()([x1, x2, x3, x4])

    # Apply another 1x1 convolution and batch normalization to refine the features
    # To be independant of the input-resolution, we need to have 2 separate Conv2D
    # If we add their results, it is mathematically equivalent to a single Conv2D
    x          = layers.Conv2D(out_filters, (1, 1), padding='same', kernel_initializer='he_normal', use_bias=False)(x)
    x_img_pool = layers.Conv2D(out_filters, (1, 1), padding='same', kernel_initializer='he_normal', use_bias=False)(img_pool)

    # BatchNormalization is shared for thoses 2 Conv2D
    shared_bn = layers.BatchNormalization()

    x          = shared_bn(x)
    x_img_pool = shared_bn(x_img_pool)

    # Add the results before applying the non-linearity
    x = layers.Add()([x,x_img_pool])
    x = layers.Activation('relu')(x)

    return x

def get_deeplabv3(input_shape: list = None, backbone: str = None, version: str = None, alpha: float = None,
                   dropout: float = None, pretrained: bool = None, num_classes: int = None,
                   output_stride: int = 16, aspp_version: str = 'v1', **kwargs) -> tf.keras.Model:
    """
    Constructs a DeepLabV3 model with a specified backbone.

    Parameters:
    - input_shape (list): The shape of the input data as a list, e.g., [224, 224, 3].
    - backbone (str): The backbone network to use, e.g., 'mobilenet'.
    - version (str): The version of the backbone, e.g., 'v2' for MobileNetV2.
    - alpha (float): The width multiplier for the MobileNetV2 (controls the width of the network).
    - dropout (float): The dropout rate to use after the ASPP (Atrous Spatial Pyramid Pooling) layer.
    - pretrained_weights (str): The path to the pre-trained weights file, or 'imagenet' for ImageNet weights.
    - num_classes (int): The number of classes for the output layer.
    - output_stride (int): The output stride for the network. Default is 16, but other values like 8 can be used for denser feature extraction.
    - aspp_version (str): The version of ASPP to use ('v1' or 'v2').

    Returns:
    - tf.keras.Model: A Keras model instance of DeepLabV3 with the specified configuration.
    """
    # Define the input layer with the given input shape
    inputs = layers.Input(shape=input_shape)
    pretrained_weights = 'imagenet' if pretrained else None
    # Load the MobileNetV2 or ResNet50 model with pre-trained weights on ImageNet
    if backbone == "mobilenet" and version == "v2":
        base_model = MobileNetV2(input_shape=input_shape, alpha=alpha, include_top=False, weights=pretrained_weights,
                                 input_tensor=inputs)
        if output_stride == 8:
            x = base_model.get_layer('block_6_expand_relu').output
        else:
            x = base_model.get_layer('block_13_expand_relu').output
    elif backbone == "resnet50" and version == "v1":
        base_model = ResNet50(input_shape=input_shape, include_top=False, weights=pretrained_weights,
                              input_tensor=inputs)
        if output_stride == 8:
            x = base_model.get_layer('conv3_block4_out').output
        else:
            x = base_model.get_layer('conv4_block6_out').output
    else:
        raise ValueError(f"Unsupported backbone {backbone} with version {version}")

    # Set all layers in the base model to be trainable
    set_all_layers_trainable_parameter(base_model, trainable=True)

    # Get the shape of the current feature map for the ASPP pooling size
    _, h, w, _ = backend.int_shape(x)
    aspp_size = (h, w)

    # Apply ASPP with the given feature map size and a fixed number of filters (256)
    x = _aspp(x, 256, aspp_size, version=aspp_version)

    # Apply dropout with the given rate
    x = layers.Dropout(dropout)(x)

    # Add a convolutional layer with `num_classes` filters for classification
    x = layers.Conv2D(num_classes, 1, strides=1, kernel_initializer='he_normal')(x)

    # Upsample the feature map to the original size of the input
    x = layers.UpSampling2D(size=(output_stride, output_stride), interpolation='bilinear')(x)

    # Create the model with inputs and outputs
    model = models.Model(inputs, x)

    return model