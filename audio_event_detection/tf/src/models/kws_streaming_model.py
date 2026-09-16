# Copyright 2026 The Google Research Authors.
# Licensed under the Apache License, Version 2.0; see kws_streaming/upstream/LICENSE.
# Modified for STM AED: functional Keras graph, external log-mel input, NHWC adapter.
"""Google kws_streaming BC-ResNet, non-streaming ('same') configuration.

Derived from google-research/kws_streaming/models/bc_resnet.py and
layers/sub_spectral_normalization.py at revision
08a8d6736475776f42ffac23b2c13111a28e5795. Pristine sources are in
kws_streaming/upstream/. See docs/README_BCRESNET_TF.md for adaptation details.
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers


def _ssn(x, name, groups=5):
    # Google folds the contiguous frequency/channel axes directly. Preserve
    # its exact reshape order (different from the previous implementation).
    time, freq, channels = (int(d) for d in x.shape[1:])
    if freq % groups:
        raise ValueError(f"Frequency {freq} must be divisible by SSN groups={groups}")
    x = layers.Reshape((time, freq // groups, channels * groups), name=name + "_fold")(x)
    x = layers.BatchNormalization(name=name + "_bn")(x)
    return layers.Reshape((time, freq, channels), name=name + "_unfold")(x)


def _block(x, filters, dilation, stride, dropout, name, transition):
    identity = x
    if transition:
        x = layers.Conv2D(filters, 1, strides=stride, padding="valid", use_bias=False,
                          name=name + "_transition")(x)
        x = layers.BatchNormalization(name=name + "_transition_bn")(x)
        x = layers.ReLU(name=name + "_transition_relu")(x)
    x = layers.DepthwiseConv2D((1, 3), dilation_rate=(1,1), padding="same",
                               use_bias=False, name=name + "_frequency_dw")(x)
    x = _ssn(x, name + "_ssn")
    residual = x
#    x = layers.AveragePooling2D((1, int(x.shape[2])), name=name + "_frequency_mean")(x)
#   Start of averagepooling2d replacement
    n = int(x.shape[2])
    x = layers.DepthwiseConv2D(
        kernel_size=(1, n),
        strides=(1, 1),
        padding="valid",
        depth_multiplier=1,
        use_bias=False,
        depthwise_initializer=tf.keras.initializers.Constant(1.0 / n),
        trainable=False,
        name=name + "_frequency_mean_dw",
    )(x)

#   End of averagepooling2d replacement
    x = layers.DepthwiseConv2D((3, 1), dilation_rate=dilation, padding="same",
                               use_bias=False, name=name + "_temporal_dw")(x)
    x = layers.BatchNormalization(name=name + "_temporal_bn")(x)
    x = layers.Activation("swish", name=name + "_swish")(x)
    x = layers.Conv2D(filters, 1, padding="valid" if transition else "same",
                      use_bias=False, name=name + "_pointwise")(x)
    x = layers.SpatialDropout2D(dropout, name=name + "_dropout")(x)
    if not transition:
        # Keep the full-size tensor first and the frequency-broadcast tensor
        # second. ADD is commutative and this ordering matches Neural-ART's
        # supported unidirectional broadcasting convention.
        x = layers.Add(name=name + "_identity_add")([identity, x])
    # In a transition block, this ADD performs the frequency broadcast. In a
    # normal block, x was already expanded to the residual tensor's shape by
    # the identity ADD above.
    x = layers.Add(name=name + "_residual_add")([residual, x])
    return layers.ReLU(name=name + "_relu")(x)


def get_custom_model(num_classes=None, input_shape=None, dropout=0.1,
                     use_garbage_class=False, multi_label=False, tau=8, **kwargs):
    """Build Google's BC-ResNet-tau defaults for STM (frequency,time,channel).

    blocks_n counts NORMAL blocks, plus one transition per stage. Dropout is
    supplied by the training config. Softmax matches STM's cross-entropy loss.
    """
    if tau not in (1, 1.5, 2, 3, 6, 8):
        raise ValueError("tau must be one of: 1, 1.5, 2, 3, 6, 8")
    base_c = int(8 * tau)
    stem_channels = base_c * 2
    stage_channels = (
        base_c,
        int(base_c * 1.5),
        base_c * 2,
        int(base_c * 2.5),
    )
    classifier_channels = base_c * 4


    if num_classes is None or int(num_classes) < 1:
        raise ValueError("num_classes must be positive")
    if input_shape is None or len(input_shape) != 3:
        raise ValueError("input_shape must be (40, time_frames, 1)")
    shape = tuple(int(d) for d in input_shape)
    if shape[0] != 40 or shape[1] < 1 or shape[2] != 1:
        raise ValueError("input_shape must be (40, positive_time_frames, 1)")
    dropout = 0.1 if dropout is None else float(dropout)
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    inputs = layers.Input(shape=shape, name="log_mel_patch")
    x = layers.Permute((2, 1, 3), name="stm_to_kws_time_frequency")(inputs)
    # Upstream stem has bias, no batch normalization and no activation.
    x = layers.Conv2D(stem_channels, 5, strides=(1, 2), padding="same", name="stem")(x)
    for stage, (n, filters, dilation, stride) in enumerate(zip(
            (2, 2, 4, 4), stage_channels,
            ((1, 1), (2, 1), (3, 1), (3, 1)),
            ((1, 1), (1, 2), (1, 2), (1, 1))), 1):
   # for stage, (n, filters, dilation, stride) in enumerate(zip(
   #          (2, 2, 4, 4),
   #          (8, 12, 16, 20),
   #          ((1, 1), (1, 1), (1, 1), (1, 1)),
   #          ((1, 1), (1, 2), (1, 2), (1, 1))), 1):
        x = _block(x, filters, dilation, stride, dropout, f"stage{stage}_transition", True)
        for block in range(n):
            x = _block(x, filters, dilation, (1, 1), dropout,
                       f"stage{stage}_normal{block + 1}", False)
    x = layers.DepthwiseConv2D(5, padding="same", name="classifier_dw")(x)
    n = int(x.shape[2])
    x = layers.DepthwiseConv2D(
        kernel_size=(1, n),
        strides=(1, 1),
        padding="valid",
        depth_multiplier=1,
        use_bias=False,
        depthwise_initializer=tf.keras.initializers.Constant(1.0 / n),
        trainable=False,
        name="classifier_frequency_mean_dw",
    )(x)
    x = layers.Conv2D(classifier_channels, 1, use_bias=False, name="classifier_projection")(x)
    x = layers.GlobalAveragePooling2D(keepdims=True, name="classifier_time_mean")(x)
    classes = int(num_classes) + int(bool(use_garbage_class))
    x = layers.Conv2D(classes, 1, use_bias=False, name="classifier_logits")(x)
    x = layers.Reshape((classes,), name="flatten_logits")(x)
    outputs = layers.Activation("sigmoid" if multi_label else "softmax", name="predictions")(x)
    return tf.keras.Model(inputs, outputs, name=f"kws_streaming_bc_resnet_{tau:g}")
