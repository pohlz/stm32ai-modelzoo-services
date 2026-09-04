# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
"""BC-ResNet model for the Audio Event Detection custom-model hook.

This is a TensorFlow/Keras implementation of BC-ResNet from:
    B. Kim et al., "Broadcasted Residual Learning for Efficient Keyword
    Spotting," Interspeech 2021, doi:10.21437/Interspeech.2021-383.

The tensor layout is adapted to the STM32 model-zoo AED pipeline (NHWC, with
frequency before time). The original implementation is available from
https://github.com/Qualcomm-AI-research/bcresnet.
"""

from typing import Optional, Sequence, Tuple

import tensorflow as tf
from tensorflow.keras import layers


_STAGE_REPEATS = (2, 2, 4, 4)
_STRIDED_STAGES = (1, 2)
_SSN_GROUPS = 5


def _scaled_channels(base_channels: int, width_multiplier: float) -> Tuple[int, ...]:
    """Return the six channel widths used by BC-ResNet-tau."""
    base = max(1, int(round(base_channels * width_multiplier)))
    return (
        base * 2,
        base,
        int(base * 1.5),
        base * 2,
        int(base * 2.5),
        base * 4,
    )


def _conv_bn_activation(
    x: tf.Tensor,
    filters: int,
    kernel_size,
    name: str,
    strides=(1, 1),
    dilation_rate=(1, 1),
    groups: int = 1,
    activation: Optional[str] = "relu",
) -> tf.Tensor:
    """Convolution followed by batch normalization and an optional activation."""
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="same",
        dilation_rate=dilation_rate,
        groups=groups,
        use_bias=False,
        name=f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    if activation:
        x = layers.Activation(activation, name=f"{name}_{activation}")(x)
    return x


def _sub_spectral_norm(
    x: tf.Tensor,
    channels: int,
    frequency_bins: int,
    name: str,
    groups: int = _SSN_GROUPS,
) -> tf.Tensor:
    """Apply five-band SubSpectralNorm using only serializable Keras layers.

    SubSpectralNorm folds each frequency sub-band into the channel dimension,
    applies independent BatchNorm statistics/affines, and restores NHWC layout.
    """
    if frequency_bins % groups:
        raise ValueError(
            f"{name}: frequency dimension {frequency_bins} must be divisible "
            f"by the {groups} SubSpectralNorm groups."
        )

    band_bins = frequency_bins // groups
    x = layers.Permute((3, 1, 2), name=f"{name}_to_chw")(x)
    x = layers.Reshape(
        (channels * groups, band_bins, -1), name=f"{name}_fold_bands"
    )(x)
    x = layers.BatchNormalization(axis=1, name=f"{name}_bn")(x)
    x = layers.Reshape(
        (channels, frequency_bins, -1), name=f"{name}_unfold_bands"
    )(x)
    return layers.Permute((2, 3, 1), name=f"{name}_to_hwc")(x)


def _bc_res_block(
    x: tf.Tensor,
    out_channels: int,
    stage_index: int,
    frequency_stride: int,
    dropout: float,
    name: str,
) -> tf.Tensor:
    """Build equation (2) from the paper: x + f2(x) + BC(f1(avg(f2(x))))."""
    in_channels = int(x.shape[-1])
    in_frequency = int(x.shape[1])
    transition = in_channels != out_channels
    shortcut = x

    # A transition changes channels before f2 and intentionally has no identity
    # shortcut, matching the paper and authors' reference implementation.
    if transition:
        x = _conv_bn_activation(
            x, out_channels, 1, name=f"{name}_transition"
        )

    # f2: 3x1 frequency-depthwise convolution plus SubSpectralNorm.
    # TensorFlow's depthwise kernel requires equal spatial strides, so perform
    # the paper's frequency-only downsampling as a separate 1x1 subsampling
    # operation. A 1x1 pool does not combine neighboring frequency values.
    x = layers.DepthwiseConv2D(
        (3, 1),
        strides=(1, 1),
        padding="same",
        use_bias=False,
        name=f"{name}_frequency_dw",
    )(x)
    if frequency_stride == 2:
        x = layers.MaxPooling2D(
            pool_size=(1, 1),
            strides=(2, 1),
            padding="same",
            name=f"{name}_frequency_downsample",
        )(x)
    out_frequency = (in_frequency + frequency_stride - 1) // frequency_stride
    x = _sub_spectral_norm(
        x, out_channels, out_frequency, name=f"{name}_ssn"
    )
    auxiliary_2d_residual = x

    # Average over frequency to create the temporal representation.
    temporal = layers.AveragePooling2D(
        pool_size=(out_frequency, 1), name=f"{name}_frequency_average"
    )(x)

    # f1: dilated 1x3 temporal depthwise convolution, BN, swish, pointwise
    # convolution, and channel-wise dropout.
    temporal = layers.DepthwiseConv2D(
        (1, 3),
        padding="same",
        dilation_rate=(1, 2**stage_index),
        use_bias=False,
        name=f"{name}_temporal_dw",
    )(temporal)
    temporal = layers.BatchNormalization(name=f"{name}_temporal_bn")(temporal)
    temporal = layers.Activation("swish", name=f"{name}_swish")(temporal)
    temporal = layers.Conv2D(
        out_channels, 1, use_bias=False, name=f"{name}_pointwise"
    )(temporal)
    if dropout:
        temporal = layers.SpatialDropout2D(
            dropout, name=f"{name}_channel_dropout"
        )(temporal)

    # Explicit expansion avoids relying on implicit broadcasting during export.
    temporal = layers.UpSampling2D(
        size=(out_frequency, 1), interpolation="nearest", name=f"{name}_broadcast"
    )(temporal)
    residuals = [auxiliary_2d_residual, temporal]
    if not transition:
        residuals.insert(0, shortcut)
    x = layers.Add(name=f"{name}_add")(residuals)
    return layers.ReLU(name=f"{name}_out")(x)


def get_custom_model(
    num_classes: int = None,
    input_shape: Tuple[int, int, int] = None,
    dropout: Optional[float] = 0.1,
    use_garbage_class: bool = False,
    multi_label: bool = False,
    base_channels: int = 8,
    width_multiplier: float = 1.0,
    stage_repeats: Sequence[int] = _STAGE_REPEATS,
    **kwargs,
) -> tf.keras.Model:
    """Create BC-ResNet-1 (or a width-scaled BC-ResNet-tau).

    The paper's architecture consumes 40-bin log-mel patches in STM32 AED's
    ``(frequency, time, channel)`` layout. Time length may vary, but must be
    static for the current model-zoo training/export flow.

    Args:
        num_classes: Dataset class count before an optional garbage class.
        input_shape: Expected to be ``(40, time_frames, 1)``.
        dropout: Channel-wise dropout in every BC-ResBlock (paper default 0.1).
        use_garbage_class: Add one output for the AED garbage class.
        multi_label: Use sigmoid outputs when true. Note that the current AED
            trainer still rejects multi-label loss.
        base_channels: BC-ResNet-1 base width (paper default 8).
        width_multiplier: Width scaling coefficient tau.
        stage_repeats: Blocks in the four stages (paper default 2, 2, 4, 4).

    Returns:
        An uncompiled ``tf.keras.Model`` for the ST AED trainer.
    """
    if num_classes is None or int(num_classes) < 1:
        raise ValueError("num_classes must be a positive integer")
    if input_shape is None or len(input_shape) != 3:
        raise ValueError("input_shape must be (40, time_frames, 1)")

    input_shape = tuple(int(dim) for dim in input_shape)
    if input_shape[0] != 40:
        raise ValueError(
            "The paper-defined BC-ResNet requires 40 mel bins. Set "
            "model.input_shape[0] and feature_extraction.n_mels to 40."
        )
    if input_shape[2] != 1:
        raise ValueError("BC-ResNet expects one log-mel input channel")
    if len(stage_repeats) != 4 or any(int(n) < 1 for n in stage_repeats):
        raise ValueError("stage_repeats must contain four positive integers")
    if base_channels < 1 or width_multiplier <= 0:
        raise ValueError("base_channels and width_multiplier must be positive")

    dropout = 0.1 if dropout is None else float(dropout)
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in the range [0, 1)")

    channels = _scaled_channels(base_channels, width_multiplier)
    inputs = tf.keras.Input(shape=input_shape, name="log_mel_patch")

    # Front-end 5x5 convolution: 1x40xW -> 16x20xW for BC-ResNet-1.
    x = _conv_bn_activation(
        inputs,
        channels[0],
        5,
        strides=(2, 1),
        name="stem",
    )

    # Four stages with [2, 2, 4, 4] blocks. Stages 2 and 3 downsample only
    # frequency; temporal dilation is [1, 2, 4, 8].
    for stage_index, (repeats, out_channels) in enumerate(
        zip(stage_repeats, channels[1:5])
    ):
        for block_index in range(int(repeats)):
            stride = 2 if stage_index in _STRIDED_STAGES and block_index == 0 else 1
            x = _bc_res_block(
                x,
                out_channels=out_channels,
                stage_index=stage_index,
                frequency_stride=stride,
                dropout=dropout,
                name=f"stage{stage_index + 1}_block{block_index + 1}",
            )

    # The authors' released architecture uses a 5x5 depthwise classifier
    # convolution with valid frequency padding and same temporal padding.
    x = layers.ZeroPadding2D(padding=((0, 0), (2, 2)), name="classifier_time_pad")(x)
    x = layers.DepthwiseConv2D(
        (5, 5), padding="valid", use_bias=False, name="classifier_depthwise"
    )(x)
    x = _conv_bn_activation(
        x, channels[-1], 1, name="classifier_projection"
    )
    x = layers.GlobalAveragePooling2D(name="global_average")(x)

    effective_classes = int(num_classes) + int(bool(use_garbage_class))
    activation = "sigmoid" if multi_label else "softmax"
    outputs = layers.Dense(
        effective_classes, activation=activation, name="predictions"
    )(x)

    return tf.keras.Model(inputs, outputs, name="bc_resnet_1")
