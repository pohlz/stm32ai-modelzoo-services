# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import numpy as np
import tensorflow as tf
from typing import Optional, Dict, Any

def preprocess_image(img: np.ndarray, height: int, width: int, aspect_ratio: Optional[str] = None,
                     interpolation: Optional[str] = None, scale: float = 1.0, offset: int = 0,
                     ) -> tf.Tensor:
    """
    Prepares an image for model input.

    Args:
        img (np.ndarray): Image to be prepared.
        height (int): Height in pixels.
        width (int): Width in pixels.
        aspect_ratio (Optional[str]): "fit" or "crop".
        interpolation (Optional[str]): Resizing interpolation method.
        scale (float): Rescaling pixels value.
        offset (int): Offset value on pixels.
        perform_scaling (bool): Whether to rescale or not the image.

    Returns:
        tf.Tensor: The prepared image.
    """
    print(height, width)
    if aspect_ratio == "fit":
        img = tf.image.resize(img, [height, width])
    else:
        img = tf.image.resize_with_crop_or_pad(img, height, width)

    return img


def preprocess_input(image: np.ndarray, input_details: Optional[Dict[str, Any]]) -> tf.Tensor:
    """
    Preprocesses an input image according to input details.

    Args:
        image (np.ndarray): Input image as a NumPy array.
        input_details (Optional[Dict[str, Any]]): Dictionary containing input details, including quantization and dtype.

    Returns:
        tf.Tensor: Preprocessed image as a TensorFlow tensor.
    """

    if input_details is not None:
        if input_details['dtype'] in [np.uint8, np.int8]:
            image_processed = (image / input_details['quantization'][0]) + input_details['quantization'][1]
            image_processed = np.clip(
                np.round(image_processed), np.iinfo(input_details['dtype']).min,
                np.iinfo(input_details['dtype']).max
            )
        else:
            image_processed = image
        image_processed = tf.cast(image_processed, dtype=input_details['dtype'])
    else:
        image_processed = image

    image_processed = tf.expand_dims(image_processed, 0)

    return image_processed
