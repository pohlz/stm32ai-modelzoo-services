# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
import os
import numpy as np
import tensorflow as tf
from typing import Optional, Dict, Any, Tuple



def read_image(image_path: str, channels: int) -> np.ndarray:
    """
    Reads an image from a file and converts it to the specified number of channels.

    Args:
        image_path (str): Path to the image file.
        channels (int): Number of channels (e.g., 3 for RGB).

    Returns:
        np.ndarray: The read and converted image.
    """
    import cv2
    img = cv2.imread(image_path)
    if channels != 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def preprocess_image(img: np.ndarray = None, height: int = None, width: int = None, aspect_ratio: str = None,
                     interpolation: str = None, scale: float = None, offset: int = None,
                     perform_scaling: bool = True) -> tf.Tensor:

    """
    Predicts a class for all the images that are inside a given directory.
    The model used for the predictions can be either a .h5 or .tflite file.

    Args:
        img (np.ndarray): image to be prepared
        height (int): height in pixels
        width (int): width in pixels
        aspect_ratio (str): "fit' or "crop"
        interpolation (str): resizing interpolation method
        scale (float): rescaling pixels value
        offset (int): offset value on pixels
        perform_scaling (bool): whether to rescale or not the image

    Returns:
        img_processed (tf.Tensor): the prepared image

    """

    if aspect_ratio == "fit":
        img = tf.image.resize(img, [height, width], method=interpolation, preserve_aspect_ratio=False)
    else:
        img = tf.image.resize_with_crop_or_pad(img, height, width)

    # Rescale the image
    if perform_scaling:
        img_processed = scale * tf.cast(img, tf.float32) + offset
    else:
        img_processed = img

    return img_processed


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


