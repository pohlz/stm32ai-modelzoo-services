# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
import os
import tensorflow as tf
from typing import Iterator, Tuple

def depth_prediction_dataloader(prediction_path: str, color_mode: str = "rgb") -> Iterator[Tuple[tf.Tensor, str]]:
    """
    Loads images from prediction_path and yields (raw_image, path) pairs for prediction.
    Args:
        prediction_path: Directory containing images to predict.
        color_mode: "rgb" or "grayscale".
    Yields:
        (raw_image, image_path)
    """
    if not os.path.isdir(prediction_path):
        raise ValueError(f"Prediction path '{prediction_path}' does not exist or is not a directory.")
    channels = 1 if color_mode == "grayscale" else 3
    image_files = [f for f in os.listdir(prediction_path)
                  if os.path.isfile(os.path.join(prediction_path, f))]
    if not image_files:
        raise ValueError(f"No valid image files found in prediction path '{prediction_path}'.")
    for fname in image_files:
        img_path = os.path.join(prediction_path, fname)
        try:
            data = tf.io.read_file(img_path)
            img = tf.image.decode_image(data, channels=channels)
            yield img, img_path
        except Exception:
            print(f"[WARN] : Failed to load image '{img_path}'. Skipping.")
            continue