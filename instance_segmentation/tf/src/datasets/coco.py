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
from typing import Tuple
from instance_segmentation.tf.src.preprocessing.preprocess import preprocess_image, read_image


def load_coco(prediction_path: dict = None,
                    scale: float = None, 
                    offset: float = None,
                    image_size: tuple = None,
                    aspect_ratio: str = None,
                    interpolation: str = None
                    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:

    image_files = [os.path.join(prediction_path, f) for f in os.listdir(prediction_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    processed_predict = []
    for image_path in image_files:
        img = read_image(image_path=image_path, channels=3)
        img_processed = preprocess_image(
            img=img,
            height=image_size[0],
            width=image_size[1],
            aspect_ratio=aspect_ratio,
            interpolation=interpolation,
            scale=scale,
            offset=offset,
            perform_scaling=True
        )
        processed_predict.append((img_processed, image_path))
    return {'train': None, 'valid': None, 'quantization': None, 'test': None, 'predict': processed_predict}