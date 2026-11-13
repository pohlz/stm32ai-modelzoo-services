# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
import numpy as np
import tensorflow as tf
from matplotlib import gridspec, pyplot as plt
import warnings
from omegaconf import DictConfig
import cv2
from typing import List

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from depth_estimation.tf.src.preprocessing.preprocess import preprocess_image



def postprocess_output_values(output: np.ndarray) -> np.ndarray:
    output = np.squeeze(output)
    output = output.astype(np.float32)
    output = output - np.min(output)
    if np.max(output) > 0:
        output = output / np.max(output)
        output = cv2.GaussianBlur(output, ksize=(3, 3), sigmaX=0)
    return output


def generate_output_image(
    image_path: str = None,
    output: np.ndarray = None,
    cfg: DictConfig = None,
    input_size: List[int] = None,
    output_details: dict = None
) -> None:
    """
    Post-processing to convert raw output to segmentation output and then display input image with segmentation overlay
    """
    # directory for saving prediction outputs
    prediction_result_dir = f'{cfg.output_dir}/predictions/'
    os.makedirs(prediction_result_dir, exist_ok=True)

    # Load the original image
    original_image = tf.io.read_file(image_path)
    original_image = tf.image.decode_image(original_image, channels=3)
    original_image = preprocess_image(
        original_image,
        height=input_size[0],
        width=input_size[1],
        aspect_ratio=cfg.preprocessing.resizing.aspect_ratio,
        interpolation=cfg.preprocessing.resizing.interpolation,
        scale=None,
        offset=None,
        perform_scaling=False
    )
    original_image = original_image.numpy().astype(np.uint8)

    if not cfg.general.display_figures:
        plt.ioff()

    plt.figure(figsize=(20, 10))
    grid_spec = gridspec.GridSpec(1, 2, width_ratios=[10, 10])

    # Plot input image
    plt.subplot(grid_spec[0])
    plt.imshow(original_image)
    plt.axis('off')
    plt.title('Original image')

    plt.subplot(grid_spec[1])
    plt.imshow(output, cmap="plasma")
    plt.axis('off')
    plt.title('Output image')

    # Save figure in the predictions directory
    fig_image_name = os.path.split(image_path)[1]
    pred_res_filename = f'{prediction_result_dir}/{os.path.basename(fig_image_name.split(".")[0])}.png'
    plt.savefig(pred_res_filename, bbox_inches='tight')

    if cfg.general.display_figures:
        plt.waitforbuttonpress()

    plt.close()



