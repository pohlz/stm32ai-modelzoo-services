# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
import numpy as np


def postprocess_and_save(generated_tensor: np.ndarray, diameter: int, sigma_color: int, sigma_space: int,
                alpha: float, beta: float, saturation_scale: float,
                prediction_result_dir: str, file: str) -> None:
    """
    Post-process the predictions and save the results.

    Args:
        generated_tensor (np.ndarray): The generated image with style applied (int8 format).
        diameter (int): The diameter for the bilateral filter.
        sigma_color (int): The sigma color for the bilateral filter.
        sigma_space (int): The sigma space for the bilateral filter.
        alpha (float):  A scaling factor (contrast control).
        beta (float): A value added to each pixel (brightness control).
        saturation_scale (float): The saturation scale.
        prediction_result_dir (str): Directory to save the result image.
        file (str): File name for the result image.

    Returns:
        None
    """
    import cv2
    # Ensure the tensor is squeezed to remove unnecessary dimensions
    generated_image = generated_tensor.squeeze()

    # Convert the int8 tensor to uint8 for OpenCV compatibility
    # Since int8 ranges from -128 to 127, we need to normalize it to 0-255
    generated_image = (generated_image.astype(np.float32) + 128).astype(np.uint8)

    # If the tensor is in CHW format (channels first), transpose it to HWC (channels last)
    if generated_image.ndim == 3 and generated_image.shape[0] in [1, 3]:  # Assuming 1 or 3 channels
        generated_image = generated_image.transpose(1, 2, 0)

    # Apply the bilateral filter
    generated_image = cv2.cvtColor(generated_image, cv2.COLOR_RGB2BGR)
    filtered_image = cv2.bilateralFilter(generated_image, diameter, sigma_color, sigma_space)

    # Remove the color noise from the filtered image
    yuv_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv_image)
    u = cv2.fastNlMeansDenoising(u, None, 5, 7, 21)
    v = cv2.fastNlMeansDenoising(v, None, 5, 7, 21)
    denoised_yuv = cv2.merge([y, u, v])
    denoised_image = cv2.cvtColor(denoised_yuv, cv2.COLOR_YUV2BGR)

    # Increase the saturation of the denoised image
    hsv_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2HSV)
    hsv_image = hsv_image.astype('float32')
    hsv_image[:, :, 1] *= saturation_scale
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1], 0, 255)
    hsv_image = hsv_image.astype('uint8')
    sat_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    # Increase the contrast and generate the final image
    final_image = cv2.convertScaleAbs(sat_image, alpha=alpha, beta=beta)

    # Create the output directory if it doesn't exist
    os.makedirs(prediction_result_dir, exist_ok=True)
    image_name = f"{file}"  # Ensure a valid image extension like .png or .jpg

    # Construct the full output file path
    output_file = os.path.join(prediction_result_dir, image_name)

    # Save the image
    cv2.imwrite(output_file, final_image)
    print(f"Neural style applied to image {file}")