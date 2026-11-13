# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
import numpy as np
from typing import Tuple, List, Optional

class PaletteManager:
    def __init__(self) -> None:
        """
        Initialize the PaletteManager with a set of predefined hex color values and a pose palette.
        """
        hex_codes = (
            "042AFF", "0BDBEB", "F3F3F3", "00DFB7", "111F68", "FF6FDD", "FF444F",
            "CCED00", "00F344", "BD00FF", "00B4FF", "DD00BA", "00FFFF", "26C000",
            "01FFB3", "7D24FF", "7B0068", "FF1B6C", "FC6D2F", "A2FF0B"
        )
        self.colors: List[Tuple[int, int, int]] = [self._hex_to_rgb(f"#{code}") for code in hex_codes]
        self.color_count: int = len(self.colors)
        self.pose_colors: np.ndarray = np.array(
            [
                [255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0],
                [255, 153, 255], [153, 204, 255], [255, 102, 255], [255, 51, 255],
                [102, 178, 255], [51, 153, 255], [255, 153, 153], [255, 102, 102],
                [255, 51, 51], [153, 255, 153], [102, 255, 102], [51, 255, 51],
                [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]
            ],
            dtype=np.uint8,
        )

    def get_color(self, index: int, as_bgr: bool = False) -> Tuple[int, int, int]:
        """
        Retrieve a color from the palette based on the provided index.

        Args:
            index (int): The index of the desired color in the palette.
            as_bgr (bool): If True, return the color in BGR format. Default is False.

        Returns:
            Tuple[int, int, int]: The color in RGB or BGR format.
        """
        color = self.colors[index % self.color_count]
        return (color[2], color[1], color[0]) if as_bgr else color

    @staticmethod
    def _hex_to_rgb(hex_code: str) -> Tuple[int, int, int]:
        """
        Convert a hex color code to an RGB tuple.

        Args:
            hex_code (str): The hex color code.

        Returns:
            Tuple[int, int, int]: The RGB representation of the color.
        """
        return tuple(int(hex_code[i:i+2], 16) for i in (1, 3, 5))

    def get_pose_color(self, index: int) -> Tuple[int, int, int]:
        """
        Retrieve a pose color from the pose palette based on the provided index.

        Args:
            index (int): The index of the desired pose color in the pose palette.

        Returns:
            Tuple[int, int, int]: The pose color in RGB format.
        """
        return tuple(self.pose_colors[index % len(self.pose_colors)])

    def add_custom_color(self, hex_code: str) -> None:
        """
        Add a custom color to the palette.

        Args:
            hex_code (str): The hex color code of the custom color to add.
        """
        self.colors.append(self._hex_to_rgb(hex_code))
        self.color_count = len(self.colors)


def _load_classes(file_path: str) -> List[str]:
    """
    Load class names from a file.

    Args:
        file_path (str): Path to the file containing class names.

    Returns:
        List[str]: List of class names.
    """
    with open(file_path, 'r') as f:
        classes = f.read().strip().split('\n')
    return classes

def custom_draw(image: np.ndarray, boxes: List[Tuple[float, float, float, float, float, float]], masks: List[np.ndarray],
                color_palette: callable, prediction_result_dir: str, file: str, class_file: Optional[str] = None) -> None:
    """
    Draws bounding boxes and masks on an image and saves the result.

    Args:
        image (np.ndarray): The input image.
        boxes (List[Tuple[float, float, float, float, float, float]]): List of bounding boxes with confidence and class.
        masks (List[np.ndarray]): List of masks corresponding to the bounding boxes.
        color_palette (callable): Function to get color for a class.
        prediction_result_dir (str): Directory to save the result image.
        file (str): File name for the result image.
        class_file (Optional[str]): File containing class names. If None, use class indices.

    Returns:
        None
    """
    import cv2
    # Load class names if class_file is provided
    class_names = _load_classes(class_file) if class_file else None

    overlay_image = image.copy()
    for (*box, conf, cls_), mask in zip(boxes, masks):
        color = color_palette(int(cls_), as_bgr=True)
        class_label = class_names[int(cls_)] if class_names else str(int(cls_))
        for c in range(3):
            overlay_image[:, :, c] = np.where(mask == 1, overlay_image[:, :, c] * 0.5 + color[c] * 0.5, overlay_image[:, :, c])
        cv2.rectangle(overlay_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 1, cv2.LINE_AA)
        cv2.putText(overlay_image, f"{class_label}: {conf:.3f}", (int(box[0]), int(box[1] - 9)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    
    overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
    
    # Create the output directory if it doesn't exist
    os.makedirs(prediction_result_dir, exist_ok=True)
    image_name = os.path.basename(file)
    output_file = os.path.join(prediction_result_dir, image_name)
    
    # Save the image
    success = cv2.imwrite(output_file, overlay_image)
    if success:
        print(f"Prediction successfully saved to {os.path.abspath(output_file)}")
    else:
        print(f"Failed to save prediction to {os.path.abspath(output_file)}")

def multiply_tensors(masks_in: np.ndarray, reshaped_protos: np.ndarray) -> np.ndarray:
    """
    Multiplies two tensors manually.

    Args:
        masks_in (np.ndarray): First tensor.
        reshaped_protos (np.ndarray): Second tensor.

    Returns:
        np.ndarray: Result of the multiplication.
    """
    # Initialize the result tensor with zeros
    result = np.zeros((masks_in.shape[0], reshaped_protos.shape[1]))

    # Perform matrix multiplication manually
    for i in range(masks_in.shape[0]):  # Iterate over rows of masks_in
        for j in range(reshaped_protos.shape[1]):  # Iterate over columns of reshaped_protos
            for k in range(masks_in.shape[1]):  # Iterate over columns of masks_in / rows of reshaped_protos
                result[i, j] += masks_in[i, k] * reshaped_protos[k, j]

    return result


def cxcywh_to_xyxy(x: np.ndarray) -> np.ndarray:
    """
    Converts bounding boxes from cxcywh format to xyxy format.

    Args:
        x (np.ndarray): Bounding boxes in cxcywh format.

    Returns:
        np.ndarray: Bounding boxes in xyxy format.
    """
    if len(x) > 0:
        # Convert center x, center y, width, height to x_min, y_min, x_max, y_max
        x[..., [0, 1]] -= x[..., [2, 3]] / 2
        x[..., [2, 3]] += x[..., [0, 1]]
    return x
