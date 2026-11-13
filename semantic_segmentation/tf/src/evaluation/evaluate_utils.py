# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
import warnings
import numpy as np

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def prediction_accuracy_on_batch(pred_mask: np.ndarray = None, true_mask: np.ndarray = None) -> float:
    """
        Evaluation of the prediction accuracy on a batch of network outputs.

        Parameters:
        pred_mask (np.array): batch of network outputs, containing on each pixel the detected class id
        true_mask (np.array): batch of corresponding ground truth

        Returns:
        accuracy_on_batch (float): estimated accuracy on batch.
    """
    accuracy_on_batch = np.mean((pred_mask == true_mask).astype(np.float32))

    return accuracy_on_batch


def iou_per_class(pred_mask: np.ndarray = None, true_mask: np.ndarray = None, num_classes: int = None) -> list:
    """
        Evaluation of IOU per class on a batch of network outputs.

        Parameters:
        pred_mask (np.array): batch of network outputs, containing on each pixel the detected class id
        true_mask (np.array): batch of corresponding ground truth
        num_classes (int): number of classes in the dataset

        Returns:
        ious_on_batch_class (list): estimated accuracy on batch.
    """

    ious_on_batch_class = []

    # Calculate IoU for each class
    for class_id in range(num_classes):
        true_class = true_mask == class_id
        pred_class = pred_mask == class_id
        intersection = np.logical_and(true_class, pred_class)
        union = np.logical_or(true_class, pred_class)
        union_sum = np.sum(union)

        iou_class = np.sum(intersection) / union_sum if union_sum > 0 else 0.
        # IoU makes only sense if there actually are occurrences of class_id in the image
        if np.sum(true_class) != 0:
            ious_on_batch_class.append(iou_class)

    return ious_on_batch_class