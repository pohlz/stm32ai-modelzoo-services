# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import numpy as np
import tensorflow as tf


def preprocess_input(patch: np.ndarray, input_details: dict) -> tf.Tensor:
    """
    Quantizes a patch according to input details.

    Args:
        patch (np.ndarray): Input patch as a NumPy array.
        input_details (dict): Dictionary containing input details, including quantization and dtype.

    Returns:
        tf.Tensor: Quantized patch as a TensorFlow tensor.
    """
    if input_details['dtype'] in [np.uint8, np.int8]:
        patch_processed = (patch / input_details['quantization'][0]) + input_details['quantization'][1]
        patch_processed = np.clip(np.round(patch_processed), np.iinfo(input_details['dtype']).min,
                                  np.iinfo(input_details['dtype']).max)
    else:
        # I would use an actual warning here but they are silenced in the main scripts...
        print("[WARNING] : Quantization dtype isn't one of 'int8', 'uint8'. \n \
               Input patches have not been quantized, this may lead to wrong results.")
        patch_processed = patch
    patch_processed = tf.cast(patch_processed, dtype=input_details['dtype'])
    # Should not need this since we are batching
    # patch_processed = tf.expand_dims(patch_processed, 0)

    return patch_processed


def postprocess_output(output: np.ndarray,
                       multi_label: bool = None,
                       multilabel_threshold: float = 0.5) -> np.ndarray:
    """
    Postprocesses the model output to obtain the predicted label.

    Args:
        output (np.ndarray): The output tensor from the model.
        multi_label (bool, optional): Whether the task is multi-label. Defaults to None.
        multilabel_threshold (float, optional): Threshold for multi-label classification. Defaults to 0.5.

    Returns:
        np.ndarray: The predicted label(s).
    """
    if not multi_label:
        predicted_label = np.argmax(output, axis=1)
    else:
        predicted_label = np.where(output < multilabel_threshold, 0, 1)

    return predicted_label
