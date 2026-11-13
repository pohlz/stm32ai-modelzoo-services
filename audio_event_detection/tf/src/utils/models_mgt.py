#  /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import tensorflow as tf
from audio_event_detection.tf.src.data_augmentation import SpecAugment, VolumeAugment
AED_CUSTOM_OBJECTS={'SpecAugment': SpecAugment,
                    'VolumeAugment': VolumeAugment,
                   }


def get_loss(multi_label:bool) -> tf.keras.losses:
    """
    Returns the appropriate loss function based on the number of classes in the dataset.

    Args:
        multi_label : bool, set to True if the dataset is multi-label.

    Returns:
       loss:  The appropriate keras loss function based on the number of classes in the dataset.
    """
    if multi_label:
        raise NotImplementedError("Multi-label classification not implemented yet, but will be in a future update.")
        # Remove the error once it's implemented
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    else:
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        
    return loss