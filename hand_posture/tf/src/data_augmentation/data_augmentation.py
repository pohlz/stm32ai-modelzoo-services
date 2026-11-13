# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

from typing import Dict
from tensorflow import keras
from tensorflow.keras import layers



def get_data_augmentation(data_augmentation: Dict = None):
    """ Define Sequential data-augmentation layers model. 
    args:
        data_augmentation: Dictionary containing data-augmentation configuration.
    returns:
        A Keras Sequential model containing data-augmentation layers.
    """

    augmentation_layers = []
    if data_augmentation.config['random_flip']:
        augmentation_layers.append(layers.RandomFlip(data_augmentation.config['random_flip']['mode']))

    data_augmentation = keras.Sequential(augmentation_layers)
    data_augmentation._name = "Data_augmentation"

    return data_augmentation

