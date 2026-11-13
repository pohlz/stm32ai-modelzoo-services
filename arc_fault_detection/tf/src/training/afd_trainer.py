# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

from omegaconf import DictConfig
import tensorflow as tf

from .train import train


class AFDTrainer:
    """
    Trainer wrapper for Arc Fault Detection (TensorFlow).
    """

    def __init__(self, cfg: DictConfig = None, model: tf.keras.Model = None, dataloaders: dict = None):
        """
        Initialize the trainer with configuration, model, and dataloaders.

        Args:
            cfg (DictConfig): Training configuration.
            model (tf.keras.Model): Model instance to train.
            dataloaders (dict): Datasets for training and validation.
        """
        self.cfg = cfg
        self.model = model
        self.dataloaders = dataloaders

    def train(self):
        """
        Train the model using the existing training logic.

        Returns
        -------
        tf.keras.Model
            The best trained model.
        """
        return train(cfg=self.cfg, model=self.model, dataloaders=self.dataloaders)
