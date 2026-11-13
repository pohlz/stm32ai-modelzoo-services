# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

from omegaconf import DictConfig
import numpy as np
import tensorflow as tf
from tabulate import tabulate

from common.model_utils.tf_model_loader import load_model_from_path


class AFDKerasPredictor:
    """Predictor for Arc Fault Detection Keras models."""

    def __init__(self, cfg: DictConfig = None, model: tf.keras.Model = None, dataloaders: dict = None):
        """
        Initialize the predictor with configuration, model, and dataloaders.

        Args:
            cfg (DictConfig): Prediction configuration.
            model (tf.keras.Model): Keras model to run prediction.
            dataloaders (dict): Datasets for prediction.
        """
        self.cfg = cfg
        self.model = model
        self.dataloaders = dataloaders

    def _get_probs(self, model: tf.keras.Model, x):
        """
        Run model inference and return class probabilities.

        Args:
            model (tf.keras.Model): Keras model used for prediction.
            x: Input data for inference.

        Returns:
            np.ndarray: Prediction probabilities.
        """
        return model.predict(x, verbose=0)

    def _format_prediction_table(self, probs: np.ndarray, class_names):
        """
        Format prediction probabilities into a readable table.

        Args:
            probs (np.ndarray): Prediction probabilities.
            class_names (list): List of class name labels.

        Returns:
            str: Formatted table output.
        """
        if probs.ndim == 2:
            probs = probs[:, None, :]

        n_samples, n_channels, n_classes = probs.shape

        body = []
        for i in range(n_samples):
            row = []
            for ch in range(n_channels):
                ch_probs = probs[i, ch]
                idx = int(np.argmax(ch_probs))
                one_hot = [1 if c == idx else 0 for c in range(n_classes)]
                row.extend([
                    class_names[idx] if idx < len(class_names) else str(idx),
                    one_hot,
                    np.round(ch_probs, 2),
                ])
            body.append(row)

        header_channel = []
        header_type = []
        for ch in range(n_channels):
            header_channel.extend(["", f"Channel {ch+1}", ""])
            header_type.extend(["Prediction", "One-hot", "Scores"])

        full_table = [header_channel, header_type] + body
        return tabulate(full_table, tablefmt="grid", showindex=False)

    def predict(self):
        """
        Run prediction using the provided Keras model.

        Returns:
            None
        """
        model = self.model
        x = self.dataloaders["predict"]
        class_names = list(self.cfg.dataset.class_names)
        probs = self._get_probs(model, x)
        print(self._format_prediction_table(probs, class_names))
