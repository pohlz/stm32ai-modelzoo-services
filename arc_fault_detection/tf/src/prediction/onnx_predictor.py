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
import onnx
import onnxruntime
from tabulate import tabulate

from common.evaluation import predict_onnx
from common.model_utils.tf_model_loader import load_model_from_path


class AFDONNXPredictor:
    """Predictor for Arc Fault Detection ONNX models."""

    def __init__(self, cfg: DictConfig = None, model: onnxruntime.InferenceSession = None, dataloaders: dict = None):
        """
        Initialize the predictor with configuration, model, and dataloaders.

        Args:
            cfg (DictConfig): Prediction configuration.
            model (onnxruntime.InferenceSession): ONNX runtime session.
            dataloaders (dict): Datasets for prediction.
        """
        self.cfg = cfg
        self.model = model
        self.dataloaders = dataloaders

    def _sanitize_onnx_opset_imports(self, onnx_model_path: str, target_opset: int) -> None:
        """
        Rewrite ONNX opset imports to match the target opset.

        Args:
            onnx_model_path (str): Path to the ONNX model file.
            target_opset (int): Opset version to enforce.

        Returns:
            None
        """
        onnx_model = onnx.load(onnx_model_path)
        del onnx_model.opset_import[:]
        opset = onnx_model.opset_import.add()
        opset.domain = ''
        opset.version = target_opset
        onnx.save(onnx_model, onnx_model_path)

    def _get_probs(self, session: onnxruntime.InferenceSession, x):
        """
        Run ONNX session inference and return class probabilities.

        Args:
            session (onnxruntime.InferenceSession): ONNX runtime session.
            x: Input data for inference.

        Returns:
            np.ndarray: Prediction probabilities.
        """
        return predict_onnx(session, x)

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
        Run prediction using the provided ONNX model.

        Returns:
            None
        """
        model_path = getattr(self.model, "model_path", None)           
        self._sanitize_onnx_opset_imports(onnx_model_path=model_path, target_opset=17)
        session = onnxruntime.InferenceSession(model_path)
        x = self.dataloaders["predict"]
        class_names = list(self.cfg.dataset.class_names)
        probs = self._get_probs(session, x)
        print(self._format_prediction_table(probs, class_names))
