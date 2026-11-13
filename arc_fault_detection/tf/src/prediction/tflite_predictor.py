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


class AFDTFLitePredictor:
    """Predictor for Arc Fault Detection TFLite models."""

    def __init__(self, cfg: DictConfig = None, model: tf.lite.Interpreter = None, dataloaders: dict = None):
        """
        Initialize the predictor with configuration, model, and dataloaders.

        Args:
            cfg (DictConfig): Prediction configuration.
            model (tf.lite.Interpreter): TFLite model interpreter.
            dataloaders (dict): Datasets for prediction.
        """
        self.cfg = cfg
        self.model = model
        self.dataloaders = dataloaders

    def _quantize_input(self, x: np.ndarray, input_details: dict) -> np.ndarray:
        """
        Quantize input data using TFLite input quantization parameters.

        Args:
            x (np.ndarray): Input data to quantize.
            input_details (dict): TFLite input tensor details.

        Returns:
            np.ndarray: Quantized input data.
        """
        if input_details["dtype"] in (np.int8, np.uint8):
            scale, zp = input_details["quantization"]
            x_q = (x / scale + zp).round()
            info = np.iinfo(input_details["dtype"])
            x_q = np.clip(x_q, info.min, info.max).astype(input_details["dtype"])
            return x_q
        return x.astype(input_details["dtype"])

    def _dequantize_output(self, raw: np.ndarray, output_details: dict) -> np.ndarray:
        """
        Dequantize output data using TFLite output quantization parameters.

        Args:
            raw (np.ndarray): Raw output tensor values.
            output_details (dict): TFLite output tensor details.

        Returns:
            np.ndarray: Dequantized output data.
        """
        if output_details["dtype"] in (np.int8, np.uint8):
            scale, zp = output_details["quantization"]
            if scale and scale > 0:
                return (raw.astype(np.float32) - zp) * scale
        return raw.astype(np.float32)

    def _get_probs(self, interpreter: tf.lite.Interpreter, x: np.ndarray) -> np.ndarray:
        """
        Run TFLite inference and return dequantized probabilities.

        Args:
            interpreter (tf.lite.Interpreter): TFLite interpreter.
            x (np.ndarray): Input data for inference.

        Returns:
            np.ndarray: Prediction probabilities.
        """
        input_detail = interpreter.get_input_details()[0]
        output_detail = interpreter.get_output_details()[0]
        expected_shape = list(input_detail["shape"])
        expected_shape[0] = x.shape[0]
        interpreter.resize_tensor_input(input_detail["index"], expected_shape)
        interpreter.allocate_tensors()
        x_proc = self._quantize_input(x, input_detail)
        interpreter.set_tensor(input_detail["index"], x_proc)
        interpreter.invoke()
        raw_out = interpreter.get_tensor(output_detail["index"])
        return self._dequantize_output(raw_out, output_detail)

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
        Run prediction using the provided TFLite model.

        Returns:
            None
        """
        interpreter = self.model
        x = self.dataloaders["predict"]
        class_names = list(self.cfg.dataset.class_names)
        probs = self._get_probs(interpreter, x)
        print(self._format_prediction_table(probs, class_names))
