# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import tensorflow as tf
import mlflow
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from common.utils import log_to_file
from common.utils.visualize_utils import plot_confusion_matrix


class AFDTFliteEvaluator:
    """Evaluator for Arc Fault Detection TFLite models."""

    def __init__(self, cfg: DictConfig = None, model: tf.lite.Interpreter = None, dataloaders: dict = None):
        """
        Initialize the evaluator with configuration, model, and dataloaders.

        Args:
            cfg (DictConfig): Evaluation configuration.
            model (tf.lite.Interpreter): TFLite model interpreter.
            dataloaders (dict): Datasets for evaluation.
        """
        self.cfg = cfg
        self.model = model
        self.dataloaders = dataloaders

    def _resolve_eval_dataset(self):
        """
        Select the evaluation dataset name and instance.

        Returns:
            tuple: Dataset label ("test_set" or "valid_set") and the dataset object.
        """
        name_ds = 'test_set' if self.dataloaders.get('test') is not None else 'valid_set'
        eval_ds = self.dataloaders[name_ds[:-4]]
        return name_ds, eval_ds

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

    def evaluate(self):
        """
        Run evaluation using the provided TFLite model.

        Returns:
            float: Accuracy percentage on the evaluation dataset.
        """
        output_dir = HydraConfig.get().runtime.output_dir
        class_names = self.cfg.dataset.class_names
        name_ds, eval_ds = self._resolve_eval_dataset()
        interpreter = self.model

        tf.print(f'[INFO] : Evaluating the quantized model using {name_ds}...')

        features = []
        labels = []
        for x, y in eval_ds.as_numpy_iterator():
            features.append(x)
            labels.append(y)

        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0).flatten()

        input_detail = interpreter.get_input_details()[0]
        output_detail = interpreter.get_output_details()[0]

        expected_shape = list(input_detail["shape"])
        expected_shape[0] = features.shape[0]
        # Get shape of a batch
        interpreter.resize_tensor_input(input_detail["index"], expected_shape)
        interpreter.allocate_tensors()

        tf.print(f"[INFO] : Quantization input details : {input_detail['quantization']}")
        tf.print(f"[INFO] : Dtype input details : {input_detail['dtype']}")

        x_proc = self._quantize_input(features, input_detail)
        interpreter.set_tensor(input_detail["index"], x_proc)
        interpreter.invoke()
        raw_out = interpreter.get_tensor(output_detail["index"])
        probs = self._dequantize_output(raw_out, output_detail)

        # Compute the accuracy
        pred_idx = np.argmax(probs, axis=-1).flatten()
        acc_score = round(accuracy_score(labels, pred_idx) * 100, 2)

        # Print metrics & log in MLFlow
        print(f"[INFO] : Accuracy of quantized model = {acc_score}%")
        mlflow.log_metric(f"quant_acc_{name_ds}", acc_score)

        log_to_file(output_dir, "" + f"Quantized model {name_ds}:")
        log_to_file(output_dir, f"Accuracy of quantized model : {acc_score} %")
        # Compute and plot the confusion matrices
        cm = confusion_matrix(labels, pred_idx)
        confusion_matrix_title = ("Quantized model confusion matrix \n"
                                 f"On dataset : {name_ds} \n"
                                 f"Quantized model accuracy : {acc_score}")

        plot_confusion_matrix(cm=cm,
                              class_names=class_names,
                              title=confusion_matrix_title,
                              model_name=f"quant_model_confusion_matrix_{name_ds}",
                              output_dir=output_dir)

        return acc_score
