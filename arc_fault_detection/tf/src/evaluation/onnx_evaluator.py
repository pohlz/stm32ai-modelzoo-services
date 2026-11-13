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
import mlflow
import onnxruntime
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from common.evaluation import model_is_quantized, predict_onnx
from common.utils import tf_dataset_to_np_array
from common.utils.visualize_utils import plot_confusion_matrix
import onnx


class AFDONNXEvaluator:
    """Evaluator for Arc Fault Detection ONNX models."""

    def __init__(self, cfg: DictConfig = None, model: onnxruntime.InferenceSession = None, dataloaders: dict = None):
        """
        Initialize the evaluator with configuration, model, and dataloaders.

        Args:
            cfg (DictConfig): Evaluation configuration.
            model (onnxruntime.InferenceSession): ONNX runtime session.
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

    def evaluate(self):
        """
        Run evaluation using the provided ONNX model.

        Returns:
            float: Accuracy percentage on the evaluation dataset.
        """
        output_dir = HydraConfig.get().runtime.output_dir
        class_names = self.cfg.dataset.class_names
        name_ds, eval_ds = self._resolve_eval_dataset()

        model_path = getattr(self.model, "model_path", None)           
        self._sanitize_onnx_opset_imports(onnx_model_path=model_path, target_opset=17)
        session = onnxruntime.InferenceSession(model_path)
        # Evaluate the model on the input data
        data, labels = tf_dataset_to_np_array(eval_ds, nchw=False)
        preds = predict_onnx(session, data)
        y_pred = np.argmax(preds, axis=-1).flatten()
        labels = labels.flatten()
        # Compute the accuracy
        eval_accuracy = round(accuracy_score(labels, y_pred) * 100, 2)
        model_type = 'Quantized' if model_is_quantized(model_path) else 'Float'
        print(f'[INFO] : {model_type} accuracy: {eval_accuracy} %')

        log_file_name = f"{output_dir}/stm32ai_main.log"
        with open(log_file_name, 'a', encoding='utf-8') as f:
            f.write(f'{model_type} ONNX model\n accuracy: {eval_accuracy} %\n')

        acc_metric_name = f"quant_acc_{name_ds}" if model_is_quantized(model_path) else f"float_acc_{name_ds}"
        mlflow.log_metric(acc_metric_name, eval_accuracy)
        # Calculate the confusion matrix.
        cm = confusion_matrix(labels, y_pred)
        # Log the confusion matrix as an image summary.
        confusion_matrix_title = (f"{model_type} confusion matrix \n"
                                 f"On dataset : {name_ds} \n"
                                 f"accuracy : {eval_accuracy}")

        plot_confusion_matrix(cm=cm,
                              class_names=class_names,
                              title=confusion_matrix_title,
                              model_name=f"{model_type.lower()}_confusion_matrix_{name_ds}",
                              output_dir=output_dir)

        return eval_accuracy
