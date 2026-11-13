# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
import mlflow
import numpy as np
import onnx
import onnxruntime
from omegaconf import DictConfig

from common.evaluation import model_is_quantized, predict_onnx
from common.utils import tf_dataset_to_np_array, plot_confusion_matrix, log_to_file, compute_confusion_matrix2
from audio_event_detection.tf.src.preprocessing import preprocess_input
from .base import BaseAEDEvaluator


class AEDONNXEvaluator(BaseAEDEvaluator):
    """
    Evaluator for ONNX models (float or quantized).

    Notes
    -----
    Supports host and target evaluation via AI Runner.
    """
    def __init__(self, cfg: DictConfig = None, model: onnxruntime.InferenceSession = None, dataloaders: dict = None):
        """
        Initialize ONNX evaluator and runtime session/interpreter.

        Parameters
        ----------
        cfg : DictConfig
            User configuration.
        model : onnxruntime.InferenceSession,
            ONNX runtime session or model wrapper.
        dataloaders : dict
            Datasets and clip labels dictionary.
        """
        super().__init__(cfg=cfg,
                         model=model,
                         dataloaders=dataloaders)
        
        self.sess = self._sanitize_onnx_opset_imports(self.model, target_opset=17)
        self.model_type = 'Quantized' if model_is_quantized(self.sess._model_path) else 'Float'
        self.quantized_model = model # For compatibility with _get_interpreter
        self.target = self._get_target()
        self.ai_runner = self._get_interpreter(self.target)

        # Sort class names alphabetically just in case 
        self.class_names = sorted(self.class_names)


    def _get_preds_on_host(self):
        """
        Run inference on host using onnxruntime.

        Returns
        -------
        tuple
            `(preds, patch_labels)` as numpy arrays.
        """
        input_samples, patch_labels = tf_dataset_to_np_array(self.eval_ds, nchw=False)
        # Sanitize opset imports and get a new session

        preds = predict_onnx(self.sess, input_samples)

        return preds, patch_labels

    def _get_preds_on_target(self):
        """
        Run inference on target via AI Runner.

        Returns
        -------
        tuple
            `(preds, patch_labels)` as numpy arrays.
        """
        # Get ai runner input details and mangle it back into 
        # the tf input detail dict format to pass to preprocess_input
        input_samples, patch_labels = tf_dataset_to_np_array(self.eval_ds, nchw=False)
        ai_runner_input_details = self.ai_runner.get_inputs()
        input_details = {}
        input_details["dtype"] = ai_runner_input_details[0].dtype
        input_details["quantization"] = [ai_runner_input_details[0].scale, ai_runner_input_details[0].zero_point]
        input_samples = preprocess_input(input_samples, input_details)

        preds, _ = self.ai_runner.invoke(input_samples)
        preds = preds[0]
        # Yes it HAS to be a tuple
        dims_to_squeeze = tuple(np.arange(1, preds.ndim - 1))

        preds = np.squeeze(preds, axis=dims_to_squeeze)

        return preds, patch_labels

    def evaluate(self):
        """
        Evaluate ONNX model on the selected dataset and target.
        Returns clip and patch-level accuracies.

        Returns
        -------
        tuple
            `(patch_acc, clip_acc)` in percent; `clip_acc` may be `None`.
        """
        print(f"[INFO] : Evaluating ONNX model using {self.name_ds}...")
        if self.target in ['stedgeai_n6', 'stedgeai_host', 'stedgeai_h7p']:
            print(f"[INFO] Evaluating ONNX model on target {self.target}")
        # Get model preds
        if self.target == "host":
            preds, patch_labels = self._get_preds_on_host()
        else:
            preds, patch_labels = self._get_preds_on_target()

        # Convert patch labels and preds from one-hot to integer labels
        # We still need to keep the one-hot labels for aggregation

        patch_level_accuracy = round(self._compute_accuracy_score(patch_labels, preds) * 100, 2)
        print(f'[INFO] : {self.model_type} patch-level evaluation accuracy: {patch_level_accuracy} %')

        if self.clip_labels is not None:
            # Compute clip-level accuracy
            # Aggregate clip-level labels
            aggregated_labels = self.aggregate_predictions(preds=patch_labels,
                                                    clip_labels=self.clip_labels,
                                                    multi_label=self.multi_label,
                                                    is_truth=True)
            aggregated_preds = self.aggregate_predictions(preds=preds,
                                                    clip_labels=self.clip_labels,
                                                    multi_label=self.multi_label,
                                                    is_truth=False)
            clip_level_accuracy = round(self._compute_accuracy_score(aggregated_labels, aggregated_preds) * 100, 2)
            print(f'[INFO] : {self.model_type} clip-level evaluation accuracy: {clip_level_accuracy} %')

        #stm32ai_main.log logging
        log_to_file(self.output_dir,  "" + f"ONNX model {self.name_ds}:")
        log_to_file(self.output_dir, f"Patch-level accuracy of {self.model_type} ONNX model : {round(patch_level_accuracy * 100, 2)} %")
        if self.clip_labels is not None:
            log_to_file(self.output_dir, f"Clip-level accuracy of {self.model_type} ONNX model : {round(clip_level_accuracy * 100, 2)} %")
        
        # MLFlow logging
        t = "quant" if self.model_type == "Quantized" else "float"
        acc_metric_name = f"{t}_patch_acc_{self.name_ds}"
        mlflow.log_metric(acc_metric_name, patch_level_accuracy)
        if self.clip_labels is not None:
            acc_metric_name = f"{t}_clip_acc_{self.name_ds}"
            mlflow.log_metric(acc_metric_name, clip_level_accuracy)

        # Compute & plot confusion matrices
        self.patch_level_cm = compute_confusion_matrix2(patch_labels, preds)
        if self.clip_labels is not None:
            self.clip_level_cm = compute_confusion_matrix2(aggregated_labels, aggregated_preds)

        self.patch_level_title = ("Quantized model patch-level confusion matrix \n"
                            f"On dataset : {self.name_ds} \n"
                            f"Quantized model patch-level accuracy : {patch_level_accuracy}")
        if self.clip_labels is not None:
            self.clip_level_title = ("Quantized model clip-level confusion matrix \n"
                                f"On dataset : {self.name_ds} \n"
                                f"Quantized model clip-level accuracy : {clip_level_accuracy}")
        
        if self.display_figures:
            self._display_figures()

        print("[INFO] : Evaluation complete")
        if self.clip_labels is not None:
            return patch_level_accuracy, clip_level_accuracy
        else:
            return patch_level_accuracy, None


    def _display_figures(self):
        """
        Plot and save confusion matrices with proper titles.
        """
        t = "quant" if self.model_type == "Quantized" else "float"
        plot_confusion_matrix(cm=self.patch_level_cm,
                            class_names=self.class_names,
                            title=self.patch_level_title,
                            model_name=f"{t}_model_patch_confusion_matrix_{self.name_ds}",
                            output_dir=self.output_dir)
        if self.clip_labels is not None:
            plot_confusion_matrix(cm=self.clip_level_cm,
                                class_names=self.class_names,
                                title=self.clip_level_title,
                                model_name=f"{t}_model_clip_confusion_matrix_{self.name_ds}",
                                output_dir=self.output_dir)
    @staticmethod
    def _sanitize_onnx_opset_imports(onnx_model, target_opset: int):
        '''
        Remove all the un-necessary opset imports from an onnx model resulting due to tf2onnx operation.
        Accepts either a file path or onnxruntime.InferenceSession.
        Returns a new onnxruntime.InferenceSession if a session or path is provided.
        '''
        onnx_model_path = onnx_model.model_path
        onnx_model_obj = onnx.load(onnx_model_path)
        del onnx_model_obj.opset_import[:]
        opset = onnx_model_obj.opset_import.add()
        opset.domain = ''
        opset.version = target_opset
        onnx.save(onnx_model_obj, onnx_model_path)
        return onnxruntime.InferenceSession(onnx_model_path)


