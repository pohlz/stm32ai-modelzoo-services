# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
import warnings
import tensorflow as tf
import mlflow
import numpy as np
import tqdm
from omegaconf import DictConfig
from common.utils import tf_dataset_to_np_array, plot_confusion_matrix, log_to_file, compute_confusion_matrix2
from audio_event_detection.tf.src.preprocessing import preprocess_input
from .base import BaseAEDEvaluator
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class AEDTFliteEvaluator(BaseAEDEvaluator):
    """
    Evaluator for TFLite models (quantized).

    Notes
    -----
    Supports host and target evaluation via AI Runner.
    """
    def __init__(self, cfg: DictConfig = None, model: tf.keras.Model = None, dataloaders: dict = None):
        """
        Initialize TFLite evaluator.

        Parameters
        ----------
        cfg : DictConfig,
            User configuration.
        model : tf.lite.Interpreter
            TFLite interpreter/model used for evaluation.
        dataloaders : dict
            Datasets and clip labels dictionary.
        """
        super().__init__(cfg=cfg,
                         model=model,
                         dataloaders=dataloaders)
        
        self.quantized_model = self.model # TFLite model, renamed for clarity
        self.target = self._get_target()
        self.ai_runner = self._get_interpreter(self.target)

        # Keep the configured order. Dataset one-hot labels and confusion-matrix
        # indices use cfg.dataset.class_names; sorting here mislabels the axes.
        self.class_names = list(self.class_names)

    def _get_preds_on_host(self):
        """
        Run inference locally using the TFLite interpreter.

        Returns
        -------
        tuple
            `(preds, patch_labels)` as numpy arrays.
        """
        input_details = self.quantized_model.get_input_details()[0]
        output_index_quant = self.quantized_model.get_output_details()[0]["index"]
        input_index_quant = input_details["index"]

        # Get shape of a batch
        batch_shape = input_details["shape"]
        batch_shape[0] = self.batch_size
        self.quantized_model.resize_tensor_input(input_index_quant, batch_shape)

        tf.print(f"[INFO] : Quantization input details : {input_details['quantization']}")
        tf.print(f"[INFO] : Dtype input details : {input_details['dtype']}")

        self.quantized_model.allocate_tensors()
        batch_preds = []
        batch_labels = []

        for patches, labels in tqdm.tqdm(self.eval_ds, total=len(self.eval_ds)):
            # If the last batch does not have enough patches, resize tensor input one last time
            if len(patches) != self.batch_size:
                batch_shape[0] = len(patches)
                self.quantized_model.resize_tensor_input(input_index_quant, batch_shape)
                self.quantized_model.allocate_tensors()
            patches_processed = preprocess_input(patches, input_details)
            self.quantized_model.set_tensor(input_index_quant, patches_processed)
            self.quantized_model.invoke()
            test_pred_score = self.quantized_model.get_tensor(output_index_quant)
            batch_preds.append(test_pred_score)
            batch_labels.append(labels.numpy())

        patch_labels = np.concatenate(batch_labels, axis=0)
        preds = np.concatenate(batch_preds, axis=0)

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
        Evaluate quantized TFLite model on the selected dataset.

        Returns
        -------
        tuple or float
            `(patch_acc, clip_acc)` when clip labels exist; otherwise `patch_acc`.
        """
        tf.print(f'[INFO] : Evaluating the quantized model using {self.name_ds}...')
        if self.target in ['stedgeai_n6', 'stedgeai_host', 'stedgeai_h7p']:
            print(f"[INFO] Evaluating TFLite model on target {self.target}")
        # Get model preds
        if self.target == "host":
            preds, patch_labels = self._get_preds_on_host()
        else:
            preds, patch_labels = self._get_preds_on_target()

        # Compute patch-level accuracy
        patch_level_accuracy = self._compute_accuracy_score(patch_labels,
                                                    preds,
                                                    is_multilabel=self.multi_label)

        # Compute clip-level accuracy
        # Aggregate clip-level labels
        if self.clip_labels is not None:
            aggregated_labels = self.aggregate_predictions(preds=patch_labels,
                                                    clip_labels=self.clip_labels,
                                                    multi_label=self.multi_label,
                                                    is_truth=True)
            aggregated_preds = self.aggregate_predictions(preds=preds,
                                                    clip_labels=self.clip_labels,
                                                    multi_label=self.multi_label,
                                                    is_truth=False)
            clip_level_accuracy = self._compute_accuracy_score(aggregated_labels,
                                                        aggregated_preds,
                                                        is_multilabel=self.multi_label)
        # Print metrics & log in MLFlow

        print(f"[INFO] : Patch-level Accuracy of quantized model = {round(patch_level_accuracy * 100, 2)}%")
        if self.clip_labels is not None:
            print(f"[INFO] : Clip-level Accuracy of quantized model = {round(clip_level_accuracy * 100, 2)}%")

        mlflow.log_metric(f"quant_patch_acc_{self.name_ds}", round(patch_level_accuracy * 100, 2))
        if self.clip_labels is not None:
            mlflow.log_metric(f"quant_clip_acc_{self.name_ds}", round(clip_level_accuracy * 100, 2))

        log_to_file(self.output_dir,  "" + f"Quantized model {self.name_ds}:")
        log_to_file(self.output_dir, f"Patch-level accuracy of quantized model : {round(patch_level_accuracy * 100, 2)} %")
        if self.clip_labels is not None:
            log_to_file(self.output_dir, f"Clip-level accuracy of quantized model : {round(clip_level_accuracy * 100, 2)} %")
        
        # Compute and plot the confusion matrices
        
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
        
        # plot_confusion_matrix writes the STM report PNG. Do this regardless
        # of display_figures: that option controls GUI display, not artifacts.
        self._display_figures()
            
        print("[INFO] : Evaluation complete")
        if self.clip_labels is not None:
            return patch_level_accuracy, clip_level_accuracy 
        else:
            return patch_level_accuracy   
    
    def _display_figures(self):
        """
        Plot and save confusion matrices for the quantized model.
        """
        plot_confusion_matrix(cm=self.patch_level_cm,
                            class_names=self.class_names,
                            title=self.patch_level_title,
                            model_name=f"quant_model_patch_confusion_matrix_{self.name_ds}",
                            output_dir=self.output_dir)
        if self.clip_labels is not None:
            plot_confusion_matrix(cm=self.clip_level_cm,
                                class_names=self.class_names,
                                title=self.clip_level_title,
                                model_name=f"quant_model_clip_confusion_matrix_{self.name_ds}",
                                output_dir=self.output_dir)
