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
from omegaconf import DictConfig
from common.utils import plot_confusion_matrix, log_to_file, compute_confusion_matrix2, count_h5_parameters
from audio_event_detection.tf.src.utils import get_loss

from .base import BaseAEDEvaluator

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class AEDKerasEvaluator(BaseAEDEvaluator):
    """
    AED Evaluator for Keras models

    Args:
        cfg (DictConfig): Model zoo user config.
        model (object): The Keras model to evaluate.
        dataloaders (dict): Dictionary containing datasets and clip labels for testing and validation.
    """
    
    def __init__(self, cfg: DictConfig = None, model: tf.keras.Model = None, dataloaders: dict = None):
        """
        Initialize Keras AED evaluator.

        Parameters
        ----------
        cfg : DictConfig,
            User configuration.
        model : tf.keras.Model
            Keras model to evaluate.
        dataloaders : dict
            Datasets and clip labels dictionary.
        """
        super().__init__(cfg=cfg,
                         model=model,
                         dataloaders=dataloaders)

    def _display_figures(self):
        """
        Plot and save patch- and clip-level confusion matrices.
        """
        plot_confusion_matrix(cm=self.patch_level_cm,
                            class_names=self.class_names,
                            title=self.patch_level_title,
                            model_name=f"float_model_patch_confusion_matrix_{self.name_ds}",
                            output_dir=self.output_dir)
        if self.clip_labels is not None:
            plot_confusion_matrix(cm=self.clip_level_cm,
                                class_names=self.class_names,
                                title=self.clip_level_title,
                                model_name=f"float_model_clip_confusion_matrix_{self.name_ds}",
                                output_dir=self.output_dir)

    def evaluate(self):
        """
        Evaluate the float Keras model on the selected dataset. Returns clip and patch accuracy.

        Returns
        -------
        tuple
            `(patch_acc, clip_acc)` in percent; `clip_acc` may be `None`.
        """
        count_h5_parameters(output_dir=self.output_dir, model=self.model)
        loss = get_loss(multi_label=self.multi_label)
        self.model.compile(loss=loss, metrics=['accuracy'])

        # Evaluate the model on the test data
        tf.print(f'[INFO] : Evaluating the float model using {self.name_ds}...')
        preds = self.model.predict(self.eval_ds)

        # Compute loss
        patch_labels = np.concatenate([y for X, y in self.eval_ds])
        loss_value = loss(patch_labels, preds)

        # Convert preds to numpy
        try:
            preds = preds.numpy()
        except:
            pass

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

        print(f"[INFO] : Patch-level Accuracy of float model = {round(patch_level_accuracy * 100, 2)}%")

        if self.clip_labels is not None:
            print(f"[INFO] : Clip-level Accuracy of float model = {round(clip_level_accuracy * 100, 2)}%")
        print(f"[INFO] : Loss of float model = {loss_value}")

        mlflow.log_metric(f"float_patch_acc_{self.name_ds}", round(patch_level_accuracy * 100, 2))
        if self.clip_labels is not None:
            mlflow.log_metric(f"float_clip_acc_{self.name_ds}", round(clip_level_accuracy * 100, 2))
        mlflow.log_metric(f"float_loss_{self.name_ds}", loss_value)

        log_to_file(self.output_dir, f"Float model {self.name_ds}:")
        log_to_file(self.output_dir, f"Patch-level accuracy of float model : {round(patch_level_accuracy * 100, 2)} %")
        if self.clip_labels is not None:
            log_to_file(self.output_dir, f"Clip-level accuracy of float model : {round(clip_level_accuracy * 100, 2)} %")
        log_to_file(self.output_dir, f"Loss of float model : {loss_value} ")

        # Compute and plot the confusion matrices

        self.patch_level_cm = compute_confusion_matrix2(patch_labels, preds)

        self.patch_level_title = (f"Float model patch-level confusion matrix \n"
                            f"On dataset : {self.name_ds} \n"
                            f"Float model patch-level accuracy : { patch_level_accuracy}")
        if self.clip_labels is not None:
            self.clip_level_cm = compute_confusion_matrix2(aggregated_labels, aggregated_preds)
            self.clip_level_title = (f"Float model clip-level confusion matrix \n"
                                f"On dataset : {self.name_ds} \n"
                                f"Float model clip-level accuracy : {clip_level_accuracy}")
            
        # plot_confusion_matrix writes the STM report PNG. Do this regardless
        # of display_figures: that option controls GUI display, not artifacts.
        self._display_figures()
            
        print("[INFO] : Evaluation complete")
        if self.clip_labels is not None:
            return patch_level_accuracy, clip_level_accuracy
        else:
            return patch_level_accuracy, None
        

