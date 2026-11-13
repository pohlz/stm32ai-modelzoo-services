# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/


import os
import numpy as np
import numpy.random as rnd
import tensorflow as tf
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score
from common.utils import ai_runner_interp


class BaseAEDEvaluator:
    '''Base evaluator class for AED evaluators. Contains boilerplate common to all evaluators'''

    def __init__(self, cfg: DictConfig = None, model = None, dataloaders: dict = None):
        """
        Initialize evaluator with config, model, and datasets.

        Parameters
        ----------
        cfg : DictConfig
            Hydra/OmegaConf configuration object.
        model : object
            Model/session to evaluate (framework-specific).
        dataloaders : dict
            Dict with keys `test_ds`, `test_clip_labels`, `val_ds`, `val_clip_labels`.
        """

        self.cfg = cfg
        self.model = model
        self.test_ds = dataloaders['test_ds']
        self.test_clip_labels = dataloaders['test_clip_labels']
        self.val_ds = dataloaders['val_ds']
        self.val_clip_labels = dataloaders['val_clip_labels']

        self.output_dir = HydraConfig.get().runtime.output_dir
        self.class_names = cfg.dataset.class_names
        self.display_figures = cfg.general.display_figures
        self.multi_label = cfg.dataset.multi_label
        if cfg.training:
            if cfg.training.batch_size:
                self.batch_size = cfg.training.batch_size
        elif cfg.general.batch_size:
            self.batch_size = cfg.general.batch_size
        else:
            self.batch_size = 16
        
        if self.multi_label:     
            raise NotImplementedError("Multi-label inference not implemented yet")
        
        self.eval_ds = None
        self.name_ds = None

        self._prepare_evaluation()
        self._ensure_output_dir()

    def _ensure_output_dir(self):
        """
        Ensures that the output directory exists.
        """
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

    def _prepare_evaluation(self):
        """
        Prepares the evaluation process by selecting the appropriate dataset.
        """
        # Use the test dataset if available; otherwise, use the validation dataset

        if self.test_ds:
            self.eval_ds = self.test_ds
            self.clip_labels = self.test_clip_labels
            self.name_ds = "test_set"
        else:
            self.eval_ds = self.val_ds
            self.clip_labels = self.val_clip_labels
            self.name_ds = "validation_set"

    def _get_target(self):
        """
        Retrieves the evaluation target from the configuration.
        """
        if self.cfg.evaluation and self.cfg.evaluation.target:
            return self.cfg.evaluation.target
        return "host"

    def _get_interpreter(self, target):
        """
        Retrieves the AI runner interpreter for the specified target.
        """
        name_model = os.path.basename(self.quantized_model.model_path)
        return ai_runner_interp(target, name_model)

    @staticmethod
    def _majority_vote(preds: tf.Tensor, 
                    multi_label: bool = False,
                    return_proba: bool = False):
        '''
        Concatenates several one-hot prediction labels into one, according to majority vote.
        Args:
            preds: np.ndarray or tf.Tensor, shape (n_preds, n_classes). Array of one-hot prediction vectors to concatenate.
            multi_label: bool, set to True if prediction vectors are multi_label.
            return_proba: bool, if True return probabilities instead of onehot predictions.
        Returns:
            onehot_vote: One-hot encoded aggregated predictions. Only returned if return_proba is False.
            aggregated_preds: Averaged predictions. Only returned if return_proba is True.
        '''

        if not multi_label:
            # If we only have one label per sample, pick the one with the most votes
            try:
                preds = preds.numpy()
            except:
                pass
            n_classes = preds.shape[1]

            if return_proba:
                aggregated_preds = np.mean(preds, axis=0)
                return aggregated_preds
            
            aggregated_preds = np.sum(preds, axis=0)
            
            # Fancy version of argmax w/ random selection in case of tie
            vote = rnd.choice(np.flatnonzero(aggregated_preds == aggregated_preds.max()))
            onehot_vote = np.zeros(n_classes)
            onehot_vote[vote] = 1
            return onehot_vote

        else:
            # Else, return the label vector where classes predicted over half the time remain
            try:
                preds = preds.numpy()
            except:
                pass
            n_classes = preds.shape[1]
            aggregated_preds = np.mean(preds, axis=0)
            if return_proba:
                return aggregated_preds
            onehot_vote = (aggregated_preds >= 0.5).astype(np.int32)
            return onehot_vote


    def aggregate_predictions(self, preds, clip_labels, multi_label=False, is_truth=False,
                              return_proba=False):
        '''
        Aggregate predictions from patch level to clip level.
        Pass is_truth=True if aggregating true labels to skip some computation.
        Args:
            preds: tf.Tensor or np.ndarray shape (n_preds, n_classes). Array of one-hot prediction vectors to concatenate.
            clip_labels: np.ndarray, shape (n_preds). A vector indicating which clip each patch belongs to.
            multi_label: bool, set to True if preds are multi-label.
            is_truth: bool, set to True if preds are true labels. Skips some computation.
            return_proba: bool, if True returns probabilities instead of one-hot labels.
        Returns:
            aggregated_preds: np.ndarray, shape (n_clips) Aggregated predictions, one prediction per clip.
        '''
        n_clips = np.max(clip_labels) + 1
        aggregated_preds = np.empty((n_clips, preds.shape[1]))
        if not is_truth:
            for i in range(n_clips):
                patches_to_aggregate = preds[np.where(clip_labels == i)[0]]
                vote = self._majority_vote(preds=patches_to_aggregate,
                                    multi_label=multi_label,
                                    return_proba=return_proba)
                aggregated_preds[i] = vote
        else:
            for i in range(n_clips):
                if len(np.where(clip_labels == i)[0]) > 0:
                    aggregated_preds[i] = preds[np.where(clip_labels == i)[0][0]]
                else:
                    raise ValueError(
                        "One clip had no patches. \ Check your silence removal and feature extraction settings."
                        )
        return aggregated_preds

    @staticmethod
    def _compute_accuracy_score(y_true, y_pred, is_multilabel=False):
        """
        Wrapper function around sklearn.metrics.accuracy_score.
        Args:
            y_true: np.ndarray, true labels (one-hot or integer encoded).
            y_pred: np.ndarray, predicted labels (one-hot or integer encoded).
            is_multilabel: bool, set to True if multi-label classification.
        Returns:
            float: accuracy score.
        """
        if not is_multilabel:
            y_true = np.argmax(y_true, axis=1)
            y_pred = np.argmax(y_pred, axis=1)

            return accuracy_score(y_true, y_pred)
        else:
            raise NotImplementedError("Not implemented yet for multi_label=True")
