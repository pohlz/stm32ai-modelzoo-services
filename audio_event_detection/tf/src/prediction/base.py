# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
from omegaconf import DictConfig
from tabulate import tabulate
import numpy as np
import tensorflow as tf
import numpy.random as rnd
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class BaseAEDPredictor:
    '''Base AED predictor class containing methods common to all predictors
       Subsequent predictor classes should inherit from this one
    '''
    def __init__(self, cfg: DictConfig = None, model: tf.keras.Model = None, dataloaders: dict = None):
        """
        Initialize predictor with configuration, model, and dataloaders.

        Parameters
        ----------
        cfg : DictConfig
            User configuration.
        model : tf.keras.Model
            Model used to generate predictions.
        dataloaders : dict
            Dictionary with keys `pred_ds`, `pred_clip_labels`, `pred_filenames`.
        """

        self.cfg = cfg
        self.model = model
        self.dataloaders = dataloaders
        self.pred_ds = self.dataloaders["pred_ds"]
        self.clip_labels = self.dataloaders["pred_clip_labels"]
        self.filenames = self.dataloaders["pred_filenames"]
        if not self.filenames:
            raise ValueError("Unable to make predictions, could not find any audio file in the "
                            f"files directory.")

        self.class_names = sorted(self.cfg.dataset.class_names)
        self.results_table = []

        # Warn user about sorting class names
        print("[INFO] : Class names sorted alphabetically. \n \
            If the model you are using has been trained using the model zoo, \n \
            there will be no issue. Otherwise, the predicted class' name might not correspond to the \n \
            predicted one-hot vector.")
        print(f"Class names : {self.class_names}")

    def _get_preds(self):
        """
        Generate patch-level predictions on the prediction dataset.
        This method is implemented in the children classes.
        """
        raise NotImplementedError("Implement this method in the inheriting predictor.")
    
    def predict(self):
        """
        Run prediction, aggregate patch predictions to clip level, and display results.

        Returns
        -------
        None
        """
        preds = self._get_preds()
        aggregated_probas = self.aggregate_predictions(preds,
                                                clip_labels=self.clip_labels,
                                                multi_label=self.cfg.dataset.multi_label,
                                                is_truth=False,
                                                return_proba=True)
        aggregated_preds = self.aggregate_predictions(preds,
                                                clip_labels=self.clip_labels,
                                                multi_label=self.cfg.dataset.multi_label,
                                                is_truth=False,
                                                return_proba=False)
        # Add result to the table
        for i in range(aggregated_preds.shape[0]):
            self.results_table.append([self.class_names[np.argmax(aggregated_preds[i])],
                                    aggregated_preds[i].tolist(),
                                    round(float(np.max(aggregated_probas[i])),3),
                                    self.filenames[i]])
        self.display_results()
        print("[INFO] : Prediction complete")

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
    
    def display_results(self):
        """
        Display aggregated prediction results in a tabulated format.

        Returns
        -------
        None
        """
        print(tabulate(self.results_table, headers=["Prediction", "One-hot prediction", "Score", "Audio file"]))
