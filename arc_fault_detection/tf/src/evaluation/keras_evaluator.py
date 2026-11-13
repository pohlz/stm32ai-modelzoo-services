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


from common.utils import count_h5_parameters, log_to_file
from common.utils.visualize_utils import plot_confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np


class AFDKerasEvaluator:
    """Evaluator for Arc Fault Detection Keras models."""

    def __init__(self, cfg: DictConfig = None, model: tf.keras.Model = None, dataloaders: dict = None):
        """
        Initialize the evaluator with configuration, model, and dataloaders.

        Args:
            cfg (DictConfig): Evaluation configuration.
            model (tf.keras.Model): Keras model to evaluate.
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

    def _compute_confusion_matrix(self, test_set: tf.data.Dataset = None,
                                  model: tf.keras.models.Model = None):
        """
        Compute confusion matrix and accuracy for a dataset.

        Args:
            test_set (tf.data.Dataset): Dataset used for evaluation.
            model (tf.keras.models.Model): Model used for prediction.

        Returns:
            tuple: Confusion matrix and accuracy percentage.
        """
        test_pred = []
        test_labels = []
        for data in test_set:
            test_pred_score = model.predict_on_batch(data[0])
            test_pred.append(np.argmax(test_pred_score, axis=-1))
            batch_labels = data[1]
            test_labels.append(batch_labels)

        labels = np.concatenate(test_labels, axis=0).flatten()
        logits = np.concatenate(test_pred, axis=0).flatten()
        acc_score = round(accuracy_score(labels, logits) * 100, 2)
        cm = confusion_matrix(labels, logits)
        return cm, acc_score

    def evaluate(self):
        """
        Run evaluation using the provided Keras model.

        Returns:
            float: Accuracy percentage on the evaluation dataset.
        """
        output_dir = HydraConfig.get().runtime.output_dir
        class_names = self.cfg.dataset.class_names
        name_ds, eval_ds = self._resolve_eval_dataset()
        model = self.model

        count_h5_parameters(output_dir=output_dir, model=model)
        # Evaluate the model on the test data
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        model.compile(loss=loss_fn, metrics=['accuracy'])
        tf.print(f'[INFO] : Evaluating the float model using {name_ds}...')
        loss, accuracy = model.evaluate(eval_ds)
        # Calculate the confusion matrix.
        cm, test_accuracy = self._compute_confusion_matrix(test_set=eval_ds, model=model)
        # Log the confusion matrix as an image summary.
        model_name = f"float_model_confusion_matrix_{name_ds}"
        plot_confusion_matrix(cm=cm,
                              class_names=class_names,
                              model_name=model_name,
                              title=f'{model_name}\naccuracy: {test_accuracy}',
                              output_dir=output_dir)
        print(f"[INFO] : Accuracy of float model = {test_accuracy}%")
        print(f"[INFO] : Loss of float model = {loss}")
        mlflow.log_metric(f"float_acc_{name_ds}", test_accuracy)
        mlflow.log_metric(f"float_loss_{name_ds}", loss)
        log_to_file(output_dir, f"Float model {name_ds}:")
        log_to_file(output_dir, f"Accuracy of float model : {test_accuracy} %")
        log_to_file(output_dir, f"Loss of float model : {round(loss, 2)} ")

        return accuracy
