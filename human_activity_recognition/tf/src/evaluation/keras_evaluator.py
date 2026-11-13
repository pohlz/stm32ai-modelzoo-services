# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

# Import necessary libraries
import os
import warnings
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

# Suppress warnings and TensorFlow logs
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import mlflow

# Import utility functions
from human_activity_recognition.tf.src.utils import get_loss
from common.utils import (
    compute_confusion_matrix, count_h5_parameters, plot_confusion_matrix,
    log_to_file, display_figures
)  # Common utilities for evaluation and logging


# Define a class for evaluating Keras models
class KerasModelEvaluator:
    """
    A class to evaluate TensorFlow Keras models.

    Args:
        cfg (DictConfig): Configuration object for evaluation.
        model (object): The Keras model to evaluate.
        dataloaders (dict): Dictionary containing datasets for testing and validation.
    """
    def __init__(self, cfg: DictConfig = None, model: object = None, 
                 dataloaders: dict = None):
        self.cfg = cfg
        self.model = model
        self.test_ds = dataloaders['test']
        self.valid_ds = dataloaders['valid']
        self.output_dir = HydraConfig.get().runtime.output_dir
        self.class_names = cfg.dataset.class_names
        self.display_figures = cfg.general.display_figures
        self.eval_ds = None
        self.name_ds = None

    def _prepare_evaluation(self):
        """
        Prepares the evaluation process by selecting the appropriate dataset.
        """
        # Use the test dataset if available; otherwise, use the validation dataset

        if self.test_ds:
            self.eval_ds = self.test_ds
            self.name_ds = "test_set"
        else:
            self.eval_ds = self.valid_ds
            self.name_ds = "validation_set"
    
    def _compile_model(self):
        """
        Compiles the model with the appropriate loss function and metrics.
        """
        loss = get_loss(num_classes=len(self.class_names))
        self.model.compile(loss=loss, metrics=['accuracy'])

    def _run_evaluate(self):
        """
        Runs the evaluation process and computes metrics.

        Returns:
            float: Accuracy of the model on the evaluation dataset.
        """
        # Count the number of parameters in the model and log them
        count_h5_parameters(output_dir=self.output_dir, 
                            model=self.model)   
        self._compile_model()
        
        # Evaluate the model on the selected dataset
        tf.print(f'[INFO] : Evaluating the float model using {self.name_ds}...')
        loss, accuracy = self.model.evaluate(self.eval_ds)
        cm, accuracy = compute_confusion_matrix(test_set=self.eval_ds, model=self.model)

        # Plot and display the confusion matrix if enabled
        if self.display_figures:
            model_name = f"float_model_confusion_matrix_{self.name_ds}"
            plot_confusion_matrix(
                cm=cm,
                class_names=self.class_names,
                model_name=model_name,
                title=f'{model_name}\naccuracy: {accuracy}',
                output_dir=self.output_dir
            )
            display_figures(self.cfg)

        # Log evaluation results
        print(f"[INFO] : Accuracy of float model on {self.name_ds} = {accuracy}%")
        print(f"[INFO] : Loss of float model on {self.name_ds} = {loss}")
        mlflow.log_metric(f"float_acc_{self.name_ds}", accuracy)
        mlflow.log_metric(f"float_loss_{self.name_ds}", loss)
        log_to_file(self.output_dir, f"Float model {self.name_ds}:")
        log_to_file(self.output_dir, f"Accuracy of float model : {accuracy} %")
        log_to_file(self.output_dir, f"Loss of float model : {round(loss,2)} ")
        return accuracy

    def evaluate(self):
        """
        Executes the full evaluation process.
        
        Returns:
            float: Accuracy of the model on the evaluation dataset.
        """
        self._prepare_evaluation()  # Prepare the evaluation process
        acc = self._run_evaluate()  # Run the evaluation
        print('[INFO] : Evaluation complete.')
        return acc  # Return the accuracy