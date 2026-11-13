
# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

# Import necessary libraries
import os
from timeit import default_timer as timer
from datetime import timedelta
from typing import List

from hydra.core.hydra_config import HydraConfig
from munch import DefaultMunch
from omegaconf import DictConfig
import numpy as np 
import tensorflow as tf

# Suppress TensorFlow warnings to reduce log clutter
import logging
logging.getLogger('mlflow.tensorflow').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Import utility functions and modules
from common.utils import (
    log_to_file, log_last_epoch_history, LRTensorBoard, check_training_determinism,
    model_summary, collect_callback_args, vis_training_curves
)
from common.training import (
    get_optimizer, lr_schedulers
)
from human_activity_recognition.tf.src.utils import get_loss

# Function to create Keras callbacks
def _get_callbacks(callbacks_dict: DictConfig, output_dir: str = None, logs_dir: str = None,
                   saved_models_dir: str = None) -> List[tf.keras.callbacks.Callback]:
    """
    Creates a list of Keras callbacks for training.

    Args:
        callbacks_dict (DictConfig): Configuration for callbacks.
        output_dir (str): Directory for saving outputs.
        logs_dir (str): Directory for saving logs.
        saved_models_dir (str): Directory for saving models.

    For each callback, the attributes and their values used in the config
    file are used to create a string that is the callback instantiation as
    it would be written in a Python script. Then, the string is evaluated.
    If the evaluation succeeds, the callback object is returned. If it fails,
    an error is thrown with a message saying that the name and/or arguments
    of the callback are incorrect.

    Returns:
        List[tf.keras.callbacks.Callback]: List of callbacks.
    """
    message = "\nPlease check the 'training.callbacks' section of your configuration file."
    lr_scheduler_names = lr_schedulers.get_scheduler_names()
    num_lr_schedulers = 0

    # Generate the callbacks used in the config file (there may be none)
    callback_list = []
    if callbacks_dict is not None:
        if type(callbacks_dict) != DefaultMunch:
            raise ValueError(f"\nInvalid callbacks syntax{message}")
        for name in callbacks_dict.keys():
            if name in ("ModelCheckpoint", "TensorBoard", "CSVLogger"):
                raise ValueError(f"\nThe `{name}` callback is built-in and can't be redefined.{message}")
            elif name in lr_scheduler_names:
                text = f"lr_schedulers.{name}"
            elif name == 'MultiResCallback':
                text = f"{name}"
            else:
                text = f"tf.keras.callbacks.{name}"

            # Add the arguments to the callback string
            # and evaluate it to get the callback object
            text += collect_callback_args(name, args=callbacks_dict[name], message=message)
            try:
                callback = eval(text)
            except ValueError as error:
                raise ValueError(f"\nThe callback name `{name}` is unknown, or its arguments are incomplete "
                                 f"or invalid\nReceived: {text}{message}") from error
            callback_list.append(callback)

            if name in lr_scheduler_names + ["ReduceLROnPlateau", "LearningRateScheduler"]:
                num_lr_schedulers += 1
            
    # Check that there is only one scheduler
    if num_lr_schedulers > 1:
        raise ValueError(f"\nFound more than one learning rate scheduler{message}")

    # Add built-in callbacks that saves the best model obtained so far
    callback_list.append(tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(output_dir, saved_models_dir, "best_model.keras"),
        save_best_only=True,
        monitor="val_accuracy",
        mode="max"
    ))
    # Add the Keras callback that saves the model at the end of the epoch
    callback_list.append(tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(output_dir, saved_models_dir, "last_model.keras"),
        save_best_only=False,
        monitor="val_accuracy",
        mode="max"
    ))
    # Add the TensorBoard callback
    callback_list.append(LRTensorBoard(log_dir=os.path.join(output_dir, logs_dir)))
    # Add the CVSLogger callback (must be last in the list 
    # of callbacks to make sure it records the learning rate)
    callback_list.append(tf.keras.callbacks.CSVLogger(os.path.join(output_dir, logs_dir, "metrics", "train_metrics.csv")))

    return callback_list


# Main class for training image classification models
class HARTrainer:
    def __init__(self, cfg, model=None, dataloaders=None):
        """
        Initializes the trainer with configuration, model, and datasets.

        Args:
            cfg: Configuration object.
            model: TensorFlow model.
            dataloaders: Dictionary containing training, validation, and test datasets.
        """
        self.cfg = cfg
        self.model = model
        self.train_ds = dataloaders['train']
        self.valid_ds = dataloaders['valid']
        self.test_ds = dataloaders['test']

        self.output_dir = HydraConfig.get().runtime.output_dir
        self.saved_models_dir = cfg.general.saved_models_dir
        self.class_names = cfg.dataset.class_names
        self.num_classes = len(self.class_names)
        self.callbacks = None
        self.history = None

    def prepare(self):
        """
        Prepares the model, datasets, and callbacks for training.
        """
        # Print dataset statistics
        print("Dataset stats:")
        train_size = sum([x.shape[0] for x, _ in self.train_ds])
        valid_size = sum([x.shape[0] for x, _ in self.valid_ds])
        if self.test_ds:
            test_size = sum([x.shape[0] for x, _ in self.test_ds])

        print("  classes:", self.num_classes)
        print("  training set size:", train_size)
        print("  validation set size:", valid_size)
        if self.test_ds:
            print("  test set size:", test_size)
        else:
            print("  no test set")

        # Log dataset information
        if self.cfg.dataset.dataset_name:
            log_to_file(self.output_dir, f"Dataset : {self.cfg.dataset.dataset_name}")

        # Prepare the model
        if self.cfg.model:
            cfm = self.cfg.model
            print(f"[INFO] : Using `{cfm.model_name}` model")
            log_to_file(self.cfg.output_dir, (f"Model name : {cfm.model_name}"))
        elif self.cfg.model.model_path:
            # self.model = change_model_number_of_classes(self.model, self.num_classes)
            print(f"[INFO] : Initialized model with weights from model file {self.cfg.model.model_path}")
            log_to_file(self.cfg.output_dir, (f"Weights from model file : {self.cfg.model.model_path}"))

        model_summary(self.model)

        # Generate callbacks
        self.callbacks = _get_callbacks(
            callbacks_dict=self.cfg.training.callbacks,
            output_dir=self.output_dir,
            saved_models_dir=self.saved_models_dir,
            logs_dir=self.cfg.general.logs_dir
        )

    def enable_determinism(self):
        """
        Enables deterministic operations for reproducibility.
        """
        if self.cfg.general.deterministic_ops:
            tf.config.experimental.enable_op_determinism()

    def fit(self):
        """
        Trains the model using the training dataset.
        """
        print("Starting training...")
        start_time = timer()
        steps_per_epoch = self.cfg.training.dryrun if self.cfg.training.dryrun else None
        self.model.compile(
            optimizer=get_optimizer(self.cfg.training.optimizer),
            loss=get_loss(self.num_classes),
            metrics=['accuracy']
        )
        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.valid_ds,
            epochs=self.cfg.training.epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=self.callbacks
        )
        last_epoch = log_last_epoch_history(self.cfg, self.output_dir)
        end_time = timer()
        fit_run_time = int(end_time - start_time)
        average_time_per_epoch = round(fit_run_time / (int(last_epoch) + 1), 2)
        print("Training runtime: " + str(timedelta(seconds=fit_run_time)))
        log_to_file(self.cfg.output_dir, (
            f"Training runtime : {fit_run_time} s\n" +
            f"Average time per epoch : {average_time_per_epoch} s"
        ))
        vis_training_curves(history=self.history, output_dir=self.output_dir)

    def save_and_evaluate(self):
        """
        Saves the best model and evaluates it on validation and test datasets.
        """
        # Load the best model checkpoint and return it
        models_dir = os.path.join(self.output_dir, self.saved_models_dir)
        checkpoint_filepath = os.path.join(models_dir, "best_model.keras")
        best_model = tf.keras.models.load_model(checkpoint_filepath)
        setattr(best_model, 'model_path', checkpoint_filepath)
        print('[INFO] : Training complete.')
        return best_model

    def train(self):
        """
        Executes the full training pipeline: prepare, train, save, and evaluate.
        """
        self.prepare()
        self.enable_determinism()
        self.fit()
        return self.save_and_evaluate()
