# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
import sys
from pathlib import Path
from timeit import default_timer as timer
from datetime import timedelta
from typing import Tuple, List, Dict, Optional

from hydra.core.hydra_config import HydraConfig
from munch import DefaultMunch
from omegaconf import DictConfig
import numpy as np
import tensorflow as tf

from common.utils import log_to_file, log_last_epoch_history, LRTensorBoard, check_training_determinism, \
                         model_summary, collect_callback_args, vis_training_curves
from common.training import set_frozen_layers, get_optimizer, set_all_layers_trainable_parameter
from hand_posture.tf.src.utils import get_loss
from hand_posture.tf.src.data_augmentation import get_data_augmentation



def _add_preprocessing_layers(
                model: tf.keras.Model,
                input_shape: Tuple = None,
                data_augmentation: Dict = None,
                batches_per_epoch: float = None):
    """
    This function adds the rescaling and data augmentation preprocessing layers.

    Arguments:
        model (tf.keras.Model): the model preprocessing layers will be added to.
        input_shape (Tuple): input shape of the model.
        scale (float): scale to use for rescaling the images.
        offset (float): offset to use for rescaling the images.
        data_augmentation (Dict): dictionary containing the data augmentation functions.
        batches_per_epoch: number of training batches per epoch.
        
    Returns:
        None
    """

    model_layers = []

    # Get data augmentation //
    if data_augmentation:
        model_layers.append(tf.keras.Input(shape=input_shape))
        model_layers.append(get_data_augmentation(data_augmentation))
        model_layers.append(model)
    else:
        model_layers.append(model)

    augmented_model = tf.keras.Sequential(model_layers, name="augmented_model")
    augmented_model.summary()

    return augmented_model


def _get_callbacks(callbacks_dict: DictConfig, output_dir: str = None, logs_dir: str = None,
                  saved_models_dir: str = None) -> List[tf.keras.callbacks.Callback]:
    """
    This function creates the list of Keras callbacks to be passed to 
    the fit() function including:
      - the Model Zoo built-in callbacks that can't be redefined by the
        user (ModelCheckpoint, TensorBoard, CSVLogger).
      - the callbacks specified in the config file that can be either Keras
        callbacks or custom callbacks (learning rate schedulers).

    For each callback, the attributes and their values used in the config
    file are used to create a string that is the callback instantiation as
    it would be written in a Python script. Then, the string is evaluated.
    If the evaluation succeeds, the callback object is returned. If it fails,
    an error is thrown with a message saying that the name and/or arguments
    of the callback are incorrect.

    The user may use the ModelSaver callback to periodically save the model 
    at the end of an epoch. If it is not used, a default ModelSaver is added
    that saves the model at the end of each epoch. The model file is named
    last_model.h5 and is saved in the output_dir/saved_models_dir directory
    (same directory as best_augmented_model.h5 and best_model.h5).

    The function also checks that there is only one learning rate scheduler
    in the list of callbacks.

    Args:
        callbacks_dict (DictConfig): dictionary containing the 'training.callbacks'
                                     section of the configuration.
        output_dir (str): directory root of the tree where the output files are saved.
        logs_dir (str): directory the TensorBoard logs and training metrics are saved.
        saved_models_dir (str): directory where the trained models are saved.

    Returns:
        List[tf.keras.callbacks.Callback]: A list of Keras callbacks to pass
                                           to the fit() function.
    """

    message = "\nPlease check the 'training.callbacks' section of your configuration file."
    num_lr_schedulers = 0

    # Generate the callbacks used in the config file (there may be none)
    callback_list = []
    if callbacks_dict is not None:
        if type(callbacks_dict) != DefaultMunch:
            raise ValueError(f"\nInvalid callbacks syntax{message}")
        for name in callbacks_dict.keys():
            if name in ("ModelCheckpoint", "TensorBoard", "CSVLogger"):
                raise ValueError(f"\nThe `{name}` callback is built-in and can't be redefined.{message}")
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

            if name in ["ReduceLROnPlateau", "LearningRateScheduler"]:
                num_lr_schedulers += 1

    # Check that there is only one scheduler
    if num_lr_schedulers > 1:
        raise ValueError(f"\nFound more than one learning rate scheduler{message}")

    # Add the Keras callback that saves the best model obtained so far
    callback = tf.keras.callbacks.ModelCheckpoint(
                        filepath=os.path.join(output_dir, saved_models_dir, "best_augmented_model.keras"),
                        save_best_only=True,
                        save_weights_only=False,
                        monitor="val_accuracy",
                        mode="max")
    callback_list.append(callback)

    # Add the Keras callback that saves the model at the end of the epoch
    # callback = tf.keras.callbacks.ModelCheckpoint(
    #                     filepath=os.path.join(output_dir, saved_models_dir, "last_augmented_model.keras"),
    #                     save_best_only=False,
    #                     save_weights_only=False,
    #                     monitor="val_accuracy",
    #                     mode="max")
    # callback_list.append(callback)

    # Add the TensorBoard callback
    callback = LRTensorBoard(log_dir=os.path.join(output_dir, logs_dir))
    callback_list.append(callback)

    # Add the CVSLogger callback (must be last in the list
    # of callbacks to make sure it records the learning rate)
    callback = tf.keras.callbacks.CSVLogger(os.path.join(output_dir, logs_dir, "metrics", "train_metrics.csv"))
    callback_list.append(callback)

    return callback_list

# Main class for training image classification models
class HPRTrainer:
    """ Class to train hand posture recognition models using TensorFlow.
    """
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
        if self.cfg.model.model_name:
            cfm = self.cfg.model
            print(f"[INFO] : Using `{cfm.model_name}` model")
            log_to_file(self.cfg.output_dir, (f"Model name : {cfm.model_name}"))
            print("[INFO] : No pretrained weights were loaded, training from randomly initialized weights.")
        
        elif self.cfg.model.model_path:
            if self.cfg.training.resume_training:
                print(f"[INFO] : Resuming training from model file {self.cfg.model.model_path}")
                log_to_file(self.cfg.output_dir, (f"Model file : {self.cfg.model.model_path }"))
            else: 
                print(f"[INFO] : Initialized model with weights from model file {self.cfg.model.model_path}")
                log_to_file(self.cfg.output_dir, (f"Weights from model file : {self.cfg.model.model_path}"))

        set_all_layers_trainable_parameter(self.model, trainable=True)
        if self.cfg.training.frozen_layers:
            set_frozen_layers(self.model, frozen_layers=self.cfg.training.frozen_layers)
        
        # display model summary
        if self.cfg.training.resume_training:
            model_summary(self.model)
            # Show summary for all layers except the first
            for layer in self.model.layers[1:]:
                model_summary(layer)
        else:
            model_summary(self.model)

        # get model input shape
  
        if not self.cfg.training.resume_training:
            # add preprocessing layers
            self.augmented_model = _add_preprocessing_layers(
                                                        model=self.model,
                                                        input_shape=self.cfg.model.input_shape,
                                                        data_augmentation=self.cfg.data_augmentation,
                                                        batches_per_epoch=len(self.train_ds)
                                                        )


        else: # If we're resuming training we don't need to reappend the data augmentation layers.
           self.augmented_model = self.model
        
        # Compile the augmented model
        self.augmented_model.compile(loss=get_loss(num_classes=self.num_classes),
                                                   metrics=['accuracy'],
                                                   optimizer=get_optimizer(cfg=self.cfg.training.optimizer))
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
        history = self.augmented_model.fit(self.train_ds,
                                    validation_data=self.valid_ds,
                                    epochs=self.cfg.training.epochs,
                                    steps_per_epoch=steps_per_epoch, 
                                    callbacks=self.callbacks)

        # print('\n[INFO] : Training interrupted')  
        #save the last epoch history in the log file
        last_epoch=log_last_epoch_history(self.cfg, self.output_dir)
        end_time = timer()
        
        #calculate and log the runtime in the log file
        fit_run_time = int(end_time - start_time)
        average_time_per_epoch = round(fit_run_time / (int(last_epoch) + 1),2)
        print("Training runtime: " + str(timedelta(seconds=fit_run_time))) 
        log_to_file(self.cfg.output_dir,
                    (f"Training runtime : {fit_run_time} s\n" + f"Average time per epoch : {average_time_per_epoch} s"))          

        # Visualize training curves
        vis_training_curves(history=history, output_dir=self.cfg.output_dir)
    

    def save_and_evaluate(self):
        """
        Saves the best model and evaluates it on validation and test datasets.
        """
        # Load the checkpoint model (best model obtained)
        models_dir = os.path.join(self.output_dir, self.saved_models_dir)
        checkpoint_filepath = os.path.join(models_dir, "best_augmented_model.keras")

        checkpoint_model = tf.keras.models.load_model(checkpoint_filepath,)
 
        # Get the checkpoint model w/o preprocessing layers
        best_model = checkpoint_model.layers[-1]
        best_model.compile(loss=get_loss(self.num_classes), metrics=['accuracy'])
        best_model_path = os.path.join(self.output_dir,
                                       "{}/{}".format(self.saved_models_dir, "best_model.keras"))
        best_model.save(best_model_path)
        setattr(best_model, 'model_path', best_model_path)
        return best_model

    def train(self):
        """
        Executes the full training pipeline: prepare, train, save, and evaluate.
        """
        self.prepare()
        self.enable_determinism()
        self.fit()
        return self.save_and_evaluate()