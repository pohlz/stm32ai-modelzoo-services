# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
from timeit import default_timer as timer
from datetime import timedelta
from typing import List
from pathlib import Path
from hydra.core.hydra_config import HydraConfig
from munch import DefaultMunch
from omegaconf import DictConfig
import tensorflow as tf
import logging


from common.utils import log_to_file, log_last_epoch_history, LRTensorBoard, check_training_determinism, \
                         model_summary, collect_callback_args, vis_training_curves
from common.training import get_optimizer, lr_schedulers, set_all_layers_trainable_parameter
from audio_event_detection.tf.src.utils import get_loss, AED_CUSTOM_OBJECTS
from audio_event_detection.tf.src.data_augmentation import get_data_augmentation

# Suppress TensorFlow warnings to reduce log clutter
logging.getLogger('mlflow.tensorflow').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)


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
            if name in lr_scheduler_names:
                text = f"lr_schedulers.{name}"
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

    # Add the Keras callback that saves the best model obtained so far
    callback = tf.keras.callbacks.ModelCheckpoint(
                        filepath=os.path.join(output_dir, saved_models_dir, "best_augmented_model.keras"),
                        save_best_only=True,
                        save_weights_only=False,
                        monitor="val_accuracy",
                        mode="max")
    callback_list.append(callback)

    # Add the Keras callback that saves the model at the end of the epoch
    callback = tf.keras.callbacks.ModelCheckpoint(
                        filepath=os.path.join(output_dir, saved_models_dir, "last_augmented_model.keras"),
                        save_best_only=False,
                        save_weights_only=False,
                        monitor="val_accuracy",
                        mode="max")
    callback_list.append(callback)

    # Add the TensorBoard callback
    callback = LRTensorBoard(log_dir=os.path.join(output_dir, logs_dir))
    callback_list.append(callback)

    # Add the CVSLogger callback (must be last in the list 
    # of callbacks to make sure it records the learning rate)
    callback = tf.keras.callbacks.CSVLogger(os.path.join(output_dir, logs_dir, "metrics", "train_metrics.csv"))
    callback_list.append(callback)

    return callback_list


class AEDTrainer:
    """
    Trainer for Audio Event Detection (AED) models.

    Notes
    -----
    Coordinates callbacks, augmentation, compilation, fitting, and checkpoint
    handling for Keras-based AED models.
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
        self.train_ds = dataloaders['train_ds']

        self.val_ds = dataloaders['val_ds']
        self.val_clip_labels = dataloaders['val_clip_labels']

        self.output_dir = HydraConfig.get().runtime.output_dir
        self.saved_models_dir = cfg.general.saved_models_dir
        self.class_names = cfg.dataset.class_names
        self.num_classes = len(self.class_names)
        self.augmented_model = None
        self.history = None

        # Prepare (including an optional resume-time unfreeze) before printing
        # the summary so the reported trainable counts match actual training.
        self._prepare_model()
        self._log_and_print_info()
        self.callbacks = _get_callbacks(callbacks_dict=cfg.training.callbacks,
                              output_dir=self.output_dir,
                              saved_models_dir=self.saved_models_dir,
                              logs_dir=cfg.general.logs_dir)
        
        self._enable_determinism()

    def train(self):
        """
        Run full training workflow: fit, visualize curves, and save checkpoints.

        Returns
        -------
        tf.keras.Model
            The best model (without preprocessing layers), compiled for evaluation.
        """
        self._fit()
        # Visualize training curves
        vis_training_curves(history=self.history, output_dir=self.output_dir)
        self._save_checkpoints()
        print("[INFO] : Training complete.")
        return self.best_model


    def _save_checkpoints(self):
        """
        Save best checkpointed model under `best_model.keras`.

        Returns
        -------
        None
        """
        # Load the checkpoint model (best model obtained)
        models_dir = os.path.join(self.output_dir, self.saved_models_dir)
        checkpoint_filepath = Path(models_dir) / "best_augmented_model.keras"

        checkpoint_model = tf.keras.models.load_model(
            checkpoint_filepath,
            custom_objects=AED_CUSTOM_OBJECTS)

        # Get the checkpoint model w/o preprocessing layers

        self.best_model = checkpoint_model.layers[-1]
        self.best_model.compile(loss=get_loss(multi_label=self.cfg.dataset.multi_label),
                        metrics=['accuracy'])
        best_model_path = Path(self.output_dir) / self.saved_models_dir / "best_model.keras"
        self.best_model.save(best_model_path)
        setattr(self.best_model, 'model_path', str(best_model_path))


    def _fit(self):
        """
        Fit the augmented model and record training history.

        Returns
        -------
        None
        """
        # Train the model
        print("Starting training...")
        start_time = timer()
        steps_per_epoch = self.cfg.training.dryrun if self.cfg.training.dryrun else None
        self.history = self.augmented_model.fit(self.train_ds,
                                        validation_data=self.val_ds,
                                        epochs=self.cfg.training.epochs,
                                        steps_per_epoch=steps_per_epoch,
                                        callbacks=self.callbacks)
    
        #save the last epoch history in the log file
        last_epoch=log_last_epoch_history(self.cfg, self.output_dir)
        end_time = timer()
        fit_run_time = int(end_time - start_time)
        average_time_per_epoch = round(fit_run_time / (int(last_epoch) + 1),2)
        print("Training runtime: " + str(timedelta(seconds=fit_run_time)))
        log_to_file(self.output_dir, (f"Training runtime : {fit_run_time} s\n" + f"Average time per epoch : {average_time_per_epoch} s"))          


    def _log_and_print_info(self):
        ''' Logs and prints various info about the model & the dataset used for training
            and prints a summary of the model
        '''
        # Info messages about the model that was loaded
        if self.cfg.model.model_name:
            print(f"[INFO] : Using `{self.cfg.model.model_name}` model")
            log_to_file(self.cfg.output_dir, (f"Model name : {self.cfg.model.model_name}"))  
            if self.cfg.model.pretrained:
                print(f"[INFO] : Initialized model with pretrained weights")
                log_to_file(self.cfg.output_dir,(f"Using pretrained weights"))
            else:
                print("[INFO] : No pretrained weights were loaded, training from randomly initialized weights.")
        elif self.cfg.model.model_path:
            if self.cfg.training.resume_training:
                print(f"[INFO] : Resuming training from model file {self.cfg.model.model_path}")
                log_to_file(self.cfg.output_dir, (f"Model file : {self.cfg.model.model_path }"))
            else: 
                print(f"[INFO] : Initialized model with weights from model file {self.cfg.model.model_path}")
                log_to_file(self.cfg.output_dir, (f"Weights from model file : {self.cfg.model.model_path}"))
        if self.cfg.dataset.dataset_name: 
            log_to_file(self.output_dir, f"Dataset : {self.cfg.dataset.dataset_name}")

        # Display a summary of the model
        if self.cfg.training.resume_training:
            model_summary(self.model)
            # Resumed augmented models may contain zero, one, or several
            # preprocessing layers. Show the innermost model without assuming
            # that it is always stored at outer layer index 1 or 2.
            for layer in reversed(self.model.layers):
                if isinstance(layer, tf.keras.Model):
                    model_summary(layer)
                    break
        else:
            model_summary(self.model)

    def _prepare_model(self):
        """
        Build the augmented model with optional data augmentation layers and compile it.

        Returns
        -------
        None
        """
        # Initialize the augmented model that includes data augmentation layers
        model_layers = []

        if not self.cfg.training.resume_training:
            # Append eventual data augmentation layers to the model
            if self.cfg.data_augmentation:
                data_augmentation_layers = get_data_augmentation(
                    cfg=self.cfg.data_augmentation,
                    db_scale=self.cfg.feature_extraction.to_db
                    )
                model_layers.extend(data_augmentation_layers)
            # Add the actual model to the model
            model_layers.append(self.model)

            self.augmented_model = tf.keras.Sequential(model_layers, name="augmented_model")

        else: # If we're resuming training we don't need to reappend the data augmentation layers.
           self.augmented_model = self.model
           # A model saved after head-only transfer learning retains the frozen
           # state of its YAMNet backbone. Allow a second-stage resume config to
           # explicitly unfreeze the complete loaded model before recompiling.
           if self.cfg.training.fine_tune:
               set_all_layers_trainable_parameter(self.augmented_model, trainable=True)


        # Compile the model with data augmentation layers added
        self.augmented_model.compile(
            loss=get_loss(multi_label=self.cfg.dataset.multi_label),
            metrics=['accuracy'],
            optimizer=get_optimizer(cfg=self.cfg.training.optimizer)
            )
    
    def _enable_determinism(self):
        """
        Enable deterministic TensorFlow ops when configured and verify determinism.

        Returns
        -------
        None
        """
        # check if determinism can be enabled
        if self.cfg.general.deterministic_ops:
            sample_ds = self.train_ds.take(1)
            tf.config.experimental.enable_op_determinism()
            if not check_training_determinism(self.augmented_model, sample_ds):
                print("[WARNING]: Some operations cannot be run deterministically. Setting deterministic_ops to False.")
                tf.config.experimental.enable_op_determinism.__globals__["_pywrap_determinism"].enable(False)



