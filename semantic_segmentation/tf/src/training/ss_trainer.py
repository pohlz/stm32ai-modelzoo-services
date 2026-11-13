# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2025 STMicroelectronics.
#  * All rights reserved.
#  *--------------------------------------------------------------------------------------------*/

import os
from pathlib import Path
from timeit import default_timer as timer
from datetime import timedelta
from typing import List, Optional, Dict

import tensorflow as tf
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from munch import DefaultMunch

# Suppress TF warnings
import logging
logging.getLogger('mlflow.tensorflow').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from common.utils import (
    log_to_file, log_last_epoch_history, LRTensorBoard,
    model_summary, collect_callback_args, vis_training_curves,
    check_training_determinism
)
from common.training import set_frozen_layers, get_optimizer, lr_schedulers
from semantic_segmentation.tf.src.utils import change_model_number_of_classes
from semantic_segmentation.tf.src.training.train_model import SegmentationTrainingModel


# Define a custom callback for multi-resolution training
class MultiResCallback(tf.keras.callbacks.Callback):
    """
    A custom Keras callback to dynamically change the input resolution
    of the model during training.

    Args:
        image_sizes (List[int]): List of resolutions to cycle through.
        period (int): Number of batches before changing resolution.
        name (str, optional): Name of the callback.
    """
    def __init__(self, image_sizes, period, name=None):
        super().__init__()
        self.resolutions = image_sizes
        self.period = period

    def on_train_batch_begin(self, batch, logs=None):
        # Change the resolution of the input layer based on the batch number
        res = self.resolutions[((batch - 1) // self.period) % len(self.resolutions)]
        self.model.set_resolution(res)


class SaveBaseModelWithWeightsCallback(tf.keras.callbacks.Callback):
    """
    Custom callback to save the base_model as a full Keras model when a new best is found.
    """
    def __init__(self, base_model, save_path, filepath=None):
        super().__init__()
        self.base_model = base_model
        self.save_path = save_path
        self.filepath = filepath
        
    def on_epoch_end(self, epoch, logs=None):
        self.base_model.load_weights(self.filepath)
        self.base_model.save(self.save_path)



def _get_callbacks(callbacks_dict: DictConfig, base_model: tf.keras.Model = None, output_dir: str = None, logs_dir: str = None,
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

    # Add the Keras callback that saves the best model obtained so far
    callback = tf.keras.callbacks.ModelCheckpoint(
                        filepath=os.path.join(output_dir, saved_models_dir, "best_weights.weights.h5"),
                        save_best_only=True,save_weights_only=True,monitor="val_loss", mode="min")
    callback_list.append(callback)

    # Add the Keras callback that saves the model at the end of the epoch
    callback = tf.keras.callbacks.ModelCheckpoint(
                        filepath=os.path.join(output_dir, saved_models_dir, "last_weights.weights.h5"),
                        save_best_only=False,save_weights_only=True,monitor="val_loss",mode="min")
    callback_list.append(callback)

    # Add the custom callback to save the base_model as a full Keras model when a new best is found
    callback = SaveBaseModelWithWeightsCallback(
                    base_model=base_model,
                    save_path=os.path.join(output_dir, saved_models_dir, "best_model.keras"),
                    filepath=os.path.join(output_dir, saved_models_dir, "best_weights.weights.h5")
                )
    callback_list.append(callback)

    # Add the custom callback to save the base_model as a full Keras model at the end of each epoch
    callback = SaveBaseModelWithWeightsCallback(
                    base_model=base_model,
                    save_path=os.path.join(output_dir, saved_models_dir, "last_model.keras"),
                    filepath=os.path.join(output_dir, saved_models_dir, "last_weights.weights.h5")
                )
    callback_list.append(callback)
    # Add the TensorBoard callback
    callback = LRTensorBoard(log_dir=os.path.join(output_dir, logs_dir))
    callback_list.append(callback)

    # Add the CVSLogger callback (must be last in the list 
    # of callbacks to make sure it records the learning rate)
    callback = tf.keras.callbacks.CSVLogger(os.path.join(output_dir, logs_dir, "metrics", "train_metrics.csv"))
    callback_list.append(callback)
    return callback_list



class SSTrainer:
    """
    Object-oriented semantic segmentation trainer.

    Public workflow:
        trainer.prepare()
        trainer.enable_determinism()
        trainer.fit()
        best_model = trainer.save_and_evaluate()
        # or simply: best_model = trainer.train()

    SegmentationTrainingModel wraps base model with preprocessing and data augmentation.
    """
    def __init__(self, cfg: DictConfig, model: tf.keras.Model, dataloaders: Dict[str, tf.data.Dataset]):
        """
        Initialize trainer with configuration, base model and dataloaders.

        Args:
            cfg: Hydra DictConfig containing all sections.
            model: Base segmentation backbone/head tf.keras.Model.
            dataloaders: Dict with keys 'train', 'valid', optional 'test' mapping to tf.data.Dataset.
        """
        self.cfg = cfg
        self.base_model = model
        self.train_ds = dataloaders.get('train')
        self.valid_ds = dataloaders.get('valid')
        self.test_ds = dataloaders.get('test')
        self.output_dir = Path(HydraConfig.get().runtime.output_dir)
        self.saved_models_dir = os.path.join(self.output_dir, cfg.general.saved_models_dir)
        self.callbacks = None
        self.history = None
        self.train_model = None
        self.class_names = cfg.dataset.class_names
        self.num_classes = len(self.class_names)

    def prepare(self):
        """
        Prepare training artifacts:
          - Create output directories.
          - Log dataset/model info.
          - Adjust number of classes.
          - Freeze layers if requested.
          - Wrap model in SegmentationTrainingModel (adds preprocessing & augmentation).
          - Compile wrapped model.
          - Instantiate callbacks.
        """
        Path(self.saved_models_dir).mkdir(parents=True, exist_ok=True)
        train_batches = sum(1 for _ in self.train_ds) if self.train_ds is not None else 0
        valid_batches = sum(1 for _ in self.valid_ds) if self.valid_ds is not None else 0
        test_batches = sum(1 for _ in self.test_ds) if self.test_ds is not None else 0
        print("Dataset stats:")
        print("  classes:", self.num_classes)
        print("  training batches:", train_batches)
        print("  validation batches:", valid_batches)
        print("  test batches:" if self.test_ds else "  no test set", test_batches if self.test_ds else "")

        if getattr(self.cfg.model, "model_path", None):
            log_to_file(self.output_dir, f"Model file : {self.cfg.model.model_path}")
        if getattr(self.cfg.dataset, "dataset_name", None):
            log_to_file(self.output_dir, f"Dataset : {self.cfg.dataset.dataset_name}")

        self.base_model = change_model_number_of_classes(self.base_model, self.cfg.dataset.num_classes)
        self.base_model.compile()
        base_model_path = os.path.join(self.saved_models_dir, "base_model.keras")
        self.base_model.save(base_model_path)

        if getattr(self.cfg.training, "frozen_layers", None):
            set_frozen_layers(self.base_model, frozen_layers=self.cfg.training.frozen_layers)

        model_summary(self.base_model)

        scale = self.cfg.preprocessing.rescaling.scale
        offset = self.cfg.preprocessing.rescaling.offset
        pixels_range = (offset, scale * 255 + offset)

        self.train_model = SegmentationTrainingModel(
            self.base_model,
            num_classes=self.num_classes,
            image_size=self.cfg.model.input_shape[:2],
            data_augmentation_cfg=self.cfg.data_augmentation.config,
            loss_weights=None,
            pixels_range=pixels_range
        )
        self.train_model.compile(optimizer=get_optimizer(self.cfg.training.optimizer))
        # Configure MultiResCallback if applicable
        data_aug_args = DefaultMunch.fromDict(self.cfg.data_augmentation.config)
        if data_aug_args.random_periodic_resizing is not None:
            rpr = DefaultMunch.fromDict(data_aug_args.random_periodic_resizing)
            if rpr.image_sizes is not None:
                self.cfg.training.callbacks['MultiResCallback'] = DefaultMunch.fromDict({
                    'image_sizes': rpr.image_sizes,
                    'period': rpr.period if rpr.period is not None else 10
                })
            else:
                print("[WARNING]: 'random_periodic_resizing' can't be used because [image_sizes] argument is missing.")
        self.callbacks = _get_callbacks(
            callbacks_dict=self.cfg.training.callbacks,
            base_model=self.base_model,
            output_dir=self.output_dir,
            logs_dir=self.cfg.general.logs_dir,
            saved_models_dir=self.saved_models_dir
        )

    def enable_determinism(self):
        """
        Enable deterministic TensorFlow operations if cfg.general.deterministic_ops is True.

        Falls back to non-deterministic if verification fails.
        """
        if getattr(self.cfg.general, "deterministic_ops", False):
            sample = self.train_ds.take(1)
            tf.config.experimental.enable_op_determinism()
            if not check_training_determinism(self.train_model, sample):
                print("[WARNING] Some ops are not deterministic, disabling determinism.")
                tf.config.experimental.enable_op_determinism.__globals__['_pywrap_determinism'].enable(False)

    def fit(self):
        """
        Execute Keras fit loop on wrapped training model.

        Handles optional dry-run (steps_per_epoch override), logs runtime,
        and records final epoch metrics. Optionally plots curves.
        """
        print("[INFO] : Starting training")
        start_time = timer()
        steps_per_epoch = self.cfg.training.dryrun if getattr(self.cfg.training, "dryrun", None) else None
        self.history = self.train_model.fit(
            self.train_ds,
            validation_data=self.valid_ds,
            epochs=self.cfg.training.epochs,
            callbacks=self.callbacks,
            steps_per_epoch=steps_per_epoch
        )
        last_epoch = log_last_epoch_history(self.cfg, self.output_dir)
        end_time = timer()
        fit_run_time = int(end_time - start_time)
        avg_time = round(fit_run_time / (int(last_epoch) + 1), 2)
        print("Training runtime:", str(timedelta(seconds=fit_run_time)))
        log_to_file(self.output_dir, f"Training runtime : {fit_run_time} s\nAverage time per epoch : {avg_time} s")
        if self.cfg.general.display_figures:
            vis_training_curves(history=self.history, output_dir=self.output_dir)

    def save(self):
        """
        Save best and last models by loading stored weights into base model.

        Evaluates best model on validation and test datasets if provided.

        Returns:
            best_model (tf.keras.Model): Unwrapped model loaded with best weights.
        """
        best_weights_path = os.path.join(self.saved_models_dir, "best_weights.weights.h5")
        last_weights_path = os.path.join(self.saved_models_dir, "last_weights.weights.h5")
        best_model_path = os.path.join(self.saved_models_dir, "best_model.keras")
        last_model_path = os.path.join(self.saved_models_dir, "last_model.keras")

        self.base_model.load_weights(best_weights_path)
        self.base_model.save(best_model_path)
        self.base_model.load_weights(last_weights_path)
        self.base_model.save(last_model_path)

        print("[INFO] Saved trained models:")
        print("  best model:", best_model_path)
        print("  last model:", last_model_path)

        best_model = tf.keras.models.load_model(best_model_path, compile=False)
        setattr(best_model, 'model_path', best_model_path)
        print('[INFO] : Training complete.')
        return best_model

    def train(self):
        """
        Convenience orchestration method running:
          prepare -> enable_determinism -> fit -> save_and_evaluate

        Returns:
            best_model (tf.keras.Model)
        """
        self.prepare()
        self.enable_determinism()
        self.fit()
        return self.save()
