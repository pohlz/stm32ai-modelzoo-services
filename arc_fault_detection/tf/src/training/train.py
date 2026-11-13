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

from common.training import get_optimizer, lr_schedulers
from arc_fault_detection.tf.src.evaluation import AFDKerasEvaluator


def _get_layer_weight_shapes(layer: tf.keras.layers.Layer) -> List[Tuple[int, ...]]:
    """
    Return a list of weight shapes for a given Keras layer.

    Args:
        layer (tf.keras.layers.Layer): Layer to inspect.

    Returns:
        List[Tuple[int, ...]]: Weight shapes for the layer.
    """
    return [tuple(weight.shape) for weight in layer.weights]


def _are_models_compatible(base_model: tf.keras.Model, resume_model: tf.keras.Model) -> Tuple[bool, str]:
    """
    Compare two Keras models for compatibility based on layer name/type/weight shapes.
    """
    if len(base_model.layers) != len(resume_model.layers):
        return False, (
            f"Layer count mismatch: {len(base_model.layers)} vs {len(resume_model.layers)}"
        )

    for idx, (base_layer, resume_layer) in enumerate(zip(base_model.layers, resume_model.layers)):
        if base_layer.name != resume_layer.name:
            return False, (
                f"Layer name mismatch at index {idx}: {base_layer.name} vs {resume_layer.name}"
            )
        if base_layer.__class__.__name__ != resume_layer.__class__.__name__:
            return False, (
                f"Layer type mismatch at index {idx} ({base_layer.name}): "
                f"{base_layer.__class__.__name__} vs {resume_layer.__class__.__name__}"
            )
        base_shapes = _get_layer_weight_shapes(base_layer)
        resume_shapes = _get_layer_weight_shapes(resume_layer)
        if base_shapes != resume_shapes:
            return False, (
                f"Weight shapes mismatch at index {idx} ({base_layer.name}): "
                f"{base_shapes} vs {resume_shapes}"
            )

    return True, "Models are compatible."


def _load_keras_model_for_resume(model_path: str) -> Optional[tf.keras.Model]:
    """
    Load a Keras model from a supported checkpoint path for resume training.

    Args:
        model_path (str): Path to a .h5 or .keras model file.

    Returns:
        Optional[tf.keras.Model]: Loaded model or None if format is unsupported.
    """
    file_extension = str(model_path).split('.')[-1]
    if file_extension not in ['h5', 'keras']:
        print(f"[WARNING] : Unsupported resume model format: {model_path}")
        return None
    return tf.keras.models.load_model(model_path, compile=False)

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
    last_model.keras and is saved in the output_dir/saved_models_dir directory
    (same directory as best_model.keras).

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
                        filepath=os.path.join(output_dir, saved_models_dir, "best_model.keras"),
                        save_best_only=True,
                        save_weights_only=False,
                        monitor="val_accuracy",
                        mode="max")
    callback_list.append(callback)

    # Add the Keras callback that saves the model at the end of the epoch
    callback = tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(output_dir, saved_models_dir, "last_epoch_model.keras"),
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
    callback = tf.keras.callbacks.CSVLogger(os.path.join(output_dir,
                                                         logs_dir, "metrics", "train_metrics.csv"))
    callback_list.append(callback)

    return callback_list


def train(cfg: DictConfig = None, model: tf.keras.Model = None, dataloaders: dict = None) -> tf.keras.Model:
    """
    Train a model using the provided configuration and dataloaders.

    Args:
        cfg (DictConfig): The entire configuration file dictionary.
        model (tf.keras.Model): Model instance to train.
        dataloaders (dict): Dict of datasets (expects at least `train` and `valid`,
            and optionally `test`).

    Returns:
           tf.keras.Model: The best trained model, with an added `model_path` attribute.
    """

    #output_dir = cfg.output_dir
    output_dir = HydraConfig.get().runtime.output_dir
    saved_models_dir = cfg.general.saved_models_dir
    class_names = cfg.dataset.class_names

    print("Dataset stats:")
    train_size = sum([x.shape[0] for x, _ in dataloaders['train']])
    valid_size = sum([x.shape[0] for x, _ in dataloaders['valid']])
    if dataloaders.get('test'):
        test_size = sum([x.shape[0] for x, _ in dataloaders['test']])

    print("  classes:", class_names)
    print("  training set size:", train_size)
    print("  validation set size:", valid_size)
    if dataloaders.get('test'):
        print("  test set size:", test_size)
    else:
        print("  no test set")

    # Info messages about the model that was loaded
    if cfg.model.model_name:
        print(f"[INFO] : Using `{cfg.model.model_name}` model")
        log_to_file(cfg.output_dir, (f"Model name : {cfg.model.model_name}"))
        if not cfg.training.initialize_parameters_from:
            print("[INFO] : No pretrained weights were loaded, training from randomly initialized weights.")
    elif cfg.model.model_path:
        print(f"[INFO] : Loaded model file {cfg.model.model_path}")
        log_to_file(cfg.output_dir ,(f"Model file : {cfg.model.model_path}"))
        if not cfg.training.initialize_parameters_from:
            print("[INFO] : No resume training weights were loaded, training from model file weights.")
    if cfg.dataset.dataset_name:
        log_to_file(output_dir, f"Dataset : {cfg.dataset.dataset_name}")

    # Apply resume weights if provided and compatible
     
    if cfg.training.initialize_parameters_from:
        print(f"[INFO] : Resuming training from weights at {cfg.training.initialize_parameters_from}")
        log_to_file(cfg.output_dir, (f"Resume training from : {cfg.training.initialize_parameters_from}"))
        resume_model = _load_keras_model_for_resume(cfg.training.initialize_parameters_from)
        if resume_model is None:
            warning = f"Resume weights skipped: unsupported format at {cfg.training.initialize_parameters_from}"
            print(f"[WARNING] : {warning}")
            log_to_file(cfg.output_dir, (f"Resume weights warning : {warning}"))
            if cfg.model.model_name:
                print("[INFO] : Keeping randomly initialized weights.")
            else:
                print(f"[INFO] : Keeping weights from model file {cfg.model.model_path}")
        else:
            compatible, reason = _are_models_compatible(model, resume_model)
            if compatible:
                model.set_weights(resume_model.get_weights())
                print(f"[INFO] : Applied resume weights from {cfg.training.initialize_parameters_from}")
                log_to_file(cfg.output_dir, (f"Resume weights : {cfg.training.initialize_parameters_from}"))
            else:
                warning = f"Resume weights skipped: incompatible model architecture. {reason}"
                print(f"[WARNING] : {warning}")
                log_to_file(cfg.output_dir, (f"Resume weights warning : {warning}"))
                if cfg.model.model_name:
                    print("[INFO] : Keeping randomly initialized weights.")
                else:
                    print(f"[INFO] : Keeping weights from model file {cfg.model.model_path}")



    # Display a summary of the model
    model_summary(model)

    # Compile the model
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'],
                  optimizer=get_optimizer(cfg=cfg.training.optimizer))

    callbacks = _get_callbacks(callbacks_dict=cfg.training.callbacks,
                              output_dir=output_dir,
                              saved_models_dir=saved_models_dir,
                              logs_dir=cfg.general.logs_dir)

    # check if determinism can be enabled
    if cfg.general.deterministic_ops:
        sample_ds = dataloaders['train'].take(1)
        tf.config.experimental.enable_op_determinism()
        if not check_training_determinism(model, sample_ds):
            print("[WARNING]: Some operations cannot be run deterministically.\
                   Setting deterministic_ops to False.")
            tf.config.experimental.enable_op_determinism.__globals__["_pywrap_determinism"].enable(False)

    # Train the model
    print("Starting training...")
    start_time = timer()
    history = model.fit(dataloaders['train'],
                        validation_data=dataloaders['valid'],
                        epochs=cfg.training.epochs,
                        callbacks=callbacks)
    end_time = timer()
    
    #save the last epoch history in the log file
    last_epoch = log_last_epoch_history(cfg, output_dir)
    fit_run_time = int(end_time - start_time)
    average_time_per_epoch = round(fit_run_time / (int(last_epoch) + 1),2)
    print("Training runtime: " + str(timedelta(seconds=fit_run_time)))
    log_to_file(cfg.output_dir, (f"Training runtime : {fit_run_time} s\n" + f"Average time per epoch : {average_time_per_epoch} s"))

    # Visualize training curves
    vis_training_curves(history=history, output_dir=output_dir)
    best_model_path = os.path.join(output_dir, saved_models_dir, "best_model.keras")
    best_model = tf.keras.models.load_model(best_model_path)
    best_model.save(best_model_path)
    setattr(best_model, 'model_path', str(best_model_path))

    # Evaluate keras best model on the validation set
    validator = AFDKerasEvaluator(cfg=cfg, model=best_model, dataloaders={"valid": dataloaders['valid']})
    validator.evaluate()
    if dataloaders.get('test'):
        # Evaluate keras best model on the test set
        tester = AFDKerasEvaluator(cfg=cfg, model=best_model, dataloaders={"test": dataloaders['test']})
        tester.evaluate()

    return best_model
