# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
import sys
from timeit import default_timer as timer
from datetime import timedelta
from typing import Tuple, List, Dict, Optional

from hydra.core.hydra_config import HydraConfig
from munch import DefaultMunch
from omegaconf import DictConfig
import numpy as np
import tensorflow as tf
import mlflow

# Suppress Tensorflow warnings
import logging
logging.getLogger('mlflow.tensorflow').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from common.utils import log_to_file, log_last_epoch_history, LRTensorBoard, check_training_determinism, \
                         model_summary, collect_callback_args, vis_training_curves
from common.training import set_frozen_layers, set_dropout_rate, get_optimizer, lr_schedulers, set_all_layers_trainable_parameter
from re_identification.tf.src.preprocessing import apply_rescaling
from re_identification.tf.src.utils import get_loss, get_triplet_loss, change_model_number_of_classes
from re_identification.tf.src.data_augmentation import DataAugmentationLayer
from re_identification.tf.src.evaluation import evaluate_h5_model, evaluate_h5_model_reid






def _add_preprocessing_layers(
                model: tf.keras.Model,
                input_shape: Tuple = None, 
                scale: float = None,
                offset: float = None,
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
    model_layers.append(tf.keras.Input(shape=input_shape))

    # Add the rescaling layer
    model_layers.append(tf.keras.layers.Rescaling(scale, offset))

    # If data augmentation is used, add the custom .
    if data_augmentation:
        # Add the data augmentation layer
        pixels_range = (offset, scale * 255 + offset)
        model_layers.append(
            DataAugmentationLayer(
                data_augmentation_fn=data_augmentation.function_name,
                config=data_augmentation.config,
                pixels_range=pixels_range,
                batches_per_epoch=batches_per_epoch
            )
        )
    model_layers.append(model)
    augmented_model = tf.keras.Sequential(model_layers, name="augmented_model")

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
                        monitor="val_loss",
                        mode="min")
    callback_list.append(callback)

    # Add the Keras callback that saves the model at the end of the epoch
    callback = tf.keras.callbacks.ModelCheckpoint(
                        filepath=os.path.join(output_dir, saved_models_dir, "last_augmented_model.keras"),
                        save_best_only=False,
                        save_weights_only=False,
                        monitor="val_loss",
                        mode="min")
    callback_list.append(callback)

    # Add the TensorBoard callback
    callback = LRTensorBoard(log_dir=os.path.join(output_dir, logs_dir))
    callback_list.append(callback)

    # Add the CVSLogger callback (must be last in the list 
    # of callbacks to make sure it records the learning rate)
    callback = tf.keras.callbacks.CSVLogger(os.path.join(output_dir, logs_dir, "metrics", "train_metrics.csv"))
    callback_list.append(callback)

    return callback_list


def train(cfg: DictConfig = None, model: tf.keras.Model = None, train_ds: tf.data.Dataset = None,
          valid_ds: tf.data.Dataset = None, test_query_ds: tf.data.Dataset = None,
          test_gallery_ds: tf.data.Dataset = None) -> str:
    """
    Trains the model using the provided configuration and datasets.

    Args:
        cfg (DictConfig): The entire configuration file dictionary.
        train_ds (tf.data.Dataset): training dataset loader.
        valid_ds (tf.data.Dataset): validation dataset loader.
        test_ds (Optional, tf.data.Dataset): test dataset dataset loader.

    Returns:
        Path to the best model obtained
    """

    output_dir = HydraConfig.get().runtime.output_dir
    saved_models_dir = cfg.general.saved_models_dir
    class_names = cfg.dataset.class_names
    class_names_test = cfg.dataset.class_names_test if cfg.dataset.class_names_test else class_names
    input_shape = cfg.model.input_shape
    num_classes = len(class_names)

    if 'triplet_loss' in cfg.training:
        triplet_margin = cfg.training.triplet_loss.margin if 'margin' in cfg.training.triplet_loss else 0.3
        triplet_strategy = cfg.training.triplet_loss.strategy if 'strategy' in cfg.training.triplet_loss else 'hard'
        distance_metric = cfg.training.triplet_loss.distance_metric if 'distance_metric' in cfg.training.triplet_loss else 'cosine'

    print("Dataset stats:")
    train_size = sum([x.shape[0] for x, _ in train_ds])
    valid_size = sum([x.shape[0] for x, _ in valid_ds])
    if test_query_ds and test_gallery_ds:
        test_query_size = sum([x.shape[0] for x, _ in test_query_ds])
        test_gallery_size = sum([x.shape[0] for x, _ in test_gallery_ds])

    print("  classes:", num_classes)
    print("  training set size:", train_size)
    print("  validation set size:", valid_size)
    if test_query_ds and test_gallery_ds:
        print("  test query set size:", test_query_size)
        print("  test gallery set size:", test_gallery_size)
    else:
        print("  no test set")


    # Info messages about the model that was loaded
    if cfg.model.model_name:
        cfm = cfg.model
        print(f"[INFO] : Using `{cfm.model_name}` model")
        log_to_file(cfg.output_dir, (f"Model name : {cfm.model_name}"))
        if cfm.pretrained:
            print(f"[INFO] : Initialized model with pretrained weights")
            log_to_file(cfg.output_dir,(f"Pretrained weights"))
        elif cfm.model_path:
            print(f"[INFO] : Initialized model with weights from model file {cfm.model_path}")
            log_to_file(cfg.output_dir, (f"Weights from model file : {cfm.model_path}"))
        else:
            print("[INFO] : No pretrained weights were loaded, training from randomly initialized weights.")

    elif cfg.model.model_path and not cfg.training.resume_training:
        print(f"[INFO] : Using model file {cfg.model.model_path}")
        log_to_file(cfg.output_dir, (f"Model file : {cfg.model.model_path}"))
        x = model.output
        # add dropout layer if dropout is specified
        if cfg.training.dropout is not None:
            # add dropout layer to the output of model
            x = tf.keras.layers.Dropout(cfg.training.dropout)(x)
        # add dense for classification
        x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        model = tf.keras.Model(inputs=model.input, outputs=x)

    elif cfg.training.resume_training and cfg.model.model_path:
        print(f"[INFO] : Resuming training from model file {cfg.model.model_path}")
        log_to_file(cfg.output_dir, (f"Model file : {cfg.model.model_path}"))

    if cfg.dataset.dataset_name: 
        log_to_file(output_dir, f"Dataset : {cfg.dataset.dataset_name}")
    if not cfg.training.resume_training:
        #model.trainable = True
        set_all_layers_trainable_parameter(model, trainable=True)
        if cfg.training.frozen_layers:
            set_frozen_layers(model, frozen_layers=cfg.training.frozen_layers)
        if cfg.training.dropout is not None:
            set_dropout_rate(model, dropout_rate=cfg.training.dropout)
        if cfg.training.dropout is not None and cfg.training.dropout > 0:
            model = tf.keras.Model(inputs=model.input, outputs=[model.layers[-1].output, model.layers[-3].output]) 
        else:
            model = tf.keras.Model(inputs=model.input, outputs=[model.layers[-1].output, model.layers[-2].output])
    # Display a summary of the model
    if cfg.training.resume_training:
        model_summary(model)
        if len(model.layers) == 2:
            model_summary(model.layers[1])
        else:
            model_summary(model.layers[2])
    else:
        model_summary(model)

    if not cfg.training.resume_training:
        # Add the preprocessing layers to the model
        augmented_model = _add_preprocessing_layers(model,
                                input_shape=input_shape,
                                scale=cfg.preprocessing.rescaling.scale,
                                offset=cfg.preprocessing.rescaling.offset,
                                data_augmentation=cfg.data_augmentation,
                                batches_per_epoch=train_size)
        # Compile the augmented model
        augmented_model.compile(loss=[get_loss(num_classes),
                                      get_triplet_loss(margin=triplet_margin, mining=triplet_strategy, distance_metric=distance_metric)],
                                metrics=['accuracy', None],
                                optimizer=get_optimizer(cfg=cfg.training.optimizer))
    else:
        # The preprocessing layers are already included.
        # We don't compile the model otherwise we would 
        # loose the loss and optimizer states.
        augmented_model = model

        # Some saved models may not contain compile state.
        # In that case, compile with the current training config.
        if getattr(augmented_model, "optimizer", None) is None:
            print("[INFO] : Loaded resumed model has no compile state. Compiling with current training settings.")
            augmented_model.compile(
                loss=[
                    get_loss(num_classes),
                    get_triplet_loss(margin=triplet_margin, mining=triplet_strategy, distance_metric=distance_metric)
                ],
                metrics=['accuracy', None],
                optimizer=get_optimizer(cfg=cfg.training.optimizer)
            )

    callbacks = _get_callbacks(callbacks_dict=cfg.training.callbacks,
                                output_dir=output_dir,
                                saved_models_dir=saved_models_dir,
                                logs_dir=cfg.general.logs_dir)

    # check if determinism can be enabled
    if cfg.general.deterministic_ops:
        sample_ds = train_ds.take(1)
        tf.config.experimental.enable_op_determinism()
        if not check_training_determinism(augmented_model, sample_ds):
            print("[WARNING]: Some operations cannot be run deterministically. Setting deterministic_ops to False.")
            tf.config.experimental.enable_op_determinism.__globals__["_pywrap_determinism"].enable(False)

    # Train the model 
    print("Starting training...")
    start_time = timer()
    
    try:
        steps_per_epoch = cfg.training.dryrun if cfg.training.dryrun else None
        history = augmented_model.fit(train_ds,
                                  validation_data=valid_ds,
                                  epochs=cfg.training.epochs,
                                  steps_per_epoch=steps_per_epoch,
                                  callbacks=callbacks)
    except: 
        print('\n[INFO] : Training interrupted')

    #save the last epoch history in the log file
    last_epoch=log_last_epoch_history(cfg, output_dir)
    end_time = timer()
    
    #calculate and log the runtime in the log file
    fit_run_time = int(end_time - start_time)
    average_time_per_epoch = round(fit_run_time / (int(last_epoch) + 1),2)
    print("Training runtime: " + str(timedelta(seconds=fit_run_time)))
    log_to_file(cfg.output_dir, (f"Training runtime : {fit_run_time} s\n" + f"Average time per epoch : {average_time_per_epoch} s"))

    # Visualize training curves
    vis_training_curves(history=history, output_dir=output_dir)

    # Load the checkpoint model (best model obtained)
    models_dir = os.path.join(output_dir, saved_models_dir)
    checkpoint_filepath = os.path.join(models_dir, "best_augmented_model.keras")

    checkpoint_model = tf.keras.models.load_model(
        checkpoint_filepath,
        custom_objects={
            'DataAugmentationLayer': DataAugmentationLayer,
            'get_triplet_loss': get_triplet_loss
        })
 
    # Get the checkpoint model w/o preprocessing layers
    best_model_classification = checkpoint_model.layers[-1]
    best_model_classification = tf.keras.Model(inputs=best_model_classification.input, outputs=best_model_classification.output[0])
    best_model_classification.compile(loss=(get_loss(num_classes)), metrics=['accuracy'])
    best_model_classification_path = os.path.join(output_dir,
                                   "{}/{}".format(saved_models_dir, "best_model_classification.keras"))
    best_model_classification.save(best_model_classification_path)
    if cfg.training.dropout:
        best_model = tf.keras.Model(inputs=best_model_classification.input, outputs=best_model_classification.layers[-3].output)
    else:
        best_model = tf.keras.Model(inputs=best_model_classification.input, outputs=best_model_classification.layers[-2].output)
    best_model_path = os.path.join(output_dir,
                                   "{}/{}".format(saved_models_dir, "best_model.keras"))
    best_model.save(best_model_path)

    # Save a copy of the best model if requested
    if cfg.training.trained_model_path_classification:
        best_model_classification.save(cfg.training.trained_model_path_classification)
        print("[INFO] : Saved trained model in file {}".format(cfg.training.trained_model_path_classification))
    if cfg.training.trained_model_path:
        best_model.save(cfg.training.trained_model_path)
        print("[INFO] : Saved trained re-identification model in file {}".format(cfg.training.trained_model_path))

    # Pre-process validation dataset
    valid_ds = apply_rescaling(dataset=valid_ds, scale=cfg.preprocessing.rescaling.scale,
                               offset=cfg.preprocessing.rescaling.offset)
    setattr(best_model, 'model_path', best_model_path)
    # Evaluate h5 or keras best model on the validation set
    # dataset_single_label = original_dataset.map(lambda x, y: (x, y[0]))
    valid_ds_single_label = valid_ds.map(lambda x, y: (x, y[0]))
    evaluate_h5_model(model_path=best_model_classification_path, eval_ds=valid_ds_single_label,
                     class_names=class_names, output_dir=output_dir, name_ds="validation_set")

    if test_query_ds and test_gallery_ds:
        # Pre-process test dataset
        test_query_ds = apply_rescaling(dataset=test_query_ds, scale=cfg.preprocessing.rescaling.scale,
                                  offset=cfg.preprocessing.rescaling.offset)
        
        test_gallery_ds = apply_rescaling(dataset=test_gallery_ds, scale=cfg.preprocessing.rescaling.scale,
                                  offset=cfg.preprocessing.rescaling.offset)

        if cfg.evaluation and 'reid_distance_metric' in cfg.evaluation:
            distance_metric = cfg.evaluation.reid_distance_metric
        else:
            distance_metric = 'cosine'

        print (f"[INFO] : Using `{distance_metric}` distance metric for re-identification evaluation")
        # Evaluate h5 or keras best model on the test set
        best_model_test_acc = evaluate_h5_model_reid(model_path=best_model_path, 
                                                     eval_query_ds=test_query_ds,
                                                     eval_gallery_ds=test_gallery_ds,
                                                     distance_metric=distance_metric,
                                                     output_dir=output_dir, 
                                                     name_ds="test_set", 
                                                     display_figures=cfg.general.display_figures)

    return best_model
