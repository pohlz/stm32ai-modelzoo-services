# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from omegaconf import DictConfig
from .utils import load_subset_dataloaders
from munch import DefaultMunch


def load_tfs_like(cfg: DictConfig, 
                  image_size: tuple[int],
                  val_batch_size: int) -> dict:
    """
    Load an already-prepared TFS dataset.

    This loader expects:
      - raw images in the source image directories:
          * dataset.train_images_path
          * dataset.val_images_path
          * dataset.test_images_path
        (depending on the requested split and mode)
      - corresponding .tfs files stored in the split-specific TFS directories:
          * dataset.train_annotations_path
          * dataset.val_annotations_path
          * dataset.test_annotations_path

    The loader does not perform any dataset conversion.

    Expected usage:
      - training pairs are built from train_images_path + train_annotations_path
      - validation pairs are built from val_images_path + val_annotations_path, if explicit validation exists
        otherwise validation is created by splitting the training TFS set
      - test pairs are built from test_images_path + test_annotations_path

    Args:
        cfg (DictConfig): Configuration object containing dataset parameters including:
            - dataset.format: "tfs"
            - dataset.train_images_path: Path to training images
            - dataset.val_images_path: Path to validation images
            - dataset.test_images_path: Path to test images
            - dataset.train_annotations_path: Path to training TFS files
            - dataset.val_annotations_path: Path to validation TFS files
            - dataset.test_annotations_path: Path to test TFS files
            - dataset.validation_split: Validation split ratio when no explicit validation TFS path is provided
            - dataset.class_names: List of class names to use
            - dataset.exclude_unlabeled: Whether to exclude unlabeled samples
            - dataset.max_detections: Maximum number of detections per image
            - dataset.seed: Random seed
            - operation_mode: One of the supported modes or chains

    Returns:
        dict[str, tf.data.Dataset]:
            A dictionary containing the available TensorFlow datasets
            (training, validation, test, quantization, prediction) depending on the mode.
    """

    if not hasattr(cfg, 'operation_mode'):
        raise ValueError("cfg.operation_mode must be specified")

    # Check for unsupported download_data option
    if hasattr(cfg.dataset, 'download_data') and cfg.dataset.download_data:
        raise NotImplementedError("Downloading dataset is unsupported for TFS-ready format. "
                                  "Please prepare the dataset manually in the expected format.")

    mode_str = cfg.operation_mode.lower()

    mode_groups = DefaultMunch.fromDict({
        "training": ["training", "chain_tqeb", "chain_tqe"],
        "evaluation": ["evaluation", "chain_tqeb", "chain_tqe", "chain_eqe", "chain_eqeb"],
        "quantization": ["quantization", "chain_tqeb", "chain_tqe", "chain_eqe",
                         "chain_qb", "chain_eqeb", "chain_qd"],
        "benchmarking": ["benchmarking", "chain_tqeb", "chain_qb", "chain_eqeb"],
        "deployment": ["deployment", "chain_qd"],
        "prediction": ["prediction"],
        "compression": ["compression"]
    })

    # Conditional addition based on cfg.quantization.operating_mode
    if getattr(cfg.quantization, "operating_mode", None) == "full_auto":
        additional_items = ["quantization", "chain_qd", "chain_qb"]
        
        for item in additional_items:
            if item not in mode_groups.evaluation:
                mode_groups.evaluation.append(item)

    def is_mode_in_group(group_name):
        return mode_str in mode_groups.get(group_name, [])

    is_training = is_mode_in_group("training")
    is_evaluation = is_mode_in_group("evaluation")
    is_quantization = is_mode_in_group("quantization")
    is_prediction = is_mode_in_group("prediction")

    # Verify required class names
    if not hasattr(cfg.dataset, 'class_names'):
        raise ValueError("Class names must be specified in cfg.dataset.class_names")

    # Validate paths depending on operation mode
    if is_training:
        if not hasattr(cfg.dataset, 'train_images_path'):
            raise ValueError("cfg.dataset.train_images_path must be specified in training mode")
        if not os.path.exists(cfg.dataset.train_images_path):
            raise ValueError(f"Training path {cfg.dataset.train_images_path} does not exist")
        if not hasattr(cfg.dataset, 'train_annotations_path'):
            raise ValueError("cfg.dataset.train_annotations_path must be specified in training mode")
        if not os.path.exists(cfg.dataset.train_annotations_path):
            raise ValueError(f"Training path {cfg.dataset.train_annotations_path} does not exist")
        print("Skipping dataset analysis as for TFS dataset format.\n")
        cfg.dataset.training_path = cfg.dataset.train_annotations_path

    if hasattr(cfg.dataset, 'val_images_path') and cfg.dataset.val_images_path:
        if not os.path.exists(cfg.dataset.val_images_path):
            raise ValueError(f"Val images path {cfg.dataset.val_images_path} does not exist")
        if not os.path.exists(cfg.dataset.val_annotations_path):
            raise ValueError(f"Val annoations path {cfg.dataset.val_annotations_path} does not exist")

    if is_evaluation:
        if not hasattr(cfg.dataset, 'test_images_path') and not cfg.dataset.test_images_path:
            raise ValueError("cfg.dataset.test_images_path must be specified in evaluation mode")
        if not os.path.exists(cfg.dataset.test_images_path):
            raise ValueError(f"Test path {cfg.dataset.test_images_path} does not exist")
        if not hasattr(cfg.dataset, 'test_annotations_path') and not cfg.dataset.test_annotations_path:
            raise ValueError("cfg.dataset.test_annotations_path must be specified in evaluation mode")
        if not os.path.exists(cfg.dataset.test_annotations_path):
            raise ValueError(f"Test path {cfg.dataset.test_annotations_path} does not exist")

    if is_prediction:
        if not hasattr(cfg.dataset, 'prediction_path'):
            raise ValueError("cfg.dataset.prediction_path must be specified in prediction mode")
        if not os.path.exists(cfg.dataset.prediction_path):
            raise ValueError(f"Prediction path {cfg.dataset.prediction_path} does not exist")

    if is_quantization:
        if not hasattr(cfg.dataset, 'quantization_path'):
            raise ValueError("cfg.dataset.quantization_path must be specified in quantization mode")
        if not os.path.exists(cfg.dataset.quantization_path):
            raise ValueError(f"Quantization path {cfg.dataset.quantization_path} does not exist")

    print("Loading datasets in darknet format...")
    return load_subset_dataloaders(cfg, is_training, is_evaluation,
                             is_prediction, is_quantization,
                             image_size=image_size, val_batch_size=val_batch_size)