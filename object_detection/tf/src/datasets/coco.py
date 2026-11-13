# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
import glob
import numpy as np
import tensorflow as tf
from pathlib import Path
from omegaconf import DictConfig
from object_detection.tf.src.datasets.utils import compute_labels_stats, compute_class_stats, \
                    convert_dataset_to_yolo, convert_val_dataset_to_yolo, \
                    add_tfs_files_to_dataset, load_subset_dataloaders                   
from munch import DefaultMunch

from hydra.core.hydra_config import HydraConfig

def load_coco_like(cfg: DictConfig,
                   image_size: tuple[int],
                   val_batch_size: int) -> dict:
    """
    Load a COCO-format dataset and create TFS files.

    This loader supports two COCO workflows:
      1) Official COCO layout when download_data=True:
         - train images:  <data_dir>/train2017
         - val images:    <data_dir>/val2017
         - annotations:   <data_dir>/annotations/instances_train2017.json
                          <data_dir>/annotations/instances_val2017.json

      2) Custom COCO layout when download_data=False:
         - train_images_path / train_annotations_path are mandatory
         - val_images_path / val_annotations_path are optional
         - test_images_path / test_annotations_path are required for evaluation-only mode

    The conversion flow is split-aware:
      - Depending on the operation mode, the TFS labels are written into:
          * raw_root/tfs_labels/train
          * raw_root/tfs_labels/val
          * raw_root/tfs_labels/test

    Args:
        cfg (DictConfig): Configuration object containing dataset parameters including:
            - dataset.format: "coco" or "tfs"
            - dataset.dataset_name: Dataset name
            - dataset.class_names: List of class names to keep
            - dataset.download_data: Whether to use the official COCO layout
            - dataset.exclude_unlabeled: Whether to drop unlabeled samples
            - dataset.max_detections: Maximum number of detections per image
            - dataset.train_images_path: Path to training images
            - dataset.train_annotations_path: Path to training COCO JSON annotations
            - dataset.val_images_path: Path to validation images (optional)
            - dataset.val_annotations_path: Path to validation COCO JSON annotations (optional)
            - dataset.test_images_path: Path to test images (required for evaluation-only)
            - dataset.test_annotations_path: Path to test COCO JSON annotations (required for evaluation-only)
            - dataset.validation_split: Split ratio used only when no explicit validation TFS folder exists
            - dataset.seed: Random seed
            - operation_mode: One of the supported modes or chains
            - quantization.operating_mode: Optional mode modifier

    Returns:
        dict[str, tf.data.Dataset]:
            A dictionary containing the available TensorFlow datasets
            (training, validation, test, quantization, prediction) depending on the mode.
    """

    if not hasattr(cfg, 'operation_mode'):
        raise ValueError("cfg.operation_mode must be specified")

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

    def is_mode_in_group(group_name: str) -> bool:
        return mode_str in mode_groups.get(group_name, [])

    is_training = is_mode_in_group("training")
    is_evaluation = is_mode_in_group("evaluation")
    is_quantization = is_mode_in_group("quantization")
    is_prediction = is_mode_in_group("prediction")

    # Verify required class names
    if not hasattr(cfg.dataset, 'class_names'):
        raise ValueError("Class names must be specified in cfg.dataset.class_names")

    dataset_format = getattr(cfg.dataset, "format", "").lower()

    # Common options for .tfs creation
    exclude_unlabeled = getattr(cfg.dataset, "exclude_unlabeled", False)
    max_detections = getattr(cfg.dataset, "max_detections", 20)
    
    def _derive_raw_root_from_path(path: str) -> str:
        return os.path.dirname(os.path.abspath(path))

    def _tfs_labels_dir(raw_root: str, split_name: str) -> str:
        return os.path.join(raw_root, "tfs_labels", split_name)

    if is_training or is_evaluation:
        if dataset_format == "coco":
            download_data = getattr(cfg.dataset, "download_data", False)

            if is_training:
                if download_data:
                    # Official COCO layout
                    if not hasattr(cfg.dataset, 'data_dir'):
                        raise ValueError(
                            "data_dir must be specified in cfg.dataset when download_data=True"
                        )

                    data_dir = cfg.dataset.data_dir
                    train_images = os.path.join(data_dir, 'train2017')
                    val_images = os.path.join(data_dir, 'val2017')
                    ann_dir = os.path.join(data_dir, 'annotations')

                    if (not os.path.exists(train_images) or
                        not os.path.exists(val_images) or
                        not os.path.exists(ann_dir)):
                        raise ValueError(
                            f"Downloaded COCO dataset structure not found in {data_dir}. "
                            f"Expected 'train2017', 'val2017', and 'annotations' directories."
                        )

                    cfg.dataset.train_images_path = train_images
                    cfg.dataset.val_images_path = val_images
                    cfg.dataset.train_annotations_path = os.path.join(
                        ann_dir, 'instances_train2017.json'
                    )
                    cfg.dataset.val_annotations_path = os.path.join(
                        ann_dir, 'instances_val2017.json'
                    )

                    # derive roots from official dataset location
                    raw_root = _derive_raw_root_from_path(train_images)
                    train_root = _tfs_labels_dir(raw_root, "train")
                    val_root = _tfs_labels_dir(raw_root, "val")

                else:
                    # Custom COCO layout
                    if not getattr(cfg.dataset, 'train_images_path', None):
                        raise ValueError(
                            "For custom COCO (download_data=False) in training mode, "
                            "train_images_path must be specified in cfg.dataset"
                        )
                    if not getattr(cfg.dataset, 'train_annotations_path', None):
                        raise ValueError(
                            "For custom COCO (download_data=False) in training mode, "
                            "train_annotations_path must be specified in cfg.dataset"
                        )

                    train_images = cfg.dataset.train_images_path
                    raw_root = _derive_raw_root_from_path(train_images)
                    train_root = _tfs_labels_dir(raw_root, "train")

                    # explicit validation only if val_* are provided
                    has_explicit_val = (
                        getattr(cfg.dataset, 'val_images_path', None) and
                        getattr(cfg.dataset, 'val_annotations_path', None)
                    )
                    val_root = _tfs_labels_dir(raw_root, "val") if has_explicit_val else None

                    if not has_explicit_val:
                        print("Warning: Validation COCO paths not fully provided. "
                              "Will rely on validation_split or other logic in dataloaders.")

                # Expose derived roots for downstream code
                cfg.dataset.training_path = train_root
                os.makedirs(train_root, exist_ok=True)

                if val_root:
                    cfg.dataset.validation_path = val_root
                    os.makedirs(val_root, exist_ok=True)

                # If explicit validation is configured but paths are missing, fail early
                if (getattr(cfg.dataset, 'val_images_path', None) and
                    getattr(cfg.dataset, 'val_annotations_path', None) and
                    not val_root):
                    raise ValueError(
                        "COCO validation images/annotations are provided, "
                        "but validation root could not be determined."
                    )

                print("Starting dataset conversion to YOLO Darknet format for training/validation...")
                convert_dataset_to_yolo(cfg)
                print("Dataset conversion for training/validation completed.\n")

                print("Starting dataset analysis on training dataset...")
                compute_class_stats(
                    images_path=train_images,
                    annotations_path=train_root,
                    dataset_name=getattr(cfg.dataset, 'dataset_name', None),
                    histogram_dir=HydraConfig.get().runtime.output_dir,
                )
                compute_labels_stats(
                    dataset_path=train_images,
                    annotations_path=train_root,
                    dataset_name=getattr(cfg.dataset, 'dataset_name', None),
                    histogram_dir=HydraConfig.get().runtime.output_dir,
                )
                print("Dataset analysis completed.\n")

                print(f"Creating .tfs files for the training dataset in {train_root}...")
                add_tfs_files_to_dataset(
                    dataset_path=train_images,
                    annotations_path=train_root,
                    output_path=train_root,
                    exclude_unlabeled_images=exclude_unlabeled,
                    padded_labels_size=max_detections,
                )
                # Remove temporary txt labels
                for path in glob.glob(os.path.join(train_root, "*.txt")):
                    os.remove(path)
                print(".tfs files creation for training dataset completed.")

                if val_root and getattr(cfg.dataset, 'val_images_path', None) and getattr(cfg.dataset, 'val_annotations_path', None):
                    print(f"Creating .tfs files for the validation dataset in {val_root}...")
                    val_images = getattr(cfg.dataset, "val_images_path", None)
                    add_tfs_files_to_dataset(
                        dataset_path=val_images,
                        annotations_path=val_root,
                        output_path=val_root,
                        exclude_unlabeled_images=exclude_unlabeled,
                        padded_labels_size=max_detections,
                    )
                    # Remove temporary txt labels
                    for path in glob.glob(os.path.join(val_root, "*.txt")):
                        os.remove(path)
                    print(".tfs files creation for validation dataset completed.")

            if (not is_training) and is_evaluation:
                # Evaluation-only: derive test_root from test_images_path parent, not from test_path
                test_images = getattr(cfg.dataset, "test_images_path", None)
                test_ann = getattr(cfg.dataset, "test_annotations_path", None)

                if not test_images or not os.path.isdir(test_images):
                    raise ValueError(
                        "For COCO evaluation-only mode, 'dataset.test_images_path' must be defined "
                        "and point to a valid directory containing test images.")
                if not test_ann or not os.path.isfile(test_ann):
                    raise ValueError(
                        "For COCO evaluation-only mode, 'dataset.test_annotations_path' must be defined "
                        "and point to a valid COCO JSON annotations file.")

                raw_root = _derive_raw_root_from_path(test_images)
                test_root = _tfs_labels_dir(raw_root, "test")
                cfg.dataset.test_path = test_root 
                os.makedirs(test_root, exist_ok=True)

                # Convert validation/test COCO split to YOLO under test_root
                # when running evaluation without prior training.
                if mode_str in ["evaluation", "chain_eqe", "chain_eqeb"]:
                    print("Converting COCO validation/test split to YOLO Darknet format for evaluation...")
                    convert_val_dataset_to_yolo(cfg)
                    print("Validation/test dataset conversion completed.\n")

                print(f"Creating .tfs files for the evaluation dataset in {test_root}...")
                add_tfs_files_to_dataset(
                    dataset_path=test_images,
                    annotations_path=test_root,
                    output_path=test_root,
                    exclude_unlabeled_images=exclude_unlabeled,
                    padded_labels_size=max_detections,
                )
                # Remove temporary txt labels
                for path in glob.glob(os.path.join(test_root, "*.txt")):
                    os.remove(path)

                print(".tfs files creation for evaluation dataset completed.")

        elif dataset_format == "tfs":
            # If format is tfs, directly load without conversion or analysis
            print("Dataset format is 'tfs'. Skipping conversion and analysis steps.")
            print("Loading datasets in tfs format...")
            return load_subset_dataloaders(
                cfg, is_training, is_evaluation,
                is_prediction, is_quantization,
                image_size=image_size, val_batch_size=val_batch_size,
            )
        else:
            raise ValueError(f"Unsupported dataset format '{dataset_format}' for training/evaluation mode.")

    if is_prediction:
        pred_path = getattr(cfg.dataset, 'prediction_path', None)
        if not pred_path:
            raise ValueError("cfg.dataset.prediction_path must be specified in prediction mode")
        if not os.path.exists(pred_path):
            raise ValueError(f"Prediction path {pred_path} does not exist")

    if is_quantization:
        quant_path = getattr(cfg.dataset, 'quantization_path', None)
        if not quant_path:
            raise ValueError("cfg.dataset.quantization_path must be specified in quantization mode")
        if not os.path.exists(quant_path):
            raise ValueError(f"Quantization path {quant_path} does not exist")

    print("Loading datasets in darknet-like (.tfs) format...")
    return load_subset_dataloaders(
        cfg, is_training, is_evaluation,
        is_prediction, is_quantization,
        image_size=image_size, val_batch_size=val_batch_size,
    )