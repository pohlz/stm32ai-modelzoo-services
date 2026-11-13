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
def load_pascal_voc_like(cfg: DictConfig,
                         image_size: tuple[int],
                         val_batch_size: int) -> dict:
    """
    Load a Pascal VOC-format dataset and create TFS files.

    This loader supports the official Pascal VOC layout and a custom split-based workflow:
      - Images remain in the source image directory/directories
      - The .tfs files are generated in:
          * raw_root/tfs_labels/train
          * raw_root/tfs_labels/val
          * raw_root/tfs_labels/test

    Supported workflows:
      1) Training:
         - train_images_path + train_annotations_path + train_split are required
         - explicit validation is optional:
             * val_images_path + val_annotations_path + val_split
         - if explicit val is provided, TFS are generated in raw_root/tfs_labels/val
         - otherwise validation is created later by splitting the training TFS set

      2) Evaluation-only:
         - test_images_path + test_annotations_path + test_split are required
         - TFS are generated in raw_root/tfs_labels/test

    Args:
        cfg (DictConfig): Configuration object containing dataset parameters including:
            - dataset.format: "pascal_voc" or "tfs"
            - dataset.dataset_name: Dataset name
            - dataset.class_names: List of class names to keep
            - dataset.exclude_unlabeled: Whether to drop unlabeled samples
            - dataset.max_detections: Maximum number of detections per image
            - dataset.train_images_path: Path to training images
            - dataset.train_annotations_path: Path to training Pascal VOC annotations
            - dataset.train_split: Path to ImageSets/Main/train.txt
            - dataset.val_images_path: Path to validation images (optional)
            - dataset.val_annotations_path: Path to validation annotations (optional)
            - dataset.val_split: Path to ImageSets/Main/val.txt (optional)
            - dataset.test_images_path: Path to test images (required for evaluation-only)
            - dataset.test_annotations_path: Path to test annotations (required for evaluation-only)
            - dataset.test_split: Path to ImageSets/Main/test.txt (required for evaluation-only)
            - dataset.validation_split: Ratio used when no explicit validation split is available
            - dataset.seed: Random seed
            - operation_mode: One of the supported modes or chains
            - quantization.operating_mode: Optional mode modifier

    Returns:
        dict[str, tf.data.Dataset]:
            A dictionary containing the available TensorFlow datasets
            (training, validation, test, quantization, prediction) depending on the mode.
    """

    if not hasattr(cfg, "operation_mode"):
        raise ValueError("cfg.operation_mode must be specified")

    mode_str = cfg.operation_mode.lower()

    mode_groups = DefaultMunch.fromDict({
        "training": ["training", "chain_tqeb", "chain_tqe"],
        "evaluation": ["evaluation", "chain_tqeb", "chain_tqe", "chain_eqe", "chain_eqeb"],
        "quantization": ["quantization", "chain_tqeb", "chain_tqe", "chain_eqe", "chain_qb", "chain_eqeb", "chain_qd"],
        "benchmarking": ["benchmarking", "chain_tqeb", "chain_qb", "chain_eqeb"],
        "deployment": ["deployment", "chain_qd"],
        "prediction": ["prediction"],
        "compression": ["compression"],
    })

    if getattr(cfg.quantization, "operating_mode", None) == "full_auto":
        for item in ["quantization", "chain_qd", "chain_qb"]:
            if item not in mode_groups.evaluation:
                mode_groups.evaluation.append(item)

    def is_mode_in_group(group_name: str) -> bool:
        return mode_str in mode_groups.get(group_name, [])

    is_training = is_mode_in_group("training")
    is_evaluation = is_mode_in_group("evaluation")
    is_quantization = is_mode_in_group("quantization")
    is_prediction = is_mode_in_group("prediction")

    if not hasattr(cfg.dataset, "class_names"):
        raise ValueError("Class names must be specified in cfg.dataset.class_names")

    dataset_format = getattr(cfg.dataset, "format", "").lower()

    exclude_unlabeled = getattr(cfg.dataset, "exclude_unlabeled", False)
    max_detections = getattr(cfg.dataset, "max_detections", 20)
    if max_detections is None:
        max_detections = 20


    def _derive_raw_root_from_path(path: str) -> str:
        return os.path.dirname(os.path.abspath(path))

    def _tfs_labels_dir(raw_root: str, split_name: str) -> str:
        return os.path.join(raw_root, "tfs_labels", split_name)

    if is_training or is_evaluation:
        if dataset_format == "pascal_voc":
            if is_training:
                train_images = getattr(cfg.dataset, "train_images_path", None)
                train_annotations = getattr(cfg.dataset, "train_annotations_path", None)
                train_split = getattr(cfg.dataset, "train_split", None)
                if not train_images or not os.path.isdir(train_images):
                    raise ValueError("dataset.train_images_path must point to a valid directory.")
                if not train_annotations or not os.path.isdir(train_annotations):
                    raise ValueError("dataset.train_annotations_path must point to a valid directory.")
                if not train_split or not os.path.isfile(train_split):
                    raise ValueError("dataset.train_split must point to a valid split file.")

                raw_root = _derive_raw_root_from_path(train_images)
                train_out = _tfs_labels_dir(raw_root, "train")
                os.makedirs(train_out, exist_ok=True)
                cfg.dataset.training_path = train_out

                val_images = getattr(cfg.dataset, "val_images_path", None)
                val_annotations = getattr(cfg.dataset, "val_annotations_path", None)
                val_split = getattr(cfg.dataset, "val_split", None)
                has_val = bool(val_images and val_annotations and val_split and os.path.isfile(val_split))

                if has_val:
                    raw_root_val = _derive_raw_root_from_path(val_images)
                    val_out = _tfs_labels_dir(raw_root_val, "val")
                    os.makedirs(val_out, exist_ok=True)
                    cfg.dataset.validation_path = val_out
                else:
                    val_out = None

                print("Starting Pascal VOC dataset conversion to YOLO Darknet format for training/validation...")
                convert_dataset_to_yolo(cfg)
                print("Pascal VOC dataset conversion for training/validation completed.\n")

                print("Starting dataset analysis on training dataset...")
                compute_class_stats(train_images, train_out, getattr(cfg.dataset, "dataset_name", None), "histograms")
                compute_labels_stats(train_images, train_out, getattr(cfg.dataset, "dataset_name", None), "histograms")
                print("Dataset analysis completed.\n")

                print(f"Creating .tfs files for the training dataset in {train_out}...")
                add_tfs_files_to_dataset(
                    dataset_path=train_images,
                    annotations_path=train_out,
                    output_path=train_out,
                    exclude_unlabeled_images=exclude_unlabeled,
                    padded_labels_size=max_detections,
                )
                # Remove temporary txt labels
                for path in glob.glob(os.path.join(train_out, "*.txt")):
                    os.remove(path)

                if val_out:
                    print(f"Creating .tfs files for the validation dataset in {val_out}...")
                    add_tfs_files_to_dataset(
                        dataset_path=val_images,
                        annotations_path=val_out,
                        output_path=val_out,
                        exclude_unlabeled_images=exclude_unlabeled,
                        padded_labels_size=max_detections,
                    )
                    for path in glob.glob(os.path.join(val_out, "*.txt")):
                        os.remove(path)

            if (not is_training) and is_evaluation:
                test_images = getattr(cfg.dataset, "test_images_path", None)
                test_annotations = getattr(cfg.dataset, "test_annotations_path", None)
                test_split = getattr(cfg.dataset, "test_split", None)
                if not test_images or not os.path.isdir(test_images):
                    raise ValueError("dataset.test_images_path must point to a valid directory.")
                if not test_annotations or not os.path.isdir(test_annotations):
                    raise ValueError("dataset.test_annotations_path must point to a valid directory.")
                if not test_split or not os.path.isfile(test_split):
                    raise ValueError("dataset.test_split must point to a valid split file.")

                raw_root = _derive_raw_root_from_path(test_images)
                test_out = _tfs_labels_dir(raw_root, "test")
                os.makedirs(test_out, exist_ok=True)
                cfg.dataset.test_path = test_out

                print("Converting Pascal VOC test split to YOLO Darknet format for evaluation...")
                convert_val_dataset_to_yolo(cfg)
                print("Pascal VOC test split conversion completed.\n")

                print(f"Creating .tfs files for the evaluation dataset in {test_out}...")
                add_tfs_files_to_dataset(
                    dataset_path=test_images,
                    annotations_path=test_out,
                    output_path=test_out,
                    exclude_unlabeled_images=exclude_unlabeled,
                    padded_labels_size=max_detections,
                )
                for path in glob.glob(os.path.join(test_out, "*.txt")):
                    os.remove(path)

        elif dataset_format == "tfs":
            print("Dataset format is 'tfs'. Skipping conversion and analysis steps.")
            return load_subset_dataloaders(
                cfg, is_training, is_evaluation, is_prediction, is_quantization,
                image_size=image_size, val_batch_size=val_batch_size
            )
        else:
            raise ValueError(f"Unsupported dataset format '{dataset_format}' for training/evaluation mode.")

    if is_prediction:
        pred_path = getattr(cfg.dataset, "prediction_path", None)
        if not pred_path:
            raise ValueError("cfg.dataset.prediction_path must be specified in prediction mode")
        if not os.path.exists(pred_path):
            raise ValueError(f"Prediction path {pred_path} does not exist")

    if is_quantization:
        quant_path = getattr(cfg.dataset, "quantization_path", None)
        if not quant_path:
            raise ValueError("cfg.dataset.quantization_path must be specified in quantization mode")
        if not os.path.exists(quant_path):
            raise ValueError(f"Quantization path {quant_path} does not exist")

    return load_subset_dataloaders(
        cfg, is_training, is_evaluation, is_prediction, is_quantization,
        image_size=image_size, val_batch_size=val_batch_size
    )