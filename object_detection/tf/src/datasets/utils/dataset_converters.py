# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
import sys
import json
import hydra
import shutil
import argparse
from hydra.core.hydra_config import HydraConfig
from tqdm import tqdm
from munch import DefaultMunch
from omegaconf import OmegaConf
from omegaconf import DictConfig
import xml.etree.ElementTree as ET
from PIL import Image

def classes_inspector(non_existing_classes: list = None,
                      available_classes: list = None) -> None:
    """
    Ensure all defined classes are well present in the dataset

    Args:
        non_existing_classes (list): list of non found classes from the dataset
        available_classes (list): list of detected classes in the dataset

    Returns:
        None
    """
    if len(non_existing_classes) > 0:
        print("The following classes were not found: {}".format(non_existing_classes))
        print("Please make sure that your selected classes exist in the following list: {}".format(available_classes))
        print("Exiting the script...")
        sys.exit()
    else:
        print("Converting the dataset ...")


def verify_voc_classes(xml_folder: str = None,
                       classes: list = None) -> None:
    """
    Check if all expected classes are well present in the provided dataset

    Args:
        xml_folder (str): path to the xml directory
        classes (list): list of the provided classes (from the yaml file)

    Returns:
        None
    """
    print("Analyzing the dataset ...")
    available_classes = set()
    xml_files = [file for file in os.listdir(xml_folder) if file.endswith('.xml')]
    for filename in tqdm(xml_files):
        xml_path = os.path.join(xml_folder, filename)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text
            print(name)
            available_classes.add(name)

    non_existing_classes = [c for c in classes if c not in available_classes]
    classes_inspector(non_existing_classes,
                      available_classes)

def convert_voc_split_to_yolo(images_folder: str,
                              annotations_folder: str,
                              split_file: str,
                              classes: list,
                              export_folder: str) -> None:
    """
    Convert a Pascal VOC split defined by a split file (train.txt, val.txt, test.txt)
    to YOLO text labels.
    """
    if not os.path.isfile(split_file):
        raise ValueError(f"Split file not found: {split_file}")
    if not os.path.isdir(images_folder):
        raise ValueError(f"Images folder not found: {images_folder}")
    if not os.path.isdir(annotations_folder):
        raise ValueError(f"Annotations folder not found: {annotations_folder}")
    if not classes:
        raise ValueError("classes must be a non-empty list")

    os.makedirs(export_folder, exist_ok=True)

    with open(split_file, "r") as f:
        image_ids = [line.strip().split()[0] for line in f if line.strip()]

    for img_id in tqdm(image_ids, desc=f"Converting VOC split {os.path.basename(split_file)}"):
        xml_path = os.path.join(annotations_folder, f"{img_id}.xml")
        if not os.path.isfile(xml_path):
            continue  # skip missing annotations

        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find("size")
        if size is None:
            continue

        width_node = size.find("width")
        height_node = size.find("height")
        if width_node is None or height_node is None:
            continue

        width = int(width_node.text)
        height = int(height_node.text)

        txt_path = os.path.join(export_folder, f"{img_id}.txt")

        for obj in root.findall("object"):
            name_node = obj.find("name")
            bbox = obj.find("bndbox")

            if name_node is None or bbox is None:
                continue

            name = name_node.text
            if name not in classes:
                continue

            xmin_node = bbox.find("xmin")
            ymin_node = bbox.find("ymin")
            xmax_node = bbox.find("xmax")
            ymax_node = bbox.find("ymax")

            if None in (xmin_node, ymin_node, xmax_node, ymax_node):
                continue

            xmin = float(xmin_node.text)
            ymin = float(ymin_node.text)
            xmax = float(xmax_node.text)
            ymax = float(ymax_node.text)

            x_center = (xmin + xmax) / (2.0 * width)
            y_center = (ymin + ymax) / (2.0 * height)
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height

            class_id = classes.index(name)
            with open(txt_path, "a") as label_file:
                label_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")


def convert_voc_to_yolo(xml_folder: str = None,
                        images_folder: str = None,
                        classes: list = None,
                        export_folder: str = None) -> None:
    """
    Core routine that converts voc data to yolo format and exports them

    Args:
        xml_folder (str): path to the xml directory
        images_folder (str): path to the images directory
        classes (list): list of the provided classes (from the yaml file)
        export_folder (str): path converted dataset will be stored

    Returns:
        None
    """
    verify_voc_classes(xml_folder,
                       classes)
    if not os.path.exists(export_folder):
        os.makedirs(export_folder)
    xml_files = [file for file in os.listdir(xml_folder) if file.endswith('.xml')]
    for filename in tqdm(xml_files):
        copy_image = False
        xml_path = os.path.join(xml_folder, filename)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        txt_filename = os.path.splitext(filename)[0] + '.txt'
        txt_path = os.path.join(export_folder, txt_filename)

        for obj in root.findall('object'):
            name = obj.find('name').text
            if name in classes:
                with open(txt_path, 'a') as label_file:
                    copy_image = True
                    class_id = classes.index(name)
                    bbox = obj.find('bndbox')
                    xmin = float(bbox.find('xmin').text)
                    ymin = float(bbox.find('ymin').text)
                    xmax = float(bbox.find('xmax').text)
                    ymax = float(bbox.find('ymax').text)
                    x_center = (xmin + xmax) / (2 * width)
                    y_center = (ymin + ymax) / (2 * height)
                    w = (xmax - xmin) / width
                    h = (ymax - ymin) / height
                    label_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

        if copy_image:
            image_file = os.path.splitext(filename)[0] + '.jpg'
            image_path = os.path.join(images_folder, image_file)
            shutil.copy(image_path, export_folder)


def verify_coco_classes(coco_annotations_file: str = None,
                        classes: list = None) -> None:
    """
    Check if all expected classes are well present in the provided dataset

    Args:
        coco_annotations_file (str): path to the coco annotation file
        classes (list): list of the provided classes (from the yaml file)

    Returns:
        None
    """
    print("Analyzing the dataset ...")
    with open(coco_annotations_file, 'r') as f:
        coco_data = json.load(f)

    class_names = set()
    for annotation in tqdm(coco_data['annotations']):
        category_id = annotation['category_id']
        for category in coco_data['categories']:
            if category['id'] == category_id:
                class_name = category['name']
                class_names.add(class_name)

    available_classes = list(class_names)
    non_existing_classes = [c for c in classes if c not in available_classes]
    classes_inspector(non_existing_classes,
                      available_classes)


def convert_coco_to_yolo(coco_annotations_file: str = None,
                         coco_images_dir: str = None,
                         classes: list = None,
                         export_folder: str = None) -> None:
    """
    Core routine that converts coco data to yolo format and exports them

    Args:
        coco_annotations_file (str): path to the coco annotations directory
        coco_images_dir (str): path to the images directory
        classes (list): list of the provided classes (from the yaml file)
        export_folder (str): path converted dataset will be stored

    Returns:
        None
    """
    verify_coco_classes(coco_annotations_file,
                        classes)
    if not os.path.exists(export_folder):
        os.makedirs(export_folder)
    with open(coco_annotations_file, 'r') as f:
        coco_data = json.load(f)

    for image_info in tqdm(coco_data['images']):
        copy_image = False
        image_file_name = image_info['file_name']
        label_file_name = os.path.splitext(image_file_name)[0] + '.txt'
        label_file_path = os.path.join(export_folder, label_file_name)
        for annotation in coco_data['annotations']:
            if annotation['image_id'] == image_info['id']:
                category_id = annotation['category_id']
                class_name = None
                for category in coco_data['categories']:
                    if category['id'] == category_id:
                        class_name = category['name']
                        break
                if class_name in classes:
                    class_id = classes.index(class_name)
                    x, y, w, h = annotation['bbox']
                    x_center = x + (w / 2)
                    y_center = y + (h / 2)
                    x_center /= image_info['width']
                    y_center /= image_info['height']
                    w /= image_info['width']
                    h /= image_info['height']
                    with open(label_file_path, 'a') as label_file:
                        label_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

        if copy_image:
            image_path = os.path.join(coco_images_dir, image_file_name)
            shutil.copy(image_path, export_folder)


def verify_kitti_classes(kitti_annotations_dir: str = None, classes: list = None) -> None:
    """
    Check if all expected classes are well present in the provided dataset

    Args:
        kitti_annotations_dir (str): path to the KITTI annotations directory
        classes (list): list of the provided classes (from the yaml file)

    Returns:
        None
    """
    print("Analyzing the dataset ...")
    available_classes = set()
    annotation_files = [file for file in os.listdir(kitti_annotations_dir) if file.endswith('.txt')]
    for filename in tqdm(annotation_files):
        annotation_path = os.path.join(kitti_annotations_dir, filename)
        with open(annotation_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_name = parts[0]
                available_classes.add(class_name)

    non_existing_classes = [c for c in classes if c not in available_classes]
    classes_inspector(non_existing_classes, available_classes)


def convert_kitti_to_yolo(kitti_annotations_dir: str = None,
                          kitti_images_dir: str = None,
                          classes: list = None,
                          export_folder: str = None) -> None:
    """
    Core routine that converts KITTI data to YOLO format and exports them

    Args:
        kitti_annotations_dir (str): path to the KITTI annotations directory
        kitti_images_dir (str): path to the images directory
        classes (list): list of the provided classes (from the yaml file)
        export_folder (str): path converted dataset will be stored

    Returns:
        None
    """
    verify_kitti_classes(kitti_annotations_dir, classes)
    if not os.path.exists(export_folder):
        os.makedirs(export_folder)
    annotation_files = [file for file in os.listdir(kitti_annotations_dir) if file.endswith('.txt')]
    for filename in tqdm(annotation_files):
        copy_image = False
        annotation_path = os.path.join(kitti_annotations_dir, filename)
        txt_filename = os.path.splitext(filename)[0] + '.txt'
        txt_path = os.path.join(export_folder, txt_filename)
        image_file = os.path.splitext(filename)[0] + '.jpg'
        image_path = os.path.join(kitti_images_dir, image_file)
        if not os.path.exists(image_path):
            continue
        image = Image.open(image_path)
        width_image, height_image = image.size

        with open(annotation_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_name = parts[0]
                if class_name in classes:
                    copy_image = True
                    class_id = classes.index(class_name)
                    xmin, ymin, xmax, ymax = map(float, parts[4:8])
                    width = float(parts[8])
                    height = float(parts[9])
                    if width == 0 and height == 0:
                        w = xmax - xmin
                        h = ymax - ymin
                        x_center = (xmin + xmax) / 2
                        y_center = (ymin + ymax) / 2
                    else:
                        x_center = (xmin + xmax) / (2 * width)
                        y_center = (ymin + ymax) / (2 * height)
                        w = (xmax - xmin) / width
                        h = (ymax - ymin) / height
                    x_center /= width_image
                    y_center /= height_image
                    w /= width_image
                    h /= height_image
                    with open(txt_path, 'a') as label_file:
                        label_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

        if copy_image:
            shutil.copy(image_path, export_folder)

def convert_val_dataset_to_yolo(cfg: DictConfig) -> None:
    """
    Converts only the validation/test dataset to YOLO Darknet format according to the config.
    This is used when loading the validation/test dataset only (e.g., evaluation-only).

    For COCO:
        - Validation/test set is converted to cfg.dataset.test_path.

    For Pascal VOC:
        - Validation/test set is converted to cfg.dataset.test_path.

    Args:
        cfg (DictConfig): Configuration dictionary containing:
            - dataset.format: Format of the input dataset ("coco", "pascal_voc", "yolo_darknet", ...)
            - COCO:
                - dataset.val_annotations_path: Path to validation annotations
                - dataset.val_images_path: Path to validation images
            - Pascal VOC:
                - dataset.val_xml_dir: Path to validation XML annotations directory
                - dataset.val_images_path: Path to validation images
            - dataset.class_names: List of class names
            - dataset.test_path: Output path for converted validation/test data (required in this context)

    Returns:
        None
    """

    dataset_format = getattr(cfg.dataset, "format", None)

    if dataset_format == "coco":
        test_annotations = getattr(cfg.dataset, "test_annotations_path", None)
        test_images = getattr(cfg.dataset, "test_images_path", None)

        if not test_images or not os.path.isdir(test_images):
            raise ValueError("dataset.test_images_path must point to a valid directory.")
        if not test_annotations or not os.path.isfile(test_annotations):
            raise ValueError("dataset.test_annotations_path must point to a valid JSON file.")
        if not getattr(cfg.dataset, "class_names", None):
            raise ValueError("cfg.dataset.class_names must be specified for COCO format")
        
        raw_root = os.path.dirname(os.path.abspath(test_images))
        test_out_root = os.path.join(raw_root, "tfs_labels", "test")
        os.makedirs(test_out_root, exist_ok=True)
        cfg.dataset.test_path = test_out_root

        print(f"\nConverting COCO validation/test set to YOLO format in '{test_out_root}'...")
        convert_coco_to_yolo(
            coco_annotations_file=test_annotations,
            coco_images_dir=test_images,
            classes=cfg.dataset.class_names,
            export_folder=test_out_root,
        )

    elif dataset_format == "pascal_voc":
        # Prefer test split if provided, otherwise fallback to val split
        test_images = getattr(cfg.dataset, "test_images_path", None)
        test_ann = getattr(cfg.dataset, "test_annotations_path", None)
        test_split = getattr(cfg.dataset, "test_split", None)

        if not test_images or not os.path.isdir(test_images):
            raise ValueError("dataset.test_images_path must point to a valid directory.")
        if not test_ann or not os.path.isdir(test_ann):
            raise ValueError("dataset.test_annotations_path must point to a valid directory.")
        if not test_split or not os.path.isfile(test_split):
            raise ValueError("dataset.test_split must point to a valid split file.")
        if not getattr(cfg.dataset, "class_names", None):
            raise ValueError("cfg.dataset.class_names must be specified for Pascal VOC format")
        
        raw_root = os.path.dirname(os.path.abspath(test_images))
        test_out_root = os.path.join(raw_root, "tfs_labels", "test")
        os.makedirs(test_out_root, exist_ok=True)
        cfg.dataset.test_path = test_out_root

        print(f"\nConverting Pascal VOC validation/test split to YOLO format in '{test_out_root}'...")
        convert_voc_split_to_yolo(
            images_folder=test_images,
            annotations_folder=test_ann,
            split_file=test_split,
            classes=cfg.dataset.class_names,
            export_folder=test_out_root,
        )

    elif dataset_format == "yolo_darknet":
        print("Dataset is already in YOLO format. No conversion needed.")
        return

    else:
        print(
            "Please make sure that you selected one of the following formats: {}, {}, {}, {}".format(
                "coco", "pascal_voc", "yolo_darknet", "kitti"
            )
        )
        sys.exit()

def convert_dataset_to_yolo(cfg: DictConfig) -> None:
    """
    Converts the dataset to YOLO Darknet format according to the config.
    If validation paths are provided, converts both training and validation sets.

    For COCO:
        - Training set is converted to cfg.dataset.training_path if defined,
          otherwise to cfg.dataset.data_dir.
        - Validation set is converted to cfg.dataset.validation_path if defined,
          otherwise to cfg.dataset.data_dir.
          If val_images_path and val_annotations_path are specified, validation_path
          MUST be defined.

    For Pascal VOC:
        - Training set is converted to cfg.dataset.training_path if defined,
          otherwise to cfg.dataset.data_dir.
        - Validation set is converted to cfg.dataset.validation_path if defined,
          otherwise to cfg.dataset.data_dir.
          If val_images_path and val_xml_dir are specified, validation_path
          MUST be defined.

    Args:
        cfg (DictConfig): Configuration dictionary containing:
            - dataset.format: "coco", "pascal_voc", "yolo_darknet", ...
            - COCO:
                - dataset.train_annotations_path
                - dataset.train_images_path
                - dataset.val_annotations_path (optional)
                - dataset.val_images_path (optional)
            - Pascal VOC:
                - dataset.train_xml_dir
                - dataset.train_images_path
                - dataset.val_xml_dir (optional)
                - dataset.val_images_path (optional)
            - dataset.class_names
            - dataset.data_dir
            - dataset.training_path (optional)
            - dataset.validation_path (optional)

    Returns:
        None
    """

    dataset_format = getattr(cfg.dataset, "format", None)

    def _get_output_root(split: str) -> str:
        """
        split: "train" or "val"
        """
        if split == "train":
            out = getattr(cfg.dataset, "training_path", None)
        elif split == "val":
            out = getattr(cfg.dataset, "validation_path", None)
        else:
            raise ValueError(f"Unsupported split '{split}'")
        if not out:
            out = getattr(cfg.dataset, "data_dir", None)
        if not out:
            raise ValueError(f"Could not determine output directory for split '{split}'. "
                             f"Please specify dataset.{split}ing_path or dataset.data_dir in cfg."
            )

        os.makedirs(out, exist_ok=True)
        return out

    if dataset_format == "coco":
        if not getattr(cfg.dataset, "train_annotations_path", None):
            raise ValueError("cfg.dataset.train_annotations_path must be specified for COCO format")
        if not getattr(cfg.dataset, "train_images_path", None):
            raise ValueError("cfg.dataset.train_images_path must be specified for COCO format")
        if not getattr(cfg.dataset, "class_names", None):
            raise ValueError("cfg.dataset.class_names must be specified")

        # Validation presence check (for the stricter requirement on validation_path)
        train_images = getattr(cfg.dataset, "train_images_path", None)
        train_ann = getattr(cfg.dataset, "train_annotations_path", None)
        val_imgs = getattr(cfg.dataset, "val_images_path", None)
        val_ann  = getattr(cfg.dataset, "val_annotations_path", None)

        raw_root = os.path.dirname(os.path.abspath(train_images))
        train_out_root = os.path.join(raw_root, "tfs_labels", "train")
        os.makedirs(train_out_root, exist_ok=True)

        print(f"Converting COCO training set to YOLO format in '{train_out_root}'...")
        convert_coco_to_yolo(
            coco_annotations_file=train_ann,
            coco_images_dir=train_images,
            classes=cfg.dataset.class_names,
            export_folder=train_out_root,
        )

         # Optional explicit validation split
        has_explicit_val = bool(
            val_imgs and os.path.isdir(val_imgs) and
            val_ann and os.path.isfile(val_ann)
        )

        if has_explicit_val:
            val_out_root = os.path.join(raw_root, "tfs_labels", "val")
            os.makedirs(val_out_root, exist_ok=True)
            
            print(f"\nConverting COCO validation set to YOLO format in '{val_out_root}'...")
            convert_coco_to_yolo(
                val_ann,
                val_imgs,
                cfg.dataset.class_names,
                val_out_root,
            )
        else:
            print("\nCOCO validation paths not fully specified; skipping validation conversion.")

    elif dataset_format == "pascal_voc":
        # Minimal Pascal VOC config using splits
        images_path = getattr(cfg.dataset, "train_images_path", None)
        ann_path = getattr(cfg.dataset, "train_annotations_path", None)
        train_split = getattr(cfg.dataset, "train_split", None)
        val_images_path = getattr(cfg.dataset, "val_images_path", None)
        val_ann_path = getattr(cfg.dataset, "val_annotations_path", None)
        val_split = getattr(cfg.dataset, "val_split", None)

        if not images_path or not os.path.isdir(images_path):
            raise ValueError("dataset.train_images_path must point to a valid directory.")
        if not ann_path or not os.path.isdir(ann_path):
            raise ValueError("dataset.train_annotations_path must point to a valid directory.")
        if not train_split or not os.path.isfile(train_split):
            raise ValueError("dataset.train_split must point to a valid split file.")
        if not getattr(cfg.dataset, "class_names", None):
            raise ValueError("cfg.dataset.class_names must be specified for Pascal VOC format")


        raw_root = os.path.dirname(os.path.abspath(images_path))
        train_out_root = os.path.join(raw_root, "tfs_labels", "train")
        os.makedirs(train_out_root, exist_ok=True)

        print(f"Converting Pascal VOC training split to YOLO format in '{train_out_root}'...")
        convert_voc_split_to_yolo(
            images_folder=images_path,
            annotations_folder=ann_path,
            split_file=train_split,
            classes=cfg.dataset.class_names,
            export_folder=train_out_root,
        )
        
        has_explicit_val = bool(
            val_images_path and os.path.isdir(val_images_path) and
            val_ann_path and os.path.isdir(val_ann_path) and
            val_split and os.path.isfile(val_split)
        )

        if has_explicit_val:
            val_out_root = os.path.join(raw_root, "tfs_labels", "val")
            os.makedirs(val_out_root, exist_ok=True)
            print(f"\nConverting Pascal VOC validation split to YOLO format in '{val_out_root}'...")
            convert_voc_split_to_yolo(
                images_folder=val_images_path,
                annotations_folder=val_ann_path,
                split_file=val_split,
                classes=cfg.dataset.class_names,
                export_folder=val_out_root,
            )
            
        else:
            print("\nPascal VOC validation paths/split not fully specified; skipping validation conversion.")

    elif dataset_format == "yolo_darknet":
        print("Dataset is already in YOLO format. No conversion needed.")
        return

    else:
        print("Please make sure that you selected one of the following formats: {}, {}, {}, {}".format(
             "coco", "pascal_voc", "yolo_darknet", "kitti"))
        sys.exit()