# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
import shutil
import argparse
from typing import List, Dict, Any
import numpy as np
from pycocotools.coco import COCO
from PIL import Image
from tqdm import tqdm
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def create_directory(directory_path: str) -> None:
    """
    Create a directory if it does not exist.

    Args:
        directory_path (str): Path to the directory.
    """
    os.makedirs(directory_path, exist_ok=True)


def process_coco_images_and_masks(
    coco: COCO,
    category_ids: List[int],
    image_ids: List[int],
    voc_categories: List[str],
    smallest_annotation_area: int,
    final_masks_path: str,
    final_images_path: str,
    coco_images_path: str
) -> None:
    """
    Process COCO images and masks, save masks, and copy images.

    Args:
        coco (COCO): COCO object.
        category_ids (List[int]): List of category IDs.
        image_ids (List[int]): List of image IDs.
        voc_categories (List[str]): List of VOC category names.
        smallest_annotation_area (int): Minimum area for valid annotations.
        final_masks_path (str): Path to save final masks.
        final_images_path (str): Path to save final images.
        coco_images_path (str): Path to COCO images.
    """
    for image_id in tqdm(image_ids):
        image_info = coco.loadImgs(image_id)[0]
        annotation_ids = coco.getAnnIds(imgIds=image_info['id'], catIds=category_ids, iscrowd=None)
        annotations = coco.loadAnns(annotation_ids)

        # Initialize an empty mask with the same dimensions as the image
        combined_mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)

        # Create a flag to check if the mask is to be written
        valid_mask = False

        # Accumulate masks for all selected categories in the image
        for annotation in annotations:
            mask = coco.annToMask(annotation)
            # Apply a filtering for the annotation mask to be at least of smallest_annotation_area pixels
            if annotation['area'] > smallest_annotation_area:
                valid_mask = True
            combined_mask[mask > 0] = voc_categories.index(coco.loadCats(annotation['category_id'])[0]['name']) + 1

        # If the valid_mask is true, write the image
        if valid_mask:
            # Save the combined mask image
            combined_mask_image = Image.fromarray(combined_mask)
            combined_mask_image.save(os.path.join(final_masks_path, f"{image_info['file_name'].split('.')[0]}.png"))

            # Copy the original COCO image to the final images directory
            shutil.copy(os.path.join(coco_images_path, image_info['file_name']), final_images_path)


def copy_pascal_voc_images_and_masks(
    pascal_voc_images_path: str,
    pascal_voc_masks_path: str,
    final_images_path: str,
    final_masks_path: str
) -> None:
    """
    Copy Pascal VOC images and masks to the final directories.

    Args:
        pascal_voc_images_path (str): Path to Pascal VOC images.
        pascal_voc_masks_path (str): Path to Pascal VOC masks.
        final_images_path (str): Path to save final images.
        final_masks_path (str): Path to save final masks.
    """
    for filename in os.listdir(pascal_voc_images_path):
        shutil.copy(os.path.join(pascal_voc_images_path, filename), final_images_path)

    for filename in os.listdir(pascal_voc_masks_path):
        shutil.copy(os.path.join(pascal_voc_masks_path, filename), final_masks_path)


def create_train_txt_file(
    final_masks_path: str,
    txt_val_file: str,
    final_txt_training_file: str,
    final_txt_val_file: str
) -> None:
    """
    Create train.txt file with the IDs of the masks present in the final masks directory,
    excluding the IDs present in the validation file.

    Args:
        final_masks_path (str): Path to the final masks directory.
        txt_val_file (str): Path to the validation file.
        final_txt_training_file (str): Path to save the training file.
        final_txt_val_file (str): Path to save the validation file.
    """
    # Read validation IDs
    with open(txt_val_file, 'r') as val_file:
        val_ids = set(line.strip() for line in val_file)

    # Write validation IDs to the final validation file
    with open(final_txt_val_file, 'w') as val_file:
        for val_id in val_ids:
            val_file.write(f"{val_id}\n")

    # Create training file excluding validation IDs
    with open(final_txt_training_file, 'w') as train_file:
        for mask_filename in os.listdir(final_masks_path):
            if mask_filename.endswith('.png'):
                mask_id = os.path.splitext(mask_filename)[0]
                if mask_id not in val_ids:
                    train_file.write(f"{mask_id}\n")


def main(config_path: str) -> None:
    """
    Main function to process COCO and Pascal VOC datasets and create train.txt file.

    Args:
        config_path (str): Path to the YAML configuration file.
    """
    config = load_config(config_path)

    coco_annotations_path = config['coco']['annotations_path']
    coco_images_path = config['coco']['images_path']
    pascal_voc_images_path = config['pascal_voc']['images_path']
    pascal_voc_masks_path = config['pascal_voc']['masks_path']
    txt_val_file = config['pascal_voc']['txt_val_file']
    final_images_path = config['output']['final_images_path']
    final_masks_path = config['output']['final_masks_path']
    final_txt_training_file = config['output']['final_txt_training_file']
    final_txt_val_file = config['output']['final_txt_val_file']
    smallest_annotation_area = config['smallest_annotation_area']
    voc_categories = config['voc_categories']

    # Initialize COCO API for instance annotations
    coco = COCO(coco_annotations_path)

    # Get the corresponding category IDs from COCO
    category_ids = coco.getCatIds(catNms=voc_categories)
    print(f"Selected Category IDs: {category_ids}")

    # Initialize an empty set to store all image IDs
    image_ids = set()

    # Get all images containing any of the above Category IDs
    for category_id in category_ids:
        image_ids.update(coco.getImgIds(catIds=[category_id]))

    # Convert the set to a list
    image_ids = list(image_ids)
    print(f"Number of images containing the selected categories: {len(image_ids)}")

    # Create directories for saving the final masks and images
    create_directory(final_images_path)
    create_directory(final_masks_path)

    # Process COCO images and masks
    process_coco_images_and_masks(
        coco, category_ids, image_ids, voc_categories,
        smallest_annotation_area, final_masks_path,
        final_images_path, coco_images_path
    )

    # Copy Pascal VOC images and masks to the final directories
    copy_pascal_voc_images_and_masks(
        pascal_voc_images_path, pascal_voc_masks_path,
        final_images_path, final_masks_path
    )

    # Create train.txt file with the IDs of the masks present in the final masks directory
    create_train_txt_file(final_masks_path, txt_val_file, final_txt_training_file, final_txt_val_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, default='dataset_config.yaml', help='Path to the configuration file')
    args = parser.parse_args()

    main(args.config_path)