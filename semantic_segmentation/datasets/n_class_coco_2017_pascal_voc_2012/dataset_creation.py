# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
import numpy as np
import yaml
import argparse
from tqdm import tqdm
from typing import List, Tuple

# Define the original classes
original_classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
                    "car", "cat", "chair", "cow", "dining table", "dog", "horse", "motorbike",
                    "person", "potted plant", "sheep", "sofa", "train", "tv/monitor"]

def remap_classes(mask: np.ndarray, class_map: List[str]) -> np.ndarray:
    """
    Remap class indices in the mask to new values based on the specified class map.

    Args:
        mask (np.ndarray): The original mask with class indices.
        class_map (List[str]): The list of class names to remap.

    Returns:
        np.ndarray: The remapped mask with new class indices.
    """
    remapped_mask = np.zeros_like(mask)
    for new_idx, class_name in enumerate(class_map):
        old_idx = original_classes.index(class_name)
        remapped_mask[mask == old_idx] = new_idx
    return remapped_mask

def verify_mask(mask: np.ndarray, allowed_indices: List[int]) -> Tuple[bool, np.ndarray]:
    """
    Verify that the mask contains only the allowed indices.

    Args:
        mask (np.ndarray): The mask to verify.
        allowed_indices (List[int]): The list of allowed class indices.

    Returns:
        Tuple[bool, np.ndarray]: A tuple containing a boolean indicating validity and the unique indices found in the mask.
    """
    unique_indices = np.unique(mask)
    for idx in unique_indices:
        if idx not in allowed_indices:
            return False, unique_indices
    return True, unique_indices

def exceeds_pixel_threshold(mask: np.ndarray, interesting_class_indices: List[int], pixel_threshold: int) -> bool:
    """
    Check if any interesting class exceeds the pixel threshold.

    Args:
        mask (np.ndarray): The mask to check.
        interesting_class_indices (List[int]): The list of interesting class indices.
        pixel_threshold (int): The pixel threshold to check against.

    Returns:
        bool: True if any interesting class exceeds the pixel threshold, False otherwise.
    """
    for idx in interesting_class_indices:
        if np.sum(mask == idx) > pixel_threshold:
            return True
    return False

def analyze_dataset(new_train_ids: List[str], new_val_ids: List[str], output_dir: str, class_map: List[str]) -> None:
    """
    Analyze the dataset and print statistics.

    Args:
        new_train_ids (List[str]): List of new train image IDs.
        new_val_ids (List[str]): List of new val image IDs.
        output_dir (str): Path to the directory where the output is saved.
        class_map (List[str]): The list of class names to keep.
    """
    import cv2
    print(f"Number of images in the new train set: {len(new_train_ids)}")
    print(f"Number of images in the new val set: {len(new_val_ids)}")

    class_pixel_counts = {class_name: 0 for class_name in class_map}
    class_image_counts = {class_name: 0 for class_name in class_map}
    total_pixels = 0

    for img_id in new_train_ids + new_val_ids:
        mask_path = os.path.join(output_dir, "SegmentationClassAug", f"{img_id}.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        total_pixels += mask.size
        for new_idx, class_name in enumerate(class_map):
            class_pixel_counts[class_name] += np.sum(mask == new_idx)
            if np.any(mask == new_idx):
                class_image_counts[class_name] += 1

    print("Percentage of pixels for each class in the new dataset:")
    for class_name , pixel_count in class_pixel_counts.items(): 
        if class_name != "background":
            percentage = (pixel_count / total_pixels) * 100
            print(f"{class_name}: {percentage:.2f}%")

    if len(class_map) > 2:
        print("Number of images containing each class in the new dataset:")
        for class_name, image_count in class_image_counts.items():
            print(f"{class_name}: {image_count} images")


def filter_and_save_dataset(image_dir: str, mask_dir: str, train_ids_file: str, val_ids_file: str, output_dir: str, class_map: List[str], pixel_threshold: int) -> Tuple[List[str], List[str]]:
    """
    Filters and saves a dataset of images and segmentation masks based on classes of interest and a pixel threshold.

    This function reads the IDs of the training and validation datasets, filters the images and masks based on the classes of interest and a pixel threshold. The filtered images and masks 
    are then saved in a specified output directory. The new IDs of the training and validation datasets are also saved in text files.

    Args:
        image_dir (str): Directory containing the input images.
        mask_dir (str): Directory containing the input segmentation masks.
        train_ids_file (str): Path to the text file containing the training image IDs.
        val_ids_file (str): Path to the text file containing the validation image IDs.
        output_dir (str): Directory where the filtered images and masks will be saved.
        class_map (List[str]): List of classes of interest to retain in the segmentation masks.
        pixel_threshold (int): Minimum pixel threshold for a class to be considered present in an image.

    Returns:
        Tuple[List[str], List[str]]: Two lists containing the new IDs of the training and validation datasets respectively.

    """
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "JPEGImages"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "SegmentationClassAug"), exist_ok=True)

    new_train_ids = []
    new_val_ids = []
    allowed_indices = list(range(len(class_map)))
    
    with open(train_ids_file, 'r') as f:
        train_ids = f.read().splitlines()
    
    with open(val_ids_file, 'r') as f:
        val_ids = f.read().splitlines()
    
    # Process validation set first
    for img_id in tqdm(val_ids):
        img_path = os.path.join(image_dir, f"{img_id}.jpg")
        mask_path = os.path.join(mask_dir, f"{img_id}.png")
        
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            continue
        
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        interesting_class_indices = [original_classes.index(cls) for cls in class_map if cls != "background"]
        if np.any(np.isin(mask, interesting_class_indices)):
            if not exceeds_pixel_threshold(mask, interesting_class_indices, pixel_threshold):
                continue
            
            remapped_mask = remap_classes(mask, class_map)
            is_valid, unique_indices = verify_mask(remapped_mask, allowed_indices)
            if not is_valid:
                print(f"Warning: Mask for image {img_id} contains invalid indices: {unique_indices}")
                continue
            
            new_val_ids.append(img_id)
            cv2.imwrite(os.path.join(output_dir, "JPEGImages", f"{img_id}.jpg"), image)
            cv2.imwrite(os.path.join(output_dir, "SegmentationClassAug", f"{img_id}.png"), remapped_mask)
    
    # Process train set and complete validation set if needed
    for img_id in tqdm(train_ids):
        if img_id in new_val_ids:
            continue
        
        img_path = os.path.join(image_dir, f"{img_id}.jpg")
        mask_path = os.path.join(mask_dir, f"{img_id}.png")
        
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            continue
        
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        interesting_class_indices = [original_classes.index(cls) for cls in class_map if cls != "background"]
        if np.any(np.isin(mask, interesting_class_indices)):
            if not exceeds_pixel_threshold(mask, interesting_class_indices, pixel_threshold):
                continue
            
            remapped_mask = remap_classes(mask, class_map)
            is_valid, unique_indices = verify_mask(remapped_mask, allowed_indices)
            if not is_valid:
                print(f"Warning: Mask for image {img_id} contains invalid indices: {unique_indices}")
                continue
            
            if len(new_val_ids) < len(val_ids):
                new_val_ids.append(img_id)
            else:
                new_train_ids.append(img_id)
            
            cv2.imwrite(os.path.join(output_dir, "JPEGImages", f"{img_id}.jpg"), image)
            cv2.imwrite(os.path.join(output_dir, "SegmentationClassAug", f"{img_id}.png"), remapped_mask)
    
    with open(os.path.join(output_dir, "train.txt"), 'w') as f:
        f.write("\n".join(new_train_ids))
    
    with open(os.path.join(output_dir, "val.txt"), 'w') as f:
        f.write("\n".join(new_val_ids))
    
    return new_train_ids, new_val_ids

def main(config_path: str) -> None:
    """
    Main function to process the dataset based on the configuration file.

    Args:
        config_path (str): Path to the configuration file.
    """
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    image_dir = config['image_dir']
    mask_dir = config['mask_dir']
    train_ids_file = config['train_ids_file']
    val_ids_file = config['val_ids_file']
    output_base_dir = config['output_base_dir']
    interesting_classes = config['interesting_classes']
    dataset_name = config['dataset_name']
    pixel_threshold = config.get('pixel_threshold', 1000)

    output_dir = os.path.join(output_base_dir, dataset_name)

    new_train_ids, new_val_ids = filter_and_save_dataset(image_dir, mask_dir, train_ids_file, val_ids_file, output_dir, interesting_classes, pixel_threshold)

    analyze_dataset(new_train_ids, new_val_ids, output_dir, interesting_classes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, default='dataset_config.yaml', help='Path to the configuration file')
    args = parser.parse_args()

    main(args.config_path)