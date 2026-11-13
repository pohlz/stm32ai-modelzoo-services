# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
import yaml
import argparse

def create_trainaug_txt(masks_dir, val_file, output_file):
    """
    Create a trainaug.txt file with the IDs of the images that have additional masks,
    excluding the IDs present in val.txt.

    Args:
        masks_dir (str): Directory containing the additional masks.
        val_file (str): Path to the val.txt file containing validation IDs.
        output_file (str): Path to the output trainaug.txt file.
    """
    # Read validation IDs from val.txt
    with open(val_file, 'r') as vf:
        val_ids = set(line.strip() for line in vf)

    # Get mask files and filter out validation IDs
    mask_files = [f for f in os.listdir(masks_dir) if f.endswith('.png')]
    image_ids = [os.path.splitext(f)[0] for f in mask_files if os.path.splitext(f)[0] not in val_ids]

    # Write the filtered image IDs to trainaug.txt
    with open(output_file, 'w') as f:
        for image_id in image_ids:
            f.write(f"{image_id}\n")

def load_config(config_file):
    """
    Load configuration from a YAML file.

    Args:
        config_file (str): Path to the YAML configuration file.
    
    Returns:
        dict: Configuration parameters.
    """
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main(config_path):
    """
    Main function to load configuration and create trainaug.txt.

    Args:
        config_path (str): Path to the YAML configuration file.
    """
    config = load_config(config_path)
    masks_dir = config['masks_dir']
    val_file = config['val_file']
    output_file = config['output_file']

    create_trainaug_txt(masks_dir, val_file, output_file)
    print(f"trainaug.txt has been created with {len(os.listdir(masks_dir)) - len(open(val_file).readlines())} entries.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create trainaug.txt for Pascal VOC 2012 dataset augmentation.")
    parser.add_argument('--config-path', type=str, default='dataset_config.yaml', help='Path to the configuration file')
    args = parser.parse_args()

    main(args.config_path)