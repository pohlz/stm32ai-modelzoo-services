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
from pathlib import Path
import argparse
from omegaconf import OmegaConf
from munch import DefaultMunch
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
print(ROOT_DIR)
sys.path.append(ROOT_DIR)

from common.data_augmentation import remap_pixel_values_range
from common.utils import postprocess_config_dict
from re_identification.tf.src.data_augmentation import data_augmentation
from re_identification.tf.src.preprocessing import preprocess
from re_identification.tf.src.utils import parse_dataset_section, parse_preprocessing_section, parse_data_augmentation_section


def display_images_side_by_side(image, image_aug, save_dir: str = None, filename: str = None):
    """
    This function displays (or saves) the original and augmented images side by side.

    Args:
        image: original image.
        image_aug: corresponding augmented image.
        save_dir (str): directory to save the figure. If None, plt.show() is used.
        filename (str): filename to use when saving. Required if save_dir is not None.

    Returns:
        None
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(image)
    ax1.title.set_text("original")
    ax1.axis("off")
    ax2.imshow(image_aug)
    ax2.title.set_text("augmented")
    ax2.axis("off")

    plt.tight_layout()
    if save_dir and filename:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def test_data_augmentation(config_file_path: str, seed_arg: str = None, output_dir: str = None) -> None:
    """
    Samples a batch of images with their groundtruth labels from
    the training set, applies to them the data augmentation functions
    specified in the YAML configuration file, and displays side by side
    the original images and augmented images with their groundtruth
    bounding boxes.

    Arguments:
        config_file_path:
            A string, specifies the path to the YAML configuration file.
        seed_args:
            An integer, the optional `seed` argument passed to the script.
            Set to None if the argument was not used.

    Returns:
        None
    """

    # If the `seed` argument of the script was not used,
    # random generators are not seeded.
    if seed_arg:
        tf.keras.utils.set_random_seed(seed_arg)

    # Load and postprocess the configuration file
    config_data = OmegaConf.load(config_file_path)
    config_dict = OmegaConf.to_container(config_data)
    postprocess_config_dict(config_dict, replace_none_string=True)
    cfg = DefaultMunch.fromDict(config_dict)

    # Check that there is a data augmentation section
    # and that the operation mode is set to 'training'
    if not cfg.data_augmentation and not cfg.custom_data_augmentation:
        raise ValueError("\nCould not find any data augmentation section.\n"
                         "Please check your configuration file.")
    if cfg.operation_mode != "training":
        raise ValueError("\nPlease set `operation_mode` to 'training' to run this script.")

    # Parse the needed config file sections
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
    parse_dataset_section(cfg.dataset, "training", mode_groups)
    parse_preprocessing_section(cfg.preprocessing, mode="training")
    parse_data_augmentation_section(cfg, config_dict)

    # Set the 'name' field that preprocess() expects from 'dataset_name'
    if hasattr(cfg.dataset, 'dataset_name') and not hasattr(cfg.dataset, 'name'):
        cfg.dataset.name = cfg.dataset.dataset_name

    if not os.path.isabs(cfg.dataset.training_path):
        cfg.dataset.training_path = os.path.join("../", cfg.dataset.training_path)

    scale = cfg.preprocessing.rescaling.scale
    offset = cfg.preprocessing.rescaling.offset
    pixels_range = (offset, 255*scale + offset)

    # If the `seed` argument of the script was not used,
    # different images will be sampled every time
    # the script is run.
    seed = seed_arg if seed_arg else None
    if seed_arg:
        cfg.dataset.seed = seed_arg
    else:
        cfg.dataset.seed = None

    # Create a data loader to get examples from the training set
    print("Dataset:", cfg.dataset.training_path)
    print("Dataset name:", cfg.dataset.dataset_name)
    print("Sampling seed:", seed)

    data_loader, _, _, _, _ = preprocess(cfg)

    augmentation_functions = list(cfg.data_augmentation.config.keys())
    if len(augmentation_functions) == 0:
        print("No data augmentation functions to test. Exiting script...")
        exit()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving augmentation images to: {output_dir}")
    else:
        print("Use ctrl+c to exit the script")

    img_counter = 0
    for data in data_loader:
        images = data[0]
        batch_size = tf.shape(images)[0]

        # Rescale the images
        images = scale * tf.cast(images, dtype=tf.float32) + offset

        # Apply the data augmentation functions to the images
        images_aug = data_augmentation(images, cfg.data_augmentation.config, pixels_range=pixels_range)

        # Map pixels values to the [0, 1] interval
        # to get correct displays in matplotlib
        images = remap_pixel_values_range(images, pixels_range, (0, 1))
        images_aug = remap_pixel_values_range(images_aug, pixels_range, (0, 1))

        # Plot the original and augmented images side-by-side
        for i in range(batch_size):
            if output_dir:
                filename = f"augmentation_{img_counter:04d}.png"
                display_images_side_by_side(images[i], images_aug[i],
                                          save_dir=output_dir, filename=filename)
                img_counter += 1
            else:
                display_images_side_by_side(images[i], images_aug[i])


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="../../user_config.yaml",
                        help="Path to the YAML configuration file starting from the directory " + \
                        "above this script. Default: ../user_config.yaml")
    parser.add_argument("--seed", type=str, default="",
                        help="Seed for the random generators used to sample the dataset. " + \
                        "By default, samples will be different every time the script is run.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Optional directory to save augmentation images (default: display only)")

    args = parser.parse_args()

    if not os.path.isfile(Path(args.config_file)):
        raise ValueError(f"\nCould not find configuration file {args.config_file}")

    if args.seed:
        try:
            seed = int(args.seed)
        except:
            raise ValueError(f"\nThe `seed` argument should be an integer. Received {args.seed}")
    else:
        seed = None

    test_data_augmentation(Path(args.config_file), seed_arg=seed, output_dir=args.output_dir)

if __name__ == '__main__':
    main()

