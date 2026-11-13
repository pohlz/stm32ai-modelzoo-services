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


from api import get_dataloaders
from common.data_augmentation import remap_pixel_values_range
from common.utils import postprocess_config_dict
from image_classification.tf.src.utils import parse_dataset_section, parse_preprocessing_section, parse_data_augmentation_section
from image_classification.tf.src.preprocessing import preprocess
from image_classification.tf.src.data_augmentation import data_augmentation


def display_images_side_by_side(image, image_aug, save_dir=None, filename=None):
    """
    This function displays the original and augmented images side by side or saves them.

    Args:
        image: original image.
        image_aug: corresponding augmented image.
        save_dir (str): directory to save the images. If None, images are displayed.
        filename (str): filename to use when saving. Required if save_dir is not None.

    Returns:
        None
    """

    # Ensure NumPy arrays (for tf.Tensor inputs)
    image = np.array(image)
    image_aug = np.array(image_aug)

    # Clip values to [0, 1] range for proper visualization
    image = np.clip(image, 0, 1)
    image_aug = np.clip(image_aug, 0, 1)

    f = plt.figure()
    f.add_subplot(1, 2, 1)
    plt.imshow(image, cmap='gray', vmin=0, vmax=1)
    plt.title("original")
    f.add_subplot(1, 2, 2)
    plt.imshow(image_aug, cmap='gray', vmin=0, vmax=1)
    plt.title("augmented")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        if filename is None:
            filename = "comparison.png"
        out_path = os.path.join(save_dir, filename)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
        plt.close()
    else:
        plt.show(block=True)
        plt.close()


def test_data_augmentation(config_file_path: str, seed_arg: str = None, num_image: int = 0, output_dir: str = None) -> None:
    """
        Samples a batch of images with their groundtruth labels from
        the training set, applies to them the data augmentation functions
        specified in the YAML configuration file, and displays or saves side by side
        the original images and augmented images.

        Args:
            config_file_path (str): specifies the path to the YAML configuration file.
            seed_arg (int): the optional `seed` argument passed to the script. Set to None if the argument was not used.
            num_image (int): number of images used for testing data augmentation
            output_dir (str): directory to save the images. If None, images are displayed.

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
    mode_groups = DefaultMunch.fromDict({"training": ["training"], "evaluation": ["evaluation"], "quantization": ["quantization"]})
    parse_dataset_section(cfg.dataset, "training", mode_groups)
    parse_preprocessing_section(cfg.preprocessing, mode="training")
    parse_data_augmentation_section(cfg, config_dict)

    if not os.path.isabs(cfg.dataset.training_path):
        cfg.dataset.training_path = os.path.join("../", cfg.dataset.training_path)

    scale = cfg.preprocessing.rescaling.scale
    offset = cfg.preprocessing.rescaling.offset
    pixels_range = (offset, 255*scale + offset)

    # If the `seed` argument of the script was not used,
    # different images will be sampled every time
    # the script is run.
    if seed_arg:
        cfg.dataset.seed = seed_arg
    else:
        cfg.dataset.seed = None

    # Create a data loader to get examples from the training set
    print("Dataset:", cfg.dataset.training_path)
    print("Sampling seed:", seed_arg if seed_arg else "None")

    if output_dir:
        print(f"Saving augmentation test images to: {output_dir}")
    else:
        print("Display mode: Images will be shown (use ctrl+c to exit)")

     # Set the use_case and framework before calling the dataloader
    cfg.use_case = "image_classification"
    cfg.model.framework = "tf"

    data_loader = get_dataloaders(cfg)

    augmentation_functions = list(cfg.data_augmentation.config.keys())
    if len(augmentation_functions) == 0:
        print("No data augmentation functions to test. Exiting script...")
        exit()

    for data in data_loader['train']:
        images = data[0]
        batch_size = tf.shape(images)[0]

        # Convert to float32 if needed (dataloader already normalizes to [0, 1])
        images = tf.cast(images, dtype=tf.float32)

        # Apply the data augmentation functions to the images
        images_aug = data_augmentation(images, cfg.data_augmentation.config, pixels_range=(0, 1))

        # Clip to [0, 1] range for display
        images = tf.clip_by_value(images, 0, 1)
        images_aug = tf.clip_by_value(images_aug, 0, 1)

        # Display or save the original and augmented images side-by-side
        img_counter = 0
        if num_image == 0 :
            for i in range(batch_size):
                if output_dir:
                    filename = f"augmentation_test_{img_counter:04d}.png"
                    display_images_side_by_side(images[i], images_aug[i], save_dir=output_dir, filename=filename)
                else:
                    display_images_side_by_side(images[i], images_aug[i])
                img_counter += 1
        else:
            for i in range(0, num_image):
                if output_dir:
                    filename = f"augmentation_test_{img_counter:04d}.png"
                    display_images_side_by_side(images[i], images_aug[i], save_dir=output_dir, filename=filename)
                else:
                    display_images_side_by_side(images[i], images_aug[i])
                img_counter += 1
            break


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="../../user_config.yaml",
                        help="Path to the YAML configuration file starting from the directory " + \
                        "above this script. Default: ../user_config.yaml")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for the random generators used to sample the dataset. " + \
                        "By default, samples will be different every time the script is run.")
    parser.add_argument("--num_image", type=int, default=0,
                        help="Number of displayed images. Default: 0, all the dataset.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save the augmentation test images. If not provided, images will be displayed.")

    args = parser.parse_args()

    if not os.path.isfile(Path(args.config_file )):
        raise ValueError(f"\nCould not find configuration file {args.config_file}")

    test_data_augmentation(Path(args.config_file), seed_arg=args.seed, num_image=args.num_image, output_dir=args.output_dir)

if __name__ == '__main__':
    main()

