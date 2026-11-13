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
import random
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
from pose_estimation.tf.src.data_augmentation import data_augmentation
from pose_estimation.tf.src.utils import plot_keypoints, parse_data_augmentation_section, parse_preprocessing_section, parse_dataset_section


def plot_keypoints_on_axis(ax, keypoints, image_width, image_height):
    """
    Plots keypoints on a matplotlib axis.

    Arguments:
        ax: matplotlib axis
        keypoints:
            Groundtruth keypoints
            A Tensor with shape [num_labels, 3*keypoints]
        image_width: width of the image
        image_height: height of the image
    """
    sh = tf.shape(keypoints)
    keypoints = tf.reshape(keypoints,[sh[0],sh[1]//3,3]) # shape [num_labels, keypoints, 3]

    keypoints_x = keypoints[...,0].numpy() * image_width
    keypoints_y = keypoints[...,1].numpy() * image_height

    ax.scatter(keypoints_x[:,0],keypoints_y[:,0])
    ax.scatter(keypoints_x[:,1::2],keypoints_y[:,1::2],c='red')
    ax.scatter(keypoints_x[:,2::2],keypoints_y[:,2::2],c='green')


def plot_image_and_labels(image: np.array, labels: np.array,
                          image_aug: np.array, labels_aug: np.array,
                          grayscale: bool = None,
                          save_dir: str = None, filename: str = None) -> None:
    """
    Displays side by side the original and augmented image
    with their groundtruth keypoints.

    Arguments:
        image:
            Original image.
            A numpy array with shape [width, height, channels]
        labels:
            Groundtruth labels of the original image.
            A numpy array with shape [num_labels, ...]
        image_aug:
            Augmented image.
            A numpy array with shape [width, height, channels]
        labels_aug:
            Groundtruth labels of the augmented image.
            A numpy array with shape [num_labels, ...]
        grayscale:
            Boolean indicating if image is grayscale
        save_dir:
            Optional directory to save the figure to.
        filename:
            Optional filename for saving the figure.

    Returns:
        None
    """

    # Calculate the dimensions of the displayed images
    image_width, image_height = np.shape(image)[:2]
    display_size = 9
    if image_width >= image_height:
        x_size = display_size
        y_size = round((image_width / image_height) * display_size)
    else:
        y_size = display_size
        x_size = round((image_height / image_width) * display_size)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(x_size, y_size))

    if grayscale:
        ax1.imshow(image, cmap='gray')
    else:
        ax1.imshow(image)
    plot_keypoints_on_axis(ax1, labels[:, 5:], image_width, image_height)
    ax1.title.set_text("Original")

    if grayscale:
        ax2.imshow(image_aug, cmap='gray')
    else:
        ax2.imshow(image_aug)
    plot_keypoints_on_axis(ax2, labels_aug[:, 5:], image_width, image_height)
    ax2.title.set_text("Augmented")

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
    keypoints.

    Arguments:
        config_file_path:
            A string, specifies the path to the YAML configuration file.
        seed_arg:
            An integer, the optional `seed` argument passed to the script.
            Set to None if the argument was not used.
        output_dir:
            Optional directory to save augmentation images.

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
    parse_data_augmentation_section(cfg)

    if not os.path.isabs(cfg.dataset.training_path):
        cfg.dataset.training_path = os.path.join("../../", cfg.dataset.training_path)

    cpp = cfg.preprocessing.rescaling
    pixels_range = (cpp.offset, 255*cpp.scale + cpp.offset)

    # If the `seed` argument of the script was not used,
    # different images will be sampled every time
    # the script is run.
    seed = seed_arg if seed_arg else None

    # Create a data loader to get examples from the training set
    print("Dataset path:", cfg.dataset.training_path)
    print("Dataset name:", cfg.dataset.dataset_name)
    print("Dataset format:", cfg.dataset.format)
    print("Sampling seed:", seed)

    cfg.use_case = "pose_estimation"
    cfg.model.framework = "tf"
    data_loader = get_dataloaders(cfg)

    if "random_periodic_resizing" in cfg.data_augmentation.config:
        print("[INFO] : Ignoring random_periodic_resizing function")
        del cfg.data_augmentation.config.random_periodic_resizing

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
    for data in data_loader['train']:
        images = data[0]
        labels = data[1]
        batch_size = tf.shape(images)[0]

        # Apply the data augmentation functions to the images and labels
        images_aug, labels_aug = data_augmentation(
                                images, labels,
                                cfg.data_augmentation.config,
                                pixels_range=pixels_range)

        # Map pixels values to the [0, 1] interval
        # to get correct displays in matplotlib
        images = remap_pixel_values_range(images, pixels_range, (0, 1))
        images_aug = remap_pixel_values_range(images_aug, pixels_range, (0, 1))

        # Plot the images and their groundtruth labels
        for i in range(batch_size):
            if output_dir:
                filename = f"augmentation_{img_counter:04d}.png"
                plot_image_and_labels(images[i], labels[i],
                                      images_aug[i], labels_aug[i],
                                      save_dir=output_dir, filename=filename)
                img_counter += 1
            else:
                plot_image_and_labels(images[i], labels[i],
                                      images_aug[i], labels_aug[i])


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="../../user_config.yaml",
                        help="Path to the YAML configuration file starting from the directory " + \
                        "above this script. Default: ../../user_config.yaml")
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
