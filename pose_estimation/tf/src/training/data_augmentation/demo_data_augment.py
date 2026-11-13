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
import argparse
from munch import DefaultMunch
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
print(ROOT_DIR)
sys.path.append(ROOT_DIR)

from api import get_dataloaders
from common.data_augmentation import random_color, random_erasing, remap_pixel_values_range
from pose_estimation.tf.src.data_augmentation import pose_random_affine, pose_random_misc


def plot_keypoints(ax, keypoints, image_width, image_height):
    """
    Displays side by side the original and augmented image
    with their groundtruth bounding boxes.

    Arguments:
        ax: matplotlib axis
        keypoints:
            Groundtruth keypoints
            A Tensor with shape [num_labels, 3*keypoints]
    """
    sh = tf.shape(keypoints)
    keypoints = tf.reshape(keypoints,[sh[0],sh[1]//3,3]) # shape [num_labels, keypoints, 3]

    keypoints_x = keypoints[...,0].numpy() * image_width
    keypoints_y = keypoints[...,1].numpy() * image_height

    ax.scatter(keypoints_x[:,0],keypoints_y[:,0])
    ax.scatter(keypoints_x[:,1::2],keypoints_y[:,1::2],c='red')
    ax.scatter(keypoints_x[:,2::2],keypoints_y[:,2::2],c='green')

    return 0

def plot_images_and_labels(image, labels, image_aug, labels_aug, grayscale=None, legend=None, save_dir=None, filename=None):
    """
    Displays or saves side by side the original and augmented image
    with their groundtruth bounding boxes.

    Arguments:
        images:
            Original image.
            A numpy array with shape [width, height, channels]
        labels:
            Groundtruth labels of the original image.
            A numpy array with shape [num_labels, 5]
        images_aug:
            Augmented images.
            A numpy rray with shape [width, height, channels]
        images_aug:
            Groundtruth labels of the augmented images.
            A numpy array with shape [num_labels, 5]
        class_names:
            A list of strings, the class names.
        save_dir (str): directory to save the figure. If None, plt.show() is used.
        filename (str): filename to use when saving. Required if save_dir is not None.

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

    # # Sample box colors
    # num_boxes = np.shape(labels)[0]
    # colors = []
    # for i in range(num_boxes):
    #     colors.append(random.choice(['b', 'g', 'r', 'c', 'm', 'y', 'w']))

    if grayscale:
        ax1.imshow(image, cmap='gray')
    else:
        ax1.imshow(image)
    plot_keypoints(ax1, labels[:, 5:], image_width, image_height)
    ax1.title.set_text("original")

    if grayscale:
        ax2.imshow(image_aug, cmap='gray')
    else:
        ax2.imshow(image_aug)
    plot_keypoints(ax2, labels_aug[:, 5:], image_width, image_height)
    ax2.title.set_text(legend)

    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        if filename is None:
            filename = "comparison.png"
        out_path = os.path.join(save_dir, filename)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def augment_images(images, labels, fn_name=None, pixels_range=None):

    if fn_name == "random_contrast":
        images_aug = random_color.random_contrast(images, factor=0.7, pixels_range=pixels_range)
        return images_aug, labels

    elif fn_name == "random_brightness":
        images_aug = random_color.random_brightness(images, factor=0.4, pixels_range=pixels_range)
        return images_aug, labels

    elif fn_name == "random_gamma":
        images_aug = random_color.random_gamma(images, gamma=(0.2, 2.0), pixels_range=pixels_range)
        return images_aug, labels

    elif fn_name == "random_hue":
        images_aug = random_color.random_hue(images, delta=0.1, pixels_range=pixels_range)
        return images_aug, labels

    elif fn_name == "random_saturation":
        images_aug = random_color.random_saturation(images, delta=0.4, pixels_range=pixels_range)
        return images_aug, labels

    elif fn_name == "random_value":
        images_aug = random_color.random_value(images, delta=0.4, pixels_range=pixels_range)
        return images_aug, labels

    elif fn_name == "random_hsv":
        images_aug = random_color.random_hsv(
                        images, hue_delta=0.1, saturation_delta=0.1, value_delta=0.2, pixels_range=pixels_range)
        return images_aug, labels

    elif fn_name == "random_rgb_to_hsv":
        images_aug = random_color.random_rgb_to_hsv(images, pixels_range=pixels_range, change_rate=1.0)
        return images_aug, labels

    elif fn_name == "random_rgb_to_grayscale":
        images_aug = random_color.random_rgb_to_grayscale(images, pixels_range=pixels_range, change_rate=1.0)
        return images_aug, labels

    elif fn_name == "random_sharpness":
        images_aug = random_color.random_sharpness(images, factor=(1.0, 4.0), pixels_range=pixels_range)
        return images_aug, labels

    elif fn_name == "random_posterize":
        images_aug = random_color.random_posterize(images, bits=(1, 8), pixels_range=pixels_range)
        return images_aug, labels

    elif fn_name == "random_invert":
        images_aug = random_color.random_invert(images, pixels_range=pixels_range, change_rate=1.0)
        return images_aug, labels

    elif fn_name == "random_solarize":
        images_aug = random_color.random_solarize(images, pixels_range=pixels_range, change_rate=1.0)
        return images_aug, labels

    elif fn_name == "random_autocontrast":
        images_aug = random_color.random_autocontrast(images, cutoff=10, pixels_range=pixels_range, change_rate=1.0)
        return images_aug, labels

    elif fn_name == "random_blur":
        images_aug = pose_random_misc.objdet_random_blur(images, filter_size=(2, 4), pixels_range=pixels_range, change_rate=1.0)
        return images_aug, labels

    elif fn_name == "random_gaussian_noise":
        images_aug = pose_random_misc.objdet_random_gaussian_noise(images, stddev=(0.02, 0.1), pixels_range=pixels_range, change_rate=1.0)
        return images_aug, labels

    elif fn_name == "random_crop":
        return pose_random_misc.objdet_random_crop(images, labels, change_rate=1.0)

    elif fn_name == "random_flip":
        return pose_random_affine.pose_random_flip(images, labels, mode="horizontal", change_rate=0.5)

    elif fn_name == "random_translation":
        return pose_random_affine.objdet_random_translation(images, labels, width_factor=0.2, height_factor=0.2)

    elif fn_name == "random_rotation":
        return pose_random_affine.pose_random_rotation(images, labels, factor=0.1, fill_mode='constant',interpolation='bilinear',fill_value=0.0,change_rate=1.0,rotation_type='2D')

    elif fn_name == "random_shear":
        return pose_random_affine.objdet_random_shear(images, labels, factor=0.075, axis='xy')

    elif fn_name == "random_shear_x":
        return pose_random_affine.objdet_random_shear(images, labels, factor=0.075, axis='x')

    elif fn_name == "random_shear_y":
        return pose_random_affine.objdet_random_shear(images, labels, factor=0.075, axis='y')

    elif fn_name == "random_zoom":
        return pose_random_affine.objdet_random_zoom(images, labels, width_factor=0.3)

    elif fn_name == "random_rectangle_erasing":
        images_aug = random_erasing.random_rectangle_erasing(
                        images,
                        nrec=(0, 3),
                        area=(0.05, 0.2),
                        wh_ratio=(0.2, 1.5),
                        pixels_range=pixels_range)
        return images_aug, labels


def demo_data_augmentation(dataset_path, grayscale=None, num_images=None, output_dir=None):
    """
    Samples a batch of images with their groundtruth labels from
    the training set, applies to them the data augmentation functions
    specified in the YAML configuration file, and displays or saves side by side
    the original images and augmented images with their groundtruth
    bounding boxes.
    """

    # function_names = [
    #     "random_contrast", "random_brightness", "random_gamma", "random_hue",
    #     "random_saturation", "random_value", "random_hsv", "random_rgb_to_hsv",
    #     "random_rgb_to_grayscale", "random_sharpness", "random_posterize",
    #     "random_invert", "random_solarize", "random_autocontrast", "random_blur",
    #     "random_gaussian_noise", "random_crop", "random_flip", "random_translation",
    #     "random_rotation", "random_shear", "random_shear_x", "random_shear_y",
    #     "random_zoom", "random_rectangle_erasing"]

    function_names = ["random_flip","random_rotation"]

    color_only_functions = [
        "random_hue", "random_saturation", "random_value", "random_hsv",
        "random_rgb_to_hsv", "random_rgb_to_grayscale", "random_autocontrast"]

    # If grayscale was requested, remove the functions
    # that are only applicable to color images from
    # the list of function names.
    if grayscale:
        for fn in color_only_functions:
            function_names.remove(fn)

    scale = 1/127.5
    offset = -1
    pixels_range = (offset, scale * 255 + offset)

    # Create a configuration dictionary with the
    # information needed to create the data loader
    cfg = DefaultMunch.fromDict({
                "use_case" : "pose_estimation",
                "operation_mode": "training",
                "general" :{
                    "global_seed": 123
                },
                "model": {
                    "model_path": None,
                    "input_shape": (224, 224, 3),
                    "framework": "tf"
                },
                "dataset": {
                    "dataset_name" : "coco",
                    "keypoints": 17,
                    "training_path": dataset_path,
                    "validation_path": None,
                    "validation_split": 0.2,
                    "test_path": None,
                    "prediction_path": None,
                    "quantization_path": None,
                    "quantization_split": 0.2,
                    "seed": None
                },
                "preprocessing": {
                    "rescaling": { "scale": scale, "offset": offset  },
                    "resizing": { "interpolation": "bilinear", "aspect_ratio": "fit" },
                    "color_mode": "grayscale" if grayscale else "rgb",
                },
                "training": {
                    "batch_size": num_images,
                }
          })

    data_loader = get_dataloaders(cfg)

    # Create output directory if saving is enabled
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving augmentation demo images to: {output_dir}")
    else:
        print("Display mode: Images will be shown (use ctrl+c to exit)")

    print("Demonstrating data augmentation functions:")
    for i, fn in enumerate(function_names):
        print("  " + fn)

        for data in data_loader['train']:

            images, labels = data
            batch_size = tf.shape(images)[0]

            images_aug, labels_aug = augment_images(images, labels, fn_name=fn, pixels_range=pixels_range)

            # Map pixels values to the [0, 1] interval
            # to get correct displays in matplotlib
            images = remap_pixel_values_range(images, pixels_range, (0, 1))
            images_aug = remap_pixel_values_range(images_aug, pixels_range, (0, 1))

            # Plot or save the images and their groundtruth labels
            for k in range(batch_size):
                if output_dir:
                    # One folder per augmentation function
                    fn_dir = os.path.join(output_dir, fn)
                    filename = f"{fn}_img{k}.png"
                    plot_images_and_labels(images[k], labels[k], images_aug[k], labels_aug[k],
                                           grayscale=grayscale, legend=fn,
                                           save_dir=fn_dir, filename=filename)
                else:
                    plot_images_and_labels(images[k], labels[k], images_aug[k], labels_aug[k],
                                           grayscale=grayscale, legend=fn)

            # Stop when all the data augmentation functions have been demo'ed
            if i == len(function_names) - 1:
                exit()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='', required=True,
                        help='path to the dataset to sample')
    parser.add_argument('--grayscale', action='store_true',
                        help='demo data augmentation functions on grayscale images')
    parser.add_argument('--num_images', type=int, default=4,
                        help='number of images to display for each data augmentation function (default: 4)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='optional directory to save augmentation demo images (default: display only)')

    args = parser.parse_args()

    if not os.path.isdir(args.dataset_path):
        raise ValueError(f"\nCould not find dataset directory: {args.dataset_path}")

    demo_data_augmentation(args.dataset_path,
                           grayscale=args.grayscale,
                           num_images=args.num_images,
                           output_dir=args.output_dir)

if __name__ == '__main__':
    main()

