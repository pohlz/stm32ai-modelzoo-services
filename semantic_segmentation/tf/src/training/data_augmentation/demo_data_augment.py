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
from semantic_segmentation.tf.src.preprocessing import get_ds
from common.data_augmentation import random_color, random_misc, remap_pixel_values_range
from semantic_segmentation.tf.src.data_augmentation import segm_random_affine, segm_random_misc



def _plot_images_and_labels(image, label, image_aug, label_aug, grayscale=None, legend=None,
                            save_dir=None, filename=None):
    """
    Displays side by side the original and augmented image
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

    fig, axes = plt.subplots(2, 2, figsize=(x_size, y_size))
    (ax1, ax2), (ax3, ax4) = axes

    if grayscale:
        ax1.imshow(image, cmap='gray')
        ax1.title.set_text("Original")
        ax2.imshow(image_aug, cmap='gray')
    else:
        ax1.imshow(image)
        ax1.title.set_text("Original")
        ax2.imshow(image_aug)

    ax2.title.set_text(legend)
    ax3.imshow(label)
    ax4.imshow(label_aug)
    ax3.imshow(label)
    ax4.imshow(label_aug)

    plt.tight_layout()
    if save_dir and filename:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def _augment_images(images, labels, fn_name=None):

    if fn_name == "random_contrast":
        images_aug = random_color.random_contrast(images, factor=0.7)
        return images_aug, labels

    elif fn_name == "random_brightness":
        images_aug = random_color.random_brightness(images, factor=0.4)
        return images_aug, labels

    elif fn_name == "random_gamma":
        images_aug = random_color.random_gamma(images, gamma=(0.2, 2.0))
        return images_aug, labels

    elif fn_name == "random_hue":
        images_aug = random_color.random_hue(images, delta=0.1)
        return images_aug, labels

    elif fn_name == "random_saturation":
        images_aug = random_color.random_saturation(images, delta=0.2)
        return images_aug, labels

    elif fn_name == "random_value":
        images_aug = random_color.random_value(images, delta=0.2)
        return images_aug, labels

    elif fn_name == "random_hsv":
        images_aug = random_color.random_hsv(
                        images, hue_delta=0.1, saturation_delta=0.2, value_delta=0.2)
        return images_aug, labels

    elif fn_name == "random_rgb_to_hsv":
        images_aug = random_color.random_rgb_to_hsv(images, change_rate=1.0)
        return images_aug, labels

    elif fn_name == "random_rgb_to_grayscale":
        images_aug = random_color.random_rgb_to_grayscale(images, change_rate=1.0)
        return images_aug, labels

    elif fn_name == "random_sharpness":
        images_aug = random_color.random_sharpness(images, factor=(1.0, 4.0))
        return images_aug, labels

    elif fn_name == "random_posterize":
        images_aug = random_color.random_posterize(images, bits=(1, 8))
        return images_aug, labels

    elif fn_name == "random_invert":
        images_aug = random_color.random_invert(images, change_rate=1.0)
        return images_aug, labels

    elif fn_name == "random_solarize":
        images_aug = random_color.random_solarize(images, change_rate=1.0)
        return images_aug, labels

    elif fn_name == "random_autocontrast":
        images_aug = random_color.random_autocontrast(images, change_rate=1.0)
        return images_aug, labels

    elif fn_name == "random_blur":
        images_aug = random_misc.random_blur(images, filter_size=(2, 4))
        return images_aug, labels

    elif fn_name == "random_gaussian_noise":
        images_aug = random_misc.random_gaussian_noise(images, stddev=(0.02, 0.1))
        return images_aug, labels

    elif fn_name == "random_crop":
        return segm_random_misc.segm_random_crop(images, labels, change_rate=1.0)

    elif fn_name == "random_flip":
        return segm_random_affine.segm_random_flip(images, labels, mode="horizontal_and_vertical", change_rate=1.0)

    elif fn_name == "random_translation":
        return segm_random_affine.segm_random_translation(images, labels, width_factor=0.2, height_factor=0.2)

    elif fn_name == "random_rotation":
        return segm_random_affine.segm_random_rotation(images, labels, factor=0.075)

    elif fn_name == "random_shear":
        return segm_random_affine.segm_random_shear(images, labels, factor=0.075, axis='xy')

    elif fn_name == "random_shear_x":
        return segm_random_affine.segm_random_shear(images, labels, factor=0.075, axis='x')

    elif fn_name == "random_shear_y":
        return segm_random_affine.segm_random_shear(images, labels, factor=0.075, axis='y')

    elif fn_name == "random_zoom":
        return segm_random_affine.segm_random_zoom(images, labels, width_factor=0.3)


def _demo_data_augmentation(images_path, masks_path, grayscale=False, num_images=4, output_dir=None):
    """
    Samples a batch of images with their groundtruth labels from
    the training set, applies to them the data augmentation functions
    specified in the YAML configuration file, and displays side by side
    the original images and augmented images with their groundtruth
    bounding boxes.
    """

    function_names = [
        "random_contrast", "random_brightness", "random_gamma", "random_hue",
        "random_saturation", "random_value", "random_hsv", "random_rgb_to_hsv",
        "random_rgb_to_grayscale", "random_sharpness", "random_posterize",
        "random_invert", "random_solarize", "random_autocontrast", "random_blur",
        "random_gaussian_noise", "random_crop", "random_flip", "random_translation",
        "random_rotation", "random_shear", "random_shear_x", "random_shear_y",
        "random_zoom"
    ]

    color_only_functions = [
        "random_hue", "random_saturation", "random_value", "random_hsv",
        "random_rgb_to_hsv", "random_rgb_to_grayscale", "random_autocontrast"
    ]

    # If grayscale was requested, remove the functions
    # that are only applicable to color images from
    # the list of function names.
    if grayscale:
        for fn in color_only_functions:
            function_names.remove(fn)

    # Create a temporary files_path with list of image IDs (filenames without extension)
    import tempfile
    temp_dir = tempfile.mkdtemp()
    files_path = os.path.join(temp_dir, "ids.txt")

    # Get list of image files from the images directory
    image_files = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    # Extract image IDs (filenames without extension)
    image_ids = [os.path.splitext(f)[0] for f in image_files]

    # Write image IDs to the temporary file
    with open(files_path, 'w') as f:
        for img_id in image_ids[:num_images]:  # Limit to num_images
            f.write(img_id + '\n')

    # Create a configuration dictionary with the
    # information needed to create the data loader
    scale = 1./255
    offset = 0
    cfg = DefaultMunch.fromDict({
                "use_case" : "semantic_segmentation",
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
                    "dataset_name": "person_coco_2017_pascal_voc_2012",
                    "class_names": ["background", "person"],
                    "training_path": images_path,
                    "training_masks_path": masks_path,
                    "training_files_path": files_path,
                    "validation_path": None,
                    "validation_masks_path": None,
                    "validation_files_path": None,
                    "validation_split": 0.0,
                    "test_path": None,
                    "test_masks_path": None,
                    "test_files_path": None,
                    "prediction_path": None,
                    "prediction_files_path": None,
                    "quantization_path": None,
                    "quantization_files_path": None,
                    "quantization_split": 0.0,
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

    # Create the data loader
    data_loader= get_dataloaders(cfg)

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

            images_aug, labels_aug = _augment_images(images, labels, fn_name=fn)

            # Plot the images and labels
            for k in range(batch_size):
                if output_dir:
                    # One folder per augmentation function
                    fn_dir = os.path.join(output_dir, fn)
                    filename = f"{fn}_img{k}.png"
                    _plot_images_and_labels(images[k], labels[k], images_aug[k], labels_aug[k],
                                        grayscale=grayscale, legend=fn,
                                        save_dir=fn_dir, filename=filename)
                else:
                    _plot_images_and_labels(images[k], labels[k], images_aug[k], labels_aug[k],
                                        grayscale=grayscale, legend=fn)

            # Stop when all the data augmentation functions have been demo'ed
            if i == len(function_names) - 1:
                exit()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, default='', required=True,
                        help='images directory path (in .jpg format)')
    parser.add_argument('--masks', type=str, default='', required=True,
                        help='masks directory path (in .png format)')
    parser.add_argument('--grayscale', action='store_true',
                        help='demo data augmentation functions on grayscale images')
    parser.add_argument('--num_images', type=int, default=4,
                        help='number of images to display for each data augmentation function (default: 4)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='optional directory to save augmentation demo images (default: display only)')

    args = parser.parse_args()

    if not os.path.isdir(args.images):
        raise ValueError(f"\nCould not find images directory: {args.images}")

    if not os.path.isdir(args.masks):
        raise ValueError(f"\nCould not find masks directory: {args.masks}")

    _demo_data_augmentation(args.images, args.masks,
                           grayscale=args.grayscale, num_images=args.num_images,
                           output_dir=args.output_dir)

if __name__ == '__main__':
    main()

