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

from common.data_augmentation import random_color, random_affine, random_erasing, random_misc
from re_identification.tf.src.preprocessing import preprocess



def display_images_side_by_side(image, image_aug, grayscale=None, legend=None, save_dir=None, filename=None):
    """
    This function displays (or saves) the original and augmented images side by side.

    Args:
        images (Tuple): original images.
        images_aug (Tuple): corresponding augmented images.
        grayscale (bool): if True, display in grayscale.
        legend (str): title for the augmented image.
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

    if grayscale:
        ax1.imshow(image, cmap='gray')
    else:
        ax1.imshow(image)
    ax1.title.set_text("original")

    if grayscale:
        ax2.imshow(image_aug, cmap='gray')
    else:
        ax2.imshow(image_aug)
    ax2.title.set_text(legend)

    plt.tight_layout()
    if save_dir and filename:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def augment_images(images, fn_name=None):

    if fn_name == "random_contrast":
        return random_color.random_contrast(images, factor=0.7)

    elif fn_name == "random_brightness":
        return random_color.random_brightness(images, factor=0.4)

    elif fn_name == "random_gamma":
        return random_color.random_gamma(images, gamma=(0.2, 2.0))

    elif fn_name == "random_hue":
        return random_color.random_hue(images, delta=0.1)

    elif fn_name == "random_saturation":
        return random_color.random_saturation(images, delta=0.2)

    elif fn_name == "random_value":
        return random_color.random_value(images, delta=0.2)

    elif fn_name == "random_hsv":
        return random_color.random_hsv(
                        images, hue_delta=0.1, saturation_delta=0.2, value_delta=0.2)

    elif fn_name == "random_rgb_to_hsv":
        return random_color.random_rgb_to_hsv(images, change_rate=1.0)

    elif fn_name == "random_rgb_to_grayscale":
        return random_color.random_rgb_to_grayscale(images, change_rate=1.0)

    elif fn_name == "random_sharpness":
        return random_color.random_sharpness(images, factor=(1.0, 4.0))

    elif fn_name == "random_posterize":
        return random_color.random_posterize(images, bits=(1, 8))

    elif fn_name == "random_invert":
        return random_color.random_invert(images, change_rate=1.0)

    elif fn_name == "random_solarize":
        return random_color.random_solarize(images, change_rate=1.0)

    elif fn_name == "random_autocontrast":
        return random_color.random_autocontrast(images, change_rate=1.0)

    elif fn_name == "random_blur":
        return random_misc.random_blur(images, filter_size=(2, 4))

    elif fn_name == "random_gaussian_noise":
        return random_misc.random_gaussian_noise(images, stddev=(0.02, 0.1))

    elif fn_name == "random_jpeg_quality":
        return random_misc.random_jpeg_quality(images, jpeg_quality=(20, 100))

    elif fn_name == "random_crop":
        return random_misc.random_crop(images, change_rate=1.0)

    elif fn_name == "random_flip":
        return random_affine.random_flip(images, mode="horizontal_and_vertical", change_rate=1.0)

    elif fn_name == "random_translation":
        return random_affine.random_translation(images, width_factor=0.2, height_factor=0.2)

    elif fn_name == "random_rotation":
        return random_affine.random_rotation(images, factor=0.075)

    elif fn_name == "random_shear":
        return random_affine.random_shear(images, factor=0.075, axis='xy')

    elif fn_name == "random_shear_x":
        return random_affine.random_shear(images, factor=0.075, axis='x')

    elif fn_name == "random_shear_y":
        return random_affine.random_shear(images, factor=0.075, axis='y')

    elif fn_name == "random_zoom":
        return random_affine.random_zoom(images, width_factor=0.3)

    elif fn_name == "random_rectangle_erasing":
        return random_erasing.random_rectangle_erasing(images, nrec=(0, 4))

    elif fn_name == "random_bounded_crop":
        return random_affine.random_bounded_crop(images, width_factor=[-0.8,0.0])


def demo_data_augmentation(dataset_path, grayscale=None, num_images=None, output_dir=None):
    """
    Samples a batch of images, applies to them the data augmentation
    functions specified in the YAML configuration file, and displays
    side by side the original images and augmented images.
    """

    function_names = [
        "random_contrast", "random_brightness", "random_gamma", "random_hue",
        "random_saturation", "random_value", "random_hsv", "random_rgb_to_hsv",
        "random_rgb_to_grayscale", "random_sharpness", "random_posterize",
        "random_invert", "random_solarize", "random_autocontrast", "random_blur",
        "random_gaussian_noise", "random_jpeg_quality", "random_crop", "random_flip",
        "random_translation", "random_rotation", "random_shear", "random_shear_x",
        "random_shear_y", "random_zoom", "random_rectangle_erasing", "random_bounded_crop"
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

    # Get the class names
    class_names = [p for p in os.listdir(dataset_path)
                      if os.path.isdir(os.path.join(dataset_path, p))]

    # If the dataset is flat (no subdirectories), extract class IDs from image filenames
    if not class_names:
        # For flat reid datasets, extract unique class IDs from filenames (format: id_camera_frame.jpg)
        image_files = [f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        class_ids = set()
        for img_file in image_files:
            # Extract the first part (class ID) from filename like "00008_01_16.jpeg"
            class_id = img_file.split('_')[0]
            class_ids.add(class_id)
        class_names = sorted(list(class_ids))

    if not class_names:
        raise ValueError(f"No classes found in dataset at {dataset_path}")

    # Create a configuration dictionary with the
    # information needed to create the data loader
    scale = 1./255
    offset = 0
    cfg = DefaultMunch.fromDict({
                "use_case": "re_identification",
                "model": { "model_path": None,
                           "input_shape": (224, 224, 3),
                           "framework": "tf" },
                "operation_mode": "training",
                "dataset": {
                    "name": "DeepSportradar",
                    "training_path": dataset_path,
                    "validation_path": None,
                    "quantization_path": None,
                    "test_query_path": None,
                    "test_gallery_path": None,
                    "class_names": class_names,
                    "class_names_test": [],
                    "validation_split": 0.2,
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
    cfg.use_case = "re_identification"
    cfg.model.framework = "tf"

    # Create a data loader to get examples from the training set
    print("Dataset:", cfg.dataset.training_path)
    data_loader, _, _, _, _ = preprocess(cfg)

    # Create output directory if saving is enabled
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving augmentation demo images to: {output_dir}")
    else:
        print("Display mode: Images will be shown (use ctrl+c to exit)")

    print("Demonstrating data augmentation functions:")
    for i, fn in enumerate(function_names):
        print("  " + fn)

        for data in data_loader:
            images = data[0]
            images = scale * tf.cast(images, dtype=tf.float32) + offset

            images_aug = augment_images(images, fn_name=fn)

            # Plot or save the images and their groundtruth labels
            for k in range(cfg.training.batch_size):
                if output_dir:
                    # One folder per augmentation function
                    fn_dir = os.path.join(output_dir, fn)
                    filename = f"{fn}_img{k}.png"
                    display_images_side_by_side(images[k], images_aug[k], grayscale=grayscale, legend=fn,
                                               save_dir=fn_dir, filename=filename)
                else:
                    display_images_side_by_side(images[k], images_aug[k], grayscale=grayscale, legend=fn)

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

    demo_data_augmentation(args.dataset_path, grayscale=args.grayscale, num_images=args.num_images,
                          output_dir=args.output_dir)

if __name__ == '__main__':
    main()
