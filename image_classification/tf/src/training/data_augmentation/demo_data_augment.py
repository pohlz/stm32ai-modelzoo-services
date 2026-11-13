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
from common.data_augmentation import random_color, random_affine, random_erasing, random_misc


def display_images_side_by_side(image,
                                image_aug,
                                grayscale=None,
                                legend=None,
                                save_dir=None,
                                filename=None):
    """
    Displays (or saves) the original and augmented images side by side.

        Args:
            image (np.ndarray or tf.Tensor): original image.
            image_aug (np.ndarray or tf.Tensor): augmented image.
            grayscale (bool): if True, display in grayscale.
            legend (str): title for the augmented image.
            save_dir (str): directory to save the figure. If None, plt.show() is used.
            filename (str): filename to use when saving. Required if save_dir is not None.
    """

    # Ensure NumPy arrays (for tf.Tensor inputs)
    image = np.array(image)
    image_aug = np.array(image_aug)

    # Clip values to [0, 1] range for proper visualization
    image = np.clip(image, 0, 1)
    image_aug = np.clip(image_aug, 0, 1)

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
        ax1.imshow(image, cmap='gray', vmin=0, vmax=1)
    else:
        ax1.imshow(image, vmin=0, vmax=1)
    ax1.title.set_text("original")
    ax1.axis("off")

    if grayscale:
        ax2.imshow(image_aug, cmap='gray', vmin=0, vmax=1)
    else:
        ax2.imshow(image_aug, vmin=0, vmax=1)
    ax2.title.set_text(legend)
    ax2.axis("off")

    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        if filename is None:
            # Fallback name if not provided
            filename = "comparison.png"
        out_path = os.path.join(save_dir, filename)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def augment_images(images, fn_name=None):
    """
        Returns the augmented images according to the data augmentation fucntion specified in 'fn_name'

        Args:
            images (tf.Tensor): images on which some augmentation is to be applied
            fn_name (str): augmentation function name such as "random_brightness", "random_saturation"...

        Returns:
            (tf.Tensor): the augmented images

    """

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


def demo_data_augmentation(dataset_path, dataset_name, grayscale=None, num_images=None, output_dir=None):
    """
        Samples a batch of images, applies to them the data augmentation
        functions specified in the YAML configuration file, and displays or saves
        side by side the original images and augmented images.

        Args:
            dataset_path (str): path to the images on which we want to evaluate augmentation
            grayscale (boolean): if grayscale is True then the color_mode is "grayscale" else "rgb"
            num_images (int): size of the considered batch of images
            output_dir (str): directory to save the images. If None, images are displayed.

        Returns:
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

    if grayscale:
        for fn in color_only_functions:
            function_names.remove(fn)

    # Get the class names
    class_names = [p for p in os.listdir(dataset_path)
                      if os.path.isdir(os.path.join(dataset_path, p))]

    # Create output directory if saving is enabled
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving augmentation demo images to: {output_dir}")
    else:
        print("Display mode: Images will be shown (use ctrl+c to exit)")

    # Create a configuration dictionary with the information needed to create the data loader
    scale = 1./255
    offset = 0
    cfg = DefaultMunch.fromDict({
                "use_case": "image_classification",
                "model": { "model_path": None,
                           "input_shape": (224, 224, 3),
                           "framework": "tf" },
                "operation_mode": "training",
                "dataset": {
                    "dataset_name": dataset_name,
                    "training_path": dataset_path,
                    "class_names": class_names,
                    "seed": None
                },
                "preprocessing": {
                    "rescaling": { "scale": scale, "offset": offset  },
                    "resizing": { "interpolation": "bilinear", "aspect_ratio": "fit" },
                    "color_mode": "grayscale" if grayscale else "rgb",
                },
                "training": {
                    "batch_size": num_images,
                    "resizing": { "interpolation": "bilinear", "aspect_ratio": "fit" },
                    "color_mode": "grayscale" if grayscale else "rgb",
                }
          })

    # Create a data loader to get examples from the training set
    print("Demo data augmentation on dataset:", cfg.dataset.training_path)
    data_loader = get_dataloaders(cfg)

    print("Demonstrating data augmentation functions:")
    for i, fn in enumerate(function_names):
        print("  " + fn)

        for data in data_loader['train']:
            images, _ = data
            batch_size = tf.shape(images)[0]

            # Convert to float32 if needed (dataloader already normalizes to [0, 1])
            images = tf.cast(images, dtype=tf.float32)

            images_aug = augment_images(images, fn_name=fn)

            # Plot/save the images and their groundtruth labels
            for k in range(batch_size):
                if output_dir:
                    # One folder per augmentation function
                    fn_dir = os.path.join(output_dir, fn)
                    filename = f"{fn}_img{k}.png"
                    display_images_side_by_side(
                        images[k],
                        images_aug[k],
                        grayscale=grayscale,
                        legend=fn,
                        save_dir=fn_dir,
                        filename=filename
                    )
                else:
                    display_images_side_by_side(
                        images[k],
                        images_aug[k],
                        grayscale=grayscale,
                        legend=fn
                    )

            # Stop when all the data augmentation functions have been demo'ed
            if i == len(function_names) - 1:
                exit()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='', required=True,
                        help='path to the dataset to sample')
    parser.add_argument('--dataset_name', type=str, default='', required=True,
                        help='Name of the dataset to sample')
    parser.add_argument('--grayscale', action='store_true',
                        help='demo data augmentation functions on grayscale images')
    parser.add_argument('--num_images', type=int, default=4,
                        help='number of images to display for each data augmentation function (default: 4)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save the augmentation demo images. If not provided, images will be displayed.')

    args = parser.parse_args()

    if not os.path.isdir(args.dataset_path):
        raise ValueError(f"\nCould not find dataset directory: {args.dataset_path}")

    demo_data_augmentation(args.dataset_path, args.dataset_name, grayscale=args.grayscale, num_images=args.num_images, output_dir=args.output_dir)

if __name__ == '__main__':
    main()