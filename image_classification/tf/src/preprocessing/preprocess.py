# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
import numpy as np
import tensorflow as tf


def _apply_rescaling(dataset: tf.data.Dataset = None, 
                      scale: float = None, 
                      offset: float = None) -> tf.data.Dataset:
    """
    Applies rescaling to a dataset using a tf.keras.Sequential model.

    Args:
        dataset (tf.data.Dataset): The dataset to be rescaled.
        scale (float): The scaling factor.
        offset (float): The offset factor.

    Returns:
        The rescaled dataset.
    """
    # Define the rescaling model
    rescaling = tf.keras.Sequential([
        tf.keras.layers.Rescaling(scale, offset)
    ])

    # Apply the rescaling to the dataset
    rescaled_dataset = dataset.map(lambda x, y: (rescaling(x), y))

    return rescaled_dataset


def _apply_normalization(dataset: tf.data.Dataset = None, 
                          mean: tuple[float] = None, 
                          std: tuple[float] = None) -> tf.data.Dataset:
    """
    Applies normalization to a dataset using a tf.keras.Sequential model.

    Args:
        dataset (tf.data.Dataset): The dataset to be rescaled.
        mean (float): The mean of the three channels.
        std (float): The variance of the three channels.

    Returns:
        The rescaled dataset.
    """
    # Define the rescaling model
    if isinstance(std, float):
        variance = pow(std, 2)
    elif isinstance(std, list):
        variance = [x ** 2 for x in std]
    else:
        variance = [x ** 2 for x in std]
    normalization = tf.keras.Sequential([
        tf.keras.layers.Normalization(mean=mean, variance=variance)
    ])

    # Apply the rescaling to the dataset
    normalized_dataset = dataset.map(lambda x, y: (normalization(x), y))

    return normalized_dataset


def apply_rescaling_on_image(image: tf.Tensor = None, 
                             scale: float = None, 
                             offset: float = None) -> tf.Tensor:
    """
    Applies rescaling to an image using a tf.keras.Sequential model.

    Args:
        image (tf.Tensor): The image to be rescaled.
        scale (float): The scaling factor.
        offset (float): The offset factor.

    Returns:
        The rescaled image.
    """
    # Define the rescaling model
    rescaling = tf.keras.Sequential([
        tf.keras.layers.Rescaling(scale, offset)
    ])

    # Apply the rescaling to the image
    rescaled_image = rescaling(image)

    return rescaled_image


def preprocessing(dataset: tf.data.Dataset = None, 
                  scale: float = None, 
                  offset: float = None,
                  mean: tuple[float] = None, 
                  std: tuple[float] = None) -> tf.data.Dataset:
    """
    Applies a full preprocessing to an image using a tf.keras.Sequential model.

    Args:
        image (tf.Tensor): The image to be rescaled.
        scale (float): The scaling factor.
        offset (float): The offset factor.
        mean (float): The mean of the three channels.
        variance (float): The variance of the three channels.

    Returns:
        The rescaled and normalized image.
    """

    if dataset != None:
        dataset = _apply_rescaling(dataset=dataset, 
                                    scale=scale,
                                    offset=offset)
    
        dataset = _apply_normalization(dataset=dataset, 
                                        mean=mean,
                                        std=std)
   
    return dataset
   
   
def preprocess_input(image: np.ndarray, input_details: dict) -> tf.Tensor:
    """
    Preprocesses an input image according to input details.

    Args:
        image: Input image as a NumPy array.
        input_details: Dictionary containing input details, including quantization and dtype.

    Returns:
        Preprocessed image as a TensorFlow tensor.

    """

    image = tf.image.resize(image, (input_details['shape'][1], input_details['shape'][2]))
    # Get the dimensions
    if input_details['dtype'] in [np.uint8, np.int8]:
        image_processed = (image / input_details['quantization'][0]) + input_details['quantization'][1]
        image_processed = np.clip(np.round(image_processed), np.iinfo(input_details['dtype']).min,
                                  np.iinfo(input_details['dtype']).max)
    else:
        image_processed = image
    image_processed = tf.cast(image_processed, dtype=input_details['dtype'])
    image_processed = tf.expand_dims(image_processed, 0)
    return image_processed


def postprocess_output(output: np.ndarray, output_details: dict) -> np.ndarray:
    """
    Postprocesses the model output to obtain the predicted label.

    Args:
        output (np.ndarray): The output tensor from the model.
        output_details: Dictionary containing output details, including quantization and dtype.

    Returns:
        np.ndarray: The predicted label.
    """
    if output_details['dtype'] in [np.uint8, np.int8]:
        # Convert the output data to float32 data type and perform the inverse quantization operation
        output_flp = (np.float32(output) - output_details['quantization'][1]) * output_details['quantization'][0]
    else:
        output_flp = output

    if output.shape[1] > 1:
        predicted_label = np.argmax(output_flp, axis=1)
    else:
        predicted_label = np.where(output_flp < 0.5, 0, 1)

    return predicted_label
