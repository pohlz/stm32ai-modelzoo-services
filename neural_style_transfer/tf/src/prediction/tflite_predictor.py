# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

# Import necessary libraries
import os
import sys
from pathlib import Path
from omegaconf import DictConfig
from tabulate import tabulate
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import cv2

# Suppress warnings and TensorFlow logs for cleaner output
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import utility functions for AI runner and ONNX prediction
from common.utils import ai_runner_interp, ai_interp_input_quant, ai_interp_outputs_dequant
from neural_style_transfer.tf.src.utils import ai_runner_invoke
from neural_style_transfer.tf.src.preprocessing import preprocess_input, preprocess_image
from neural_style_transfer.tf.src.postprocessing import postprocess_and_save


class TFLiteQuantizedModelPredictor:
    """
    A class to handle predictions using a TFLite quantized model. This class includes methods for:
    - Loading and preprocessing images
    - Running inference on the TFLite model
    - Annotating and saving prediction results
    - Displaying results in a tabular format
    """
    def __init__(self, cfg, model, dataloaders):
        """
        Initialize the predictor with configuration, model, and dataloaders.

        Args:
            cfg: Configuration object containing settings for the predictor.
            model: The TFLite model to use for predictions.
            dataloaders: A dictionary containing the prediction dataset.
        """
        self.cfg = cfg
        self.model = model
        self.predict_ds = dataloaders['predict']
        self.prediction_result_dir = os.path.join(cfg.output_dir, 'predictions')
        os.makedirs(self.prediction_result_dir, exist_ok=True)
        self.results_table = []
        self.target = getattr(cfg.prediction, 'target', 'host') if hasattr(cfg, 'prediction') else 'host'
        self.model_name = os.path.basename(model.model_path)
        self.display_figures = cfg.general.display_figures

        # Initialize the TFLite interpreter for the quantized model
        self.interpreter_quant = tf.lite.Interpreter(model_path=model.model_path)
        self.interpreter_quant.allocate_tensors()
        self.input_details = self.interpreter_quant.get_input_details()[0]
        self.input_index_quant = self.input_details["index"]
        self.output_details = self.interpreter_quant.get_output_details()[0]
        self.output_index_quant = self.output_details["index"]
        self.height, self.width, _ = self.input_details['shape_signature'][1:]

        # Initialize the preprocessing parameters
        cpp = cfg.preprocessing
        self.channels = 1 if cpp.color_mode == "grayscale" else 3
        self.aspect_ratio = cpp.resizing.aspect_ratio
        self.interpolation = cpp.resizing.interpolation
        self.scale = cpp.rescaling.scale
        self.offset = cpp.rescaling.offset

        # Initialize the postprocessing parameters
        self.diameter = cfg.postprocessing.bilateral_diameter
        self.sigma_color = cfg.postprocessing.sigma_color
        self.sigma_space = cfg.postprocessing.sigma_space
        self.alpha = cfg.postprocessing.scaling_alpha
        self.beta  = cfg.postprocessing.scaling_beta
        self.saturation_scale = cfg.postprocessing.saturation_scale
        self.prediction_result_dir = os.path.join(cfg.output_dir, 'predictions')

        # Initialize the AI runner interpreter for edge devices
        self.ai_runner_interpreter = ai_runner_interp(self.target, self.model_name)

    def _load_image(self, img_path):
        """
        Load an image from the given path and convert it to RGB format.

        Args:
            img_path: Path to the image file.

        Returns:
            The loaded image in RGB format, or None if the image could not be loaded.
        """
        image = cv2.imread(img_path)
        if image is None:
            print(f"[ERROR] : Could not load image {img_path}")
            return None
        if len(image.shape) != 3:   # If the image is grayscale, convert it to BGR
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image_rgb

    def _get_stylized_img(self, preproc_img, raw_image):
        """
        Perform inference on the input image and get stylized image.

        Args:
            preproc_img: The preprocessed input image for stedgeai inference.
            raw_image: The raw input image for host inference.

        Returns:
            The prediction scores.
        """
        if self.target == 'host':
            img_process = preprocess_image(raw_image, self.height, self.width, self.aspect_ratio, self.interpolation, self.scale, self.offset)
            input_tensor = preprocess_input(img_process, self.input_details)
            self.interpreter_quant.set_tensor(self.input_index_quant, input_tensor)
            self.interpreter_quant.invoke()
            output_tensor = self.interpreter_quant.get_tensor(self.output_index_quant)

        elif self.target in ['stedgeai_host', 'stedgeai_n6', 'stedgeai_h7p']:
            img = preproc_img
            imagee = ai_interp_input_quant(self.ai_runner_interpreter, img, '.tflite')
            output_tensor = ai_runner_invoke(imagee, self.ai_runner_interpreter)[0][0]            

        else:
            raise ValueError(f"Unknown target: {self.target}")
        return output_tensor   # Remove single-dimensional entries

    def predict(self):
        """
        Perform predictions on the dataset and save the results.
        """
        for img, img_path in self.predict_ds:   # Iterate over the prediction dataset
            image_path = img_path.numpy()[0].decode()
            file_name = os.path.basename(image_path)
            image_rgb = self._load_image(image_path)
            if image_rgb is None:
                continue
            img = img.numpy()
            output_tensor = self._get_stylized_img(preproc_img=img, raw_image=image_rgb)

            postprocess_and_save(output_tensor, self.diameter, self.sigma_color, self.sigma_space,
                self.alpha, self.beta, self.saturation_scale, self.prediction_result_dir, file_name)

        print('[INFO] : Prediction complete.')