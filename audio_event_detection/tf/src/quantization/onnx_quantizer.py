#  *---------------------------------------------------------------------------------------------*/
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import pathlib
import tensorflow as tf
import os
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import onnxruntime

# Import utility functions and modules
from common.utils import tf_dataset_to_np_array
from common.quantization import quantize_onnx
from common.evaluation import model_is_quantized
from common.optimization import model_formatting_ptq_per_tensor
from common.onnx_utils import onnx_model_converter
from common.quantization import define_extra_options


# Define a class for ONNX Post-Training Quantization (PTQ)
class OnnxPTQQuantizer:
    """
    PTQ quantizer for ONNX models. Outputs QDQ-format quantized models

    Args:
        cfg (DictConfig): Configuration object for quantization.
        model (object): The model to quantize (TensorFlow or ONNX).
        dataloaders (dict): Dictionary containing datasets for quantization and testing.
        current_extra_options: ONNX quantizer extra options settings
    """
    def __init__(self, cfg: DictConfig = None, model: object = None, 
                 dataloaders: dict = None, current_extra_options: dict = None, q_mode: str = None):
        """
        Initialize ONNX PTQ quantizer.

        Parameters
        ----------
        cfg : DictConfig
            Quantization configuration.
        model : object
            Model to quantize (TensorFlow or ONNX). 
            TF models are converted to ONNX via TF2ONNX before quantization.
        dataloaders : dict
            Dataloaders containing `quantization_ds` for representative data.
        current_extra_options : dict, optional
            Extra options for ONNX quantizer.
        q_mode : str, optional
            Quantization mode; defaults to `'default'`.
        """
        self.cfg = cfg
        self.model = model
        self.dataloaders = dataloaders
        self.quantization_ds = dataloaders['quantization_ds']
        self.output_dir = HydraConfig.get().runtime.output_dir
        self.export_dir = cfg.quantization.export_dir
        self.quantized_model = None
        # For now only support default W8A8 QDQ quantization, change this 
        # if/when adding support for 4-bit quantization in the future
        self.q_mode = 'default'
        self.extra_options = current_extra_options
        self.target_opset = cfg.quantization.target_opset # Defaults to 17 in parse_config if not provided by user

        # Define extra options for ONNX quantization
        if self.extra_options is None:
            self.extra_options = define_extra_options(cfg=self.cfg)

        # Determine the quantization method based on the model type
        if isinstance(self.model, tf.keras.Model):
            self.q_method = self._quantize_keras_model
        elif isinstance(self.model, onnxruntime.InferenceSession):
            self.q_method = self._quantize_onnx_model
        else:
            raise ValueError(f"Unsupported model format: {type(self.model)}.")


    def _quantize_keras_model(self):
        """
        Converts a TF model to ONNX using tf2onnx and then quantizes it using ONNXruntime
        """
        # Optimize the model for per-tensor quantization if specified
        if self.cfg.quantization.granularity == 'per_tensor' and self.cfg.quantization.optimize:
            print("[INFO] : Optimizing the model for improved per_tensor quantization...")
            self.model = model_formatting_ptq_per_tensor(model_origin=self.model)
            models_dir = pathlib.Path(os.path.join(self.output_dir, f"{self.export_dir}/"))
            models_dir.mkdir(exist_ok=True, parents=True)
            model_path = models_dir / "optimized_model.keras"
            self.model.save(model_path)

        print("[INFO] : Starting ONNX PTQ quantization for Keras model.")
        # Convert the dataset to NumPy array
        if self.quantization_ds:
            data, _ = tf_dataset_to_np_array(self.quantization_ds, nchw=False)
        else:
            print(f'[INFO] : Quantizing by using fake dataset...')
            data = None

        # Convert the Keras model to ONNX format
        converted_model_path = os.path.join(self.output_dir, 'converted_model', 'converted_model.onnx')
        input_shape = self.model.inputs[0].shape
        print(f"[INFO] : Converting Keras model to ONNX at {converted_model_path} with input shape {input_shape}")
        onnx_model_converter(input_model_path=self.model.model_path,
                             target_opset=self.target_opset,
                             output_dir=converted_model_path,
                             static_input_shape=input_shape,
                             input_channels_last=True)

        # Perform ONNX quantization
        print(f"[INFO] : Running ONNX quantization on {converted_model_path}")
        self.quantized_model = quantize_onnx(
            quantization_samples=data,
            model_path=converted_model_path,
            configs=self.cfg,
            extra_options=self.extra_options
        )

    def _quantize_onnx_model(self):
        """
        Quantizes an ONNX model using ONNX quantization tools.
        """
        if self.cfg.quantization.quantizer.lower() == "onnx_quantizer" and self.cfg.quantization.quantization_type == "PTQ":
            print("[INFO] : Starting ONNX PTQ quantization for ONNX model.")
            if self.quantization_ds:
                data, _ = tf_dataset_to_np_array(self.quantization_ds, nchw=False)
            else:
                print(f'[INFO] : Quantizing by using fake dataset...')
                data = None

            # Ensure the ONNX model has a model_path attribute
            if getattr(self.model, 'model_path', None) is None:
                raise ValueError('ONNX InferenceSession must have a model_path attribute for quantization.')
            
            # Check if the model is already quantized
            if model_is_quantized(self.model.model_path):
                print('[INFO]: The input ONNX model is already quantized! Returning the same model!')
                self.quantized_model = self.model
            else:
                print(f"[INFO] : Running ONNX quantization on {self.model.model_path}")
                self.quantized_model = quantize_onnx(
                    quantization_samples=data,
                    model_path=self.model.model_path,
                    configs=self.cfg,
                    extra_options=self.extra_options
                )
        else:
            raise TypeError("Quantizer or quantization type not supported. "
                            "Check the `quantization` section of your user_config.yaml file!")

    def _run_quantization(self):
        """
        Wrapper to handle different quantization methods. 
        """

        # Code for support of 4-bit quantization or other quantization method to be added here

        self.q_method()


    def quantize(self):
        """
        Executes the full quantization process.

        Returns:
            onnxruntime.InferenceSession : Quantized ONNX model
        """
        print("[INFO] : Quantizing the model ... This might take few minutes ...")
        self._run_quantization()        # Run the quantization process
        print('[INFO] : Quantization complete.')
        return self.quantized_model     # Return the quantized model


