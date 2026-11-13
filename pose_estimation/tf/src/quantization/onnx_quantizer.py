# ---------------------------------------------------------------------------------------------
# Copyright (c) 2025 STMicroelectronics.
# All rights reserved.
#
# This software is licensed under terms that can be found in the LICENSE file in
# the root directory of this software component.
# If no LICENSE file comes with this software, it is provided AS-IS.
# ---------------------------------------------------------------------------------------------

import os
import onnxruntime
import tensorflow as tf
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from semantic_segmentation.tf.src.utils import tf_segmentation_dataset_to_np_array
from common.quantization import quantize_onnx, define_extra_options
from common.evaluation import model_is_quantized
from common.onnx_utils import onnx_model_converter
from common.utils import tf_dataset_to_np_array

class OnnxPTQQuantizer:
    """
    A class to handle ONNX Post-Training Quantization (PTQ).

    Args:
        cfg (DictConfig): Configuration object for quantization.
        model (tf.keras.Model): The TensorFlow model to quantize.
        dataloaders (dict): Dictionary containing datasets for quantization and testing.
        extra_options (dict): ONNX quantizer extra options settings
    """
    def __init__(self, cfg: DictConfig = None, model: object = None, 
                 dataloaders: dict = None, extra_options: dict = None):
        self.cfg = cfg
        self.model = model
        self.quantization_ds = dataloaders['quantization']
        self.extra_options = extra_options

    def quantize_keras_model(self):
        output_dir = HydraConfig.get().runtime.output_dir
        static_input_shape = self.model.inputs[0].shape
        target_opset = self.cfg.quantization.target_opset
        converted_model_path = os.path.join(output_dir, 'converted_model', 'converted_model.onnx')
        print(f"[INFO] : Converting Keras model to ONNX with static input shape {static_input_shape}")
        onnx_model_converter(
            input_model_path=self.model.model_path,
            target_opset=target_opset,
            output_dir=converted_model_path,
            static_input_shape=static_input_shape
        )
        self.extra_options = define_extra_options(self.cfg)
        result = self.quantize_onnx_model(converted_model_path)
        return result

    def quantize_onnx_model(self, model_path=None):
        if model_path is None:
            model_path = getattr(self.model, 'model_path', None)
        if model_is_quantized(model_path):
            print('[INFO] : The input model is already quantized! Returning the same model!')
            return self.model
        if self.quantization_ds:
            quant_split = self.cfg.dataset.quantization_split if self.cfg.dataset.quantization_split else 1.0
            print(f'[INFO] : Quantizing by using {quant_split * 100} % of the provided dataset...')
            data, _ = tf_dataset_to_np_array(self.quantization_ds, nchw=True)
        else:
            print(f'[INFO] : Quantizing by using fake dataset...')
            data = None
        self.extra_options = define_extra_options(self.cfg)
        quantized_model = quantize_onnx(
            configs=self.cfg,
            model_path=model_path,
            quantization_samples=data,
            extra_options=self.extra_options
        )
        print('[INFO] : Quantization complete.')
        return quantized_model

    def quantize(self):
        if isinstance(self.model, tf.keras.Model):
            return self.quantize_keras_model()
        elif isinstance(self.model, onnxruntime.InferenceSession):
            return self.quantize_onnx_model()
        else:
            raise ValueError("Model must be tf.keras.Model or onnxruntime.InferenceSession")
