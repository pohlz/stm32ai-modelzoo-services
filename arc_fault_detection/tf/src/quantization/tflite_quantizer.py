# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
from pathlib import Path
import numpy as np
import tensorflow as tf
import tqdm
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from typing import Optional

from common.optimization import model_formatting_ptq_per_tensor


def _representative_data_gen(configs: DictConfig, quantization_ds: Optional[tf.data.Dataset] = None, quantization_split: float = 1.0):
    """
    Generates representative data samples for post-training quantization.
    This generator yields input data samples, either randomly generated or from a provided dataset,
    to be used during quantization calibration.
    Args:
        configs (DictConfig): Configuration object containing model parameters, including input shape.
        quantization_ds (Optional[tf.data.Dataset]): Dataset to use for representative data. If None, random data is generated.
        quantization_split (float): Fraction of the dataset to use for quantization. If 1.0, uses the entire dataset.
            If 0.0, uses dummy random data.
        np.ndarray: A numpy array representing a single input sample, shaped according to the model's input.
    """
    if quantization_ds is None:
        print("[INFO] : Quantizing using dummy data")
        for _ in tqdm.tqdm(range(5)):
            data = np.random.rand(1, *configs.model.input_shape)
            yield [data.astype(np.float32)]
    else:
        print("[INFO] : Quantizing by using the provided dataset, this will take a while.")
        print(f"[INFO] : Using {len(quantization_ds)} patches")
        for patches, _ in tqdm.tqdm(quantization_ds, total=len(quantization_ds)):
            for patch in patches:
                yield [tf.cast(patch[np.newaxis, ...], tf.float32)]


def _tflite_ptq_quantizer(configs: DictConfig = None, model: tf.keras.Model = None,
                            quantization_ds: tf.data.Dataset = None) -> tf.lite.Interpreter:
    """
    Perform post-training quantization on a TensorFlow Lite model.

    Args:
        configs (DictConfig): Configuration dictionary containing quantization and model settings.
        model (tf.keras.Model): The TensorFlow model to be quantized.
        quantization_ds (tf.data.Dataset): The quantization dataset if it's provided by the user else the training dataset. Defaults to None
    Returns:
        tf.lite.Interpreter: The quantized TFLite model as an Interpreter object.
    """
    # Create the output directory
    output_dir = HydraConfig.get().runtime.output_dir
    tflite_models_dir = Path(output_dir) / configs.quantization.export_dir
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    tflite_model_quantized_file = tflite_models_dir / "quantized_model.tflite"

    # Create the TFLite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Set the quantization types for the input and output
    if configs.quantization_input_type == 'int8':
        converter.inference_input_type = tf.int8
    elif configs.quantization_input_type == 'uint8':
        converter.inference_input_type = tf.uint8

    if configs.quantization_output_type == 'int8':
        converter.inference_output_type = tf.int8
    elif configs.quantization_output_type == 'uint8':
        converter.inference_output_type = tf.uint8

    # Set the optimizations and representative dataset generator
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantization_split = configs.dataset.quantization_split if configs.dataset.quantization_split is not None else 1.0
    converter.representative_dataset = lambda: _representative_data_gen(configs=configs, quantization_ds=quantization_ds,
                                                               quantization_split=quantization_split)
    
    if configs.quantization_granularity == 'per_tensor':
        converter._experimental_disable_per_channel = True

    # Convert the model to a quantized TFLite model
    tflite_model_quantized = converter.convert()
    tflite_model_quantized_file.write_bytes(tflite_model_quantized)

    # Return the quantized TFLite interpreter
    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_quantized_file))
    interpreter.allocate_tensors()
    setattr(interpreter, 'model_path', str(tflite_model_quantized_file))
    print(f"[INFO] : Quantized model saved at {tflite_model_quantized_file}")
    return interpreter


class TFLitePTQQuantizer:
    """
    PTQ quantizer for TFLite models.

    Args:
        cfg (DictConfig): Configuration object for quantization.
        model (tf.keras.Model): The TensorFlow model to quantize.
        dataloaders (dict): Dictionary containing datasets for quantization and testing.
    """
    def __init__(self, cfg: DictConfig = None, model: object = None,
                 dataloaders: dict = None):
        """
        Initialize the quantizer with configuration, model, and dataloaders.

        Args:
            cfg (DictConfig): Quantization configuration.
            model (object): Model instance to quantize.
            dataloaders (dict): Datasets for quantization and validation.
        """
        self.cfg = cfg
        self.model = model
        self.dataloaders = dataloaders or {}
        self.quantized_model = None
        self.output_dir = HydraConfig.get().runtime.output_dir

    def quantize(self):
        """
        Executes the TFLite PTQ quantization process.

        Returns:
            tf.lite.Interpreter : Quantized TFLite model
        """
        model_path = self.model.model_path
        output_dir = HydraConfig.get().runtime.output_dir
        file_extension = Path(model_path).suffix
        # check if the batch dimension is included in the input shape and remove it if present
        if len(self.cfg.model.input_shape) == 4:
            setattr(self.cfg.model, 'input_shape', self.cfg.model.input_shape[1:])

        quantization_ds = self.dataloaders['quantization']

        if self.cfg.quantization.quantizer == "TFlite_converter" and self.cfg.quantization.quantization_type == "PTQ":
            if file_extension not in [".h5", ".keras"]:
                raise ValueError("For TFLite quantization, the model format must be either .h5 or .keras.")
            float_model = tf.keras.models.load_model(model_path)
            # if per-tensor quantization is required some optimizations are possible on the float model
            if self.cfg.quantization.granularity == 'per_tensor' and self.cfg.quantization.optimize:
                print("[INFO] : Optimizing the model for improved per_tensor quantization...")
                float_model = model_formatting_ptq_per_tensor(model_origin=float_model)
                optimized_model_path = os.path.join(output_dir, self.cfg.quantization.export_dir, "optimized_model.keras")
                float_model.save(optimized_model_path)

            # Quantize the model
            self.quantized_model = _tflite_ptq_quantizer(configs=self.cfg, model=float_model, quantization_ds=quantization_ds)
            return self.quantized_model
        else:
            raise NotImplementedError("Quantizer and quantization type not supported yet!")
