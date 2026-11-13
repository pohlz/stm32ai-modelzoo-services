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
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader
import onnx
import onnxruntime

from common.evaluation import model_is_quantized
from common.onnx_utils import onnx_model_converter


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
            yield data.astype(np.float32)
    else:
        print("[INFO] : Quantizing by using the provided dataset , this will take a while.")
        print(f"[INFO] : Using {len(quantization_ds)} patches")
        for patches, _ in tqdm.tqdm(quantization_ds, total=len(quantization_ds)):
            for patch in patches:
                yield tf.cast(patch[np.newaxis, ...], tf.float32).numpy()


def quantize_onnx(configs: DictConfig, model_path: str = None, quantization_ds=None):
    """
    Quantizes an ONNX model using onnx-runtime.

    Args:
        configs (DictConfig): Configuration dictionary containing quantization and model settings.
        model_path (str, optional): Path to the ONNX model file.
        quantization_ds: Calibration/representative dataset as a numpy array (optional).

    Returns:
        onnxruntime.InferenceSession: Quantized model session with model_path attribute.
    """

    # Create the output directory (like in your TFLite logic)
    output_dir = HydraConfig.get().runtime.output_dir
    onnx_models_dir = Path(output_dir) / configs.quantization.export_dir
    onnx_models_dir.mkdir(exist_ok=True, parents=True)
    quantized_model_path = onnx_models_dir / "quantized_model.onnx"

    # Define a CalibrationDataReader
    class NumpyDataReader(CalibrationDataReader):
        def __init__(self, configs, quantization_ds, quantization_split, model_path):
            """
            Initialize the calibration data reader.

            Args:
                configs (DictConfig): Quantization configuration.
                quantization_ds: Dataset for calibration samples.
                quantization_split (float): Fraction of dataset used for calibration.
                model_path (str): Path to the ONNX model to read input names.
            """
            # Create the generator inside the class
            self.enum_data = iter(
                _representative_data_gen(
                    configs=configs,
                    quantization_ds=quantization_ds,
                    quantization_split=quantization_split
                )
            )
            self.input_name = onnx.load(model_path).graph.input[0].name

        def get_next(self):
            """
            Return the next calibration batch for ONNX Runtime.

            Returns:
                dict | None: Mapping of input name to sample, or None when exhausted.
            """
            try:
                return {self.input_name: next(self.enum_data)}
            except StopIteration:
                return None

    # Usage:
    quantization_split = configs.dataset.quantization_split if configs.dataset.quantization_split is not None else 1.0
    data_reader = NumpyDataReader(
        configs=configs,
        quantization_ds=quantization_ds,
        quantization_split=quantization_split,
        model_path=model_path
    )
    # Perform static quantization
    quantize_static(
        model_input=model_path,
        model_output=str(quantized_model_path),
        calibration_data_reader=data_reader,
        per_channel=(configs.quantization.granularity == "per_channel"),
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8
    )
    # Load the modified model using ONNX Runtime Check if the model is valid
    model = onnxruntime.InferenceSession(quantized_model_path)
    try:
        model.get_inputs()
    except Exception as e:
        print(f"[ERROR] : An error occurred while quantizing the model: {e}")
        return
    setattr(model, 'model_path', quantized_model_path)
    print(f"[INFO] : Quantized model saved at {quantized_model_path}")
    return model


class OnnxPTQQuantizer:
    """
    PTQ quantizer for ONNX models. Outputs QDQ-format quantized models.

    Args:
        cfg (DictConfig): Configuration object for quantization.
        model (object): The model to quantize (TensorFlow or ONNX).
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
        Executes the ONNX PTQ quantization process.

        Returns:
            onnxruntime.InferenceSession : Quantized ONNX model
        """
        model_path = self.model.model_path
        output_dir = HydraConfig.get().runtime.output_dir
        file_extension = Path(model_path).suffix
        # check if the batch dimension is included in the input shape and remove it if present
        if len(self.cfg.model.input_shape) == 4:
            setattr(self.cfg.model, 'input_shape', self.cfg.model.input_shape[1:])

        quantization_ds = self.dataloaders['quantization']

        if self.cfg.quantization.quantizer.lower() == "onnx_quantizer" and self.cfg.quantization.quantization_type == "PTQ":
            if file_extension in [".h5", ".keras"]:
                # Convert the model to ONNX first
                input_shape = self.model.input_shape  # include batch dimension
                print(f"Converting model to ONNX, with static input shape {input_shape}")
                converted_model_path = os.path.join(output_dir, 'converted_model', 'converted_model.onnx')
                target_opset = self.cfg.quantization.target_opset if self.cfg.quantization.target_opset else 17
                onnx_model_converter(input_model_path=model_path, target_opset=target_opset,
                                     output_dir=converted_model_path, static_input_shape=input_shape,
                                     input_channels_last=True)
                self.quantized_model = quantize_onnx(configs=self.cfg, quantization_ds=quantization_ds, model_path=converted_model_path)
                return self.quantized_model

            elif file_extension == '.onnx':
                if model_is_quantized(model_path):
                    print('[INFO] : The input model is already quantized!\n\tReturning the same model!')
                    return self.model
                self.quantized_model = quantize_onnx(configs=self.cfg, quantization_ds=quantization_ds, model_path=model_path)
                return self.quantized_model
            else:
                raise ValueError("Unsupported model format for ONNX quantization. Supported formats are .h5, .keras, and .onnx")
        else:
            raise NotImplementedError("Quantizer and quantization type not supported yet!")
