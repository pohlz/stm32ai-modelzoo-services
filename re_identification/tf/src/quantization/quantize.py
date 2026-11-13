#  *---------------------------------------------------------------------------------------------*/
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import pathlib
import numpy as np
import tensorflow as tf
import tqdm
import sys
import os
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from typing import Optional
import keras

from common.utils import get_model_name_and_its_input_shape, tf_dataset_to_np_array
from common.quantization import quantize_onnx
from common.evaluation import model_is_quantized
from common.optimization import model_formatting_ptq_per_tensor
from common.onnx_utils import onnx_model_converter
from re_identification.tf.src.preprocessing import apply_rescaling


def _tflite_ptq_quantizer(model: tf.keras.Model = None, quantization_ds: tf.data.Dataset = None,
                         output_dir: str = None, export_dir: Optional[str] = None, input_shape: tuple = None,
                         quantization_granularity: str = None, quantization_input_type: str = None,
                         quantization_output_type: str = None, num_threads: int = 1) -> None:
    """
    Perform post-training quantization on a TensorFlow Lite model.

    Args:
        model (tf.keras.Model): The TensorFlow model to be quantized.
        quantization_ds (tf.data.Dataset): The quantization dataset if it's provided by the user else the training
        dataset. Defaults to None
        output_dir (str): Path to the output directory. Defaults to None.
        export_dir (str): Name of the export directory. Defaults to None.
        input_shape (tuple: The input shape of the model. Defaults to None.
        quantization_granularity (str): 'per_tensor' or 'per_channel'. Defaults to None.
        quantization_input_type (str): The quantization type for the input. Defaults to None.
        quantization_output_type (str): The quantization type for the output. Defaults to None.

    Returns:
        None
    """

    def representative_data_gen():
        """
        Generate representative data for post-training quantization.

        Yields:
            List[tf.Tensor]: A list of TensorFlow tensors representing the input data.
        """
        if not quantization_ds:
            for _ in tqdm.tqdm(range(5)):
                data = np.random.rand(1, input_shape[0], input_shape[1], input_shape[2])
                yield [data.astype(np.float32)]

        else:
            for images, labels in tqdm.tqdm(quantization_ds, total=len(quantization_ds)):
                for image in images:
                    image = tf.cast(image, dtype=tf.float32)
                    image = tf.expand_dims(image, 0)
                    yield [image]

    # Create the TFLite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Create the output directory
    tflite_models_dir = pathlib.Path(os.path.join(output_dir, "{}/".format(export_dir)))
    tflite_models_dir.mkdir(exist_ok=True, parents=True)

    # Set the quantization types for the input and output
    if quantization_input_type == 'int8':
        converter.inference_input_type = tf.int8
    elif quantization_input_type == 'uint8':
        converter.inference_input_type = tf.uint8
    else:
        pass
    if quantization_output_type == 'int8':
        converter.inference_output_type = tf.int8
    elif quantization_output_type == 'uint8':
        converter.inference_output_type = tf.uint8
    else:
        pass

    # Set the optimizations and representative dataset generator
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen

    # Set the quantization per tensor if requested
    if quantization_granularity == 'per_tensor':
        converter._experimental_disable_per_channel = True

    # Convert the model to a quantized TFLite model
    tflite_model_quantized = converter.convert()
    tflite_model_quantized_file = tflite_models_dir / "quantized_model.tflite"
    tflite_model_quantized_file.write_bytes(tflite_model_quantized)

    # Load the quantized model as a TFLite Interpreter and set model_path attribute
    model_interpreter = tf.lite.Interpreter(model_path=str(tflite_model_quantized_file), num_threads=num_threads)
    model_interpreter.model_path = str(tflite_model_quantized_file)
    return model_interpreter


def quantize(cfg: DictConfig = None, 
             model: tf.keras.Model = None,
             quantization_ds: Optional[tf.data.Dataset] = None,
             extra_options: dict = None) -> str:
    """
    Quantize the TensorFlow model with training data.

    Args:
        cfg (DictConfig): The configuration dictionary. Defaults to None.
        quantization_ds (tf.data.Dataset): The quantization dataset if it's provided by the user else the training
        dataset. Defaults to None.
        float_model_path (str, optional): Model path to quantize
        extra_options: contains so-called extra option dictionary for onnx quantizer. Defaults to None

    Returns:
        quantized model path (str)
    """

    model_path = model.model_path
    _, input_shape = get_model_name_and_its_input_shape(model_path=model_path)
    output_dir = HydraConfig.get().runtime.output_dir
    export_dir = cfg.quantization.export_dir
    num_threads = getattr(cfg.general, 'num_threads_tflite', 1)

    if quantization_ds:
        quantization_ds = apply_rescaling(dataset=quantization_ds,
                                          scale=cfg.preprocessing.rescaling.scale,
                                          offset=cfg.preprocessing.rescaling.offset)
        quant_split = cfg.dataset.quantization_split if cfg.dataset.quantization_split else 1.0
        print(f'[INFO] : Quantizing by using {quant_split * 100} % of the provided dataset...')
        quantization_ds_size = len(quantization_ds)
        quantization_ds = quantization_ds.take(int(quantization_ds_size * float(quant_split)))

    print("[INFO] : Quantizing the model ... This might take few minutes ...")

    # Check the model file extension
    file_extension = str(model_path).split('.')[-1]
    if file_extension in ['h5','keras']:
        # we expect a float Keras model
        float_model = tf.keras.models.load_model(model_path, compile=False)
        quantization_granularity = cfg.quantization.granularity
        quantization_optimize = cfg.quantization.optimize
        print(f'[INFO] : Quantization granularity : {quantization_granularity}')
        # if per-tensor quantization is required some optimizations are possible on the float model
        if quantization_granularity == 'per_tensor' and quantization_optimize:
            print("[INFO] : Optimizing the model for improved per_tensor quantization...")
            float_model = model_formatting_ptq_per_tensor(model_origin=float_model)
            models_dir = pathlib.Path(os.path.join(output_dir, "{}/".format(export_dir)))
            models_dir.mkdir(exist_ok=True, parents=True)
            model_path = models_dir / "optimized_model.keras"
            float_model.save(model_path)

    if cfg.quantization.quantizer.lower() == "onnx_quantizer" and cfg.quantization.quantization_type == "PTQ":
        # Convert the dataset to numpy array
        target_opset = cfg.quantization.target_opset
        if quantization_ds:
            data, _ = tf_dataset_to_np_array(quantization_ds, nchw=True)
        else:
            print(f'[INFO] : Quantizing by using fake dataset...')
            data = None

        # Check the model file extension
        if file_extension in ['h5','keras']:
            # Convert the model and then quantize
            converted_model_path = os.path.join(output_dir, 'converted_model', 'converted_model.onnx')

            # model = keras.models.load_model(model_path)
            static_input_shape = model.inputs[0].shape
            onnx_model_converter(input_model_path=model_path, target_opset=17, output_dir=converted_model_path,
                                 static_input_shape=static_input_shape)

#            model = keras.models.load_model(model_path)
#            model(model.inputs)
#            input_spec = keras.InputSpec()
#            input_spec.shape = model.inputs[0].shape
#            input_spec.dtype = model.inputs[0].dtype
#            input_spec.axes = {0:0, 1:3, 2:1, 3:2}
#            print("input_spec : ", input_spec)
#            model.export(converted_model_path, format="onnx", input_signature=[input_spec])

            quantized_model = quantize_onnx(quantization_samples=data, model_path=converted_model_path, configs=cfg, extra_options=extra_options)
            return quantized_model
        elif file_extension == 'onnx':
            # Check if the model is already quantized
            if model_is_quantized(model_path):
                print('[INFO]: The input model is already quantized!\n\tReturning the same model!')
                return model
            else:
                # Quantize the model
                quantized_model = quantize_onnx(quantization_samples=data, model_path=model.model_path, configs=cfg, extra_options=extra_options)
                return quantized_model
        else:
            raise ValueError(f"Unsupported model file extension: {file_extension}")

    elif cfg.quantization.quantizer.lower() == "tflite_converter" and cfg.quantization.quantization_type == "PTQ":
        if file_extension in ['h5','keras']:
            quantized_model = _tflite_ptq_quantizer(model=float_model, quantization_ds=quantization_ds, output_dir=output_dir,
                                                    export_dir=export_dir, input_shape=input_shape,
                                                    quantization_granularity=quantization_granularity,
                                                    quantization_input_type=cfg.quantization.quantization_input_type,
                                                    quantization_output_type=cfg.quantization.quantization_output_type,
                                                    num_threads=num_threads)

            quantized_model_path = os.path.join(output_dir, export_dir, "quantized_model.tflite")
            setattr(quantized_model, 'model_path', quantized_model_path)
            print(f'[INFO] : Quantized model path: {quantized_model_path}')
            return quantized_model
        else:
            raise ValueError(f"Unsupported model file extension: {file_extension}")
    else:
        raise TypeError("Quantizer or quantization type not supported."
                        "Check the `quantization` section of your user_config.yaml file!")
