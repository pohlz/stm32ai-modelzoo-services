# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
import os
from pyexpat import model
import sys
from hydra.core.hydra_config import HydraConfig
import hydra
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from omegaconf import DictConfig
import mlflow
import argparse
from typing import Optional
from clearml import Task
from clearml.backend_config.defs import get_active_config_file
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from api import get_model
from common.utils import mlflow_ini, set_gpu_memory_limit, get_random_seed, display_figures, log_to_file
from common.benchmarking import benchmark, cloud_connect
from common.evaluation import gen_load_val
from common.prediction import gen_load_val_predict
from common.quantization import define_extra_options
from re_identification.tf.src.preprocessing import preprocess
from re_identification.tf.src.utils import get_config
from re_identification.tf.src.training import train
from re_identification.tf.src.evaluation import evaluate
from re_identification.tf.src.quantization import quantize
from re_identification.tf.src.prediction import predict
from re_identification.tf.src.deployment import deploy



def chain_qd(cfg: DictConfig = None, model: tf.keras.Model = None, quantization_ds: tf.data.Dataset = None, hardware_type: str = "MCU") -> None:
    """
    Runs the chain_qd pipeline, including quantization, and deployment

    Args:
        cfg (DictConfig): Configuration dictionary. Defaults to None.
        model (tf.keras.Model): Keras model to be quantized. Defaults to None.
        quantization_ds:(tf.data.Dataset): quantization dataset. Defaults to None
        hardware_type (str): parameter to specify a target on which to deploy

    Returns:
        None
    """

    # Connect to STM32Cube.AI Developer Cloud
    credentials = None
    if cfg.tools.stedgeai.on_cloud:
        _, _, credentials = cloud_connect(stedgeai_core_version=cfg.tools.stedgeai.version)

    # whether data are coming from train set or quantization set, they end up in quantization_ds
    source_image = cfg.dataset.quantization_path if cfg.dataset.quantization_path else cfg.dataset.training_path
    source_image = source_image if source_image else "random generation"
    print('[INFO] : Quantization using input images coming from {}'.format(source_image))
    extra_options = define_extra_options(cfg=cfg)
    quantized_model_path = quantize(cfg=cfg, model=model, quantization_ds=quantization_ds, extra_options=extra_options)
    #quantized_model_path = quantize(cfg=cfg, quantization_ds=quantization_ds)
    print('[INFO] : Quantization complete.')
    deploy(cfg=cfg, model_path_to_deploy=quantized_model_path, credentials=credentials)
    print('[INFO] : Deployment complete.')
    if cfg.deployment.hardware_setup.board == "STM32N6570-DK":
        print('[INFO] : Please on STM32N6570-DK toggle the boot switches to the left and power cycle the board.')


def chain_qb(cfg: DictConfig = None, model: tf.keras.Model = None, quantization_ds: tf.data.Dataset = None) -> None:
    """
    Runs the chain_qb pipeline, including quantization and benchmarking.

    Args:
        cfg (DictConfig): Configuration dictionary. Defaults to None.
        quantization_ds:(tf.data.Dataset): quantization dataset. Defaults to None

    Returns:
        None
    """

    # Connect to STM32Cube.AI Developer Cloud
    credentials = None
    if cfg.tools.stedgeai.on_cloud:
        _, _, credentials = cloud_connect(stedgeai_core_version=cfg.tools.stedgeai.version)

    # whether data are coming from train set or quantization set, they end up in quantization_ds
    source_image = cfg.dataset.quantization_path if cfg.dataset.quantization_path else cfg.dataset.training_path
    source_image = source_image if source_image else "random generation"
    print('[INFO] : Quantization using input images coming from {}'.format(source_image))
    extra_options = define_extra_options(cfg=cfg)
    quantized_model = quantize(cfg=cfg, model=model, quantization_ds=quantization_ds, extra_options=extra_options)
    #quantized_model_path = quantize(cfg=cfg, quantization_ds=quantization_ds)
    print('[INFO] : Quantization complete.')
    benchmark(cfg=cfg, model_path_to_benchmark=quantized_model.model_path, credentials=credentials)
    print('[INFO] : Benchmarking complete.')


def chain_eqe(cfg: DictConfig = None, model: tf.keras.Model = None, valid_ds: tf.data.Dataset = None, quantization_ds: tf.data.Dataset = None,
              test_query_ds: Optional[tf.data.Dataset] = None, test_gallery_ds: Optional[tf.data.Dataset] = None) -> str:
    """
    Runs the chain_eqe pipeline, including evaluation of a float model, quantization and evaluation of
    the quantized model

    Args:
        cfg (DictConfig): Configuration dictionary. Defaults to None.
        valid_ds (tf.data.Dataset): Validation dataset. Defaults to None.
        test_query_ds (tf.data.Dataset): Test query dataset. Defaults to None.
        test_gallery_ds (tf.data.Dataset): Test gallery dataset. Defaults to None.
        quantization_ds:(tf.data.Dataset): quantization dataset. Defaults to None

    Returns:
        quantized_model_path (str): path to quantized model
    """
    model = get_model(cfg)
    if test_query_ds and test_gallery_ds:
        evaluate(cfg=cfg, model=model, eval_query_ds=test_query_ds, eval_gallery_ds=test_gallery_ds, name_ds="test_set")
    else:
        raise ValueError("For re-identification, test_query_ds and test_gallery_ds must be provided for evaluation")
    print('[INFO] : Evaluation complete.')
    display_figures(cfg)

    # whether data are coming from train set or quantization set, they end up in quantization_ds
    source_image = cfg.dataset.quantization_path if cfg.dataset.quantization_path else cfg.dataset.training_path
    source_image = source_image if source_image else "random generation"
    print('[INFO] : Quantization using input images coming from {}'.format(source_image))
    extra_options = define_extra_options(cfg=cfg)
    quantized_model = quantize(cfg=cfg, model=model, quantization_ds=quantization_ds, extra_options=extra_options)
    #quantized_model_path = quantize(cfg=cfg, quantization_ds=quantization_ds)
    print('[INFO] : Quantization complete.')

    if test_query_ds and test_gallery_ds:
        evaluate(cfg=cfg, model=quantized_model, eval_query_ds=test_query_ds, eval_gallery_ds=test_gallery_ds, name_ds="test_set")
    else:
        raise ValueError("For re-identification, test_query_ds and test_gallery_ds must be provided for evaluation")
    print('[INFO] : Evaluation complete.')
    display_figures(cfg)

    return quantized_model


def chain_eqeb(cfg: DictConfig = None, model: tf.keras.Model = None, valid_ds: tf.data.Dataset = None, quantization_ds: tf.data.Dataset = None,
                test_query_ds: Optional[tf.data.Dataset] = None, test_gallery_ds: Optional[tf.data.Dataset] = None) -> None:
    """
    Runs the chain_eqeb pipeline, including evaluation of the float model, quantization, evaluation of
    the quantized model and benchmarking

    Args:
        cfg (DictConfig): Configuration dictionary. Defaults to None.
        valid_ds (tf.data.Dataset): Validation dataset. Defaults to None.
        test_query_ds (tf.data.Dataset): Test query dataset. Defaults to None.
        test_gallery_ds (tf.data.Dataset): Test gallery dataset. Defaults to None.
        quantization_ds:(tf.data.Dataset): quantization dataset. Defaults to None

    Returns:
        None
    """

    # Connect to STM32Cube.AI Developer Cloud
    credentials = None
    if cfg.tools.stedgeai.on_cloud:
        _, _, credentials = cloud_connect(stedgeai_core_version=cfg.tools.stedgeai.version)

    quantized_model = chain_eqe(cfg=cfg, valid_ds=valid_ds, quantization_ds=quantization_ds, test_query_ds=test_query_ds, test_gallery_ds=test_gallery_ds)

    benchmark(cfg=cfg, model_path_to_benchmark=quantized_model.model_path, credentials=credentials)
    print('[INFO] : Benchmarking complete.')


def chain_tqe(cfg: DictConfig = None, model: tf.keras.Model = None, train_ds: tf.data.Dataset = None, valid_ds: tf.data.Dataset = None,
              quantization_ds: tf.data.Dataset = None, test_query_ds: Optional[tf.data.Dataset] = None, 
              test_gallery_ds: Optional[tf.data.Dataset] = None) -> str:
    """
    Runs the chain_tqe pipeline, including training, quantization and evaluation.

    Args:
        cfg (DictConfig): Configuration dictionary. Defaults to None.
        model (tf.keras.Model): Keras model to be trained. Defaults to None.
        train_ds (tf.data.Dataset): Training dataset. Defaults to None.
        valid_ds (tf.data.Dataset): Validation dataset. Defaults to None.
        test_query_ds (tf.data.Dataset): Test query dataset. Defaults to None.
        test_gallery_ds (tf.data.Dataset): Test gallery dataset. Defaults to None.
        quantization_ds:(tf.data.Dataset): quantization dataset. Defaults to None

    Returns:
        quantized_model_path (str): path to model quantized
    """
    if test_query_ds and test_gallery_ds:
        trained_model = train(cfg=cfg, model=model, train_ds=train_ds, valid_ds=valid_ds, test_query_ds=test_query_ds, test_gallery_ds=test_gallery_ds)
    else:
        trained_model = train(cfg=cfg, model=model, train_ds=train_ds, valid_ds=valid_ds)
    print('[INFO] : Training complete.')

    # whether data are coming from train set or quantization set, they end up in quantization_ds
    source_image = cfg.dataset.quantization_path if cfg.dataset.quantization_path else cfg.dataset.training_path
    source_image = source_image if source_image else "random generation"
    print('[INFO] : Quantization using input images coming from {}'.format(source_image))
    extra_options = define_extra_options(cfg=cfg)
    quantized_model = quantize(cfg=cfg, model=trained_model, quantization_ds=quantization_ds, extra_options=extra_options)
    #quantized_model_path = quantize(cfg=cfg, quantization_ds=quantization_ds, float_model_path=trained_model_path)
    print('[INFO] : Quantization complete.')

    if test_query_ds and test_gallery_ds:
        evaluate(cfg=cfg, model=quantized_model, eval_query_ds=test_query_ds, eval_gallery_ds=test_gallery_ds,
                  name_ds="test_set")
    else:
        raise ValueError("For re-identification, test_query_ds and test_gallery_ds must be provided for evaluation")
    print('[INFO] : Evaluation complete.')
    display_figures(cfg)

    return quantized_model


def chain_tqeb(cfg: DictConfig = None, model: tf.keras.Model = None, train_ds: tf.data.Dataset = None, valid_ds: tf.data.Dataset = None,
               quantization_ds: tf.data.Dataset = None, test_query_ds: Optional[tf.data.Dataset] = None,
                test_gallery_ds: Optional[tf.data.Dataset] = None) -> None:
    """
    Runs the chain_tqeb pipeline, including training, quantization, evaluation and benchmarking.

    Args:
        cfg (DictConfig): Configuration dictionary. Defaults to None.
        train_ds (tf.data.Dataset): Training dataset. Defaults to None.
        valid_ds (tf.data.Dataset): Validation dataset. Defaults to None.
        test_query_ds (tf.data.Dataset): Test query dataset. Defaults to None.
        test_gallery_ds (tf.data.Dataset): Test gallery dataset. Defaults to None.
        quantization_ds:(tf.data.Dataset): quantization dataset. Defaults to None

    Returns:
        None
    """

    # Connect to STM32Cube.AI Developer Cloud
    credentials = None
    if cfg.tools.stedgeai.on_cloud:
        _, _, credentials = cloud_connect(stedgeai_core_version=cfg.tools.stedgeai.version)

    quantized_model = chain_tqe(cfg=cfg, model=model, train_ds=train_ds, valid_ds=valid_ds, quantization_ds=quantization_ds,
                                    test_query_ds=test_query_ds, test_gallery_ds=test_gallery_ds)

    benchmark(cfg=cfg, model_path_to_benchmark=quantized_model.model_path, credentials=credentials)
    print('[INFO] : Benchmarking complete.')


def process_mode(mode: str = None,
                 configs: DictConfig = None,
                 train_ds: tf.data.Dataset = None,
                 valid_ds: tf.data.Dataset = None,
                 quantization_ds: tf.data.Dataset = None,
                 test_query_ds: tf.data.Dataset = None,
                 test_gallery_ds: tf.data.Dataset = None) -> None:
    """
    Process the selected mode of operation.

    Args:
        mode (str): The selected mode of operation. Must be one of 'train', 'evaluate', or 'predict'.
        configs (DictConfig): The configuration object.
        train_ds (tf.data.Dataset): The training dataset. Required if mode is 'train'.
        valid_ds (tf.data.Dataset): The validation dataset. Required if mode is 'train' or 'evaluate'.
        test_query_ds(tf.data.Dataset): The test query dataset.
        test_gallery_ds(tf.data.Dataset): The test gallery dataset.
        quantization_ds(tf.data.Dataset): The quantization dataset.

    Returns:
        None
    Raises:
        ValueError: If an invalid operation_mode is selected or if required datasets are missing.
    """
    
    mlflow.log_param("model_path", configs.model.model_path)
    # logging the operation_mode in the output_dir/stm32ai_main.log file
    log_to_file(configs.output_dir, f'operation_mode: {mode}')
    # Check the selected mode and perform the corresponding operation

    model = get_model(configs)
    if mode == 'training':
        if test_query_ds and test_gallery_ds:
            train(cfg=configs, model=model, train_ds=train_ds, valid_ds=valid_ds, test_query_ds=test_query_ds, test_gallery_ds=test_gallery_ds)
        else:
            train(cfg=configs, model=model, train_ds=train_ds, valid_ds=valid_ds)
        display_figures(configs)
        print('[INFO] : Training complete.')
    elif mode == 'evaluation':
        # Generates the model to be loaded on the stm32n6 device using stedgeai core,
        # then loads it and validates in on the device if required.
        gen_load_val(cfg=configs)
        # Launches evaluation on the target through the model zoo evaluation service
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        if test_query_ds and test_gallery_ds:
            evaluate(cfg=configs, model=model, eval_query_ds=test_query_ds, eval_gallery_ds=test_gallery_ds, name_ds="test_set")
        display_figures(configs)
        print('[INFO] : Evaluation complete.')
    elif mode == 'deployment':
        deploy(cfg=configs)
        print('[INFO] : Deployment complete.')
        if configs.deployment.hardware_setup.board == "STM32N6570-DK":
            print('[INFO] : Please on STM32N6570-DK toggle the boot switches to the left and power cycle the board.')
    elif mode == 'quantization':
        # whether data are coming from train set or quantization set, they end up in quantization_ds
        source_image = configs.dataset.quantization_path if configs.dataset.quantization_path \
            else configs.dataset.training_path
        source_image = source_image if source_image else "random generation"
        print('[INFO] : Quantization using input images coming from {}'.format(source_image))
        # set ONNX quantizer options for quantization
        extra_options = define_extra_options(cfg=configs)
        quantize(cfg=configs, model=model, quantization_ds=quantization_ds, extra_options=extra_options)
        print('[INFO] : Quantization complete.')
    elif mode == 'prediction':
        # Generates the model to be loaded on the stm32n6 device using stedgeai core,
        # then loads it and validates in on the device if required.
        gen_load_val_predict(cfg=configs)
        # Launches prediction on the target through the model zoo prediction service
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        predict(cfg=configs, model=model)
        print('[INFO] : Prediction complete.')
    elif mode == 'benchmarking':
        benchmark(cfg=configs, model_path_to_benchmark=model.model_path)
        print('[INFO] : Benchmark complete.')
    elif mode == 'chain_tqe':
        chain_tqe(cfg=configs, model=model, train_ds=train_ds, valid_ds=valid_ds, quantization_ds=quantization_ds, test_query_ds=test_query_ds, test_gallery_ds=test_gallery_ds)
        print('[INFO] : chain_tqe complete.')
    elif mode == 'chain_tqeb':
        chain_tqeb(cfg=configs, model=model, train_ds=train_ds, valid_ds=valid_ds, quantization_ds=quantization_ds, test_query_ds=test_query_ds, test_gallery_ds=test_gallery_ds)
        print('[INFO] : chain_tqeb complete.')
    elif mode == 'chain_eqe':
        chain_eqe(cfg=configs, model=model, valid_ds=valid_ds, quantization_ds=quantization_ds, test_query_ds=test_query_ds, test_gallery_ds=test_gallery_ds)
        print('[INFO] : chain_eqe complete.')
    elif mode == 'chain_eqeb':
        chain_eqeb(cfg=configs, model=model, valid_ds=valid_ds, quantization_ds=quantization_ds, test_query_ds=test_query_ds, test_gallery_ds=test_gallery_ds)
        print('[INFO] : chain_eqeb complete.')
    elif mode == 'chain_qb':
        chain_qb(cfg=configs, model=model, quantization_ds=quantization_ds)
        print('[INFO] : chain_qb complete.')
    elif mode == 'chain_qd':
        chain_qd(cfg=configs, model=model,quantization_ds=quantization_ds, hardware_type=configs.hardware_type)
        print('[INFO] : chain_qd complete.')
    # Raise an error if an invalid mode is selected
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Record the whole hydra working directory to get all info
    mlflow.log_artifact(configs.output_dir)
    if mode in ['benchmarking', 'chain_qb', 'chain_eqeb', 'chain_tqeb']:
        mlflow.log_param("stm32ai_version", configs.tools.stm32ai.version)
        mlflow.log_param("target", configs.benchmarking.board)
    # logging the completion of the chain
    log_to_file(configs.output_dir, f'operation finished: {mode}')

    # ClearML - Example how to get task's context anywhere in the file.
    # Checks if there's a valid ClearML configuration file
    if get_active_config_file() is not None: 
        print(f"[INFO] : ClearML task connection")
        task = Task.current_task()
        task.connect(configs)


@hydra.main(version_base=None, config_path="", config_name="user_config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point of the script.

    Args:
        cfg: Configuration dictionary.

    Returns:
        None
    """

    # Configure the GPU (the 'general' section may be missing)
    if "general" in cfg and cfg.general:
        # Set upper limit on usable GPU memory
        if "gpu_memory_limit" in cfg.general and cfg.general.gpu_memory_limit:
            set_gpu_memory_limit(cfg.general.gpu_memory_limit)
            print(f"[INFO] : Setting upper limit of usable GPU memory to {int(cfg.general.gpu_memory_limit)}GBytes.")
        else:
            print("[WARNING] The usable GPU memory is unlimited.\n"
                  "Please consider setting the 'gpu_memory_limit' attribute "
                  "in the 'general' section of your configuration file.")

    # Parse the configuration file
    cfg = get_config(cfg)
    cfg.output_dir = HydraConfig.get().run.dir
    mlflow_ini(cfg)

    # Checks if there's a valid ClearML configuration file
    print(f"[INFO] : ClearML config check")
    if get_active_config_file() is not None:
        print(f"[INFO] : ClearML initialization and configuration")
        # ClearML - Initializing ClearML's Task object.
        # task_name is the timestamp + project_name
        task = Task.init(project_name=cfg.general.project_name,
                         task_name=f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {cfg.general.project_name}")
        # ClearML - Optional yaml logging
        task.connect_configuration(name=cfg.operation_mode,
                                   configuration=cfg)
          
    # Seed global seed for random generators
    seed = get_random_seed(cfg)
    print(f'[INFO] : The random seed for this simulation is {seed}')
    if seed is not None:
        tf.keras.utils.set_random_seed(seed)

    # Extract the mode from the command-line arguments
    mode = cfg.operation_mode
    preprocess_output = preprocess(cfg=cfg)
    train_ds, valid_ds, quantization_ds, test_query_ds, test_gallery_ds = preprocess_output
    # Process the selected mode
    process_mode(mode=mode, configs=cfg, train_ds=train_ds, valid_ds=valid_ds, quantization_ds=quantization_ds,
                 test_query_ds=test_query_ds, test_gallery_ds=test_gallery_ds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, default='', help='Path to folder containing configuration file')
    parser.add_argument('--config-name', type=str, default='user_config', help='name of the configuration file')
    # add arguments to the parser
    parser.add_argument('params', nargs='*',
                        help='List of parameters to over-ride in config.yaml')
    args = parser.parse_args()

    # Call the main function
    main()

    # log the config_path and config_name parameters
    mlflow.log_param('config_path', args.config_path)
    mlflow.log_param('config_name', args.config_name)
    mlflow.end_run()
