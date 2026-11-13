# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
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

from clearml import Task
from clearml.backend_config.defs import get_active_config_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from api import get_model, get_trainer, get_evaluator, get_dataloaders, get_quantizer, get_predictor
from common.utils import mlflow_ini, set_gpu_memory_limit, get_random_seed, display_figures, log_to_file
from common.benchmarking import benchmark, cloud_connect
from common.evaluation import gen_load_val
from common.prediction import gen_load_val_predict
from common.quantization import define_extra_options
from pose_estimation.tf.src.utils import get_config
from pose_estimation.tf.src.deployment import deploy, deploy_mpu

from typing import Optional


def _process_mode(cfg: DictConfig = None) -> None:
    """
    Process the selected mode of operation.

    Args:
        cfg (DictConfig): The configuration object.
    Returns:
        None
    Raises:
        ValueError: If an invalid operation_mode is selected or if required datasets are missing.
    """

    mlflow.log_param("model_path", cfg.model.model_path) #getattr(model, "model_path", None))

    mode        = cfg.operation_mode
    model       = get_model(cfg)
    dataloaders = get_dataloaders(cfg)

    log_to_file(cfg.output_dir, f'operation_mode: {mode}')
    print(f"[INFO] : Model loaded from {model.model_path} with input shape {cfg.model.input_shape}")

    # Check the selected mode and perform the corresponding operation
    if mode == 'training':
        trainer = get_trainer(cfg=cfg, model=model,dataloaders=dataloaders)
        trained_model = trainer.train()
        # evaluate(cfg=cfg, model=trained_model, eval_ds=dataloaders['test'], name_ds="test_set")
        evaluator = get_evaluator(cfg=cfg, 
                                  model=trained_model, 
                                  dataloaders=dataloaders)
        evaluator.evaluate()
        display_figures(cfg)
        print('[INFO] : Training complete.')
    elif mode == 'evaluation':
        # Generates the model to be loaded on the stm32n6 device using stedgeai core,
        # then loads it and validates in on the device if required.
        gen_load_val(cfg=cfg)
        # Launches evaluation on the target through the model zoo evaluation service
        os.chdir(os.path.dirname(os.path.realpath(__file__)))  
        # evaluate(cfg=cfg, model=model, eval_ds=dataloaders['test'], name_ds="test_set")
        evaluator = get_evaluator(cfg=cfg, 
                                  model=model, 
                                  dataloaders=dataloaders)
        evaluator.evaluate()
        display_figures(cfg)
    elif mode == 'deployment':
        if cfg.hardware_type == "MPU":
            deploy_mpu(cfg=cfg)
        else:
            deploy(cfg=cfg)
        print('[INFO] : Deployment complete.')
        if cfg.deployment.hardware_setup.board == "STM32N6570-DK":
            print('[INFO] : Please on STM32N6570-DK toggle the boot switches to the left and power cycle the board.')
    elif mode == 'quantization':
        extra_options = define_extra_options(cfg=cfg)
        print('[INFO] : Using the quantization dataset to quantize the model.')
        quantizer = get_quantizer(cfg=cfg, model=model, dataloaders=dataloaders)
        quantized_model = quantizer.quantize()
    elif mode == 'prediction':
        # Generates the model to be loaded on the stm32n6 device using stedgeai core,
        # then loads it and validates in on the device if required.
        gen_load_val_predict(cfg=cfg)
        # Launches prediction on the target through the model zoo prediction service
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        # predict(cfg=cfg, model=model)
        predictor = get_predictor(cfg=cfg, 
                                  model=model, 
                                  dataloaders=dataloaders)
        predictor.predict()
    elif mode == 'benchmarking':
        benchmark(cfg=cfg, model_path_to_benchmark=model.model_path)
    elif mode == 'chain_tqeb':
        # Inline chain_tqeb: train -> quantize -> evaluate -> benchmark
        credentials = None
        if cfg.tools.stedgeai.on_cloud:
            _, _, credentials = cloud_connect(stedgeai_core_version=cfg.tools.stedgeai.version)
        trainer = get_trainer(cfg=cfg, model=model,dataloaders=dataloaders)
        trained_model = trainer.train()
        # evaluate(cfg=cfg, model=trained_model, eval_ds=dataloaders['test'], name_ds="test_set")
        evaluator = get_evaluator(cfg=cfg, 
                                  model=trained_model, 
                                  dataloaders=dataloaders)
        evaluator.evaluate()

        extra_options = define_extra_options(cfg=cfg)
        print('[INFO] : Using the quantization dataset to quantize the model.')
        quantizer = get_quantizer(cfg=cfg, model=trained_model, dataloaders=dataloaders)
        quantized_model = quantizer.quantize()
        # evaluate(cfg=cfg, model=quantized_model, eval_ds=dataloaders['test'], name_ds="test_set")
        evaluator = get_evaluator(cfg=cfg, 
                                  model=quantized_model, 
                                  dataloaders=dataloaders)
        evaluator.evaluate()

        display_figures(cfg)
        benchmark(cfg=cfg, model_path_to_benchmark=quantized_model.model_path, credentials=credentials)
        print('[INFO] : chain_tqeb complete.')
    elif mode == 'chain_tqe':
        # Inline chain_tqe: train -> quantize -> evaluate
        trainer = get_trainer(cfg=cfg, model=model,dataloaders=dataloaders)
        trained_model = trainer.train()
        # evaluate(cfg=cfg, model=trained_model, eval_ds=dataloaders['test'], name_ds="test_set")
        evaluator = get_evaluator(cfg=cfg, 
                                  model=trained_model, 
                                  dataloaders=dataloaders)
        evaluator.evaluate()

        extra_options = define_extra_options(cfg=cfg)
        print('[INFO] : Using the quantization dataset to quantize the model.')
        quantizer = get_quantizer(cfg=cfg, model=trained_model, dataloaders=dataloaders)
        quantized_model = quantizer.quantize()
        # evaluate(cfg=cfg, model=quantized_model, eval_ds=dataloaders['test'], name_ds="test_set")
        evaluator = get_evaluator(cfg=cfg, 
                                  model=quantized_model, 
                                  dataloaders=dataloaders)
        evaluator.evaluate()

        display_figures(cfg)
        print('[INFO] : chain_tqe complete.')
    elif mode == 'chain_eqe':
        # Inline chain_eqe: evaluate float -> quantize -> evaluate quantized
        # evaluate(cfg=cfg, model=model, eval_ds=dataloaders['test'], name_ds="test_set")
        evaluator = get_evaluator(cfg=cfg, 
                                  model=model, 
                                  dataloaders=dataloaders)
        evaluator.evaluate()

        display_figures(cfg)
        extra_options = define_extra_options(cfg=cfg)
        print('[INFO] : Using the quantization dataset to quantize the model.')
        quantizer = get_quantizer(cfg=cfg, model=model, dataloaders=dataloaders)
        quantized_model = quantizer.quantize()
        # evaluate(cfg=cfg, model=quantized_model, eval_ds=dataloaders['test'], name_ds="test_set")
        evaluator = get_evaluator(cfg=cfg, 
                                  model=quantized_model, 
                                  dataloaders=dataloaders)
        evaluator.evaluate()

        display_figures(cfg)
        print('[INFO] : chain_eqe complete.')
    elif mode == 'chain_qb':
        # Inline chain_qb: quantize -> benchmark
        credentials = None
        if cfg.tools.stedgeai.on_cloud:
            _, _, credentials = cloud_connect(stedgeai_core_version=cfg.tools.stedgeai.version)
        extra_options = define_extra_options(cfg=cfg)
        print('[INFO] : Using the quantization dataset to quantize the model.')
        quantizer = get_quantizer(cfg=cfg, model=model, dataloaders=dataloaders)
        quantized_model = quantizer.quantize()
        benchmark(cfg=cfg, model_path_to_benchmark=quantized_model.model_path, credentials=credentials)
        print('[INFO] : chain_qb complete.')
    elif mode == 'chain_eqeb':
        # Inline chain_eqeb: evaluate float -> quantize -> evaluate quantized -> benchmark
        credentials = None
        if cfg.tools.stedgeai.on_cloud:
            _, _, credentials = cloud_connect(stedgeai_core_version=cfg.tools.stedgeai.version)
        # evaluate(cfg=cfg, model=model, eval_ds=dataloaders['test'], name_ds="test_set")
        evaluator = get_evaluator(cfg=cfg, 
                                  model=model, 
                                  dataloaders=dataloaders)
        evaluator.evaluate()

        display_figures(cfg)
        extra_options = define_extra_options(cfg=cfg)
        print('[INFO] : Using the quantization dataset to quantize the model.')
        quantizer = get_quantizer(cfg=cfg, model=model, dataloaders=dataloaders)
        quantized_model = quantizer.quantize()
        # evaluate(cfg=cfg, model=quantized_model, eval_ds=dataloaders['test'], name_ds="test_set")
        evaluator = get_evaluator(cfg=cfg, 
                                  model=quantized_model, 
                                  dataloaders=dataloaders)
        evaluator.evaluate()
        
        display_figures(cfg)
        benchmark(cfg=cfg, model_path_to_benchmark=quantized_model.model_path, credentials=credentials)
        print('[INFO] : chain_eqeb complete.')
    elif mode == 'chain_qd':
        # Inline chain_qd: quantize -> deploy
        credentials = None
        if cfg.tools.stedgeai.on_cloud:
            _, _, credentials = cloud_connect(stedgeai_core_version=cfg.tools.stedgeai.version)
        extra_options = define_extra_options(cfg=cfg)
        print('[INFO] : Using the quantization dataset to quantize the model.')
        quantizer = get_quantizer(cfg=cfg, model=model, dataloaders=dataloaders)
        quantized_model = quantizer.quantize()
        if cfg.hardware_type == "MPU":
            deploy_mpu(cfg=cfg, model_path_to_deploy=quantized_model.model_path, credentials=credentials)
        else:
            deploy(cfg=cfg, model_path_to_deploy=quantized_model.model_path, credentials=credentials)
        print('[INFO] : Deployment complete.')
        if cfg.deployment.hardware_setup.board == "STM32N6570-DK":
            print('[INFO] : Please on STM32N6570-DK toggle the boot switches to the left and power cycle the board.')
        print('[INFO] : chain_qd complete.')
    # Raise an error if an invalid mode is selected
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Record the whole hydra working directory to get all info
    mlflow.log_artifact(cfg.output_dir)
    if mode in ['benchmarking', 'chain_qb', 'chain_eqeb', 'chain_tqeb']:
        mlflow.log_param("stedgeai_core_version", cfg.tools.stedgeai.version)
        mlflow.log_param("target", cfg.benchmarking.board)
    # logging the completion of the chain
    log_to_file(cfg.output_dir, f'operation finished: {mode}')

    # ClearML - Example how to get task's context anywhere in the file.
    # Checks if there's a valid ClearML configuration file
    if get_active_config_file() is not None: 
        print(f"[INFO] : ClearML task connection")
        task = Task.current_task()
        task.connect(cfg)


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
            print(f"[INFO] Setting upper limit of usable GPU memory to {int(cfg.general.gpu_memory_limit)}GBytes.")
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
        task = Task.init(project_name=cfg.general.project_name,
                         task_name='pe_modelzoo_task')
        # ClearML - Optional yaml logging 
        task.connect_configuration(name=cfg.operation_mode, 
                                   configuration=cfg)

    # Seed global seed for random generators
    seed = get_random_seed(cfg)
    print(f'[INFO] : The random seed for this simulation is {seed}')
    if seed is not None:
        tf.keras.utils.set_random_seed(seed)

    _process_mode(cfg=cfg)


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
