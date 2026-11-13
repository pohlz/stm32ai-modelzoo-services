# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
import sys
from hydra.core.hydra_config import HydraConfig
import hydra

import tensorflow as tf
from omegaconf import DictConfig
import mlflow
import argparse
from clearml import Task
from clearml.backend_config.defs import get_active_config_file


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from api import get_model, get_dataloaders, get_predictor
from common.utils import mlflow_ini, set_gpu_memory_limit, get_random_seed, log_to_file
from common.benchmarking import benchmark, cloud_connect
from common.prediction import gen_load_val_predict
from neural_style_transfer.tf.src.utils import get_config
from neural_style_transfer.tf.src.deployment import deploy


def process_mode(cfg: DictConfig = None,) -> None:
    """
    Process the selected mode of operation.

    Args:
        cfg (DictConfig): The configuration object.


    Returns:
        None
    Raises:
        ValueError: If an invalid operation_mode is selected or if required datasets are missing.
    """

    mode = cfg.operation_mode
    mlflow.log_param("model_path", cfg.model.model_path)
    # logging the operation_mode in the output_dir/stm32ai_main.log file
    log_to_file(cfg.output_dir, f'operation_mode: {mode}')

    # Creates model
    model = get_model(cfg=cfg)

    # Creates dataloaders
    if mode not in ["benchmarking", "deployment"]:
        dataloaders = get_dataloaders(cfg=cfg)
    
    # Check the selected mode and perform the corresponding operation
    if mode == 'deployment':
        if cfg.hardware_type == "MPU":
            raise ValueError(f"Invalid hardware. Only STM32N6570-DK boards are supported for deployment.")
        else:
            deploy(cfg=cfg, model_path_to_deploy=model.model_path)
        print('[INFO] : Deployment complete.')
        if cfg.deployment.hardware_setup.board == "STM32N6570-DK":
            print('[INFO] : Please on STM32N6570-DK toggle the boot switches to the left and power cycle the board.')
    elif mode == 'prediction':
        gen_load_val_predict(cfg=cfg, model=model)
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        predictor = get_predictor(cfg=cfg,
                                  model=model,
                                  dataloaders=dataloaders)
        predictor.predict()
    elif mode == 'benchmarking':
        benchmark(cfg=cfg, model_path_to_benchmark=model.model_path)
        print('[INFO] : Benchmark complete.')
    # Raise an error if an invalid mode is selected
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Record the whole hydra working directory to get all info
    mlflow.log_artifact(cfg.output_dir)
    if mode =='benchmarking':
        mlflow.log_param("stm32ai_version", cfg.tools.stm32ai.version)
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
                         task_name='nstyletr_modelzoo_task')
        # ClearML - Optional yaml logging
        task.connect_configuration(name=cfg.operation_mode,
                                   configuration=cfg)

    # Seed global seed for random generators
    seed = get_random_seed(cfg)
    print(f'[INFO] : The random seed for this simulation is {seed}')
    if seed is not None:
        tf.keras.utils.set_random_seed(seed)

    # Process the selected mode
    process_mode(cfg=cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, default='./', help='Path to folder containing configuration file')
    parser.add_argument('--config-name', type=str, default='user_config.yaml', help='name of the configuration file')
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
