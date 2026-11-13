# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
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

from common.utils import mlflow_ini, set_gpu_memory_limit, get_random_seed, display_figures, log_to_file
from common.benchmarking import benchmark, cloud_connect
from human_activity_recognition.tf.src.utils import get_config
from human_activity_recognition.tf.src.deployment import deploy
from api.api import get_model, get_dataloaders, get_trainer, get_evaluator

def chain_tb(cfg: DictConfig = None, dataloaders: dict = None, model: tf.keras.Model = None) -> None:
    """
    Runs the chain_tb pipeline, performs training and then benchmarking.

    Args:
        cfg (DictConfig): Configuration dictionary. Defaults to None.
        dataloaders (dict): Dictionary containing dataloaders. Defaults to None.
        model (tf.keras.Model): Model to be trained and benchmarked. Defaults to None.

    Returns:
        None
    """
    # Connect to STM32Cube.AI Developer Cloud
    credentials = None
    if cfg.tools.stedgeai.on_cloud:
        _, _, credentials = cloud_connect(stedgeai_core_version=cfg.tools.stedgeai.version)
    
    trainer = get_trainer(cfg=cfg, 
                          model=model, 
                          dataloaders=dataloaders)
    trained_model = trainer.train()
    evaluator = get_evaluator(cfg=cfg, 
                                  model=trained_model, 
                                  dataloaders=dataloaders)
    evaluator.evaluate()
    print('[INFO] : Training complete.')
    
    benchmark(cfg=cfg, model_path_to_benchmark=trained_model.model_path, credentials=credentials)
    print('[INFO] : benchmarking complete.')
    
    display_figures(cfg)
    

def process_mode(configs: DictConfig = None) -> None:
    """
    Process the selected mode of operation.

    Args:
        configs (DictConfig): The configuration object.
    Returns:
        None
    Raises:
        ValueError: If an invalid operation_mode is selected or if required datasets are missing.
    """
    mode = configs.operation_mode
    # logging the operation_mode in the output_dir/stm32ai_main.log file
    log_to_file(configs.output_dir, f'operation_mode: {mode}')
    mlflow.log_param("model_path", configs.model.model_path)
    
    # Always get the model object using get_model
    model = get_model(cfg=configs)
    saved_model_dir = os.path.join(configs.output_dir, configs.general.saved_models_dir)
    
    # get dataloaders
    if mode not in ["benchmarking", "deployment"]:
        dataloaders = get_dataloaders(cfg=configs)

    # Check the selected mode and perform the corresponding operation
    if mode == 'training':
        trainer = get_trainer(cfg=configs, 
                            model=model, 
                          dataloaders=dataloaders)
        trained_model = trainer.train()
        display_figures(configs)
        evaluator = get_evaluator(cfg=configs, 
                                  model=trained_model, 
                                  dataloaders=dataloaders)
        evaluator.evaluate()
        print('[INFO] : Training complete.')
    elif mode == 'evaluation':
        evaluator = get_evaluator(cfg=configs, 
                                  model=model, 
                                  dataloaders=dataloaders)
        evaluator.evaluate()
        print('[INFO] : Evaluation complete.')
    elif mode == 'deployment':
        deploy(cfg=configs, model_path_to_deploy=model.model_path )
        print('[INFO] : Deployment complete.')
    elif mode == 'benchmarking':
        benchmark(cfg=configs, model_path_to_benchmark=model.model_path)
        print('[INFO] : Benchmark complete.')
    elif mode == 'chain_tb':
        chain_tb(cfg=configs,
                 dataloaders=dataloaders,
                 model=model)
        print('[INFO] : chain_tb complete.')
    # Raise an error if an invalid mode is selected
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Record the whole hydra working directory to get all info
    mlflow.log_artifact(configs.output_dir) 
    if mode in ['benchmarking', 'chain_tb']:
        mlflow.log_param("stedgeai_core_version", configs.tools.stedgeai.version)
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
        task = Task.init(project_name=cfg.general.project_name,
                         task_name='har_modelzoo_task')
        # ClearML - Optional yaml logging 
        task.connect_configuration(name=cfg.operation_mode, 
                                   configuration=cfg)

    # Seed global seed for random generators
    seed = get_random_seed(cfg)
    print(f'[INFO] : The random seed for this simulation is {seed}')
    if seed is not None:
        tf.keras.utils.set_random_seed(seed)

    # Process the selected mode
    process_mode(configs=cfg)


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
