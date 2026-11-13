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

import tensorflow as tf
from omegaconf import DictConfig
import mlflow
import argparse
from clearml import Task
from clearml.backend_config.defs import get_active_config_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from api import get_model, get_dataloaders, get_quantizer, get_trainer, get_evaluator, get_predictor
from common.utils import mlflow_ini, set_gpu_memory_limit, get_random_seed, display_figures, log_to_file
from common.benchmarking import benchmark, cloud_connect
from arc_fault_detection.tf.src.utils import get_config


def _process_mode(configs: DictConfig = None) -> None:
    """
    Process the selected mode of operation.

    Args:
        configs (DictConfig): The configuration object.
    Returns:
        None.
    Raises:
        ValueError: If an invalid mode is selected.
    """

    # logging the operation_mode in the output_dir/stm32ai_main.log file
    mode=configs.operation_mode
    log_to_file(configs.output_dir, f'operation_mode: {mode}')
    # Track model path 
    mlflow.log_param("model_path", configs.model.model_path)
    model = get_model(configs)
    dataloaders = get_dataloaders(cfg=configs)
    # Check the selected mode and perform the corresponding operation
    if mode == 'training':
        configs.training.trainer_name = 'afd_trainer'
        trainer = get_trainer(cfg=configs,
                              model=model,
                              dataloaders=dataloaders)
        trainer.train()
        display_figures(configs)
        print('[INFO] : Training complete.')
    elif mode == 'evaluation':
        evaluator = get_evaluator(cfg=configs,
                                  model=model,
                                  dataloaders=dataloaders)
        evaluator.evaluate()
        display_figures(configs)
        print('[INFO] : Evaluation complete.')
    elif mode == 'benchmarking':
        benchmark(cfg=configs, model_path_to_benchmark=model.model_path)
    elif mode == 'quantization':
        quantizer = get_quantizer(cfg=configs,
                                  model=model,
                                  dataloaders=dataloaders)
        quantized_model = quantizer.quantize()
        print('[INFO] : Quantization complete.')
    elif mode == "prediction":
        predictor = get_predictor(cfg=configs,
                                  model=model,
                                  dataloaders=dataloaders)
        predictor.predict()
        print('[INFO] : Prediction complete.')
    elif mode == 'chain_tb':
        chain_tb(cfg=configs, model=model, dataloaders=dataloaders)
        print('[INFO] : chain_tb complete.')
    elif mode == 'chain_tbqeb':
        chain_tbqeb(cfg=configs, model=model, dataloaders=dataloaders)
        print('[INFO] : chain_tbqeb complete.')
    elif mode == 'chain_tqe':
        chain_tqe(cfg=configs, model=model, dataloaders=dataloaders)
        print('[INFO] : chain_tqe complete.')
    elif mode == 'chain_eqe':
        chain_eqe(cfg=configs, float_model=model, dataloaders=dataloaders)
        print('[INFO] : chain_eqe complete.')
    elif mode == 'chain_qb':
        chain_qb(cfg=configs, float_model=model, dataloaders=dataloaders)
        print('[INFO] : chain_qb complete.')
    elif mode == 'chain_eqeb':
        chain_eqeb(cfg=configs, float_model=model, dataloaders=dataloaders)
        print('[INFO] : chain_eqeb complete.')
    else:
        raise ValueError(f"Invalid mode: {mode}")

    mlflow.log_artifact(configs.output_dir)
    if mode in ['benchmarking', 'chain_tb', 'chain_qb', 'chain_eqeb', 'chain_tbqeb']:
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


def chain_tb(cfg: DictConfig = None, model: tf.keras.Model = None, dataloaders: dict = None) -> None:
    """
    Runs the chain_tb pipeline, performs training and then benchmarking.

    Args:
        cfg (DictConfig): Configuration object.
        model (tf.keras.Model): Model to train.
        dataloaders (dict): Data loaders for training/validation/testing.
    Returns:
        None.
    """

    print('[INFO] : Running chain_tb')
    # Connect to STEdgeAI Developer Cloud
    credentials = None
    if cfg.tools and cfg.tools.stedgeai and cfg.tools.stedgeai.on_cloud:
        _, _, credentials = cloud_connect(stedgeai_core_version=cfg.tools.stedgeai.version)
    cfg.training.trainer_name = 'afd_trainer'
    trainer = get_trainer(cfg=cfg,
                          model=model,
                          dataloaders=dataloaders)
    trained_model = trainer.train()
    print('[INFO] : Training complete.')
    benchmark(cfg=cfg, model_path_to_benchmark=trained_model.model_path, credentials=credentials)
    

def chain_tbqeb(cfg: DictConfig = None, model: tf.keras.Model = None, dataloaders: dict = None):
    """
    Runs the chain_tbqeb pipeline: training, benchmarking, quantization, evaluation, and final benchmarking.

    Args:
        cfg (DictConfig): Configuration object.
        model (tf.keras.Model): Model to train.
        dataloaders (dict): Data loaders for all stages.
        
    Returns:
        None.
    """

    print('[INFO] : Running chain_tbqeb')
    # Connect to STEdgeAI Developer Cloud
    credentials = None
    if cfg.tools and cfg.tools.stedgeai and cfg.tools.stedgeai.on_cloud:
        _, _, credentials = cloud_connect(stedgeai_core_version=cfg.tools.stedgeai.version)

    cfg.training.trainer_name = 'afd_trainer'
    trainer = get_trainer(cfg=cfg,
                          model=model,
                          dataloaders=dataloaders)
    trained_model = trainer.train()
    print('[INFO] : Training complete.')
    benchmark(cfg=cfg, model_path_to_benchmark=trained_model.model_path, credentials=credentials)
    quantizer = get_quantizer(cfg=cfg,
                              model=trained_model,
                              dataloaders=dataloaders)
    quantized_model = quantizer.quantize()
    print('[INFO] : Quantization complete.')
    evaluator = get_evaluator(cfg=cfg,
                              model=quantized_model,
                              dataloaders=dataloaders)
    evaluator.evaluate()
    print('[INFO] : Evaluation complete.')    
    display_figures(cfg)
    benchmark(cfg=cfg, model_path_to_benchmark=quantized_model.model_path, credentials=credentials)

def chain_tqe(cfg: DictConfig = None, model: tf.keras.Model = None, dataloaders: dict = None) -> None:
    """
    Runs the chain_tqe pipeline: training, quantization, and evaluation.

    Args:
        cfg (DictConfig): Configuration object.
        model (tf.keras.Model): Model to train.
        dataloaders (dict): Data loaders for all stages.

    Returns:
        None.
    """

    print('[INFO] : Running chain_tqe')
    cfg.training.trainer_name = 'afd_trainer'
    trainer = get_trainer(cfg=cfg,
                          model=model,
                          dataloaders=dataloaders)
    trained_model = trainer.train()
    print('[INFO] : Training complete.')
    quantizer = get_quantizer(cfg=cfg,
                              model=trained_model,
                              dataloaders=dataloaders)
    quantized_model = quantizer.quantize()
    print('[INFO] : Quantization complete.')
    evaluator = get_evaluator(cfg=cfg,
                              model=quantized_model,
                              dataloaders=dataloaders)
    evaluator.evaluate()
    print('[INFO] : Evaluation complete.')
    display_figures(cfg)


def chain_eqe(cfg: DictConfig = None, float_model = None, dataloaders: dict = None) -> None:
    """
    Runs the chain_eqe pipeline: evaluates a float model, quantizes it, and evaluates the quantized model.

    Args:
        cfg (DictConfig): Configuration object.
        float_model: Model to evaluate and quantize.
        dataloaders (dict): Data loaders for all stages.

    Returns:
        None.
    """

    print('[INFO] : Running chain_eqe')
    evaluator = get_evaluator(cfg=cfg,
                              model=float_model,
                              dataloaders=dataloaders)
    evaluator.evaluate()
    print('[INFO] : Evaluation complete.')
    quantizer = get_quantizer(cfg=cfg,
                              model=float_model,
                              dataloaders=dataloaders)
    quantized_model = quantizer.quantize()
    print('[INFO] : Quantization complete.')
    evaluator = get_evaluator(cfg=cfg,
                              model=quantized_model,
                              dataloaders=dataloaders)
    evaluator.evaluate()
    print('[INFO] : Evaluation complete.')
    display_figures(cfg)


def chain_qb(cfg: DictConfig = None, float_model = None, dataloaders: dict = None) -> None:
    """
    Runs the chain_qb pipeline: quantizes a float model and benchmarks the quantized model.

    Args:
        cfg (DictConfig): Configuration object.
        float_model: Model to quantize.
        dataloaders (dict): Data loaders for quantization.
    
    Returns:
        None.
    """

    print('[INFO] : Running chain_qb')
    # Connect to STEdgeAI Developer Cloud
    credentials = None
    if cfg.tools and cfg.tools.stedgeai and cfg.tools.stedgeai.on_cloud:
        _, _, credentials = cloud_connect(stedgeai_core_version=cfg.tools.stedgeai.version)

    quantizer = get_quantizer(cfg=cfg,
                              model=float_model,
                              dataloaders=dataloaders)
    quantized_model = quantizer.quantize()
    print('[INFO] : Quantization complete.')
    benchmark(cfg=cfg, model_path_to_benchmark=quantized_model.model_path, credentials=credentials)


def chain_eqeb(cfg: DictConfig = None, float_model = None, dataloaders: dict = None) -> None:
    """
    Runs the chain_eqeb pipeline: evaluates a float model, quantizes it, evaluates the quantized model, and benchmarks it.

    Args:
        cfg (DictConfig): Configuration object.
        float_model (tf.keras.Model): Model to evaluate and quantize.
        dataloaders (dict): Data loaders for all stages.
    
    Returns:
        None.
    """

    print('[INFO] : Running chain_eqeb')
    # Connect to STEdgeAI Developer Cloud
    credentials = None
    if cfg.tools and cfg.tools.stedgeai and cfg.tools.stedgeai.on_cloud:
        _, _, credentials = cloud_connect(stedgeai_core_version=cfg.tools.stedgeai.version)

    evaluator = get_evaluator(cfg=cfg,
                              model=float_model,
                              dataloaders=dataloaders)
    evaluator.evaluate()
    print('[INFO] : Evaluation complete.')
    display_figures(cfg)
    quantizer = get_quantizer(cfg=cfg,
                              model=float_model,
                              dataloaders=dataloaders)
    quantized_model = quantizer.quantize()
    print('[INFO] : Quantization complete.')
    evaluator = get_evaluator(cfg=cfg,
                              model=quantized_model,
                              dataloaders=dataloaders)
    evaluator.evaluate()
    print('[INFO] : Evaluation complete.')
    display_figures(cfg)
    benchmark(cfg=cfg, model_path_to_benchmark=quantized_model.model_path, credentials=credentials)


@hydra.main(version_base=None, config_path="", config_name="user_config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for the script. Initializes configuration, TensorFlow settings, and runs the selected mode.

    Args:
        cfg (DictConfig): Configuration object.
    
    Returns:
        None.
    """

    cfg = _fw_agnostic_initializations(cfg)
    _tf_specific_initializations(cfg)
    _process_mode(configs=cfg)


def _fw_agnostic_initializations(cfg: DictConfig = None) -> DictConfig:
    """
    Framework-agnostic initializations.

    This function performs initializations that are independent of the specific deep learning framework being used.
    It includes parsing the configuration file, setting up MLFlow, and initializing ClearML if a valid configuration
    file is found.

    Args:
        cfg (DictConfig): Configuration object.

    Returns:
        DictConfig: Updated configuration object with initialized settings.
    """

    # Parse the configuration file and set the output directory
    cfg = get_config(cfg)
    cfg.output_dir = HydraConfig.get().run.dir

    # Initialize MLFlow
    mlflow_ini(cfg)

    # Check if there's a valid ClearML configuration file and initialize ClearML
    print(f"[INFO] : ClearML config check")
    if get_active_config_file() is not None:
        print(f"[INFO] : ClearML initialization and configuration")
        task = Task.init(project_name=cfg.general.project_name,
                         task_name='afd_modelzoo_task')
        task.connect_configuration(name=cfg.operation_mode, 
                                   configuration=cfg)
    
    return cfg


def _tf_specific_initializations(cfg: DictConfig = None) -> None:
    """
    TensorFlow-specific initializations.

    This function performs initializations specific to TensorFlow, such as configuring GPU memory limits
    and setting a random seed for reproducibility.

    Args:
        cfg (DictConfig): Configuration object.
    """

    # Check if the 'general' section exists in the configuration
    if "general" in cfg and cfg.general:
        # Set an upper limit on GPU memory usage if specified in the configuration
        if "gpu_memory_limit" in cfg.general and cfg.general.gpu_memory_limit:
            set_gpu_memory_limit(cfg.general.gpu_memory_limit)
            print(f"[INFO] : Setting upper limit of usable GPU memory to {int(cfg.general.gpu_memory_limit)}GBytes.")
        else:
            # Warn the user if GPU memory usage is unlimited
            print("[WARNING] The usable GPU memory is unlimited.\n"
                "Please consider setting the 'gpu_memory_limit' attribute "
                "in the 'general' section of your configuration file.")

    # Set a random seed for reproducibility
    seed = get_random_seed(cfg)
    print(f'[INFO] : The random seed for this simulation is {seed}')
    if seed is not None:
        tf.keras.utils.set_random_seed(seed)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, default='', help='Path to folder containing configuration file')
    parser.add_argument('--config-name', type=str, default='user_config', help='name of the configuration file')
    # add arguments to the parser
    parser.add_argument('params', nargs='*', help='List of parameters to over-ride in config.yaml')
    args = parser.parse_args()  
    # Call the main function
    main()
    # log the config_path and config_name parameters
    mlflow.log_param('config_path', args.config_path)
    mlflow.log_param('config_name', args.config_name)
    mlflow.end_run()
