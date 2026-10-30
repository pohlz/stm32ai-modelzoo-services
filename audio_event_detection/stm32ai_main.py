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
from api import get_model, get_dataloaders, get_trainer, get_quantizer, get_evaluator, get_predictor
from common.utils import mlflow_ini, set_gpu_memory_limit, get_random_seed, display_figures, log_to_file
from common.benchmarking import benchmark
from common.benchmarking.common_benchmark import _get_credentials
from common.evaluation import gen_load_val
from common.prediction import gen_load_val_predict
from audio_event_detection.tf.src.utils import get_config, AED_CUSTOM_OBJECTS
from audio_event_detection.tf.src.deployment import deploy

from audio_event_detection.tf.src.training.select_quantized_candidate import (
    select_quantized_candidate,
)

def _process_mode(mode: str = None, cfg: DictConfig = None) -> None:
    """
    Process the selected mode of operation.

    Args:
        mode (str): The selected mode of operation. Must be one of 'train', 'evaluate', or 'predict'.
        cfg (DictConfig): The configuration object.
    
    Returns:
        None
        
    Raises:
        ValueError: If an invalid mode is selected
    """
        
    # Logging the operation_mode in the output_dir/stm32ai_main.log file
    mode = cfg.operation_mode
    mlflow.log_param("model_path", cfg.model.model_path)
    log_to_file(cfg.output_dir, f'operation_mode: {mode}')

    # # Connect to STM32Cube.AI Developer Cloud if needed
    if cfg.tools and cfg.tools.stedgeai and cfg.tools.stedgeai.on_cloud:
        credentials = _get_credentials()

    # Get model
    model = get_model(cfg=cfg)
    saved_model_dir = os.path.join(cfg.output_dir, cfg.general.saved_models_dir)

    # Get dataloaders
    if mode not in ["benchmarking", "deployment"]:
        dataloaders = get_dataloaders(cfg=cfg)

    if mode == "chain_tqe":
    cache_dir = Path(cfg.output_dir) / "selection_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for key in ("val_ds", "quantization_ds"):
        source = dataloaders.get(key)
        if source is None:
            raise ValueError(f"Candidate selection requires {key}.")

        frozen = source.cache(str(cache_dir / key))
        batch_count = 0

        # Exhaust the finite dataset to finish writing the cache.
        for _ in frozen:
            batch_count += 1

        if batch_count == 0:
            raise ValueError(f"{key} is empty.")

        dataloaders[key] = frozen

    # Execute Services
    if mode == 'training':
        # We only have 1 trainer, add it to config in case it wasn't present in the YAML
        cfg.training.trainer_name = 'aed_trainer'
        trainer = get_trainer(cfg=cfg, 
                              model=model, 
                              dataloaders=dataloaders)
        trained_model = trainer.train()
        display_figures(cfg)
        evaluator = get_evaluator(cfg=cfg, 
                                  model=trained_model, 
                                  dataloaders=dataloaders)
        evaluator.evaluate()
    elif mode == 'evaluation':
        gen_load_val(cfg=cfg, model=model)
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        evaluator = get_evaluator(cfg=cfg, 
                                  model=model, 
                                  dataloaders=dataloaders)
        evaluator.evaluate()

    elif mode == 'deployment':
        deploy(cfg=cfg, model_path_to_deploy=model.model_path)
        print('[INFO] : Deployment complete.')
        if cfg.deployment.hardware_setup.board == "STM32N6570-DK":
            print('[INFO] : If using STM32N6570-DK, please toggle the boot switches to the left and power cycle the board.')
        
    elif mode == 'quantization':
        quantizer = get_quantizer(cfg=cfg, 
                                  model=model, 
                                  dataloaders=dataloaders)
        quantized_model = quantizer.quantize()

    elif mode == 'prediction':
        gen_load_val_predict(cfg=cfg, model=model)
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        predictor = get_predictor(cfg=cfg, 
                                  model=model, 
                                  dataloaders=dataloaders)
        predictor.predict()
        
    elif mode == 'benchmarking':
        benchmark(cfg=cfg,model_path_to_benchmark=model.model_path, custom_objects=AED_CUSTOM_OBJECTS)

    elif mode == 'chain_tqe':
        # We only have 1 trainer, add it to config in case it wasn't present in the YAML
        cfg.training.trainer_name = 'aed_trainer'
        trainer = get_trainer(cfg=cfg, 
                              model=model, 
                              dataloaders=dataloaders)
        trained_model = trainer.train()
        display_figures(cfg)

	quantized_model = select_quantized_candidate(
    		cfg=cfg,
    		dataloaders=dataloaders,
	)

	# Optional final test evaluation, only after validation-based selection.
	if dataloaders.get("test_ds") is not None:
   		evaluator = get_evaluator(
        		cfg=cfg,
        		model=quantized_model,
        		dataloaders=dataloaders,
    		)
    		evaluator.evaluate()

	print("[INFO] : chain_tqe complete.")

#        evaluator = get_evaluator(cfg=cfg, 
#                                  model=trained_model, 
#                                  dataloaders=dataloaders)
#        evaluator.evaluate()
#        quantizer = get_quantizer(cfg=cfg, 
#                                  model=trained_model, 
#                                  dataloaders=dataloaders)
#        quantized_model = quantizer.quantize()
#        evaluator = get_evaluator(cfg=cfg, 
#                                  model=quantized_model, 
#                                  dataloaders=dataloaders)
#        evaluator.evaluate()
#        print('[INFO] : chain_tqe complete.')

    elif mode == 'chain_tqeb' or mode =='chain_tbqeb': # Compat with old config files
        # We only have 1 trainer, add it to config in case it wasn't present in the YAML
        cfg.training.trainer_name = 'aed_trainer'
        trainer = get_trainer(cfg=cfg, 
                              model=model, 
                              dataloaders=dataloaders)
        trained_model = trainer.train()
        display_figures(cfg)

        evaluator = get_evaluator(cfg=cfg, 
                                  model=trained_model, 
                                  dataloaders=dataloaders)
        evaluator.evaluate()
        quantizer = get_quantizer(cfg=cfg, 
                                  model=trained_model, 
                                  dataloaders=dataloaders)
        quantized_model = quantizer.quantize()
        evaluator = get_evaluator(cfg=cfg, 
                                  model=quantized_model, 
                                  dataloaders=dataloaders)
        evaluator.evaluate()
        benchmark(cfg=cfg, model_path_to_benchmark=quantized_model.model_path, credentials=credentials)
        print('[INFO] : chain_tqeb complete.')

    elif mode == 'chain_eqe':
        evaluator = get_evaluator(cfg=cfg, 
                                  model=model, 
                                  dataloaders=dataloaders)
        evaluator.evaluate()
        quantizer = get_quantizer(cfg=cfg, 
                                  model=model, 
                                  dataloaders=dataloaders)
        quantized_model = quantizer.quantize()
        evaluator = get_evaluator(cfg=cfg, 
                                  model=quantized_model, 
                                  dataloaders=dataloaders)
        evaluator.evaluate()
        print('[INFO] : chain_eqe complete.')
    elif mode == 'chain_eqeb':
        evaluator = get_evaluator(cfg=cfg, 
                                  model=model, 
                                  dataloaders=dataloaders)
        evaluator.evaluate()
        quantizer = get_quantizer(cfg=cfg, 
                                  model=model, 
                                  dataloaders=dataloaders)
        quantized_model = quantizer.quantize()
        evaluator = get_evaluator(cfg=cfg, 
                                  model=quantized_model, 
                                  dataloaders=dataloaders)
        evaluator.evaluate()
        benchmark(cfg=cfg, model_path_to_benchmark=quantized_model.model_path, credentials=credentials)
        print('[INFO] : chain_eqeb complete.')

    elif mode == 'chain_qb':
        quantizer = get_quantizer(cfg=cfg, 
                                  model=model, 
                                  dataloaders=dataloaders)
        quantized_model = quantizer.quantize()
        benchmark(cfg=cfg, model_path_to_benchmark=quantized_model.model_path, credentials=credentials)
        print('[INFO] : chain_qb complete.')
    elif mode == 'chain_qd':
        quantizer = get_quantizer(cfg=cfg, 
                                  model=model, 
                                  dataloaders=dataloaders)
        quantized_model = quantizer.quantize()
        
        deploy(cfg=cfg, model_path_to_deploy=quantized_model.model_path, credentials=credentials)
        print('[INFO] : chain_qd complete.')

    # Raise an error if an invalid mode is selected
    else: 
        raise ValueError(f"Invalid mode: {mode}")

    # Record the whole hydra working directory to get all info
    mlflow.log_artifact(cfg.output_dir)
    if mode in ['benchmarking', 'chain_qb', 'chain_eqeb', 'chain_tqeb']:
        mlflow.log_param("stedgeai_core_version", cfg.tools.stedgeai.version)
        mlflow.log_param("target", cfg.benchmarking.board)
    
    # Logging the completion of the chain
    log_to_file(cfg.output_dir, f'operation finished: {mode}')

    # ClearML
    if get_active_config_file() is not None: 
        print(f"[INFO] : ClearML task connection")
        task = Task.current_task()
        task.connect(cfg)



@hydra.main(version_base=None, config_path="", config_name="user_config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point of the script.

    Args:
        cfg (DictConfig): Configuration dictionary.

    Returns:
        None
    """

    cfg = _fw_agnostic_initializations(cfg)

    _tf_specific_initializations(cfg)

    _process_mode(cfg=cfg)

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
                         task_name='aed_modelzoo_task')
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
    parser.add_argument('params', nargs='*',
                        help='List of parameters to over-ride in config.yaml')
    args = parser.parse_args()

    # Call the main function
    main()

    # log the config_path and config_name parameters
    mlflow.log_param('config_path', args.config_path)
    mlflow.log_param('config_name', args.config_name)
    mlflow.end_run()
