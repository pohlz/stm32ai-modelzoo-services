#  /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
from pathlib import Path
import zipfile
from hydra.core.hydra_config import HydraConfig
from common.utils import postprocess_config_dict, check_config_attributes, parse_tools_section, parse_benchmarking_section, \
                         parse_mlflow_section, parse_top_level, parse_general_section, parse_training_section, \
                         check_hardware_type, parse_model_section
from omegaconf import OmegaConf, DictConfig
from munch import DefaultMunch


def _parse_dataset_section(cfg: DictConfig, mode: str = None, mode_groups: DictConfig = None) -> None:
    '''
    Parse and validate the `dataset` section of the configuration.

    Args:
        cfg (DictConfig): Full configuration object.
        mode (str): Selected operation mode.
        mode_groups (DictConfig): Groups of operation modes (training/evaluation/etc.).
    '''

    legal = ["dataset_name", "class_names", "training_path", "validation_path", "quantization_path", "prediction_path",
             "validation_split", "test_path", "test_split", "quantization_split", "classes_file_path", "to_cache", "seed"]

    required = []
    one_or_more = []
    if mode in mode_groups.training:
        required += ["training_path", "class_names"]
    elif mode in mode_groups.evaluation:
        one_or_more += ["test_path", "validation_path"]
    elif mode in mode_groups.prediction:
        required += ["prediction_path", "class_names"]
        
    # if normalization is enabled in preprocessing, then training_path is required
    if hasattr(cfg, 'preprocessing') and hasattr(cfg.preprocessing, 'normalization'):
        if cfg.preprocessing.normalization:
            required += ["training_path"]

    cfg = cfg.dataset
    check_config_attributes(cfg, specs={"legal": legal,
                            "all": required, "one_or_more": one_or_more},
                            section="dataset")

    # Set default values of missing optional attributes
    if not cfg.dataset_name:
        cfg.dataset_name = "<unnamed>"

    if not cfg.validation_split:
        cfg.validation_split = 0.2
    if not cfg.test_split:
        cfg.test_split = 0.2
    cfg.seed = cfg.seed if cfg.seed else 123
  
    # Check if we have at least two classes and set num_classes
    if  cfg.class_names:    
        if isinstance(cfg.class_names, list) and len(cfg.class_names) > 1:
            cfg.num_classes = len(cfg.class_names)
        else:
            raise ValueError(f"\nYour dataset must have at least two classes. Received: {len(cfg.class_names)}\n"
                                "Please check the 'dataset.class_names' section of your configuration file.")
            
    # Check the value of validation_split if it is set
    if cfg.validation_split:
        split = cfg.validation_split
        if split <= 0.0 or split >= 1.0:
            raise ValueError(f"\nThe value of `validation_split` should be > 0 and < 1. Received {split}\n"
                                "Please check the 'dataset' section of your configuration file.")

    # Check the value of test_split if it is set
    if cfg.test_split:
        split = cfg.test_split
        if split <= 0.0 or split >= 1.0:
            raise ValueError(f"\nThe value of `test_split` should be > 0 and < 1. Received {split}\n"
                                "Please check the 'dataset' section of your configuration file.")

    dataset_paths = []
    # Datasets used in a training
    if mode in mode_groups.training:
        dataset_paths += [(cfg.training_path, "training"),]
        if cfg.validation_path:
            dataset_paths += [(cfg.validation_path, "validation"),]
        if cfg.test_path:
            dataset_paths += [(cfg.test_path, "test"),]

    # Datasets used in an evaluation
    if mode in mode_groups.evaluation:
        if cfg.test_path:
            dataset_paths += [(cfg.test_path, "test"),]
        elif cfg.validation_path:
            dataset_paths += [(cfg.validation_path, "validation"),]
        else:
            dataset_paths += [(cfg.training_path, "training"),]

    # Datasets used in a quantization
    if mode in mode_groups.quantization:
        if cfg.quantization_path:
            dataset_paths += [(cfg.quantization_path, "quantization"),]
    
    # Dataset used in a prediction
    if mode in mode_groups.prediction:
        dataset_paths += [(cfg.prediction_path, "prediction"),]

    # Check that the dataset root directories exist
    for path, name in dataset_paths:
        message = f"\nPlease check the 'dataset.{name}_path' attribute in your configuration file."
        print(f"[INFO] : Checking {name} dataset path: {path}")
        if path:
            folder = os.path.dirname(path)
            zip_path = folder + '.zip'
            if not (os.path.isfile(path) or os.path.isfile(zip_path)):
                raise FileNotFoundError(f"\nUnable to find the root directory of the {name} set\n"
                                        f"Received path: {path}{message}")
            elif os.path.isfile(zip_path):
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    files_in_zip = [f for f in zip_ref.namelist() if not f.endswith('/')]
                if os.path.relpath(path, folder) not in files_in_zip:
                    raise FileNotFoundError(f"\nUnable to find the root directory of the {name} set inside the ZIP file\n"
                                            f"Received path: {path} in ZIP: {zip_path}{message}")


def _parse_preprocessing_section(cfg: DictConfig) -> None:
    '''
    Parse and validate the `preprocessing` section of the configuration.

    Args:
        cfg (DictConfig): Configuration dictionary containing the preprocessing info.
    '''

    legal = ["downsampling", "normalization", "time_domain"]
    check_config_attributes(cfg, specs={"legal": legal, "all": legal}, section="preprocessing")


def get_config(config_data: DictConfig) -> DefaultMunch:
    """
    Converts the configuration data, performs some checks and reformats
    some sections so that they are easier to use later on.

    Args:
        config_data (DictConfig): dictionary containing the entire configuration file.

    Returns:
        DefaultMunch: The configuration object.
    """

    config_dict = OmegaConf.to_container(config_data)

    # Restore booleans, numerical expressions and tuples
    # Expand environment variables
    postprocess_config_dict(config_dict)

    # Top level section parsing
    cfg = DefaultMunch.fromDict(config_dict)
    mode_groups = DefaultMunch.fromDict({
        "training": ["training", "chain_tb", "chain_tbqeb", "chain_tqe"],
        "evaluation": ["evaluation", "chain_tqe", "chain_eqe", "chain_eqeb"],
        "benchmarking": ["benchmarking", "chain_tbqeb", "chain_qb", "chain_eqeb", "chain_tb"],
        "quantization": ["quantization", "chain_tbqeb", "chain_tqe", "chain_eqe", "chain_qb", "chain_eqeb"],
        "deployment": [],
        "prediction": ["prediction"],
        "compression": []
    })
    mode_choices = [
        "training", "evaluation", "benchmarking", "chain_tb", "quantization",
        "chain_tbqeb", "chain_tqe", "chain_eqe", "chain_qb", "chain_eqeb", "prediction"]
    legal = ["general", "model", "operation_mode", "dataset", "preprocessing", "training",
             "prediction", "tools", "benchmarking", "mlflow", "quantization"]
    parse_top_level(cfg, 
                    mode_groups=mode_groups,
                    mode_choices=mode_choices,
                    legal=legal)
    print(f"[INFO] : Running `{cfg.operation_mode}` operation mode")
    cfg.use_case = "arc_fault_detection"

    # Model section parsing
    legal = ["framework", "model_name", "input_shape", "model_path"]
    required = []
    if cfg.model:
        parse_model_section(cfg.model, cfg.operation_mode, mode_groups, legal, required)

    # General section parsing
    if not cfg.general:
        cfg.general = DefaultMunch.fromDict({"project_name": "AFD"})
    legal = ["project_name", "model_path", "logs_dir", "saved_models_dir", "deterministic_ops",
            "display_figures", "global_seed", "gpu_memory_limit", "num_threads_tflite"]
    required = []
    parse_general_section(cfg.general, 
                          mode=cfg.operation_mode, 
                          mode_groups=mode_groups,
                          legal=legal,
                          required=required,
                          output_dir = HydraConfig.get().runtime.output_dir)

    # Select hardware_type from yaml information
    check_hardware_type(cfg, mode_groups)

    # Preprocessing section parsing
    if not cfg.operation_mode in mode_groups.benchmarking:
        _parse_preprocessing_section(cfg.preprocessing)

    # Dataset section parsing
    if not cfg.dataset:
        cfg.dataset = DefaultMunch.fromDict({})
    _parse_dataset_section(cfg, mode=cfg.operation_mode, mode_groups=mode_groups)
    
    # Training section parsing
    if cfg.operation_mode in mode_groups.training:
        legal = ["model", "batch_size", "epochs", "optimizer",
                 "callbacks", "initialize_parameters_from"]
        parse_training_section(cfg.training, 
                               legal=legal)

    # Tools section parsing
    if cfg.operation_mode in (mode_groups.benchmarking):
        parse_tools_section(cfg.tools, 
                            cfg.operation_mode,
                            cfg.hardware_type)

    # Benchmarking section parsing
    if cfg.operation_mode in mode_groups.benchmarking:
        parse_benchmarking_section(cfg.benchmarking)
        if cfg.hardware_type == "MPU" :
            if not (cfg.tools.stm32ai.on_cloud):
                print("Target selected for benchmark :", cfg.benchmarking.board)
                print("Offline benchmarking for MPU is not yet available please use online benchmarking")
                exit(1)

    # MLFlow section parsing
    parse_mlflow_section(cfg.mlflow)
    return cfg
