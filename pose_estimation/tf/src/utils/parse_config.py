# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
from copy import deepcopy
from pathlib import Path
import re
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, DictConfig
from munch import DefaultMunch
from typing import Dict

from common.utils import postprocess_config_dict, check_config_attributes, parse_tools_section, parse_benchmarking_section, \
                      parse_mlflow_section, parse_top_level, parse_general_section, parse_quantization_section, \
                      parse_training_section, parse_prediction_section, parse_deployment_section, check_hardware_type, \
                      parse_evaluation_section, check_attributes, parse_model_section


def _parse_postprocessing_section(cfg: DictConfig) -> None:
    # cfg: 'postprocessing' section of the configuration file

    legal = ["kpts_conf_thresh","confidence_thresh","NMS_thresh","max_detection_boxes","plot_metrics"]
    required = []

    check_config_attributes(cfg, specs={"legal": legal, "all": required}, section="postprocessing")

    cfg.plot_metrics = cfg.plot_metrics if cfg.plot_metrics is not None else False


def _parse_dataset_section(cfg: DictConfig, hardware_type:str, mode: str = None, mode_groups: DictConfig = None) -> None:
    # cfg: dictionary containing the 'dataset' section of the configuration file
    legal = ["dataset_name", "keypoints", "keypoints_file_path", "training_path", "validation_path", "validation_split", "test_path",
             "quantization_path", "quantization_split", "prediction_path", "seed", "class_names"]

    required = []
    one_or_more = []
    if mode in mode_groups.training:
        required += ["training_path", "keypoints", ]
    elif mode in mode_groups.evaluation:
        one_or_more += ["training_path", "test_path", "keypoints", ]
    if mode not in ["quantization", "benchmarking", "chain_qb", "deployment", "chain_qd"]:
        required += ["keypoints", ]
    if mode in mode_groups.deployment and hardware_type == "MPU":
        required += ["keypoints_file_path"]
    check_config_attributes(cfg, specs={"legal": legal, "all": required, "one_or_more": one_or_more},
                            section="dataset")

    # Set default values of missing optional attributes
    if not cfg.dataset_name:
        cfg.dataset_name = "<unnamed>"
    if not cfg.validation_split:
        cfg.validation_split = 0.2
    cfg.seed = cfg.seed if cfg.seed else 123

    # Check the value of validation_split if it is set
    if cfg.validation_split:
        split = cfg.validation_split
        if split <= 0.0 or split >= 1.0:
            raise ValueError(f"\nThe value of `validation_split` should be > 0 and < 1. Received {split}\n"
                             "Please check the 'dataset' section of your configuration file.")

    # Check the value of quantization_split if it is set
    if cfg.quantization_split:
        split = cfg.quantization_split
        if split <= 0.0 or split > 1.0:
            raise ValueError(f"\nThe value of `quantization_split` should be > 0 and <= 1. Received {split}\n"
                             "Please check the 'dataset' section of your configuration file.")


def _parse_preprocessing_section(cfg: DictConfig, mode:str = None) -> None:
    # cfg: 'preprocessing' section of the configuration file
    legal = ["rescaling", "resizing", "color_mode"]
    if mode == 'deployment':
        # removing the obligation to have rescaling for the 'deployment' mode
        required=["resizing", "color_mode"]
        check_config_attributes(cfg.preprocessing, specs={"legal": legal, "all": required}, section="preprocessing")
    else:
        required=legal
        check_config_attributes(cfg.preprocessing, specs={"legal": legal, "all": required}, section="preprocessing")
        legal = ["scale", "offset"]
        check_config_attributes(cfg.preprocessing.rescaling, specs={"legal": legal, "all": legal}, section="preprocessing.rescaling")

    legal = ["interpolation", "aspect_ratio"]
    check_config_attributes(cfg.preprocessing.resizing, specs={"legal": legal, "all": legal}, section="preprocessing.resizing")

    if cfg.hardware_type == "MCU":
        if cfg.preprocessing.resizing.aspect_ratio not in ("fit", "crop", "padding"):
            raise ValueError("\nSupported methods for resizing images are 'fit', 'crop' and 'padding'. "
                            f"Received {cfg.preprocessing.resizing.aspect_ratio}\n"
                            "Please check the `resizing.aspect_ratio` attribute in "
                            "the 'preprocessing' section of your configuration file.")

    elif cfg.hardware_type == "MPU":
        if cfg.preprocessing.resizing.aspect_ratio not in ["fit","padding"]:
            raise ValueError("The only value of aspect_ratio that are supported at this point are 'fit' and 'padding'"
                            "('crop' is not supported).")
    else:
        raise ValueError("The only value of aspect_ratio that are supported at this point are 'fit' and 'padding'"
                "('crop' is not supported).")

    # Check resizing interpolation value
    interpolation_methods = ["bilinear", "nearest", "area", "lanczos3", "lanczos5", "bicubic", "gaussian",
                             "mitchellcubic"]
    if cfg.preprocessing.resizing.interpolation not in interpolation_methods:
        raise ValueError(f"\nUnknown value for `interpolation` attribute. Received {cfg.preprocessing.resizing.interpolation}\n"
                         f"Supported values: {interpolation_methods}\n"
                         "Please check the 'preprocessing.resizing' section of your configuration file.")

    # Check color mode value
    color_modes = ["grayscale", "rgb", "rgba"]
    if cfg.preprocessing.color_mode not in color_modes:
        raise ValueError(f"\nUnknown value for `color_mode` attribute. Received {cfg.preprocessing.color_mode}\n"
                         f"Supported values: {color_modes}\n"
                         "Please check the 'preprocessing' section of your configuration file.")


def _parse_data_augmentation_section(cfg: DictConfig) -> None:
    """
    This function checks the data augmentation section of the config file.
    The attribute that introduces the section is either `data_augmentation`
    or `custom_data_augmentation`. If it is `custom_data_augmentation`,
    the name of the data augmentation function that is provided must be
    different from `data_augmentation` as this is a reserved name.

    Arguments:
        cfg (DictConfig): The entire configuration file as a DefaultMunch dictionary.
        config_dict (Dict): The entire configuration file as a regular Python dictionary.

    Returns:
        None
    """

    if cfg.data_augmentation and cfg.custom_data_augmentation:
        raise ValueError("\nThe `data_augmentation` and `custom_data_augmentation` attributes "
                         "are mutually exclusive.\nPlease check your configuration file.")
    
    if cfg.data_augmentation:
        data_aug = DefaultMunch.fromDict({})
        # The name of the Model Zoo data augmentation function is 'data_augmentation'.
        data_aug.function_name = "data_augmentation"
        data_aug.config = deepcopy(cfg.data_augmentation)

    if cfg.custom_data_augmentation:
        check_attributes(cfg.custom_data_augmentation,
                         expected=["function_name"],
                         optional=["config"],
                         section="custom_data_augmentation")
        if cfg.custom_data_augmentation["function_name"] == "data_augmentation":
            raise ValueError("\nThe function name `data_augmentation` is reserved.\n"
                             "Please use another name (attribute `function_name` in "
                             "the 'custom_data_augmentation' section).")
                                                          
        data_aug = DefaultMunch.fromDict({})
        data_aug.function_name = cfg.custom_data_augmentation.function_name
        if cfg.custom_data_augmentation.config:
            data_aug.config = deepcopy(cfg.custom_data_augmentation.config)
    
    cfg.data_augmentation = data_aug


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
        "training": ["training", "chain_tqeb", "chain_tqe"],
        "evaluation": ["evaluation", "chain_tqeb", "chain_tqe", "chain_eqe", "chain_eqeb"],
        "quantization": ["quantization", "chain_tqeb", "chain_tqe", "chain_eqe",
                         "chain_qb", "chain_eqeb", "chain_qd"],
        "benchmarking": ["benchmarking", "chain_tqeb", "chain_qb", "chain_eqeb"],
        "deployment": ["deployment", "chain_qd"],
        "prediction": ["prediction"],
        "compression": []
    })
    mode_choices = ["training", "evaluation", "deployment",
                    "quantization", "benchmarking", "chain_tqeb", "chain_tqe",
                    "chain_eqe", "chain_qb", "chain_eqeb", "chain_qd", "prediction"]
    legal = ["general","model", "operation_mode", "dataset", "preprocessing", "data_augmentation",
             "training", "postprocessing", "quantization", "evaluation", "prediction", "tools",
             "benchmarking", "deployment", "mlflow", "hydra"]
    parse_top_level(cfg,
                    mode_groups=mode_groups,
                    mode_choices=mode_choices,
                    legal=legal)
    print(f"[INFO] : Running `{cfg.operation_mode}` operation mode")
    cfg.use_case = "pose_estimation"
    if cfg.model:
        legal = ["framework", "model_path", "model_type", "model_name", "pretrained", "input_shape", "final_activation"]
        parse_model_section(cfg.model, mode=cfg.operation_mode, mode_groups=mode_groups, legal=legal, required=["model_type"])
    # General section parsing
    if not cfg.general:
        cfg.general = DefaultMunch.fromDict({})
    legal = ["project_name", "logs_dir", "saved_models_dir", "deterministic_ops",
            "display_figures", "global_seed", "gpu_memory_limit", "num_threads_tflite"]
    required = []
    parse_general_section(cfg.general,
                          mode=cfg.operation_mode,
                          mode_groups=mode_groups,
                          legal=legal,
                          required=required,
                          output_dir = HydraConfig.get().runtime.output_dir)

    # Select hardware_type from yaml information
    check_hardware_type(cfg,
                        mode_groups)

    # Dataset section parsing
    if not cfg.dataset:
        cfg.dataset = DefaultMunch.fromDict({})
    _parse_dataset_section(cfg.dataset, cfg.hardware_type,
                          mode=cfg.operation_mode,
                          mode_groups=mode_groups,)

    # Preprocessing section parsing
    _parse_preprocessing_section(cfg,
                                mode=cfg.operation_mode)

    # # Training section parsing
    if cfg.operation_mode in mode_groups.training:
        if cfg.data_augmentation or cfg.custom_data_augmentation:
            _parse_data_augmentation_section(cfg)
        if cfg.model.framework == "tf":
            legal = ["batch_size", "epochs", "optimizer", "dropout", "frozen_layers",
                 "callbacks", "dryrun", 'trainer_name']
        cfg.training.trainer_name = "pe_trainer"
        parse_training_section(cfg.training,
                               legal=legal)

    # Postprocessing section parsing
    #if cfg.operation_mode in (mode_groups.training + mode_groups.evaluation + mode_groups.quantization + mode_groups.deployment + mode_groups.prediction):
    if cfg.operation_mode in (mode_groups.prediction):
        _parse_postprocessing_section(cfg.postprocessing)

    # Quantization section parsing
    if cfg.operation_mode in mode_groups.quantization:
        legal = ["quantizer", "quantization_type", "quantization_input_type", "quantization_output_type", "granularity",
                 "export_dir", "optimize", "target_opset", "onnx_extra_options"]
        parse_quantization_section(cfg.quantization,
                                   legal=legal)

    # Evaluation section parsing
    if cfg.operation_mode in mode_groups.evaluation:
        if not "evaluation" in cfg:
            cfg.evaluation = DefaultMunch.fromDict({})
        legal = ["gen_npy_input", "gen_npy_output", "npy_in_name", "npy_out_name", "target", 
                 "profile", "input_type", "output_type", "input_chpos", "output_chpos"]
        parse_evaluation_section(cfg.evaluation,
                                 legal=legal)

    # Prediction section parsing
    if cfg.operation_mode == "prediction":
        if not "prediction" in cfg:
            cfg.prediction = DefaultMunch.fromDict({})
        parse_prediction_section(cfg.prediction)

    # Tools section parsing
#    if cfg.operation_mode in (mode_groups.benchmarking + mode_groups.deployment) \
#        or cfg.operation_mode == "evaluation" \
#        or cfg.operation_mode == "prediction":
    if (
        cfg.operation_mode in (mode_groups.benchmarking + mode_groups.deployment)
        or (
            cfg.operation_mode == "evaluation"
            and "evaluation" in cfg
            and "target" in cfg.evaluation
            and cfg.evaluation.target != "host"
        )
        or (
            cfg.operation_mode == "prediction"
            and "prediction" in cfg
            and "target" in cfg.prediction
            and cfg.prediction.target != "host"
        )
    ):
        parse_tools_section(cfg.tools, 
                            cfg.operation_mode,
                            cfg.hardware_type)

    #For MPU, check if online benchmarking is activated
    if cfg.operation_mode in mode_groups.benchmarking:
        if "STM32MP" in cfg.benchmarking.board :
            if cfg.operation_mode == "benchmarking" and not(cfg.tools.stedgeai.on_cloud):
                print("Target selected for benchmark :", cfg.benchmarking.board)
                print("Offline benchmarking for MPU is not yet available please use online benchmarking")
                exit(1)

    # Benchmarking section parsing
    if cfg.operation_mode in mode_groups.benchmarking:
        parse_benchmarking_section(cfg.benchmarking)
        if cfg.hardware_type == "MPU" :
            if not (cfg.tools.stedgeai.on_cloud):
                print("Target selected for benchmark :", cfg.benchmarking.board)
                print("Offline benchmarking for MPU is not yet available please use online benchmarking")
                exit(1)

    # Deployment section parsing
    if cfg.operation_mode in mode_groups.deployment:
        if cfg.hardware_type == "MCU":
            legal = ["c_project_path", "IDE", "verbosity", "hardware_setup"]
            legal_hw = ["serie", "board", "stlink_serial_number"]
            # Append additional items if board is "NUCLEO-N657X0-Q"
            if cfg.deployment.hardware_setup.board == "NUCLEO-N657X0-Q":
                legal_hw += ["output"]
            parse_deployment_section(cfg.deployment,
                                    legal=legal,
                                    legal_hw=legal_hw)
        else:
            legal = ["c_project_path", "board_deploy_path", "verbosity", "hardware_setup"]
            legal_hw = ["serie", "board", "ip_address", "stlink_serial_number"]
            if cfg.preprocessing.color_mode != "rgb":
                raise ValueError("\n Color mode used is not supported for deployment on MPU target \n Please use RGB format")
            if cfg.preprocessing.resizing.aspect_ratio != "fit":
                raise ValueError("\n Aspect ratio used is not supported for deployment on MPU target \n Please use FIT aspect ratio")
            parse_deployment_section(cfg.deployment,
                                    legal=legal,
                                    legal_hw=legal_hw)

    # MLFlow section parsing
    parse_mlflow_section(cfg.mlflow)

    return cfg
