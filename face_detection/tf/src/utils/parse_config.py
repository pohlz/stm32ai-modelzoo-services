#  /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
from pathlib import Path
from copy import deepcopy
from omegaconf import OmegaConf, DictConfig
from munch import DefaultMunch
import numpy as np
from hydra.core.hydra_config import HydraConfig

from common.utils import postprocess_config_dict, check_config_attributes, parse_tools_section, parse_benchmarking_section, \
                      parse_mlflow_section, parse_top_level, parse_general_section, parse_quantization_section, \
                     parse_prediction_section, parse_deployment_section, check_hardware_type, \
                      parse_evaluation_section, get_class_names_from_file, check_attributes, parse_model_section


def parse_dataset_section(cfg: DictConfig, mode: str = None,
                          mode_groups: DictConfig = None,
                          hardware_type: str = None) -> None:

    # cfg: dictionary containing the 'dataset' section of the configuration file

    legal = ["dataset_name", "class_names", "classes_file_path", "training_path", "validation_path", 
             "validation_split", "test_path", "quantization_path","prediction_path", "quantization_split", "seed"]

    required = []
    one_or_more = []
    if mode in mode_groups.training :
        required += ["training_path", ]
    elif mode in mode_groups.evaluation:
        one_or_more += ["training_path", "test_path"]
    
    check_config_attributes(cfg, specs={"legal": legal, "all": required, "one_or_more": one_or_more},
                            section="dataset")

    if not mode in ["quantization", "benchmarking", "chain_qb"]:
        one_or_more = []
        one_or_more += ["class_names", "classes_file_path"]
        check_config_attributes(cfg, specs={"legal": legal, "all": None, "one_or_more": one_or_more},
                               section="dataset")
        if cfg.class_names: 
            print("[INFO] : Using provided class names from dataset.class_names")
        elif cfg.class_names == None:
            cfg.class_names = get_class_names_from_file(cfg)
            print("[INFO] : Found {} classes in label file {}".format(len(cfg.class_names), cfg.classes_file_path))
            
    if mode in ["prediction"]:
        required += ["prediction_path",]
        # Check that the directory that contains the prediction tests files exist
        if not os.path.isdir(cfg.prediction_path):
            raise FileNotFoundError("\nUnable to find the directory containing the test files to predict\n"
                                    f"Received path: {cfg.prediction_path}\nPlease check the "
                                    "'dataset.prediction_path' attribute in your configuration file.")
    if not cfg.num_classes:
        cfg.num_classes = len(cfg.class_names) if cfg.class_names else 80
    # Set default values of missing optional attributes
    if not cfg.dataset_name:
        cfg.dataset_name = "unnamed"
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

    
    
def parse_preprocessing_section(cfg: DictConfig,
                                mode:str = None) -> None:
    # cfg: 'preprocessing' section of the configuration file
    legal = ["rescaling", "resizing", "color_mode"]
    if mode == 'deployment':
        # removing the obligation to have rescaling for the 'deployment' mode
        required=["resizing", "color_mode"]
        check_config_attributes(cfg, specs={"legal": legal, "all": required}, section="preprocessing")
    else:
        required=legal
        check_config_attributes(cfg, specs={"legal": legal, "all": required}, section="preprocessing")
        legal = ["scale", "offset"]
        check_config_attributes(cfg.rescaling, specs={"legal": legal, "all": legal}, section="preprocessing.rescaling")

    legal = ["interpolation", "aspect_ratio"]
    check_config_attributes(cfg.resizing, specs={"legal": legal, "all": legal}, section="preprocessing.resizing")
    if cfg.resizing.aspect_ratio not in ("fit", "crop", "padding"):
        raise ValueError("\nSupported methods for resizing images are 'fit', 'crop' and 'padding'. "
                         f"Received {cfg.resizing.aspect_ratio}\n"
                         "Please check the `resizing.aspect_ratio` attribute in "
                         "the 'preprocessing' section of your configuration file.")
                         
    # Check resizing interpolation value
    interpolation_methods = ["bilinear", "nearest", "area", "lanczos3", "lanczos5", "bicubic", "gaussian",
                             "mitchellcubic"]
    if cfg.resizing.interpolation not in interpolation_methods:
        raise ValueError(f"\nUnknown value for `interpolation` attribute. Received {cfg.resizing.interpolation}\n"
                         f"Supported values: {interpolation_methods}\n"
                         "Please check the 'resizing.attribute' in the 'preprocessing' section of your configuration file.")

    # Check color mode value
    color_modes = ["grayscale", "rgb", "rgba", "bgr"]
    if cfg.color_mode not in color_modes:
        raise ValueError(f"\nUnknown value for `color_mode` attribute. Received {cfg.color_mode}\n"
                         f"Supported values: {color_modes}\n"
                         "Please check the 'preprocessing' section of your configuration file.")



def _parse_postprocessing_section(cfg: DictConfig, model_type: str) -> None:
    # cfg: 'postprocessing' section of the configuration file

    legal = ["confidence_thresh", "NMS_thresh", "IoU_eval_thresh", "plot_metrics", 'max_detection_boxes', 'crop_stretch_percents']
    required = ["confidence_thresh", "NMS_thresh", "IoU_eval_thresh"]
    check_config_attributes(cfg, specs={"legal": legal, "all": required}, section="postprocessing")

    if model_type == "yunet":
        cfg.network_stride = [8,16,32]
    
    cfg.plot_metrics = cfg.plot_metrics if cfg.plot_metrics is not None else False


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
    postprocess_config_dict(config_dict, replace_none_string=True)

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
        "compression": ["compression"]
    })
    mode_choices = ["training", "evaluation", "deployment",
                    "quantization", "benchmarking", "chain_tqeb", "chain_tqe",
                    "chain_eqe", "chain_qb", "chain_eqeb", "chain_qd", "prediction", "compression"]
    legal = ["general", "model", "operation_mode", "dataset", "preprocessing", "postprocessing", "quantization", "evaluation", "prediction", "tools",
             "benchmarking", "deployment", "mlflow", "hydra"]
    parse_top_level(cfg, 
                    mode_groups=mode_groups,
                    mode_choices=mode_choices,
                    legal=legal)
    print(f"[INFO] : Running `{cfg.operation_mode}` operation mode")
    cfg.use_case = "object_detection"
    if cfg.model:
        legal = ["framework", "model_path", "model_name", "pretrained", "input_shape",
                 "depth_mul", "width_mul", "model_type"]
        required=["model_type"]
        parse_model_section(cfg.model, mode=cfg.operation_mode, mode_groups=mode_groups, legal=legal, required=required)
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
    check_hardware_type(cfg, mode_groups)
                        
    # Dataset section parsing
    if not cfg.dataset:
        cfg.dataset = DefaultMunch.fromDict({})
    
    
    parse_dataset_section(cfg.dataset,
                          mode=cfg.operation_mode,
                          mode_groups=mode_groups,
                          hardware_type=cfg.hardware_type)
                          
    # Preprocessing section parsing
    parse_preprocessing_section(cfg.preprocessing,
                                mode=cfg.operation_mode)



    # Postprocessing section parsing
    # This section is always needed except for benchmarking.
    if cfg.operation_mode in (mode_groups.training + mode_groups.evaluation +
                              mode_groups.quantization + mode_groups.deployment +
                              mode_groups.prediction + mode_groups.compression):
        if cfg.hardware_type == "MCU":
            _parse_postprocessing_section(cfg.postprocessing, cfg.model.model_type)

    # Quantization section parsing
    if cfg.operation_mode in mode_groups.quantization:
        legal = ["quantizer", "quantization_type", "quantization_input_type", "quantization_output_type", "granularity",
                 "export_dir", "optimize", "target_opset", "operating_mode",
                 "onnx_quant_parameters", "onnx_extra_options", "iterative_quant_parameters"]
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
            # Append additional items if board is "NUCLEO-H743ZI2"
            if cfg.deployment.hardware_setup.board == "NUCLEO-H743ZI2":
                raise ValueError("\n Model is not supported for deployment on H7 target")
            # Append additional items if board is "NUCLEO-N657X0-Q"
            if cfg.deployment.hardware_setup.board == "NUCLEO-N657X0-Q":
                legal_hw += ["output"]
        else:
            raise ValueError("\n Model is not supported for deployment on MPU target")
        parse_deployment_section(cfg.deployment,
                                 legal=legal,
                                 legal_hw=legal_hw)

    # MLFlow section parsing
    parse_mlflow_section(cfg.mlflow)

    return cfg
