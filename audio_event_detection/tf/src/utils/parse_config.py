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
import re
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, DictConfig
from munch import DefaultMunch
import tensorflow as tf
import pandas as pd
from typing import Dict, List

from common.utils import replace_none_string, postprocess_config_dict, check_config_attributes, parse_tools_section, \
                      parse_benchmarking_section, parse_mlflow_section, parse_top_level, parse_general_section, \
                      parse_training_section, parse_quantization_section, parse_prediction_section, parse_deployment_section, \
                      check_hardware_type, parse_evaluation_section, get_class_names_from_file, parse_model_section

def _parse_dataset_section(cfg: DictConfig, mode: str = None, mode_groups: DictConfig = None) -> None:
    # cfg: dictionary containing the 'dataset' section of the configuration file

    legal = ["dataset_name", "training_audio_path", "training_csv_path", "multi_label", "use_garbage_class", 
             "expand_last_dim", "file_extension", "to_cache", "shuffle", "batch_size",  "class_names",
             "classes_file_path", "validation_audio_path", "validation_csv_path", "validation_split",
             "test_audio_path", "test_csv_path", "quantization_audio_path", "quantization_csv_path", "prediction_audio_path",
             "quantization_split", "n_samples_per_garbage_class", "seed"]

    required = ["dataset_name", "multi_label", "use_garbage_class", "file_extension",
                "to_cache", "shuffle", "n_samples_per_garbage_class", "seed"]
    one_or_more = []
    if mode in mode_groups.training and cfg.dataset_name != "fsd50k":
        required += ["training_csv_path", "training_audio_path"]
    elif mode in mode_groups.evaluation and cfg.dataset_name != "fsd50k":
        one_or_more += ["training_csv_path", "test_csv_path"]
        # A second check explicitly for the audio paths
        check_config_attributes(cfg, specs={"legal":legal, "all":None,
                                           "one_or_more":["training_audio_path", "test_audio_path"]},
                                section="dataset")

    check_config_attributes(cfg, specs={"legal": legal, "all": required, "one_or_more": one_or_more},
                            section="dataset")
    # A third check for the class_names
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
        cfg.class_names = sorted(cfg.class_names)    
        cfg.num_classes = len(cfg.class_names) if cfg.class_names else None

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

    # Check that the training, evaluation, test and quantization sets
    # root directories exist in the attributes are set
    dataset_audio_paths = [(cfg.training_audio_path, "training audio"),
                           (cfg.validation_audio_path, "validation audio"),
                           (cfg.test_audio_path, "test audio"),
                           (cfg.quantization_audio_path, "quantization audio")]
        
    dataset_csv_paths =[(cfg.training_csv_path, "training csv file"),
                        (cfg.validation_csv_path, "validation csv file"),
                        (cfg.test_csv_path, "test csv file"),
                        (cfg.quantization_csv_path, "quantization csv file")]
    for path, name in dataset_audio_paths:
        if path and not os.path.isdir(path):
            raise FileNotFoundError(f"\nUnable to find the directory of {name}\n"
                                    f"Received path: {path}\n"
                                    "Please check the 'dataset' section of your configuration file.")
    for path, name in dataset_csv_paths:
        if path and not os.path.isfile(path):
            raise FileNotFoundError(f"\nUnable to find the {name}\n"
                                    f"Received path: {path}\n"
                                    "Please check the 'dataset' section of your configuration file.") 

def _parse_dataset_specific_fsd50k_section(cfg: DictConfig) -> None:
    # cfg: 'preprocessing' section of the configuration file
    legal = ["csv_folder", "dev_audio_folder", "eval_audio_folder", "audioset_ontology_path",
             "only_keep_monolabel"]
    # All are required if this function is called
    check_config_attributes(cfg, specs={"legal": legal, "all": legal}, section="dataset_specific.fsd50k")

def _parse_preprocessing_section(cfg: DictConfig) -> None:
    # cfg: 'preprocessing' section of the configuration file
    legal = ["gsc_default", "target_rate", "sample_length", "random_position",
             "time_shift_ms", "background_noise_path", "background_frequency",
             "background_volume", "reserved_background_files",
             "min_length", "max_length", "top_db", "frame_length", "hop_length",
             "trim_last_second", "lengthen"]

    if cfg.get("gsc_default", False):
        required = ["gsc_default", "target_rate", "sample_length", "random_position",
                    "time_shift_ms", "background_noise_path", "background_frequency",
                    "background_volume", "reserved_background_files"]
    else:
        required = ["min_length", "max_length", "target_rate", "top_db",
                    "frame_length", "hop_length", "trim_last_second", "lengthen"]

    check_config_attributes(
        cfg,
        specs={"legal": legal, "all": required},
        section="preprocessing"
    )

    if cfg.get("gsc_default", False):
        if cfg.target_rate != 16000:
            raise ValueError("GSC default preprocessing requires target_rate: 16000")
        if cfg.sample_length <= 0:
            raise ValueError("preprocessing.sample_length must be positive")
        if cfg.time_shift_ms < 0:
            raise ValueError("preprocessing.time_shift_ms must not be negative")
        if not 0.0 <= cfg.background_frequency <= 1.0:
            raise ValueError("preprocessing.background_frequency must be between 0 and 1")
        if not 0.0 <= cfg.background_volume <= 1.0:
            raise ValueError("preprocessing.background_volume must be between 0 and 1")
        if not cfg.reserved_background_files:
            raise ValueError(
                "preprocessing.reserved_background_files must reserve at least one WAV"
            )
        if not os.path.isdir(cfg.background_noise_path):
            raise FileNotFoundError(
                "Unable to find preprocessing.background_noise_path: "
                f"{cfg.background_noise_path}"
            )

def _parse_feature_extraction_section(cfg: DictConfig) -> None:
    # cfg: 'feature_extraction' section of the config file

    legal = ["patch_length", "n_mels", "overlap", "n_fft",
             "hop_length", "window_length", "window", "center",
             "pad_mode", "power", "fmin", "fmax", "norm",
             "htk", "to_db", "include_last_patch"]
    required = legal[:].remove('include_last_patch')
    # all are required
    check_config_attributes(cfg, specs={"legal": legal, "all": required}, section="feature_extraction")
    replace_none_string(cfg)

def _parse_data_augmentation_section(cfg: DictConfig) -> None:
    """
    This function checks the data augmentation section of the config file.
    
    Arguments:
        cfg (DictConfig): The entire configuration file as a DefaultMunch dictionary.

    Returns:
        None
    """
    # Check top-level data augmentation attributes, then check for each valid attribute.
    legal = ["GaussianNoise", "VolumeAugment", "SpecAug"]
    check_config_attributes(cfg.data_augmentation,
                            specs={"legal": legal, "all": legal},
                            section="data_augmentation")
    legal = ["enable", "scale"]
    all = ["enable"]
    check_config_attributes(cfg.data_augmentation.GaussianNoise,
                            specs={"legal": legal, "all": all},
                            section="data_augmentation.GaussianNoise")
    
    legal = ["enable", "min_scale", "max_scale"]
    all = ["enable"]
    check_config_attributes(cfg.data_augmentation.VolumeAugment,
                            specs={"legal": legal, "all": all},
                            section="data_augmentation.VolumeAugment")
    legal = ["enable","freq_mask_param", "time_mask_param",
             "n_freq_mask", "n_time_mask", "mask_value"]
    all = ["enable"]
    check_config_attributes(cfg.data_augmentation.SpecAug,
                            specs={"legal": legal, "all": all},
                            section="data_augmentation.SpecAug")


def _get_class_names(cfg: DictConfig, dataset_name: str, csv_path=None):
    '''Attemps to get class names for a dataset.
       If the dataset name is not previously known, attemps to get the class names 
       from the dataset's associated csv file. Expects that this csv file is in ESC-10 format.
    '''
    if dataset_name.lower() == "esc10":
        return ['dog', 'chainsaw', 'crackling_fire', 'helicopter', 'rain',
                'crying_baby', 'clock_tick', 'sneezing', 'rooster', 'sea_waves']
    if dataset_name.lower() == "fsd50k":
        # grab them from the vocabulary.csv
        vocab_path = os.path.join(cfg.dataset_specific.fsd50k.csv_folder, "vocabulary.csv")
        if not os.isfile(vocab_path):
            raise FileNotFoundError(f"Tried to infer class names for FSD50K dataset,\
                                     but could not find vocabulary file at {vocab_path}")
        vocab = pd.read_csv(vocab_path, header=None, names= ['id', 'name', 'mids'])
        return vocab["name"].unique().tolist()
    elif csv_path is not None:
        df = pd.read_csv(csv_path)
        return df["category"].unique().tolist()
    else:
        return None # This will raise the appropriate exception in get_config  


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
        "training": ["training", "chain_tbqeb", "chain_tqe"],
        "evaluation": ["evaluation", "chain_tbqeb", "chain_tqe", "chain_eqe", "chain_eqeb"],
        "quantization": ["quantization", "chain_tbqeb", "chain_tqe", "chain_eqe",
                         "chain_qb", "chain_eqeb", "chain_qd"],
        "benchmarking": ["benchmarking", "chain_tbqeb", "chain_qb", "chain_eqeb"],
        "deployment": ["deployment", "chain_qd"],
        "compression": []
    })
    mode_choices = ["training", "evaluation", "prediction", "deployment", 
                "quantization", "benchmarking", "chain_tbqeb", "chain_tqe", "chain_tqeb",
                "chain_eqe", "chain_qb", "chain_eqeb", "chain_qd"]
    legal = ["general", "model", "operation_mode", "dataset", "preprocessing", "feature_extraction", 
             "data_augmentation", "custom_data_augmentation", "training",
             "quantization", "evaluation", "prediction", "tools", "dataset_specific",
             "evaluation", "benchmarking", "deployment", "mlflow", "hydra"]
    parse_top_level(cfg, 
                    mode_groups=mode_groups,
                    mode_choices=mode_choices,
                    legal=legal)
    print(f"[INFO] : Running `{cfg.operation_mode}` operation mode")

    # General section parsing
    if not cfg.general:
        cfg.general = DefaultMunch.fromDict({"project_name": "<unnamed>"})
    legal = ["project_name", "model_path", "logs_dir", "saved_models_dir", "deterministic_ops", 
             "display_figures", "global_seed", "gpu_memory_limit", "batch_size", "num_threads_tflite"]
    required = []
    cfg.use_case = "audio_event_detection"
    parse_general_section(cfg.general, 
                          mode=cfg.operation_mode, 
                          mode_groups=mode_groups,
                          legal=legal,
                          required=required,
                          output_dir = HydraConfig.get().runtime.output_dir)

    # Model section parsing
    if cfg.model:
        legal=["framework", "model_path", "model_name", "input_shape", "pretrained", "model_type"]
        parse_model_section(cfg.model, cfg.operation_mode, mode_groups, legal=legal, required=[])

    # Select hardware_type from yaml information
    check_hardware_type(cfg,
                        mode_groups)

    # Dataset section parsing
    if cfg.operation_mode != "benchmarking":
        if not cfg.dataset:
            cfg.dataset = DefaultMunch.fromDict({})
        _parse_dataset_section(cfg.dataset, 
                              mode=cfg.operation_mode, 
                              mode_groups=mode_groups)
        # If dataset is FSD50K, parse its dedicated section
        # No need to have this section if all we're doing is deploying,
        if cfg.dataset.dataset_name.lower() == 'fsd50k' and cfg.operation_mode != "deployment":
            _parse_dataset_specific_fsd50k_section(cfg.dataset_specific.fsd50k)

        _parse_preprocessing_section(cfg.preprocessing)
        _parse_feature_extraction_section(cfg.feature_extraction)

    # Training section parsing
    if cfg.operation_mode in mode_groups.training:
        if cfg.data_augmentation:
            _parse_data_augmentation_section(cfg)
        legal = [ "batch_size", "epochs", "optimizer", "dropout", "frozen_layers",
            "callbacks", "resume_training", "fine_tune", "dryrun"]
        parse_training_section(cfg.training, 
                               legal=legal)

    # Quantization section parsing
    if cfg.operation_mode in mode_groups.quantization:
        legal = ["quantizer", "quantization_type", "quantization_input_type",
                "quantization_output_type", "export_dir", "granularity", "target_opset", "optimize",
                "operating_mode", "onnx_quant_parameters", "onnx_extra_options", "iterative_quant_parameters"]
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
        legal = ["c_project_path", "IDE", "verbosity", "hardware_setup","build_conf","unknown_class_threshold"]
        legal_hw = ["serie", "board", "stlink_serial_number"]
        parse_deployment_section(cfg.deployment,
                                 legal=legal,
                                 legal_hw=legal_hw)

    # MLFlow section parsing
    parse_mlflow_section(cfg.mlflow)

    # Check that all datasets have the required directory structure
    # Also check that all image files can be loaded if requested
    cds = cfg.dataset
    if cds: # All this breaks if no dataset section is present
        if not cds.class_names and cfg.operation_mode not in ("quantization", "benchmarking", "chain_qb"):
            # Infer the class names from a dataset if there is one
            for path in [cds.training_csv_path, cds.validation_csv_path, cds.test_csv_path, cds.quantization_csv_path]:
                cds.class_names = _get_class_names(cfg, dataset_name=cds.dataset_name, csv_path=path)
                print(f"[INFO] : Found {len(cds.class_names)} classes in dataset {path}")
                print(f"[INFO] : Automatically inferred classes {cds.class_names}")
                break
            if not cds.class_names:
                raise ValueError("\nMissing the class names. You can provide in the 'dataset' section either:\n"
                                    " - the class names using the `class_names` attribute\n"
                                    " - a path to a dataset using the `training_path`, `validation_path`, `test_path` or `quantization_path` attribute\n" 
                                    "Please update your configuration file.")

    return cfg
