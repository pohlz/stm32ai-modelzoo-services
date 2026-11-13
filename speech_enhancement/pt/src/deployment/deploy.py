# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

from pathlib import Path
from omegaconf import DictConfig
from typing import Optional
import os
from speech_enhancement.pt.src.deployment import gen_h_user_file, generate_LUT_files
from common.deployment import  stm32ai_deploy_stm32n6
from common.utils import get_model_name_and_its_input_shape, get_model_name

def deploy(cfg: DictConfig = None, model_path_to_deploy: Optional[str] = None,
           credentials: list[str] = None) -> None:
    """
    Deploy the AI model to a target device.

    Args:
        cfg (DictConfig): The configuration dictionary. Defaults to None.
        model_path_to_deploy (str, optional): Model path to deploy. Defaults to None
        credentials list[str]: User credentials for the STM32AI cloud. Unused at the moment.
            Credentials are either read from environment variables or asked in terminal.

    Returns:
        None
    """
    # Build and flash Getting Started
    board = cfg.deployment.hardware_setup.board
    stlink_serial_number = cfg.deployment.hardware_setup.stlink_serial_number
    c_project_path = cfg.deployment.c_project_path
    output_dir = cfg.output_dir
    stm32ai_output = Path(output_dir, "stm32ai_files")
    stedgeai_core_version = cfg.tools.stedgeai.version
    optimization = cfg.tools.stedgeai.optimization
    path_to_stm32ai = cfg.tools.stedgeai.path_to_stedgeai
    path_to_cube_ide = cfg.tools.path_to_cubeIDE
    verbosity = cfg.deployment.verbosity
    stm32ai_ide = cfg.deployment.IDE
    stm32ai_serie = cfg.deployment.hardware_setup.serie

    # Get model name for STM32Cube.AI STATS
    model_path = model_path_to_deploy if model_path_to_deploy else cfg.model.model_path
    model_name, input_shape = get_model_name_and_its_input_shape(model_path=model_path)

    get_model_name_output = get_model_name(model_type=str(model_name),
                                           input_shape=str(input_shape[0]),
                                           project_name=cfg.general.project_name)
    
    # Generate ai_model_config.h for C embedded application
    print("[INFO] : Generating C header file for Getting Started...")
    gen_h_user_file(config=cfg)

    print("[INFO] : Generating C LUT files for Getting Started")
    generate_LUT_files(config=cfg)


    additional_files = ["C_header/user_mel_tables.h", "C_header/user_mel_tables.c",
                        "stai_network.c",
                        "stai_network.h",
                        "network.h",
                        "network_atonbuf.xSPI2.raw"]
    
    if stm32ai_serie.upper() in ["STM32N6"] and stm32ai_ide.lower() == "gcc":
        if board == "STM32N6570-DK":
            stmaic_conf_filename = "stmaic_STM32N6570-DK.conf"
        else:
            raise TypeError("The hardware selected in cfg.deployment.hardware_setup.board is not supported yet!\n"
                            "Please choose the following boards : `STM32N6`.")

        # Run the deployment
        if board == "STM32N6570-DK":
            stm32ai_deploy_stm32n6(target=board,
                        stlink_serial_number=stlink_serial_number,
                        stedgeai_core_version=stedgeai_core_version,
                        c_project_path=c_project_path,
                        output_dir=output_dir,
                        stm32ai_output=stm32ai_output,
                        optimization=optimization,
                        path_to_stm32ai=path_to_stm32ai,
                        path_to_cube_ide=path_to_cube_ide,
                        stmaic_conf_filename=stmaic_conf_filename,
                        verbosity=verbosity,
                        debug=True,
                        model_path=model_path,
                        get_model_name_output=get_model_name_output,
                        stm32ai_ide=stm32ai_ide,
                        stm32ai_serie=stm32ai_serie, 
                        credentials=credentials,
                        additional_files=additional_files, 
                        on_cloud=cfg.tools.stedgeai.on_cloud, 
                        build_conf = cfg.deployment.build_conf)

    else:
        raise NotImplementedError(
            "Options for cfg.deployment.hardware_setup.serie \ and cfg.deployment.IDE not supported yet!\n"
            "Only options are respectively `STM32N6` and `GCC`.")


