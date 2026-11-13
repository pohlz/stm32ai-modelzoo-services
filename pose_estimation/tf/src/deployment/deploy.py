# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
import sys
import warnings
import shutil

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from typing import Optional

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from common.utils import get_model_name_and_its_input_shape, get_model_name
from common.deployment import stm32ai_deploy_stm32n6, stm32ai_deploy_mpu
from common.benchmarking import cloud_connect
from pose_estimation.tf.src.utils import gen_h_user_file_n6


def deploy(cfg: DictConfig = None, model_path_to_deploy: Optional[str] = None,
           credentials: list[str] = None) -> None:
    """
    Deploy the AI model to a target device.

    Args:
        cfg (DictConfig): The configuration dictionary. Defaults to None.
        model_path_to_deploy (str, optional): Model path to deploy. Defaults to None
        credentials list[str]: User credentials used before to connect.

    Returns:
        None
    """
    # Build and flash Getting Started
    board = cfg.deployment.hardware_setup.board
    stlink_serial_number = cfg.deployment.hardware_setup.stlink_serial_number
    c_project_path = cfg.deployment.c_project_path
    output_dir = HydraConfig.get().runtime.output_dir
    stm32ai_output = os.path.join(output_dir, "generated")
    stedgeai_core_version = cfg.tools.stedgeai.version
    optimization = cfg.tools.stedgeai.optimization
    path_to_stm32ai = cfg.tools.stedgeai.path_to_stm32ai
    path_to_cube_ide = cfg.tools.path_to_cubeIDE
    verbosity = cfg.deployment.verbosity
    stm32ai_ide = cfg.deployment.IDE
    stm32ai_serie = cfg.deployment.hardware_setup.serie
    check_large_model=False
    # Get model name for STM32Cube.AI STATS
    model_path = model_path_to_deploy if model_path_to_deploy else cfg.model.model_path
    model_name, input_shape = get_model_name_and_its_input_shape(model_path=model_path)

    get_model_name_output = get_model_name(model_type=str(model_name),
                                           input_shape=str(input_shape[0]),
                                           project_name=cfg.general.project_name)
    # Get model name
    # Generate ai_model_config.h for C embedded application
    print("[INFO] : Generating C header file for Getting Started...")
    if stm32ai_serie.upper() == "STM32H7":
        raise ValueError()
    else:
        gen_h_user_file_n6(config=cfg, quantized_model_path=model_path)

    if stm32ai_serie.upper() == "STM32N6" and stm32ai_ide.lower() == "gcc":
        if board == "STM32N6570-DK":
            stmaic_conf_filename = "stmaic_STM32N6570-DK.conf"
        elif board == "NUCLEO-N657X0-Q":
            stmaic_conf_filename = "stmaic_NUCLEO-N657X0-Q.conf"
        else:
            raise TypeError("The hardware selected in cfg.deployment.hardware_setup.board is not supported yet!\n"
                            "Please choose one of the following boards : `stmaic_STM32N6570-DK.conf` or `stmaic_NUCLEO-N657X0-Q.conf`.")

        additional_files = [
            os.path.join(output_dir, "stai_network.c"),
            os.path.join(output_dir, "stai_network.h"),
            os.path.join(output_dir, "network_atonbuf.xSPI2.raw")
        ]
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
                               debug=False,
                               model_path=model_path,
                               get_model_name_output=get_model_name_output,
                               stm32ai_ide=stm32ai_ide,
                               stm32ai_serie=stm32ai_serie,
                               on_cloud=cfg.tools.stedgeai.on_cloud,
                               build_conf = cfg.deployment.hardware_setup.output,
                               check_large_model=True,
                               cfg=cfg,
                               input_data_type='uint8',
                               output_data_type='int8',
                               inputs_ch_position='chlast',
                               outputs_ch_position='',
                               additional_files=additional_files)
    else:
        raise TypeError("Options for cfg.deployment.hardware_setup.serie and cfg.deployment.IDE not supported yet!\n"
                        "Only options are respectively `STM32N6` and `GCC`.")



def deploy_mpu(cfg: DictConfig = None, model_path_to_deploy: Optional[str] = None,
            credentials: list[str] = None) -> None:
    """
    Deploy the AI model to a MPU target device.

    Args:
        cfg (DictConfig): The configuration dictionary. Defaults to None.
        model_path_to_deploy (str, optional): Model path to deploy. Defaults to None
        credentials list[str]: User credentials used before to connect.

    Returns:
        None
    """
    # Build and flash Getting Started
    board = cfg.deployment.hardware_setup.board
    c_project_path = cfg.deployment.c_project_path
    keypoints_file_path = cfg.dataset.keypoints_file_path 
    board_deploy_path = cfg.deployment.board_deploy_path
    verbosity = cfg.deployment.verbosity
    board_serie = cfg.deployment.hardware_setup.serie
    board_ip = cfg.deployment.hardware_setup.ip_address
    optimized_model_path = c_project_path + "Optimized_models/"

    # Get model name for STM32Cube.AI STATS
    model_path = model_path_to_deploy if model_path_to_deploy else cfg.model.model_path
    model_extension = os.path.splitext(model_path)[1]
    model_name, input_shape = get_model_name_and_its_input_shape(model_path=model_path)

    if board_serie == "STM32MP2" and (model_extension == ".tflite" or model_extension == ".onnx") and cfg.tools.stedgeai.on_cloud:
        # Connect to STM32Cube.AI Developer Cloud
        login_success, ai, _ = cloud_connect(stedgeai_core_version=None, credentials=credentials)
        if login_success:
            try:
                ai.upload_model(model_path)
                model = model_name + model_extension
                res = ai.generate_nbg(model)
                ai.download_model(res, optimized_model_path + res)
                model_path=os.path.join(optimized_model_path,res)
                model_name = model_name + ".nb"
                rename_model_path=os.path.join(optimized_model_path,model_name)
                os.rename(model_path, rename_model_path)
                model_path = rename_model_path
                print("[INFO] : Optimized Model Name:", model_name)
                print("[INFO] : Optimization done ! Model available at :",optimized_model_path)

            except Exception as e:
                print(f"[FAIL] : Model optimization via Cloud failed : {e}.")
                print("[INFO] : Use default model instead of optimized ...")

    if board in ["STM32MP257F-EV1","STM32MP157F-DK2","STM32MP135F-DK"]:
        # Run the deployment
        res = stm32ai_deploy_mpu(target=board, board_ip_address=board_ip, class_names=keypoints_file_path, board_deploy=board_deploy_path, c_project_path=c_project_path,
                                    verbosity=verbosity, debug=False, model_path=model_path,cfg=cfg)
        if res == False:
            raise TypeError("Deployment on the target failed\n")
    else:
        raise TypeError("Options for cfg.deployment.hardware_setup.board and not supported !\n"
                        "Only valid options are STM32MP257F-EV1,STM32MP157F-DK2,STM32MP135F-DK \n")