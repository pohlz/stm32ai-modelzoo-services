# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import os
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import numpy as np
import tensorflow as tf

from common.utils import aspect_ratio_dict, color_mode_n6_dict
from .connections import skeleton_connections_dict


def gen_h_user_file_n6(config: DictConfig = None, quantized_model_path: str = None) -> None:
    """
    Generates a C header file containing user configuration for the AI model.

    Args:
        config: A configuration object containing user configuration for the AI model.
        quantized_model_path: The path to the quantized model file.

    """
    class Flags:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    params = Flags(**config)

    if quantized_model_path.lower().endswith('.tflite'):
        interpreter_quant = tf.lite.Interpreter(model_path=quantized_model_path)
        input_details = interpreter_quant.get_input_details()[0]
        output_details = interpreter_quant.get_output_details()[0]
        input_shape = input_details['shape']
    elif quantized_model_path.lower().endswith('.onnx'):
        import onnxruntime


        params = config

        model = onnxruntime.InferenceSession(quantized_model_path)
        inputs  = model.get_inputs()
        outputs = model.get_outputs()
        input_shape_raw = inputs[0].shape
        input_shape = [1,input_shape_raw[2],input_shape_raw[3],input_shape_raw[1]]
    else:
        raise TypeError("Please provide a TFLITE or ONNX model for N6 deployment")
    
    class_name = config.dataset.class_names[0] if config.dataset.class_names is not None else 'None'

    path = os.path.join(HydraConfig.get().runtime.output_dir, "C_header/")

    try:
        os.mkdir(path)
    except OSError as error:
        print(error)

    TFLite_Detection_PostProcess_id = False

    with open(os.path.join(path, "app_config.h"), "wt") as f:
        f.write("/**\n")
        f.write("******************************************************************************\n")
        f.write("* @file    app_config.h\n")
        f.write("* @author  GPM Application Team\n")
        f.write("*\n")
        f.write("******************************************************************************\n")
        f.write("* @attention\n")
        f.write("*\n")
        f.write("* Copyright (c) 2023 STMicroelectronics.\n")
        f.write("* All rights reserved.\n")
        f.write("*\n")
        f.write("* This software is licensed under terms that can be found in the LICENSE file\n")
        f.write("* in the root directory of this software component.\n")
        f.write("* If no LICENSE file comes with this software, it is provided AS-IS.\n")
        f.write("*\n")
        f.write("******************************************************************************\n")
        f.write("*/\n\n")
        f.write("/* ---------------    Generated code    ----------------- */\n")
        f.write("#ifndef APP_CONFIG\n")
        f.write("#define APP_CONFIG\n\n")
        f.write('#include "stai_network.h"\n')
        f.write('#include "arm_math.h"\n')
        f.write('#include "stm32_lcd.h"\n\n')
        f.write("#define USE_DCACHE\n\n")
        f.write("/*Defines: CMW_MIRRORFLIP_NONE; CMW_MIRRORFLIP_FLIP; CMW_MIRRORFLIP_MIRROR; CMW_MIRRORFLIP_FLIP_MIRROR;*/\n")
        f.write("#define CAMERA_FLIP CMW_MIRRORFLIP_NONE\n\n")
        f.write("")
        f.write("#define ASPECT_RATIO_CROP (1) /* Crop both pipes to nn input aspect ratio; Original aspect ratio kept */\n")
        f.write("#define ASPECT_RATIO_FIT (2) /* Resize both pipe to NN input aspect ratio; Original aspect ratio not kept */\n")
        f.write("#define ASPECT_RATIO_FULLSCREEN (3) /* Resize camera image to NN input size and display a fullscreen image */\n")
        f.write("#define ASPECT_RATIO_MODE {}\n".format(aspect_ratio_dict[params.preprocessing.resizing.aspect_ratio]))
        f.write("\n")

        f.write("/* Postprocessing type configuration */\n")

        if params.model.model_type == "heatmaps_spe":
            f.write("#define POSTPROCESS_TYPE POSTPROCESS_SPE_MOVENET_UI\n\n")
        elif params.model.model_type == "yolo_mpe":
            f.write("#define POSTPROCESS_TYPE POSTPROCESS_MPE_YOLO_V8_UI\n\n")
        else:
            raise TypeError("Please select one of the supported post processing options: heatmaps_spe or yolo_mpe")

        f.write("#define COLOR_BGR (0)\n")
        f.write("#define COLOR_RGB (1)\n")
        f.write("#define COLOR_MODE    {}\n".format(color_mode_n6_dict[params.preprocessing.color_mode]))
        f.write("\n/* Post processing values */\n")

        if params.model.model_type == "heatmaps_spe":
            nb_kpt = int(output_details['shape'][3])
            f.write("#define AI_POSE_PP_CONF_THRESHOLD                    ({})\n".format(float(params.postprocessing.kpts_conf_thresh)))
            f.write("#define AI_POSE_PP_POSE_KEYPOINTS_NB                 ({})\n".format(nb_kpt))
            f.write("#define AI_SPE_MOVENET_POSTPROC_HEATMAP_WIDTH        (STAI_NETWORK_IN_1_WIDTH/4)\n")
            f.write("#define AI_SPE_MOVENET_POSTPROC_HEATMAP_HEIGHT       (STAI_NETWORK_IN_1_HEIGHT/4)\n")
            f.write("#define AI_SPE_MOVENET_POSTPROC_NB_KEYPOINTS         (AI_POSE_PP_POSE_KEYPOINTS_NB)\n\n")
        elif params.model.model_type == "yolo_mpe":
            if quantized_model_path.lower().endswith('.tflite'):
                out_shape = output_details["shape"]
            elif quantized_model_path.lower().endswith('.onnx'):
                out_shape = outputs[0].shape
            else:
                raise TypeError("Please provide a TFLITE or ONNX model for N6 deployment")
            nb_kpt = params.dataset.keypoints
            f.write("#define AI_POSE_PP_CONF_THRESHOLD                    ({})\n".format(float(params.postprocessing.kpts_conf_thresh)))
            f.write("#define AI_POSE_PP_POSE_KEYPOINTS_NB                 ({})\n".format(int(nb_kpt)))
            f.write("#define AI_MPE_YOLOV8_PP_NB_CLASSES                  (1)\n")
            f.write("#define AI_MPE_YOLOV8_PP_TOTAL_BOXES                 ({})\n".format(int(out_shape[2])))
            f.write("#define AI_MPE_YOLOV8_PP_MAX_BOXES_LIMIT             ({})\n".format(int(params.postprocessing.max_detection_boxes)))
            f.write("#define AI_MPE_YOLOV8_PP_IOU_THRESHOLD               ({})\n".format(float(params.postprocessing.NMS_thresh)))
            f.write("#define AI_MPE_YOLOV8_PP_CONF_THRESHOLD              ({})\n\n".format(float(params.postprocessing.confidence_thresh)))
        else:
            raise ValueError("model_type not supported")

        try:
            skeleton_connections = skeleton_connections_dict[class_name][nb_kpt]
        except:
            print('Skeleton for the class [{}] & number of keypoints [{}] is unknown -> use [hand] & [21] or [person] & ([17] or [13])'.format(class_name,nb_kpt))
            print('You can add your own in the utils/connections.py file')
            skeleton_connections = None

        if skeleton_connections:
            f.write("#define USE_BINDINGS\n")
            f.write("#define BINDINGS_NB ({})\n".format(len(skeleton_connections)))
            f.write("#define BINDINGS const int bindings[BINDINGS_NB][3] = {\\\n")
            for i in range(len(skeleton_connections)):
                f.write("    {{ {}, {}, {} }},\\\n".format(skeleton_connections[i][0], skeleton_connections[i][1], skeleton_connections[i][2]))
            f.write("}\\\n\n")
            f.write("extern const int bindings[BINDINGS_NB][3];\n\n")
        else:
            f.write("#define BINDINGS const int **bindings = NULL\n\n")

        f.write('/* Display */\n')
        f.write('#define WELCOME_MSG_0       "Single/multi pose estimation - Hand landmark"\n')
        f.write('#define WELCOME_MSG_1       "{}"\n'.format(os.path.basename(params.model.model_path)))
        if config.deployment.hardware_setup.board == 'NUCLEO-N657X0-Q':
            f.write('#define WELCOME_MSG_2         ((char *[2]) {"Model Running in STM32 MCU", "internal memory"})')
        else:
            f.write('#define WELCOME_MSG_2       "{}"\n'.format("Model Running in STM32 MCU internal memory"))
        f.write("\n")
        f.write("#endif      /* APP_CONFIG */\n")

    return TFLite_Detection_PostProcess_id, quantized_model_path