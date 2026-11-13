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

from common.utils import color_mode_n6_dict, aspect_ratio_dict



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
    interpreter_quant = tf.lite.Interpreter(model_path=quantized_model_path)
    input_details = interpreter_quant.get_input_details()[0]
    output_details = interpreter_quant.get_output_details()
    input_shape = input_details['shape']


    class_names = params.dataset.class_names
    classes = '{\\\n'

    for i, x in enumerate(class_names):
        if i == (len(class_names) - 1):
            classes = classes + '   "' + str(x) + '"' + '}\\'
        else:
            classes = classes + '   "' + str(x) + '"' + ' ,' + ('\\\n' if (i % 5 == 0 and i != 0) else '')

    path = os.path.join(HydraConfig.get().runtime.output_dir, "C_header/")

    try:
        os.mkdir(path)
    except OSError as error:
        print(error)

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
        f.write('#include "arm_math.h"\n\n')
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

        if params.model.model_type == "yolov8n_seg":
            f.write("#define POSTPROCESS_TYPE POSTPROCESS_ISEG_YOLO_V8_UI\n\n")
        else:
            raise TypeError("please select one of this supported post processing options [yolov8n_seg]")

        f.write("#define COLOR_BGR (0)\n")
        f.write("#define COLOR_RGB (1)\n")
        f.write("#define COLOR_MODE {}\n".format(color_mode_n6_dict[params.preprocessing.color_mode]))

        f.write("#define NB_CLASSES   ({})\n".format(len(class_names)))
        f.write("#define CLASSES_TABLE const char* classes_table[NB_CLASSES] = {}\n".format(classes))

        if params.model.model_type == "yolov8n_seg":
            out_shape1 = output_details[0]["shape"]
            scale1, offset1 = output_details[0]['quantization']
            out_shape2 = output_details[1]["shape"]
            scale2, offset2 = output_details[1]['quantization']
            n_masks = out_shape2[3]
            f.write("\n/* Post processing values */\n")
            f.write("#define AI_YOLOV8_SEG_PP_TOTAL_BOXES       ({})\n".format(int(out_shape1[2])))
            f.write("#define AI_YOLOV8_SEG_PP_NB_CLASSES        ({})\n".format(int(out_shape1[1]-4-n_masks)))
            f.write("#define AI_YOLOV8_SEG_PP_MASK_NB           ({})\n".format(int(n_masks)))
            f.write("#define AI_YOLOV8_SEG_PP_MASK_SIZE         ({})\n".format(int(out_shape2[1])))
            f.write("#define AI_YOLOV8_SEG_ZERO_POINT           ({})\n".format(int(offset1)))
            f.write("#define AI_YOLOV8_SEG_SCALE                ({})\n".format(float(scale1)))
            f.write("#define AI_YOLOV8_SEG_MASK_ZERO_POINT      ({})\n".format(int(offset2)))
            f.write("#define AI_YOLOV8_SEG_MASK_SCALE           ({})\n".format(float(scale2)))
            f.write('/* --------  Tuning below can be modified by the application --------- */\n')
            f.write("#define AI_YOLOV8_SEG_PP_CONF_THRESHOLD              ({})\n".format(float(params.postprocessing.confidence_thresh)))
            f.write("#define AI_YOLOV8_SEG_PP_IOU_THRESHOLD               ({})\n".format(float(params.postprocessing.NMS_thresh)))
            f.write("#define AI_YOLOV8_SEG_PP_MAX_BOXES_LIMIT             ({})\n".format(int(params.postprocessing.max_detection_boxes)))
        else:
            raise ValueError("model_type not supported")
        f.write('/* Display */\n')
        f.write('#define WELCOME_MSG_0       "Single/multi pose estimation - Hand landmark"\n')
        f.write('#define WELCOME_MSG_1       "{}"\n'.format(os.path.basename(params.model.model_path)))
        # @Todo retieve info from stedgeai output
        if config.deployment.hardware_setup.board == 'NUCLEO-N657X0-Q':
            f.write('#define WELCOME_MSG_2         ((char *[2]) {"Model Running in STM32 MCU", "internal memory"})')
        else:
            f.write('#define WELCOME_MSG_2       "{}"\n'.format("Model Running in STM32 MCU internal memory"))
        f.write("\n")
        f.write("#endif      /* APP_CONFIG */\n")

    return quantized_model_path