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
import shutil
import numpy as np
import tensorflow as tf

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from common.utils import aspect_ratio_dict, color_mode_n6_dict


def gen_h_user_file_n6_blazeface(config: DictConfig = None, quantized_model_path: str = None) -> None:
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
    output_details = interpreter_quant.get_output_details()[0]
    input_shape = input_details['shape']

    class_names = params.dataset.class_names

    path = os.path.join(HydraConfig.get().runtime.output_dir, "C_header/")

    try:
        os.mkdir(path)
    except OSError as error:
        print(error)

    TFLite_Detection_PostProcess_id = None
    classes = '{\\\n'

    for i, x in enumerate(params.dataset.class_names):
        if i == (len(class_names) - 1):
            classes = classes + '   "' + str(x) + '"' + '}\\'
        else:
            classes = classes + '   "' + str(x) + '"' + ' ,' + ('\\\n' if (i % 5 == 0 and i != 0) else '')
    
    if params.model.model_type == "facedetect_front":
        outs_info = interpreter_quant.get_output_details()
        #print(outs_info)
        output_shapes =[]
        for buffer in outs_info:
            output_shapes.append(buffer["shape"])
        sorted_shapes = sorted(output_shapes, key=lambda arr: (arr[1], arr[2]), reverse=True)
        SSD_OPTIONS_FRONT = {
        'num_layers': 4,
        'input_size_height': 128,
        'input_size_width': 128,
        'anchor_offset_x': 0.5,
        'anchor_offset_y': 0.5,
        'strides': [8, 16, 16, 16],
        'interpolated_scale_aspect_ratio': 1.0}
        from face_detection.tf.src.postprocessing  import ssd_generate_anchors
        anchors=ssd_generate_anchors(SSD_OPTIONS_FRONT)
        anch_0_rows = int(sorted_shapes[0][1])
        anch_1_rows = int(sorted_shapes[2][1])
        anch_0 = anchors[:anch_0_rows, :]
        anch_1 = anchors[anch_0_rows:, :]
        anch_0_flat = anch_0.reshape(int(anch_0.shape[0] * anch_0.shape[1]))
        anch_1_flat = anch_1.reshape(int(anch_1.shape[0] * anch_1.shape[1]))
        # Format the array elements as strings with 'f' suffix for floats in C
        formatted_anch_0_flat = ", ".join(f"{x:.6f}" for x in anch_0_flat)
        c_anch_0_str = f"const float32_t g_Anchors_0[{int(anch_0.shape[0] * anch_0.shape[1])}] = {{ {formatted_anch_0_flat} }};"
        formatted_anch_1_flat = ", ".join(f"{x:.6f}" for x in anch_1_flat)
        c_anch_1_str = f"const float32_t g_Anchors_1[{int(anch_1.shape[0] * anch_1.shape[1])}] = {{ {formatted_anch_1_flat} }};"
        
        with open(os.path.join(path, "fd_blazeface_anchors_0.h"), "wt") as f:
            f.write("#ifndef __ANCHORS_0_H__\n")
            f.write("#define __ANCHORS_0_H__\n\n")
            f.write(c_anch_0_str)
            f.write("\n")
            f.write("#endif /* __ANCHORS_0_H__ */\n")

        with open(os.path.join(path, "fd_blazeface_anchors_1.h"), "wt") as f:
            f.write("#ifndef __ANCHORS_1_H__\n")
            f.write("#define __ANCHORS_1_H__\n\n")
            f.write(c_anch_1_str)
            f.write("\n")
            f.write("#endif /* __ANCHORS_1_H__ */\n")

        # Copy the anchors to the C project
        anchors_0_path_C = os.path.join(params.deployment.c_project_path, 'Application', params.deployment.hardware_setup.board, 'Inc', 'fd_blazeface_anchors_0.h')
        anchors_1_path_C = os.path.join(params.deployment.c_project_path, 'Application', params.deployment.hardware_setup.board, 'Inc', 'fd_blazeface_anchors_1.h')
        if os.path.exists(anchors_0_path_C):
            os.remove(anchors_0_path_C)
        if os.path.exists(anchors_1_path_C):
            os.remove(anchors_1_path_C)
        shutil.copy(os.path.join(path, "fd_blazeface_anchors_0.h"), anchors_0_path_C)
        shutil.copy(os.path.join(path, "fd_blazeface_anchors_1.h"), anchors_1_path_C)

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

        if params.model.model_type == "facedetect_front":
            f.write("#define POSTPROCESS_TYPE    POSTPROCESS_FD_BLAZEFACE_UI\n")
        elif params.model.model_type == "yunet":
            f.write("#define POSTPROCESS_TYPE    POSTPROCESS_FD_YUNET_UI\n")
        else:
            raise TypeError("Please select one of the supported model_type")
        f.write("\n")
        f.write("#define COLOR_BGR (0)\n")
        f.write("#define COLOR_RGB (1)\n")
        f.write("#define COLOR_MODE {}\n".format(color_mode_n6_dict[params.preprocessing.color_mode]))



        if params.model.model_type == "facedetect_front":
            f.write("\n/* Postprocessing FD_BLAZEFACE configuration */\n")

            outs_info = interpreter_quant.get_output_details()
            #print(outs_info)
            output_shapes =[]
            for buffer in outs_info:
                output_shapes.append(buffer["shape"])
            sorted_shapes = sorted(output_shapes, key=lambda arr: (arr[1], arr[2]), reverse=True)

            f.write("#define AI_FD_BLAZEFACE_PP_NB_KEYPOINTS      ({})\n".format(int((sorted_shapes[0][2]-4)/2)))
            f.write("#define AI_FD_BLAZEFACE_PP_NB_CLASSES        ({})\n".format(int(sorted_shapes[-1][-1])))
            f.write("#define AI_FD_BLAZEFACE_PP_IMG_SIZE          ({})\n".format(int(input_shape[1])))
            f.write("#define AI_FD_BLAZEFACE_PP_OUT_0_NB_BOXES    ({})\n".format(int(sorted_shapes[0][1])))
            f.write("#define AI_FD_BLAZEFACE_PP_OUT_1_NB_BOXES    ({})\n".format(int(sorted_shapes[-1][1])))

            f.write("#define AI_FD_BLAZEFACE_PP_MAX_BOXES_LIMIT   ({})\n".format(int(params.postprocessing.max_detection_boxes)))
            f.write("#define AI_FD_BLAZEFACE_PP_CONF_THRESHOLD    ({})\n".format(float(params.postprocessing.confidence_thresh)))
            f.write("#define AI_FD_BLAZEFACE_PP_IOU_THRESHOLD     ({})\n".format(float(params.postprocessing.NMS_thresh)))

        elif params.model.model_type == "yunet":
            f.write("\n/* Postprocessing FD_YUNET configuration */\n")

            outs_info = interpreter_quant.get_output_details()
            #print(outs_info)
            output_shapes =[]
            for buffer in outs_info:
                output_shapes.append(buffer["shape"])
            sorted_shapes = sorted(output_shapes, key=lambda arr: (arr[1], arr[2]), reverse=True)

            f.write("#define AI_FD_YUNET_PP_NB_KEYPOINTS      ({})\n".format(int(5)))
            f.write("#define AI_FD_YUNET_PP_NB_CLASSES        ({})\n".format(int(1)))
            f.write("#define AI_FD_YUNET_PP_IMG_SIZE          ({})\n".format(int(320)))
            f.write("#define AI_FD_YUNET_PP_OUT_32_NB_BOXES   ({})\n".format(int(100)))
            f.write("#define AI_FD_YUNET_PP_OUT_16_NB_BOXES   ({})\n".format(int(400)))
            f.write("#define AI_FD_YUNET_PP_OUT_8_NB_BOXES    ({})\n".format(int(1600)))

            f.write("#define AI_FD_YUNET_PP_MAX_BOXES_LIMIT   ({})\n".format(int(params.postprocessing.max_detection_boxes)))
            f.write("#define AI_FD_YUNET_PP_CONF_THRESHOLD    ({})\n".format(float(params.postprocessing.confidence_thresh)))
            f.write("#define AI_FD_YUNET_PP_IOU_THRESHOLD     ({})\n".format(float(params.postprocessing.NMS_thresh)))



        f.write('#define WELCOME_MSG_1         "{}"\n'.format(os.path.basename(params.model.model_path)))
        # @Todo retieve info from stedgeai output
        if config.deployment.hardware_setup.board == 'NUCLEO-N657X0-Q':
            f.write('#define WELCOME_MSG_2         ((char *[2]) {"Model Running in STM32 MCU", "internal memory"})')
        else:
            f.write('#define WELCOME_MSG_2       "{}"\n'.format("Model Running in STM32 MCU internal memory"))

        f.write("\n")
        f.write("#endif      /* APP_CONFIG */\n")

    return TFLite_Detection_PostProcess_id, quantized_model_path


#=============================================================================================================================================================

def gen_h_user_file_n6_yunet(config, quantized_model_path: str = None) -> None:

    """
    Generates a C header file containing user configuration for the AI model.

    Args:
        config: A configuration object containing user configuration for the AI model.
        quantized_model_path: The path to the quantized model file.

    """

    import onnxruntime

    params = config

    model = onnxruntime.InferenceSession(quantized_model_path)
    inputs  = model.get_inputs()
    outputs = model.get_outputs()
    input_shape_raw = inputs[0].shape

    sorted_outputs = sorted(outputs, key=lambda x: (x.shape[1], x.shape[2]),reverse=True)

    class_names = params.dataset.class_names

    path = os.path.join(HydraConfig.get().runtime.output_dir, "C_header/")

    try:
        os.mkdir(path)
    except OSError as error:
        print(error)

    classes = '{\\\n'

    for i, x in enumerate(params.dataset.class_names):
        if i == (len(class_names) - 1):
            classes = classes + '   "' + str(x) + '"' + '}\\'
        else:
            classes = classes + '   "' + str(x) + '"' + ' ,' + ('\\\n' if (i % 5 == 0 and i != 0) else '')
    
    if params.model.model_type == "yunet":
        from face_detection.tf.src.postprocessing  import generate_yunet_anchor
        strides = [32,16,8] #model strides
        NK = 5 #number of keypoints
        image_size =  (input_shape_raw[2], input_shape_raw[3], input_shape_raw[1])
        model_anchor_centers = generate_yunet_anchor(image_size, strides)
        anchors_sorted = sorted(model_anchor_centers, key=lambda x: (x.shape[0]),reverse=True)

        anch_s32_flat = anchors_sorted[2].reshape(int(anchors_sorted[2].shape[0] * anchors_sorted[2].shape[1]))
        anch_s16_flat = anchors_sorted[1].reshape(int(anchors_sorted[1].shape[0] * anchors_sorted[1].shape[1]))
        anch_s8_flat  = anchors_sorted[0].reshape(int(anchors_sorted[0].shape[0] * anchors_sorted[0].shape[1]))

        # Format the array elements as strings with 'f' suffix for floats in C
        formatted_anch_s32_flat = ", ".join(f"{int(x)}" for x in anch_s32_flat)
        c_anch_s32_str = f"const int16_t g_Anchors_32[{int(anchors_sorted[2].shape[0] * anchors_sorted[2].shape[1])}] = {{ {formatted_anch_s32_flat} }};"

        formatted_anch_s16_flat = ", ".join(f"{int(x)}" for x in anch_s16_flat)
        c_anch_s16_str = f"const int16_t g_Anchors_16[{int(anchors_sorted[1].shape[0] * anchors_sorted[1].shape[1])}] = {{ {formatted_anch_s16_flat} }};"

        formatted_anch_s8_flat = ", ".join(f"{int(x)}" for x in anch_s8_flat)
        c_anch_s8_str = f"const int16_t g_Anchors_8[{int(anchors_sorted[0].shape[0] * anchors_sorted[0].shape[1])}] = {{ {formatted_anch_s8_flat} }};"
        
        with open(os.path.join(path, "fd_yunet_anchors_32.h"), "wt") as f:
            f.write("#ifndef __ANCHORS_32_H__\n")
            f.write("#define __ANCHORS_32_H__\n\n")
            f.write(c_anch_s32_str)
            f.write("\n")
            f.write("#endif /* __ANCHORS_32_H__ */\n")

        with open(os.path.join(path, "fd_yunet_anchors_16.h"), "wt") as f:
            f.write("#ifndef __ANCHORS_16_H__\n")
            f.write("#define __ANCHORS_16_H__\n\n")
            f.write(c_anch_s16_str)
            f.write("\n")
            f.write("#endif /* __ANCHORS_16_H__ */\n")

        with open(os.path.join(path, "fd_yunet_anchors_8.h"), "wt") as f:
            f.write("#ifndef __ANCHORS_8_H__\n")
            f.write("#define __ANCHORS_8_H__\n\n")
            f.write(c_anch_s8_str)
            f.write("\n")
            f.write("#endif /* __ANCHORS_8_H__ */\n")

        # Copy the anchors to the C project
        anchors_s32_path_C = os.path.join(params.deployment.c_project_path, 'Application', params.deployment.hardware_setup.board, 'Inc', 'fd_yunet_anchors_32.h')
        anchors_s16_path_C = os.path.join(params.deployment.c_project_path, 'Application', params.deployment.hardware_setup.board, 'Inc', 'fd_yunet_anchors_16.h')
        anchors_s8_path_C  = os.path.join(params.deployment.c_project_path, 'Application', params.deployment.hardware_setup.board, 'Inc', 'fd_yunet_anchors_8.h')

        if os.path.exists(anchors_s32_path_C):
            os.remove(anchors_s32_path_C)
        if os.path.exists(anchors_s16_path_C):
            os.remove(anchors_s16_path_C)
        if os.path.exists(anchors_s8_path_C):
            os.remove(anchors_s8_path_C)

        shutil.copy(os.path.join(path, "fd_yunet_anchors_32.h"), anchors_s32_path_C)
        shutil.copy(os.path.join(path, "fd_yunet_anchors_16.h"), anchors_s16_path_C)
        shutil.copy(os.path.join(path, "fd_yunet_anchors_8.h"), anchors_s8_path_C)

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

        if params.model.model_type == "yunet":
            f.write("#define POSTPROCESS_TYPE    POSTPROCESS_FD_YUNET_UI\n")
        else:
            raise TypeError("Please select one of the supported model_type")
        f.write("\n")
        f.write("#define COLOR_BGR (0)\n")
        f.write("#define COLOR_RGB (1)\n")
        f.write("#define COLOR_MODE {}\n".format(color_mode_n6_dict[params.preprocessing.color_mode]))


        if params.model.model_type == "yunet":
            f.write("\n/* Postprocessing FD_YUNET configuration */\n")

            f.write("#define AI_FD_YUNET_PP_NB_KEYPOINTS      ({})\n".format(int(NK)))
            f.write("#define AI_FD_YUNET_PP_NB_CLASSES        ({})\n".format(int(1)))
            f.write("#define AI_FD_YUNET_PP_IMG_SIZE          ({})\n".format(int(image_size[1])))
            f.write("#define AI_FD_YUNET_PP_OUT_32_NB_BOXES   ({})\n".format(int(anchors_sorted[2].shape[0])))
            f.write("#define AI_FD_YUNET_PP_OUT_16_NB_BOXES   ({})\n".format(int(anchors_sorted[1].shape[0])))
            f.write("#define AI_FD_YUNET_PP_OUT_8_NB_BOXES    ({})\n".format(int(anchors_sorted[0].shape[0])))

            f.write("#define AI_FD_YUNET_PP_MAX_BOXES_LIMIT   ({})\n".format(int(params.postprocessing.max_detection_boxes)))
            f.write("#define AI_FD_YUNET_PP_CONF_THRESHOLD    ({})\n".format(float(params.postprocessing.confidence_thresh)))
            f.write("#define AI_FD_YUNET_PP_IOU_THRESHOLD     ({})\n".format(float(params.postprocessing.NMS_thresh)))



        f.write('#define WELCOME_MSG_1         "{}"\n'.format(os.path.basename(params.model.model_path)))
        # @Todo retieve info from stedgeai output
        if config.deployment.hardware_setup.board == 'NUCLEO-N657X0-Q':
            f.write('#define WELCOME_MSG_2         ((char *[2]) {"Model Running in STM32 MCU", "internal memory"})')
        else:
            f.write('#define WELCOME_MSG_2       "{}"\n'.format("Model Running in STM32 MCU internal memory"))

        f.write("\n")
        f.write("#endif      /* APP_CONFIG */\n")

    return None, quantized_model_path