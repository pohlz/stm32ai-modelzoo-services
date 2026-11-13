/**
 ******************************************************************************
 * @file    ai_model_config.h
 * @author  STMicroelectronics - AIS - MCD Team
 * @version $Version$
 * @date    $Date$
 * @brief   Configure the getting started functionality
 *
 * Each logic module of the application should define a DEBUG control byte
 * used to turn on/off the log for the module.
 *
 *********************************************************************************
 * @attention
 *
 * Copyright (c) 2023 STMicroelectronics
 * All rights reserved.
 *
 * This software is licensed under terms that can be found in the LICENSE file in
 * the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *********************************************************************************
 */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef AI_MODEL_CONFIG_H
#define AI_MODEL_CONFIG_H

#ifdef __cplusplus
extern "C" {
#endif

/* Exported constants --------------------------------------------------------*/
#define CTRL_X_CUBE_AI_MODE_NB_OUTPUT          (1U)
#define CTRL_X_CUBE_AI_MODE_OUTPUT_1           CTRL_AI_CLASS_DISTRIBUTION
#define CTRL_X_CUBE_AI_MODE_CLASS_NUMBER       (4U)
#define CTRL_X_CUBE_AI_MODE_CLASS_LIST         {"Jogging","Stationary","Stairs","Walking"}
#define CTRL_X_CUBE_AI_SENSOR_TYPE             (COM_TYPE_ACC)
#define CTRL_X_CUBE_AI_SENSOR_ODR              (26.0F)
#define CTRL_X_CUBE_AI_SENSOR_FS               (4.0F)
#define CTRL_X_CUBE_AI_NB_SAMPLES              (0U)  // number of signals you want to run inference
                                                     // zero means undefined
#define CTRL_X_CUBE_AI_PREPROC                 (CTRL_AI_GRAV_ROT_SUPPR)
#define CTRL_SEQUENCE                            {CTRL_CMD_PARAM_AI,0}
#ifdef __cplusplus
}
#endif

#endif /* AI_MODEL_CONFIG_H*/

