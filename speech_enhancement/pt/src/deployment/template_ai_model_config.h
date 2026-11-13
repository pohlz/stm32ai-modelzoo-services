/**
  ******************************************************************************
  * @file    ai_model_config.h
  * @author  STMicroelectronics - AIS - MCD Team
  * @version $Version$
  * @date    $Date$
  * @brief   Configure the getting started functionality
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2023 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef AI_MODEL_CONFIG_H
#define AI_MODEL_CONFIG_H

#ifdef __cplusplus
extern "C" {
#endif

/* Exported constants --------------------------------------------------------*/
#define CTRL_AI_HW_SELECT                        STM32N6570-DK
#define CTRL_X_CUBE_AI_MODEL_NAME                "X-CUBE-AI STFT TCN"
#define CTRL_X_CUBE_AI_MODE_NB_OUTPUT            (1U)
#define CTRL_X_CUBE_AI_MODE_OUTPUT_1             (CTRL_AI_SPECTROGRAM)
#define CTRL_X_CUBE_AI_SENSOR_TYPE               (COM_TYPE_MIC)
#define CTRL_X_CUBE_AI_AUDIO_OUT                 (COM_TYPE_HEADSET)
#define CTRL_X_CUBE_AI_SENSOR_NAME               "imp34dt05"
#define CTRL_X_CUBE_AI_SENSOR_ODR                (16000.0F)
#define CTRL_X_CUBE_AI_PREPROC                   (CTRL_AI_STFT)
#define CTRL_X_CUBE_AI_POSTPROC                  (CTRL_AI_ISTFT)
#define CTRL_X_CUBE_AI_SPECTROGRAM_HOP_LENGTH    (160U)
#define CTRL_X_CUBE_AI_SPECTROGRAM_NFFT          (512U) /* power of 2 closed to CTRL_X_CUBE_AI_WINDOW_LENGTH */
#define CTRL_X_CUBE_AI_SPECTROGRAM_WINDOW_LENGTH (400U)
#define CTRL_X_CUBE_AI_SPECTROGRAM_COL_NO_OVL    (20U)
#define CTRL_X_CUBE_AI_SPECTROGRAM_COL_OVL       (3U)

#define CTRL_X_CUBE_AI_SPECTROGRAM_WIN           (user_win)

#define CTRL_SEQUENCE                            {CTRL_CMD_PARAM_AI,0}
#ifdef __cplusplus
}
#endif

#endif /* AI_MODEL_CONFIG_H*/

