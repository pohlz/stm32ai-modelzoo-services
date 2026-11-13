/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file    target/hsp_if_conf.h.h
  * @author  MCD Application Team
  * @brief   Header file
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef HSP_IF_CONF_H
#define HSP_IF_CONF_H

#ifdef __cplusplus
 extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32u3xx_hal.h"
#include "hsp_api_def.h"

/** @addtogroup HSP_ENGINE
  * @{
  */

/** @defgroup HSP_CONF
  * @{
  */

/** @defgroup HSP_CONF_Exported_Constants
  * @{
  */
/**
  * @brief Set below the HAL Format to "1U" for HAL Cube1
  */

/**
  * @}
  */

/** @defgroup HSP_CONF_Exported_Types
  * @{
  */

/**
  * @}
  */

/** @defgroup HSP_CONF_Exported_Macros
  * @{
  */

/**
  * @}
  */

/** @defgroup HSP_CONF_Exported_Variables
  * @{
  */
hsp_core_status_t HSP_Engine_IF_Init(hsp_core_handle_t *hmw);

/**
  * @}
  */

/** @defgroup HSP_CONF_Exported_FunctionsPrototype
  * @{
  */

/**
  * @}
  */

/**
  * @}
  */

/**
  * @}
  */

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* HSP_IF_CONF_H */
