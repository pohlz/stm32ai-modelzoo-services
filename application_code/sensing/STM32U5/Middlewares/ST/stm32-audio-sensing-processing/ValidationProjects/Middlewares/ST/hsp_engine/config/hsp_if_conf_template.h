/**
  ******************************************************************************
  * @file    hsp_if_conf_template.h
  * @brief   Header file for interfaces
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
#ifndef HSP_IF_CONF_TEMPLATE_H
#define HSP_IF_CONF_TEMPLATE_H

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/* Includes ------------------------------------------------------------------*/
#include "interface.h"
#include "stm32xxxx_hal.h" /* Select the hal file corresponding to the device in use (i.e. stm32f3xx_hal.h, stm32f0xx_hal.h, ...) */
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

#endif /* HSP_IF_CONF_TEMPLATE_H */
