/* USER CODE BEGIN Header */
/**
******************************************************************************
* @file           : hsp_engine.h
* @version        : v1_0_Cube
* @brief          : Header for hsp_engine.c file.
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
/* USER CODE END Header */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __HSP_ENGINE__H__
#define __HSP_ENGINE__H__

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/

/* USER CODE BEGIN INCLUDE */

/* USER CODE END INCLUDE */

/** @addtogroup HSP_DRIVER
* @{
*/

/** @defgroup HSP
* @brief Host file for hsp otg low level driver.
* @{
*/

/** @defgroup HSP_Engine_Exported_Variables HSP_Engine_Exported_Variables
* @brief Public variables.
* @{
*/

/**
* @}
*/

/** @defgroup HSP_Engine_Exported_FunctionsPrototype HSP_Engine_Exported_FunctionsPrototype
* @brief Declaration of public functions for Usb host.
* @{
*/

/* Exported functions -------------------------------------------------------*/

/** @brief USB Host initialization function. */
void MX_HSP_Engine_Init(void);

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
#endif

#endif /* __HSP_ENGINE__H__ */

