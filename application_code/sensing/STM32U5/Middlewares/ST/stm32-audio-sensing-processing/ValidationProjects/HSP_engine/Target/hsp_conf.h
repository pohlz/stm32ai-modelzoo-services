/**
  ******************************************************************************
  * @file    hsp_conf.h
  * @author  MCD Application Team
  * @brief   Header file
  ******************************************************************************
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
#ifndef HSP_CONF_H
#define HSP_CONF_H

#ifdef __cplusplus
 extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include <stdio.h>
#include <stdint.h>

/* USER CODE BEGIN INCLUDE */

/* USER CODE END INCLUDE */

/** @addtogroup HSP_ENGINE
  * @{
  */

/** @defgroup HSP_CONF HSP Configuration
  * @{
  */

/** @defgroup HSP_CONF_Exported_Constants HSP Configuration Constants
  * @{
  */

/* ################## HSP Modules Configuration ############### */
/**
  * @brief Set below the peripheral configuration to "1U" to add the support of the module
  */
#define USE_HSP_MODULES_DIRECT_LIB               1U
#define USE_HSP_MODULES_DSP_LIB                  0U
#define USE_HSP_MODULES_CNN_LIB                  1U
#define USE_HSP_MODULES_PROCLIST_COMPLEX         0U
#define USE_HSP_MODULES_PROCLIST_CONDITIONAL     0U
#define USE_HSP_MODULES_PROCLIST_CRC             0U
#define USE_HSP_MODULES_PROCLIST_FIR             0U
#define USE_HSP_MODULES_PROCLIST_MATRIX          0U
#define USE_HSP_MODULES_PROCLIST_MEMORY          0U
#define USE_HSP_MODULES_PROCLIST_SCALAR          0U
#define USE_HSP_MODULES_PROCLIST_TRANSFORM       0U
#define USE_HSP_MODULES_PROCLIST_VECTOR          0U
/**
  * @brief Enable the use of STM32 HSP memcpy
  *        (0U to use the standard lib C memcpy, 1U to use STM32 HSP memcpy)
  */
#define USE_HSP_MEMCPY                1U

/**
  * @brief Enable the Synchronous mode of Direct Command APIs
  *        (0U to disable, 1U to enable)
  */
#define USE_HSP_ACC_DIRECT_COMMAND_SYNCHRONOUS  1U

/* HSP Performance Monitor ---------------------------------------------------*/
/**
  * @brief Enable the HSP Performance Monitor at boot step
           (0U to disable, 1U to enable)
  */
#define ENABLE_HSP_PERFORMANCE_MONITOR  0U

/* HSP Timeout ---------------------------------------------------------------*/
/**
  * @brief Allow to configure HSP Boot Timeout in millisecond
  */
#define HSP_BOOT_TIMEOUT_MS  500U

/**
  * @brief Allow to configure Timeout in millisecond for all polling functions
  */
#define HSP_TIMEOUT_MS  2000U

/* Cube AI -------------------------------------------------------------------*/
/**
  * @brief Define the BRAM Size reserved for CubeAI processing
  */
#if (defined  TEST_LFBE_SWW)
#define HSP_BRAM_AI_SIZE 2500u  /*!< Size in word */
#elif (defined  TEST_LFBE_AED)
#define HSP_BRAM_AI_SIZE 2500u  /*!< Size in word */
#else
#define HSP_BRAM_AI_SIZE 4096U  /*!< Size in word */
#endif

#define HSPx  HSP1     

/** @defgroup HSP_CONF_Exported_Macros HSP Configuration Macros
  * @{
  */
#if !defined(UNUSED)
#define UNUSED(X) ((void)X)      /* To avoid gcc/g++ warnings */
#endif /* UNUSED */

#if !defined(COUNTOF)
#define COUNTOF(a)  (sizeof(a) / sizeof(*(a)))
#endif /* COUNTOF */

/**
  * @brief Enable the load of Plugin (0U to disable, 1U to enable)
  *   If enabled the Application must allocate the array like this:
  *     1- hsp_plugin_t *name_of_array;
  *     2- hsp_plugin_t name_of_array[NB_HSP_PLUGIN];
  *     3- hsp_plugin_t name_of_array[] = { {}, {}...};
  *
  *   And the alias HSP_PLUGIN_ARRAY_NAME must be updated like this:
  *     #define HSP_PLUGIN_ARRAY_NAME  name_of_array
  */
#define USE_HSP_PLUGIN  0U
#if defined(USE_HSP_PLUGIN) && (USE_HSP_PLUGIN == 1)
/* Replace the literal "a_hsp_plugins" by the one defined by the Application */
#define HSP_PLUGIN_ARRAY_NAME  a_hsp_plugins
#define HSP_PLUGIN_ARRAY_SIZE  COUNTOF(HSP_PLUGIN_ARRAY_NAME)
#else
#define HSP_PLUGIN_ARRAY_NAME
#define HSP_PLUGIN_ARRAY_SIZE  0UL
#endif /* USE_HSP_PLUGIN */
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

#endif /* HSP_CONF_H */
