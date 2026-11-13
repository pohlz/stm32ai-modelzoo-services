/**
  ******************************************************************************
  * @file    hsp_core.h
  * @author  MCD Application Team
  * @brief   Header file for hsp_core.c
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

/* Define to prevent recursive  ----------------------------------------------*/
#ifndef HSP_CORE_H
#define HSP_CORE_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "hsp_api_def.h"
#include "hsp_conf.h"
#include "hsp_hw_if.h"

/** @addtogroup HSP_ENGINE
  * @{
  */

/** @defgroup HSP_CORE HSP Core
  * @{
  */
/** @defgroup HSP_CORE_Init HSP Init
  * @{
  */
/** @defgroup HSP_CORE_Exported_Constants HSP Exported Constants
  * @{
  */
/** @defgroup HSP_CORE_Error_Code  HSP Error Code
  * @{
  */
#define HSP_CORE_ERROR_NONE                   (0UL)            /**< HSP no error */
#define HSP_CORE_ERROR_INIT                   (0x01UL << 1U)   /**< HSP init failed as it was already done */
#define HSP_CORE_ERROR_INVALID_PARAM          (0x01UL << 2U)   /**< HSP invalid parameter */

#define HSP_CORE_PROCESS_FUNCTION_PARAMETER_NBR  16U

/**
  * @}
  */

/** @defgroup HSP_CORE_TRGO  HSP Trigger Output Constants
  * @{
  */
#define HSP_CORE_TRGO_0  (1UL)
#define HSP_CORE_TRGO_1  (1UL << 1U)
#define HSP_CORE_TRGO_2  (1UL << 2U)
#define HSP_CORE_TRGO_3  (1UL << 3U)

/**
  * @}
  */

/** @defgroup HSP_CORE_SmartClocking  HSP Core SmartClocking Constants
  * @{
  */
typedef enum
{
  HSP_CORE_SMART_CLOCKING_SPE,
  HSP_CORE_SMART_CLOCKING_MMC,
  HSP_CORE_SMART_CLOCKING_CTRL
} hsp_core_smart_clocking_t;

typedef enum
{
  HSP_CORE_SMART_CLOCKING_DISABLED = 0U,
  HSP_CORE_SMART_CLOCKING_ENABLED = 1U,
} hsp_core_smart_clocking_status_t;

/**
  * @}
  */


/** @defgroup HSP_CORE_Exported_Types HSP Core Exported Types
  * @{
  */
#define hsp_core_trgo_source_t  hsp_hw_if_output_trigger_source_t
#define hsp_core_trgo_t  hsp_hw_if_output_trigger_t

/**
  * @}
  */

/** @defgroup HSP_CORE_Exported_Macros HSP Core Exported Macros
  * @{
  */

/**
  * @}
  */

/** @defgroup HSP_CORE_Exported_Functions HSP Core Exported Functions
  * @{
  */

/** @defgroup HSP_CORE_Exported_Functions_Group1 HSP Init functions
  * @{
  */
hsp_core_status_t HSP_CORE_Init(hsp_core_handle_t *hmw);
void HSP_CORE_DeInit(hsp_core_handle_t *hmw);

/**
  * @}
  */

/** @defgroup HSP_CORE_Exported_Functions_Group2 HSP Output Trigger Management
  * @{
  */
hsp_core_status_t HSP_CORE_OUTPUT_SetConfig(hsp_core_handle_t *hmw, hsp_core_trgo_t trgo_id,
                                            hsp_core_trgo_source_t source);
hsp_core_status_t HSP_CORE_OUTPUT_Enable(hsp_core_handle_t *hmw);
hsp_core_status_t HSP_CORE_OUTPUT_Disable(hsp_core_handle_t *hmw);
uint32_t HSP_CORE_OUTPUT_IsEnabled(hsp_core_handle_t *hmw);

/**
  * @}
 */

/** @defgroup HSP_CORE_Exported_Functions_Group3 HSP Smart Clocking Management
  * @{
  */
hsp_core_status_t HSP_CORE_EnableSmartClocking(hsp_core_handle_t *hmw, hsp_core_smart_clocking_t clock);
hsp_core_status_t HSP_CORE_DisableSmartClocking(hsp_core_handle_t *hmw, hsp_core_smart_clocking_t clock);
hsp_core_smart_clocking_status_t HSP_CORE_GetSmartClockingStatus(hsp_core_handle_t *hmw,
                                                                 hsp_core_smart_clocking_t clock);

/**
  * @}
 */

/** @defgroup HSP_CORE_Exported_Functions_Group4 HSP Protection
  * @{
  */
hsp_core_status_t HSP_CORE_Lock(hsp_core_handle_t *hmw);

/**
  * @}
 */

/** @defgroup HSP_CORE_Exported_Functions_Group5 Generic and common functions
  * @{
  */
hsp_core_state_t HSP_CORE_GetState(const hsp_core_handle_t *hmw);

/**
  * @}
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

/**
  * @}
  */

#ifdef __cplusplus
}
#endif

#endif /* HSP_CORE_H */
