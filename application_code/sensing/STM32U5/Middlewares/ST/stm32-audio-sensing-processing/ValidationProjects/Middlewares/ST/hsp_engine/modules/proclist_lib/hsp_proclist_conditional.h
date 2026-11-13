/**
  ******************************************************************************
  * @file    hsp_proclist_conditional.h
  * @author  MCD Application Team
  * @brief   Header file for hsp_proclist_conditional.c
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
#ifndef HSP_PROCLIST_CONDITIONAL_H
#define HSP_PROCLIST_CONDITIONAL_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "hsp_proclist_def.h"

/** @addtogroup HSP
  * @{
  */

/** @defgroup HSP_PROCLIST
  * @{
  */

/** @defgroup HSP_MODULES_PROCLIST_CONDITIONAL_LIBRARY HSP Proclist Conditional Functions
  * @{
  */
/* Exported functions ---------------------------------------------------------*/
/** @defgroup HSP_Exported_Functions_ProcList_Conditional MW HSP Conditional family function for Processing List
  * @brief    HSP Conditional functions for processing list
  * @{
  */  
  
hsp_core_status_t HSP_SEQ_IfIFloat(hsp_core_handle_t *hmw, hsp_proclist_cond_cmd_t cmd, float32_t *var1, float32_t var2);
hsp_core_status_t HSP_SEQ_IfIUint32(hsp_core_handle_t *hmw, hsp_proclist_cond_cmd_t cmd, uint32_t *var1, uint32_t var2);
hsp_core_status_t HSP_SEQ_IfIInt32(hsp_core_handle_t *hmw, hsp_proclist_cond_cmd_t cmd, int32_t *var1, int32_t var2);
hsp_core_status_t HSP_SEQ_IfFloat(hsp_core_handle_t *hmw, hsp_proclist_cond_cmd_t cmd, float32_t *var1, float32_t *var2);
hsp_core_status_t HSP_SEQ_IfUint32(hsp_core_handle_t *hmw, hsp_proclist_cond_cmd_t cmd, uint32_t *var1, uint32_t *var2);
hsp_core_status_t HSP_SEQ_IfInt32(hsp_core_handle_t *hmw, hsp_proclist_cond_cmd_t cmd, int32_t *var1, int32_t *var2);
hsp_core_status_t HSP_SEQ_IfCnt(hsp_core_handle_t *hmw, hsp_proclist_cond_cmd_t cmd, uint32_t idx, uint32_t var2);
hsp_core_status_t HSP_SEQ_IfElse(hsp_core_handle_t *hmw);
hsp_core_status_t HSP_SEQ_IfEndif(hsp_core_handle_t *hmw);
hsp_core_status_t HSP_SEQ_Loop(hsp_core_handle_t *hmw, uint32_t count);
hsp_core_status_t HSP_SEQ_LoopEnd(hsp_core_handle_t *hmw);
hsp_core_status_t HSP_SEQ_IfLoop(hsp_core_handle_t *hmw, hsp_proclist_cond_cmd_t cmd, uint32_t *var);
hsp_core_status_t HSP_SEQ_IfILoop(hsp_core_handle_t *hmw, hsp_proclist_cond_cmd_t cmd, uint32_t val);
hsp_core_status_t HSP_SEQ_IfLoopOdd(hsp_core_handle_t *hmw);
hsp_core_status_t HSP_SEQ_IfLoopEven(hsp_core_handle_t *hmw);


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

#endif /* HSP_PROCLIST_CONDITIONAL_H */
