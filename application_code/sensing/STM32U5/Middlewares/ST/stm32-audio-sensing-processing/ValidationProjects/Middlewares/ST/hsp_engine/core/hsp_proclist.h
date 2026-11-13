/**
  ******************************************************************************
  * @file    hsp_proclist.h
  * @author  MCD Application Team
  * @brief   Header file for hsp_proclist.c
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
#ifndef HSP_PROCLIST_H
#define HSP_PROCLIST_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "hsp_def.h"

/** @addtogroup HSP_ENGINE
  * @{
  */

/** @addtogroup HSP_CORE HSP Core
  * @{
  */

/** @defgroup HSP_PROCLIST HSP Processing List Management
  * @{
  */

/** @defgroup HSP_PROCLIST_Exported_Defines HSP Processing List Exported Constants
  * @{
  */
#define HSP_SEQ_TRGIN_0  HSP_HW_IF_TRGIN_0
#define HSP_SEQ_TRGIN_1  HSP_HW_IF_TRGIN_1
#define HSP_SEQ_TRGIN_2  HSP_HW_IF_TRGIN_2
#define HSP_SEQ_TRGIN_3  HSP_HW_IF_TRGIN_3
#define HSP_SEQ_TRGIN_4  HSP_HW_IF_TRGIN_4
#define HSP_SEQ_TRGIN_5  HSP_HW_IF_TRGIN_5
#define HSP_SEQ_TRGIN_6  HSP_HW_IF_TRGIN_6
#define HSP_SEQ_TRGIN_7  HSP_HW_IF_TRGIN_7
#define HSP_SEQ_TRGIN_8  HSP_HW_IF_TRGIN_8
#define HSP_SEQ_TRGIN_9  HSP_HW_IF_TRGIN_9

#define HSP_SEQ_STREAM_BUFFER_0  HSP_HW_IF_STREAM_BUFFER_0
#define HSP_SEQ_STREAM_BUFFER_1  HSP_HW_IF_STREAM_BUFFER_1
#define HSP_SEQ_STREAM_BUFFER_2  HSP_HW_IF_STREAM_BUFFER_2
#define HSP_SEQ_STREAM_BUFFER_3  HSP_HW_IF_STREAM_BUFFER_3

#define HSP_SEQ_STREAM_BUFFER_SYNC_DISABLE  HSP_HW_IF_STREAM_BUFFER_SYNC_DISABLE
#define HSP_SEQ_STREAM_BUFFER_SYNC_ENABLE   HSP_HW_IF_STREAM_BUFFER_SYNC_ENABLE

#define HSP_SEQ_EVENT_SYNC_DISABLE  HSP_HW_IF_EVENT_SYNC_DISABLE
#define HSP_SEQ_EVENT_SYNC_ENABLE   HSP_HW_IF_EVENT_SYNC_ENABLE

#define HSP_SEQ_TRGIN_POLARITY_RISING  HSP_HW_IF_TRGIN_POLARITY_RISING
#define HSP_SEQ_TRGIN_POLARITY_FALLING  HSP_HW_IF_TRGIN_POLARITY_FALLING

#define HSP_SEQ_STREAM_BUFFER_DIRECTION_IN  HSP_HW_IF_STREAM_BUFFER_DIRECTION_IN
#define HSP_SEQ_STREAM_BUFFER_DIRECTION_OUT  HSP_HW_IF_STREAM_BUFFER_DIRECTION_OUT

#define HSP_SEQ_TASK_COMPARATOR_0  HSP_HW_IF_TASK_COMPARATOR_0
#define HSP_SEQ_TASK_COMPARATOR_1  HSP_HW_IF_TASK_COMPARATOR_1
#define HSP_SEQ_TASK_COMPARATOR_2  HSP_HW_IF_TASK_COMPARATOR_2
#define HSP_SEQ_TASK_COMPARATOR_3  HSP_HW_IF_TASK_COMPARATOR_3

#define HSP_SEQ_STREAM_Enable(hmw)   HSP_HW_IF_SEQ_STREAM_Enable(((hmw)->hdriver))
#define HSP_SEQ_STREAM_Disable(hmw)  HSP_HW_IF_SEQ_STREAM_Disable(((hmw)->hdriver))
/**
  * @}
  */

/** @defgroup HSP_PROCLIST_Exported_Macros HSP Processing List Exported Macros
  * @{
  */
/**
  * @brief   Convert an event id number to a bit mask value
  * @param   evt_id  value in range [0..31]
  * @return  bitmask value
  */
#define HSP_SEQ_EVENT_ID_TO_BITMASK(evt_id)  (0x1UL << (evt_id))

 /**
  * @}
  */

/** @defgroup HSP_PROCLIST_Exported_Types HSP Processing List Exported Types
  * @{
  */
typedef enum
{
  HSP_SEQ_EVENT_1 = 1U,
  HSP_SEQ_EVENT_2 = 2U,
  HSP_SEQ_EVENT_3 = 3U,
  HSP_SEQ_EVENT_4 = 4U,
  HSP_SEQ_EVENT_5 = 5U,
  HSP_SEQ_EVENT_6 = 6U,
  HSP_SEQ_EVENT_7 = 7U,
  HSP_SEQ_EVENT_8 = 8U,
  HSP_SEQ_EVENT_9 = 9U,
  HSP_SEQ_EVENT_10 = 10U,
  HSP_SEQ_EVENT_11 = 11U,
  HSP_SEQ_EVENT_12 = 12U,
  HSP_SEQ_EVENT_13 = 13U,
  HSP_SEQ_EVENT_14 = 14U,
  HSP_SEQ_EVENT_15 = 15U,
  HSP_SEQ_EVENT_16 = 16U,
  HSP_SEQ_EVENT_17 = 17U,
  HSP_SEQ_EVENT_18 = 18U,
  HSP_SEQ_EVENT_19 = 19U,
  HSP_SEQ_EVENT_20 = 20U,
  HSP_SEQ_EVENT_21 = 21U,
  HSP_SEQ_EVENT_22 = 22U,
} hsp_seq_event_t;

#define hsp_seq_event_sync_t  hsp_hw_if_event_sync_t
#define hsp_seq_event_config_t  hsp_hw_if_event_config_t
#define hsp_seq_event_stream_config_t  hsp_hw_if_stream_config_t
#define hsp_seq_trgin_source_t  hsp_hw_if_trgin_source_t
#define hsp_seq_event_trigger_config_t  hsp_hw_if_trigger_config_t

/**
  * @}
  */

/** @defgroup HSP_PROCLIST_Exported_Macros HSP Processing List Exported Macros
  * @{
  */

/**
  * @}
  */

/** @defgroup HSP_PROCLIST_Exported_Functions HSP Processing List Exported Functions
  * @{
  */
/** @defgroup HSP_PROCLIST_Exported_Functions_Group1 HSP Processing List Recording
  * @{
  */
hsp_core_status_t HSP_SEQ_StartRecordPL(hsp_core_handle_t *hmw, uint32_t pl_id);
hsp_core_status_t HSP_SEQ_StopRecordPL(hsp_core_handle_t *hmw);
hsp_core_status_t HSP_SEQ_RunTask(hsp_core_handle_t *hmw, uint32_t pl_id, uint32_t timeout_ms);
hsp_core_status_t HSP_SEQ_ResetPL(hsp_core_handle_t *hmw);

/**
  * @}
  */

/** @defgroup HSP_PROCLIST_Exported_Functions_Group2 HSP Processing List Event Management
  * @{
  */
/* Event configuration -------------------------------------------------------*/
hsp_core_status_t HSP_SEQ_EVENT_SetConfig_StreamBuffer(hsp_core_handle_t *hmw, hsp_seq_event_t evt_id,
                                                       const hsp_seq_event_stream_config_t *p_config);
hsp_core_status_t HSP_SEQ_EVENT_SetConfig_Trigger(hsp_core_handle_t *hmw, hsp_seq_event_t evt_id,
                                                  const hsp_seq_event_trigger_config_t *p_config);
hsp_core_status_t HSP_SEQ_EVENT_SetConfig_CSEG(hsp_core_handle_t *hmw, hsp_seq_event_t evt_id,
                                               const hsp_seq_event_sync_t *p_sync);
hsp_core_status_t HSP_SEQ_EVENT_SetConfig_SPE(hsp_core_handle_t *hmw, hsp_seq_event_t evt_id,
                                              const hsp_seq_event_sync_t *p_sync);

/* Event Control -------------------------------------------------------------*/
hsp_core_status_t HSP_SEQ_EVENT_Enable(hsp_core_handle_t *hmw, uint32_t events_mask);
hsp_core_status_t HSP_SEQ_EVENT_Disable(hsp_core_handle_t *hmw, uint32_t events_mask);
uint32_t HSP_SEQ_EVENT_GetStatus(hsp_core_handle_t *hmw);

hsp_core_status_t HSP_SEQ_EVENT_Trig_CSEG(hsp_core_handle_t *hmw, uint32_t events_mask);
hsp_core_status_t HSP_SEQ_EVENT_PollForPending(hsp_core_handle_t *hmw, uint32_t events_mask, uint32_t timeout_ms);

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

#endif /* HSP_PROCLIST_H */
