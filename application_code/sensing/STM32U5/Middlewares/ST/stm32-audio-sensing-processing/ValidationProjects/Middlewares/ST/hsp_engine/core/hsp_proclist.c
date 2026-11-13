/**
  ******************************************************************************
  * @file    hsp_proclist.c
  * @author  MCD Application Team
  * @brief   This file implements the functions for the proclist management
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

/* Includes ------------------------------------------------------------------*/
#include "hsp_proclist.h"
#include "hsp_hw_if.h"

/** @addtogroup HSP_ENGINE
  * @{
  */

/** @addtogroup HSP_CORE HSP Core
  * @{
  */

/** @addtogroup HSP_PROCLIST
  * @{
  */

/** @addtogroup HSP_PROCLIST_Exported_Functions
  * @{
  */
/** @addtogroup HSP_PROCLIST_Exported_Functions_Group1
  * @{
  */
/**
  * @brief Start the record of a processing list.
  * @param hmw              HSP handle.
  * @param pl_id  ID of the new processing list.
  * @retval HSP_CORE_OK     Success.
  * @retval HSP_CORE_ERROR  Error.
  */
hsp_core_status_t HSP_SEQ_StartRecordPL(hsp_core_handle_t *hmw, uint32_t pl_id)
{
  HSP_ASSERT_DBG_PARAM((hmw != NULL));
  HSP_CHECK_UPDATE_STATE(hmw, global_state, HSP_CORE_STATE_IDLE, HSP_CORE_STATE_PROCLIST_RECORDING);

  if (HSP_HW_IF_SEQ_StartRecord(hmw->hdriver, pl_id) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }

  return HSP_CORE_OK;
}

/**
  * @brief Stop the record of a processing list.
  * @param hmw              HSP handle.
  * @retval HSP_CORE_OK     Success.
  * @retval HSP_CORE_ERROR  Error.
  */
hsp_core_status_t HSP_SEQ_StopRecordPL(hsp_core_handle_t *hmw)
{
  HSP_ASSERT_DBG_PARAM((hmw != NULL));
  HSP_CHECK_UPDATE_STATE(hmw, global_state, HSP_CORE_STATE_PROCLIST_RECORDING, HSP_CORE_STATE_IDLE);

  if (HSP_HW_IF_SEQ_StopRecord(hmw->hdriver) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }

  return HSP_CORE_OK;
}

/**
  * @brief Run a processing list with the lowest priority.
  * @param hmw              HSP handle.
  * @retval HSP_CORE_OK     Success.
  * @retval HSP_CORE_ERROR  Running failed.
  */
hsp_core_status_t HSP_SEQ_RunTask(hsp_core_handle_t *hmw, uint32_t pl_id, uint32_t timeout_ms)
{
  HSP_ASSERT_DBG_PARAM((hmw != NULL));
  HSP_ASSERT_DBG_STATE(hmw->global_state, (uint32_t)HSP_CORE_STATE_IDLE);

  if (HSP_HW_IF_SEQ_RunTask(hmw->hdriver, pl_id, timeout_ms) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }

  return HSP_CORE_OK;
}

/**
  * @brief Reset (remove) all the processing list in HSP.
  * @param hmw              HSP handle.
  * @retval HSP_CORE_OK     Success.
  * @retval HSP_CORE_ERROR  Running failed.
  */
hsp_core_status_t HSP_SEQ_ResetPL(hsp_core_handle_t *hmw)
{
  HSP_ASSERT_DBG_PARAM((hmw != NULL));
  HSP_ASSERT_DBG_STATE(hmw->global_state, (uint32_t)HAL_HSP_STATE_READY);

  if (HSP_HW_IF_SEQ_ResetPL(hmw->hdriver) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }

  return HSP_CORE_OK;  
}
/**
  * @}
  */

/** @addtogroup HSP_PROCLIST_Exported_Functions_Group2
  * @{
  */
/* Event configuration -------------------------------------------------------*/
hsp_core_status_t HSP_SEQ_EVENT_SetConfig_StreamBuffer(hsp_core_handle_t *hmw, hsp_seq_event_t evt_id,
                                                       const hsp_seq_event_stream_config_t *p_config)
{
  HSP_ASSERT_DBG_PARAM((hmw != NULL));
  HSP_ASSERT_DBG_PARAM((p_config != NULL));

  HSP_ASSERT_DBG_STATE(hmw->global_state, (uint32_t)HSP_CORE_STATE_IDLE);

  if (HSP_HW_IF_EVENT_SetConfig_StreamBuffer(hmw->hdriver, (uint32_t)evt_id, p_config) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }

  return HSP_CORE_OK;
}

hsp_core_status_t HSP_SEQ_EVENT_SetConfig_Trigger(hsp_core_handle_t *hmw, hsp_seq_event_t evt_id,
                                          const hsp_seq_event_trigger_config_t *p_config)
{
  HSP_ASSERT_DBG_PARAM((hmw != NULL));
  HSP_ASSERT_DBG_PARAM((p_config != NULL));

  HSP_ASSERT_DBG_STATE(hmw->global_state, (uint32_t)HSP_CORE_STATE_IDLE);

  if (HSP_HW_IF_EVENT_SetConfig_Trigger(hmw->hdriver, (uint32_t)evt_id, p_config) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }

  return HSP_CORE_OK;
}

hsp_core_status_t HSP_SEQ_EVENT_SetConfig_CSEG(hsp_core_handle_t *hmw, hsp_seq_event_t evt_id,
                                               const hsp_hw_if_event_sync_t *p_sync)
{
  HSP_ASSERT_DBG_PARAM((hmw != NULL));
  HSP_ASSERT_DBG_PARAM((p_sync != NULL));

  HSP_ASSERT_DBG_STATE(hmw->global_state, (uint32_t)HSP_CORE_STATE_IDLE);

  if (HSP_HW_IF_EVENT_SetConfig_CSEG(hmw->hdriver, (uint32_t)evt_id, p_sync) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }

  return HSP_CORE_OK;
}

hsp_core_status_t HSP_SEQ_EVENT_SetConfig_SPE(hsp_core_handle_t *hmw, hsp_seq_event_t evt_id,
                                              const hsp_hw_if_event_sync_t *p_sync)
{
  HSP_ASSERT_DBG_PARAM((hmw != NULL));
  HSP_ASSERT_DBG_PARAM((p_sync != NULL));

  HSP_ASSERT_DBG_STATE(hmw->global_state, (uint32_t)HSP_CORE_STATE_IDLE);

  if (HSP_HW_IF_EVENT_SetConfig_SPE(hmw->hdriver, (uint32_t)evt_id, p_sync) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }

  return HSP_CORE_OK;
}

/* Event Control -------------------------------------------------------------*/
hsp_core_status_t HSP_SEQ_EVENT_Enable(hsp_core_handle_t *hmw, uint32_t events_mask)
{
  HSP_ASSERT_DBG_PARAM((hmw != NULL));
  HSP_ASSERT_DBG_STATE(hmw->global_state, (uint32_t)HSP_CORE_STATE_IDLE);

  if (HSP_HW_IF_EVENT_Enable(hmw->hdriver, events_mask) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }

  return HSP_CORE_OK;
}

hsp_core_status_t HSP_SEQ_EVENT_Disable(hsp_core_handle_t *hmw, uint32_t events_mask)
{
  HSP_ASSERT_DBG_PARAM((hmw != NULL));
  HSP_ASSERT_DBG_STATE(hmw->global_state, (uint32_t)HSP_CORE_STATE_IDLE);

  if (HSP_HW_IF_EVENT_Disable(hmw->hdriver, events_mask) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }

  return HSP_CORE_OK;
}

uint32_t HSP_SEQ_EVENT_GetStatus(hsp_core_handle_t *hmw)
{
  HSP_ASSERT_DBG_PARAM((hmw != NULL));

  return HSP_HW_IF_EVENT_GetStatus(hmw->hdriver);
}

hsp_core_status_t HSP_SEQ_EVENT_Trig_CSEG(hsp_core_handle_t *hmw, uint32_t events_mask)
{
  HSP_ASSERT_DBG_PARAM((hmw != NULL));
  HSP_ASSERT_DBG_STATE(hmw->global_state, (uint32_t)HSP_CORE_STATE_IDLE);

  if (HSP_HW_IF_SEQ_EVENT_Trig_CSEG(hmw->hdriver, events_mask) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }

  return HSP_CORE_OK;
}

hsp_core_status_t HSP_SEQ_EVENT_PollForPending(hsp_core_handle_t *hmw, uint32_t events_mask, uint32_t timeout_ms)
{
  HSP_ASSERT_DBG_PARAM((hmw != NULL));
  HSP_ASSERT_DBG_STATE(hmw->global_state, (uint32_t)HSP_CORE_STATE_IDLE);

//  if (HSP_HW_IF_SEQ_EVENT_PollForPending(hmw->hdriver, events_mask) != HSP_IF_OK)
//  {
//    return HSP_CORE_ERROR;
//  }

  return HSP_CORE_OK;
}


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

/**
  * @}
  */
