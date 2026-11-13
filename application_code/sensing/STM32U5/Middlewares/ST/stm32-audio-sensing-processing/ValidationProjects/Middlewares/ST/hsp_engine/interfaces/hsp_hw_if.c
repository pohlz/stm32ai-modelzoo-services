/**
  ******************************************************************************
  * @file    hsp_hw_if.c
  * @author  GPM Application Team
  * @brief   This file implements the service provide by the HSP Hardware
  *          Interface
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
#include "hsp_hw_if.h"
#include "hsp_conf.h"
#include "hsp_if_conf.h"
#include "hsp_fw_def_generic.h"


/** @addtogroup HSP_ENGINE
  * @{
  */

/** @addtogroup HSP_INTERFACES
  * @{
  */

/** @addtogroup HSP_HW_IF
  * @{
  */

/* Private types -----------------------------------------------------------*/
/** @defgroup HSP_HW_IF_Private_Types HSP_HW_IF Private Types
  * @{
  */

/**
  * @}
  */

/* Private constants ---------------------------------------------------------*/
/** @defgroup HSP_HW_IF_Private_Constants HSP_HW_IF Private Constants
  * @{
  */

#define HSP_HW_IF_EVENT_SOURCE_GET_ENUM_FOR_CSEG  HAL_HSP_EVENT_SOURCE_GET_ENUM_FOR_CSEG
/**
  * @}
  */

/* Private macros -------------------------------------------------------------*/
/** @defgroup HSP_HW_IF_Private_Macros HSP_HW_IF Private Macros
  * @{
  */
#define POSITION_FIRST_ZERO(VAL) (__CLZ(__RBIT(~(VAL))))
#if defined(HAL_HSP_SMARTCLOCKING_HSPDMA)
#define HSP_HW_IF_GET_CLOCK(clock_id)                                               \
  (((clock_id) == HSP_HW_IF_SMARTCLOCKING_CTRL)?HAL_HSP_SMARTCLOCKING_CTRL:       \
   ((clock_id) == HSP_HW_IF_SMARTCLOCKING_SPE)?HAL_HSP_SMARTCLOCKING_SPE:         \
   ((clock_id) == HSP_HW_IF_SMARTCLOCKING_MMC)?HAL_HSP_SMARTCLOCKING_MMC: bb      \
   ((clock_id) == HSP_HW_IF_SMARTCLOCKING_HSPDMA)?HAL_HSP_SMARTCLOCKING_HSPDMA:0U)
#else
#define HSP_HW_IF_GET_CLOCK(clock_id)                                          \
  (((clock_id) == HSP_HW_IF_SMARTCLOCKING_CTRL)?HAL_HSP_SMARTCLOCKING_CTRL:  \
   ((clock_id) == HSP_HW_IF_SMARTCLOCKING_SPE)?HAL_HSP_SMARTCLOCKING_SPE:    \
   ((clock_id) == HSP_HW_IF_SMARTCLOCKING_MMC)?HAL_HSP_SMARTCLOCKING_MMC:0U)
#endif /* HAL_HSP_SMARTCLOCKING_HSPDMA */
/**
  * @}
  */

/* Private functions ---------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/

/* Exported functions --------------------------------------------------------*/
/** @addtogroup HSP_HW_IF_Exported_Functions
  * @{
  */

/**
  * @brief  Deinit the interface
  * @param  hdriver   HSP Driver handle.
  * @return None
  */
void HSP_HW_IF_DeInit(void *hdriver)
{
  HSP_ASSERT_DBG_PARAM(hdriver != NULL);

  (void) HAL_HSP_DeInit((hal_hsp_handle_t *)(hdriver));
}

/**
  * @brief  Load a Firmware OpCode & data respectively in CRAM & DRAM
  * @param  hdriver   HSP Driver handle.
  * @param  p_fw      Pointer to Firmware structure
  * @return HSP_IF_OK     Load complete successfully
  * @return HSP_IF_ERROR  Load failed
  */
hsp_if_status_t HSP_HW_IF_LoadFW(void *hdriver, const hsp_hw_if_firmware_t *p_fw)
{
  assert_param(hdriver != NULL);

  if (HAL_HSP_LoadFirmware((hal_hsp_handle_t *)(hdriver), p_fw->p_fw_cram_bin, p_fw->fw_cram_size_in_word,
                           p_fw->p_fw_dram_bin, p_fw->fw_dram_size_in_word) != HAL_OK)
  {
    return HSP_IF_ERROR;
  }

  return HSP_IF_OK;
}

/**
  * @brief  Load a Plugin in CRAM.
  * @param  hdriver    HSP Driver handle.
  * @param  p_bin      Pointer to the Plugin data.
  * @param  size_word  Size in word of Plugin data.
  * @return HSP_IF_OK     Load complete successfully
  * @return HSP_IF_ERROR  Load failed
  */
hsp_if_status_t HSP_HW_IF_LoadPlugin(void *hdriver, const uint32_t *p_bin, uint32_t size_word)
{
  HSP_ASSERT_DBG_PARAM(hdriver != NULL);
  HSP_ASSERT_DBG_PARAM(p_bin != NULL);

  if (HAL_HSP_LoadPlugin((hal_hsp_handle_t *)(hdriver), p_bin, size_word) != HAL_OK)
  {
    return HSP_IF_ERROR;
  }

  return HSP_IF_OK;
}

/**
  * @brief Boot the HSP.
  * @param  hdriver    HSP Driver handle.
  * @param enableCycleCountEnable  Enable cycle count on HSP side
  * @return HSP_IF_OK     Boot complete successfully
  * @return HSP_IF_ERROR  Boot failed
  */
hsp_if_status_t HSP_HW_IF_Boot(void *hdriver, uint32_t enableCycleCount)
{
  HSP_ASSERT_DBG_PARAM(hdriver != NULL);

  hal_hsp_handle_t *hhsp = (hal_hsp_handle_t *)(hdriver);
  hal_hsp_boot_config_t boot_cfg;
  boot_cfg.boot_cmd_id = HSP_CMD_FW_START;
  boot_cfg.boot_success_code = HSP_BSTAT_BOOTOK;
  boot_cfg.perf_mon = (hal_hsp_perf_monitor_t)enableCycleCount;
  boot_cfg.perf_counter_offset = HSP_DRAM_CC_ADDR;

  if (HAL_HSP_Boot(hhsp, &boot_cfg, HSP_BOOT_TIMEOUT_MS) != HAL_OK)
  {
    return HSP_IF_ERROR;
  }
#if 0 /* VALERIE */
  hal_hsp_fw_version_t rom_version;
  if (HAL_HSP_GetCROMVersion(&(h_hw_hsp_if[hsp_id].hhsp), &rom_version) != HAL_OK)
  {
    return HSP_IF_ERROR;
  }

  if (rom_version.major != HSP_FW_DEF_ROM_VERSION_MAJOR)
  {
    return HSP_IF_ERROR;
  }

  if (rom_version.minor != HSP_FW_DEF_ROM_VERSION_MINOR)
  {
    return HSP_IF_ERROR;
  }

  hal_hsp_fw_version_t ram_version;
  if (HAL_HSP_GetCRAMVersion(&(h_hw_hsp_if[hsp_id].hhsp), &ram_version) != HAL_OK)
  {
    return HSP_IF_ERROR;
  }

  if (ram_version.major != HSP_FW_DEF_RAM_VERSION_MAJOR)
  {
    return HSP_IF_ERROR;
  }

  if (ram_version.minor != HSP_FW_DEF_RAM_VERSION_MINOR)
  {
    return HSP_IF_ERROR;
  }
#endif
  return HSP_IF_OK;
}

/* API to manage the SmartClocking -------------------------------------------*/
/**
  * @brief Enable SmartClocking capabilities of HSP blocks.
  * @param  hdriver   HSP Driver handle.
  * @param clock_id   Clock ID
  * @return HSP_IF_OK     Action complete successfully
  * @return HSP_IF_ERROR  Action failed
  */
hsp_if_status_t HSP_HW_IF_EnableSmartClocking(void *hdriver, uint32_t clock_id)
{
  HSP_ASSERT_DBG_PARAM(hdriver != NULL);

  hal_hsp_handle_t *hhsp = (hal_hsp_handle_t *)(hdriver);
  uint32_t clock = HSP_HW_IF_GET_CLOCK(clock_id);
  if (clock == 0U)
  {
    return HSP_IF_ERROR;
  }

  if (HAL_HSP_SMARTCLOCKING_Enable(hhsp, clock) != HAL_OK)
  {
    return HSP_IF_ERROR;
  }
  return HSP_IF_OK;
}

/**
  * @brief Disable SmartClocking capabilities of HSP blocks.
  * @param  hdriver   HSP Driver handle.
  * @param  clock_id  Clock ID
  * @return HSP_IF_OK     Action complete successfully
  * @return HSP_IF_ERROR  Action failed
  */
hsp_if_status_t HSP_HW_IF_DisableSmartClocking(void *hdriver, uint32_t clock_id)
{
  HSP_ASSERT_DBG_PARAM(hdriver != NULL);

  hal_hsp_handle_t *hhsp = (hal_hsp_handle_t *)(hdriver);
  uint32_t clock = HSP_HW_IF_GET_CLOCK(clock_id);
  if (clock == 0U)
  {
    return HSP_IF_ERROR;
  }

  if (HAL_HSP_SMARTCLOCKING_Disable(hhsp, clock) != HAL_OK)
  {
    return HSP_IF_ERROR;
  }

  return HSP_IF_OK;
}

/* API to manage OUTPUT ------------------------------------------------------*/
/**
  * @brief Configure the source of a TRGO output.
  * @param  hdriver   HSP Driver handle.
  * @param  trgo_id   TRGO output ID.
  * @param  source    Source value.
  * @return HSP_IF_OK     Action complete successfully
  * @return HSP_IF_ERROR  Action failed
  */
hsp_if_status_t HSP_HW_IF_OUTPUT_SetConfig(void *hdriver, hsp_hw_if_output_trigger_t trgo_id, uint32_t source)
{
  HSP_ASSERT_DBG_PARAM(hdriver != NULL);

  hal_hsp_handle_t *hhsp = (hal_hsp_handle_t *)(hdriver);

  if (HAL_HSP_OUTPUT_SetConfigTrigger(hhsp, trgo_id, (hal_hsp_output_trigger_source_t)source) != HAL_OK)
  {
    return HSP_IF_ERROR;
  }
  return HSP_IF_OK;
}

/**
  * @brief Enable GPO & TRGO outputs.
  * @param  hdriver   HSP Driver handle.
  * @return HSP_IF_OK     Action complete successfully
  * @return HSP_IF_ERROR  Action failed
  */
hsp_if_status_t HSP_HW_IF_OUTPUT_Enable(void *hdriver)
{
  HSP_ASSERT_DBG_PARAM(hdriver != NULL);

  hal_hsp_handle_t *hhsp = (hal_hsp_handle_t *)(hdriver);

  if (HAL_HSP_OUTPUT_Enable(hhsp) != HAL_OK)
  {
    return HSP_IF_ERROR;
  }
  return HSP_IF_OK;
}

/**
  * @brief Enable GPO & TRGO outputs.
  * @param  hdriver   HSP Driver handle.
  * @return HSP_IF_OK     Action complete successfully
  * @return HSP_IF_ERROR  Action failed
  */
hsp_if_status_t HSP_HW_IF_OUTPUT_Disable(void *hdriver)
{
  HSP_ASSERT_DBG_PARAM(hdriver != NULL);

  hal_hsp_handle_t *hhsp = (hal_hsp_handle_t *)(hdriver);

  if (HAL_HSP_OUTPUT_Disable(hhsp) != HAL_OK)
  {
    return HSP_IF_ERROR;
  }

  return HSP_IF_OK;
}

/**
  * @brief Return the enabling status of HSP Outputs
  * @param  hdriver   HSP Driver handle.
  * @return IF_ACTIVATION_DISABLED  HSP Outputs are disabled
  * @return IF_ACTIVATION_ENABLED   HSP Outputs are enabled
  */
if_activation_status_t HSP_HW_IF_OUTPUT_IsEnabled(void *hdriver)
{
  HSP_ASSERT_DBG_PARAM(hdriver != NULL);

  hal_hsp_handle_t *hhsp = (hal_hsp_handle_t *)(hdriver);

  if (HAL_HSP_OUTPUT_IsEnabled(hhsp) == HAL_HSP_OUTPUT_DISABLED)
  {
    return IF_ACTIVATION_DISABLED;
  }
  return IF_ACTIVATION_ENABLED;
}

/* API to manage Direct Command ----------------------------------------------*/
/**
  * @brief Send a command with MsgBox Interface.
  * @param  hdriver     HSP Driver handle.
  * @param  command_id  Command ID.
  * @return HSP_IF_OK     Action complete successfully
  * @return HSP_IF_ERROR  Action failed
  */
hsp_if_status_t HSP_HW_IF_SendCommand(void *hdriver, uint32_t command_id)
{
  HSP_ASSERT_DBG_PARAM(hdriver != NULL);

  hal_hsp_handle_t *hhsp = (hal_hsp_handle_t *)(hdriver);

  if (HAL_HSP_MSGBOX_SendCommand(hhsp, command_id, HSP_TIMEOUT_MS) != HAL_OK)
  {
    return HSP_IF_ERROR;
  }

  return HSP_IF_OK;
}

/* API to manage Processing List ---------------------------------------------*/
/**
  * @brief  Start the record of a Processing List
  * @param  hdriver  HSP Driver handle.
  * @param  pl_id    Processing List ID
  * @return HSP_IF_OK     Action complete successfully
  * @return HSP_IF_ERROR  Action failed
  */
hsp_if_status_t HSP_HW_IF_SEQ_StartRecord(void *hdriver, uint32_t pl_id)
{

  HSP_ASSERT_DBG_PARAM(hdriver != NULL);

  hal_hsp_handle_t *hhsp = (hal_hsp_handle_t *)(hdriver);

  if (HAL_HSP_TASK_Run(hhsp, HAL_HSP_TASK_SUPERVISOR, HSP_TIMEOUT_MS) != HAL_OK)
  {
    return HSP_IF_ERROR;
  }
  HAL_HSP_WriteParameter(hhsp, HAL_HSP_PARAM_0, pl_id);

  if (HAL_HSP_MSGBOX_SendCommand(hhsp, HSP_CMD_PRO_CFG_START, HSP_TIMEOUT_MS) != HAL_OK)
  {
    return HSP_IF_ERROR;
  }
  return HSP_IF_OK;
}

/**
  * @brief  Stop the record of the Processing List
  * @param  hdriver   HSP Driver handle.
  * @return HSP_IF_OK     Action complete successfully
  * @return HSP_IF_ERROR  Action failed
  */
hsp_if_status_t HSP_HW_IF_SEQ_StopRecord(void *hdriver)
{
  HSP_ASSERT_DBG_PARAM(hdriver != NULL);

  hal_hsp_handle_t *hhsp = (hal_hsp_handle_t *)(hdriver);

  if (HAL_HSP_MSGBOX_SendCommand(hhsp, HSP_CMD_PRO_CFG_END, HSP_TIMEOUT_MS) != HAL_OK)
  {
    return HSP_IF_ERROR;
  }

  return HSP_IF_OK;
}

/**
  * @brief  Reset all Processing List
  * @retval HSP_IF_OK success
  *         HSP_IF_ERROR failure
  */
hsp_if_status_t HSP_HW_IF_SEQ_ResetPL(void *hdriver)
{
  HSP_ASSERT_DBG_PARAM(hdriver != NULL);
  hal_hsp_handle_t *hhsp = (hal_hsp_handle_t *)(hdriver);

  if (HAL_HSP_TASK_Run(hhsp, HAL_HSP_TASK_SUPERVISOR, HSP_TIMEOUT_MS) != HAL_OK)
  {
    return HSP_IF_ERROR;
  }

  HAL_HSP_WriteParameter(hhsp, HAL_HSP_PARAM_1, ENABLE_HSP_PERFORMANCE_MONITOR);
  if (HAL_HSP_MSGBOX_SendCommand(hhsp, HSP_CMD_PL_RESET, HSP_TIMEOUT_MS) != HAL_OK)
  {
    return HSP_IF_ERROR;
  }
  return HSP_IF_OK;
}

hsp_if_status_t HSP_HW_IF_SEQ_RunTask(void *hdriver, uint32_t task_id, uint32_t timeout_ms)
{
  HSP_ASSERT_DBG_PARAM(hdriver != NULL);

  hal_hsp_handle_t *hhsp = (hal_hsp_handle_t *)(hdriver);

  if (HAL_HSP_TASK_Run(hhsp, (hal_hsp_task_t)task_id, timeout_ms) != HAL_OK)
  {
    return HSP_IF_ERROR;
  }

  return HSP_IF_OK;
}

/* API to manage Trigger & Event configuration -------------------------------*/
/**
  * @brief Configure an event with a STREAM Buffer as source.
  * @param  hdriver   HSP Driver handle.
  * @param  event_id  Event ID.
  * @param  p_config  Pointer to the config
  * @note  The enable of STREAM is not perform at the end of the configuration wotherwise it will be impossible
  *        to configure the next Stream Buffer.
  *        The application must do it after all STREAM Buffer configurations.
  * @return HSP_IF_OK     Action complete successfully
  * @return HSP_IF_ERROR  Action failed
  */
hsp_if_status_t HSP_HW_IF_EVENT_SetConfig_StreamBuffer(void *hdriver, uint32_t event_id,
                                                   const hsp_hw_if_stream_config_t *p_config)
{
  HSP_ASSERT_DBG_PARAM(hdriver != NULL);
  HSP_ASSERT_DBG_PARAM(p_config != NULL);

  hal_hsp_handle_t *hhsp = (hal_hsp_handle_t *)(hdriver);

  /* Config the stream */
  if (HAL_HSP_STREAM_SetConfig(hhsp, (HAL_HSP_STREAM_BufferTypeDef)(p_config->instance), &(p_config->itf)) != HAL_OK)
  {
    return HSP_IF_ERROR;
  }

  /* Configure the Event */
  if (HAL_HSP_EVENT_SetConfig(hhsp, &(p_config->evt_cfg)) != HAL_OK)
  {
    return HSP_IF_ERROR;
  }

  return HSP_IF_OK;
}

/**
  * @brief Configure an event with a TRGIN trigger as source.
  * @param  hdriver   HSP Driver handle.
  * @param  event_id  Event ID.
  * @param  p_config  Pointer to the config
  * @return HSP_IF_OK     Action complete successfully
  * @return HSP_IF_ERROR  Action failed
  */
hsp_if_status_t HSP_HW_IF_EVENT_SetConfig_Trigger(void *hdriver, uint32_t event_id,
                                              const hsp_hw_if_trigger_config_t *p_config)
{
  HSP_ASSERT_DBG_PARAM(hdriver != NULL);
  HSP_ASSERT_DBG_PARAM(p_config != NULL);

  hal_hsp_handle_t *hhsp = (hal_hsp_handle_t *)(hdriver);

  /* Config the TRGINx */
  if (HAL_HSP_TRGIN_SetConfig(hhsp, p_config->instance, &(p_config->itf)) != HAL_OK)
  {
    return HSP_IF_ERROR;
  }

  /* Configure the Event */
  if (HAL_HSP_EVENT_SetConfig(hhsp, &(p_config->evt_cfg)) != HAL_OK)
  {
    return HSP_IF_ERROR;
  }

  /* Enable the TRGINx instance */
  if (HAL_HSP_TRGIN_Enable(hhsp, p_config->instance) != HAL_OK)
  {
    return HSP_IF_ERROR;
  }

  return HSP_IF_OK;
}

/**
  * @brief Configure an event with the CSEG interface as source.
  * @param  hdriver   HSP Driver handle.
  * @param  event_id  Event ID.
  * @param  p_sync    Pointer to the Event Synchronization config
  * @return HSP_IF_OK     Action complete successfully
  * @return HSP_IF_ERROR  Action failed
  */
hsp_if_status_t HSP_HW_IF_EVENT_SetConfig_CSEG(void *hdriver, uint32_t event_id, const hsp_hw_if_event_sync_t *p_sync)
{
  HSP_ASSERT_DBG_PARAM(hdriver != NULL);
  HSP_ASSERT_DBG_PARAM(p_sync != NULL);

  hal_hsp_handle_t *hhsp = (hal_hsp_handle_t *)(hdriver);
  hal_hsp_event_config_t evt_cfg;

  evt_cfg.source = HAL_HSP_EVENT_SOURCE_GET_ENUM_FOR_CSEG(event_id);
  evt_cfg.sync.state = p_sync->state;
  evt_cfg.sync.tcu = p_sync->tcu;

  if (HAL_HSP_EVENT_SetConfig(hhsp, &evt_cfg) != HAL_OK)
  {
    return HSP_IF_ERROR;
  }

  return HSP_IF_OK;
}

/**
  * @brief Configure an event with the HSEG interface as source.
  * @param  hdriver   HSP Driver handle.
  * @param  event_id  Event ID.
  * @param  p_sync    Pointer to the Event Synchronization config
  * @return HSP_IF_OK     Action complete successfully
  * @return HSP_IF_ERROR  Action failed
  */
hsp_if_status_t HSP_HW_IF_EVENT_SetConfig_SPE(void *hdriver, uint32_t event_id, const hsp_hw_if_event_sync_t *p_sync)
{
  HSP_ASSERT_DBG_PARAM(hdriver != NULL);
  HSP_ASSERT_DBG_PARAM(p_sync != NULL);

  hal_hsp_handle_t *hhsp = (hal_hsp_handle_t *)(hdriver);
  hal_hsp_event_config_t evt_cfg;

  evt_cfg.source = HAL_HSP_EVENT_SOURCE_GET_ENUM_FOR_HSEG(event_id);
  evt_cfg.sync.state = p_sync->state;
  evt_cfg.sync.tcu = p_sync->tcu;

  if (HAL_HSP_EVENT_SetConfig(hhsp, &evt_cfg) != HAL_OK)
  {
    return HSP_IF_ERROR;
  }

  return HSP_IF_OK;
}

hsp_if_status_t HSP_HW_IF_EVENT_Enable(void *hdriver, uint32_t events_mask)
{
  HSP_ASSERT_DBG_PARAM(hdriver != NULL);

  hal_hsp_handle_t *hhsp = (hal_hsp_handle_t *)(hdriver);

  if (HAL_HSP_EVENT_Enable(hhsp, events_mask) != HAL_OK)
  {
    return HSP_IF_ERROR;
  }

  return HSP_IF_OK;
}

hsp_if_status_t HSP_HW_IF_EVENT_Disable(void *hdriver, uint32_t events_mask)
{
  HSP_ASSERT_DBG_PARAM(hdriver != NULL);

  hal_hsp_handle_t *hhsp = (hal_hsp_handle_t *)(hdriver);

  if (HAL_HSP_EVENT_Disable(hhsp, events_mask) != HAL_OK)
  {
    return HSP_IF_ERROR;
  }

  return HSP_IF_OK;
}

hsp_if_status_t HSP_HW_IF_SEQ_EVENT_Trig_CSEG(void *hdriver, uint32_t events)
{
  HSP_ASSERT_DBG_PARAM(hdriver != NULL);

  hal_hsp_handle_t *hhsp = (hal_hsp_handle_t *)(hdriver);

  if (HAL_HSP_EVENT_RequestSWTrigger(hhsp, events) != HAL_OK)
  {
    return HSP_IF_ERROR;
  }

  return HSP_IF_OK;
}


/* API to BRAM Access control ------------------------------------------------*/
hsp_if_status_t HSP_HW_IF_BRAM_EnableConflictAccessCounter(void *hdriver)
{
  HSP_ASSERT_DBG_PARAM(hdriver != NULL);

  hal_hsp_handle_t *hhsp = (hal_hsp_handle_t *)(hdriver);

  if (HAL_HSP_BRAM_EnableConflictAccessCounter(hhsp) != HAL_OK)
  {
    return HSP_IF_ERROR;
  }

  return HSP_IF_OK;
}

hsp_if_status_t HSP_HW_IF_BRAM_DisableConflictAccessCounter(void *hdriver)
{
  HSP_ASSERT_DBG_PARAM(hdriver != NULL);

  hal_hsp_handle_t *hhsp = (hal_hsp_handle_t *)(hdriver);

  if (HAL_HSP_BRAM_DisableConflictAccessCounter(hhsp) != HAL_OK)
  {
    return HSP_IF_ERROR;
  }

  return HSP_IF_OK;
}

uint32_t HSP_HW_IF_BRAM_GetConflictAccessCounter(void *hdriver)
{
  HSP_ASSERT_DBG_PARAM(hdriver != NULL);

  hal_hsp_handle_t *hhsp = (hal_hsp_handle_t *)(hdriver);

  return HAL_HSP_BRAM_GetConflictAccessCounter(hhsp);
}

hsp_if_status_t HSP_HW_IF_BRAM_SetBandwidthArbitration(void *hdriver, uint32_t mode)
{
  HSP_ASSERT_DBG_PARAM(hdriver != NULL);

  hal_hsp_handle_t *hhsp = (hal_hsp_handle_t *)(hdriver);

  if (HAL_HSP_BRAM_SetBandwidthArbitration(hhsp, (hal_hsp_bram_arbitration_t) mode) != HAL_OK)
  {
    return HSP_IF_ERROR;
  }

  return HSP_IF_OK;
}

uint32_t HSP_HW_IF_BRAM_GetBandwidthArbitration(void *hdriver)
{
  HSP_ASSERT_DBG_PARAM(hdriver != NULL);

  hal_hsp_handle_t *hhsp = (hal_hsp_handle_t *)(hdriver);

  return (uint32_t)HAL_HSP_BRAM_GetBandwidthArbitration(hhsp);
}

/* API for secure Access -----------------------------------------------------*/
hsp_if_status_t HSP_HW_IF_Lock(void *hdriver)
{
  HSP_ASSERT_DBG_PARAM(hdriver != NULL);

  hal_hsp_handle_t *hhsp = (hal_hsp_handle_t *)(hdriver);

  if (HAL_HSP_Lock(hhsp) != HAL_OK)
  {
    return HSP_IF_ERROR;
  }

  return HSP_IF_OK;
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
