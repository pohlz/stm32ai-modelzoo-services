/**
  ******************************************************************************
  * @file    hsp_core.c
  * @author  MCD Application Team
  * @brief   This file implements the HSP Core functions.
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
#include "hsp_core.h"
#include "hsp_bram_if.h"
#include "hsp_hw_if.h"
#include "hsp_fw_ram_generic.h"


/** @addtogroup HSP_ENGINE
  * @{

# How to use the HSP Engine Middleware

## Usage

1. Declare a hal_play_handle_t handle and initialize the PLAY driver with a PLAY instance by calling HAL_PLAY_Init().
  The PLAY clock is enabled inside HAL_PLAY_Init() if USE_HAL_PLAY_CLK_ENABLE_MODEL > HAL_CLK_ENABLE_NO.

2. Configure the low-level hardware(CLOCK, NVIC...):
  - Enable the PLAY Bus clock if USE_HAL_PLAY_CLK_ENABLE_MODEL = HAL_CLK_ENABLE_NO.
  - PLAYx pins configuration:
      - Enable the clock for the PLAYx GPIOs
      - Configure PLAYx pins as alternate function
  - NVIC configuration if you need to use PLAY interrupt process:
      - Configure the PLAY interrupt priority by calling HAL_NVIC_SetPriority().
      - Enable the PLAY IRQ by calling HAL_NVIC_EnableIRQ().
      - In PLAY IRQ handler, call HAL_PLAY_IRQHandler().

3. Configure the ...
4. Call the ...
5. Use operating functions:
  - Polling mode:
    - ....
  - Interrupt mode:
    - ...

6. ...

7. Call HSP_CORE_DeInit() to deinitialize the HSP Engine Middleware.

## Callback registration
  When the preprocessor directive USE_HSP_REGISTER_CALLBACKS is set to 1, the user can dynamically configure the
  driver callbacks, via its own method:

Callback name                | Default value                           | Callback registration function
---------------------------- | --------------------------------------- | ----------------------------------------------
xxxCallback                  | HAL_HSP_xxxCallback()                   | HSP_CORE_RegisterSWTriggerWriteCpltCallback()

  To unregister a callback, the default one must be registered.

  By default, after HSP_CORE_Init() and when the state is HSP_CORE_STATE_INIT, all callbacks are set to the
  corresponding weak functions.

  Callbacks can be registered in HSP_CORE_STATE_INIT or HSP_CORE_STATE_IDLE states.

  When the preprocessing directive USE_HSP_REGISTER_CALLBACKS is set to 0 or is undefined,
  the callback registration feature is not available and callbacks are set to the corresponding weak functions.

## HSP driver configuration

  Config defines                  | Where            | Default value     | Note
  ------------------------------- | ---------------- | ----------------- | --------------------------------------------
  USE_HSP_REGISTER_CALLBACKS      |    hsp_conf.h    |        0U         | Enable the registration of callbacks
  USE_HSP_CHECK_PARAM             |    hsp_conf.h    |        0U         | Check vital parameters at runtime
  USE_HSP_CHECK_PROCESS_STATE     |    hsp_conf.h    |        0U         | Check HSP MW state transition at runtime
  USE_HSP_ASSERT_DBG_PARAM        | PreProcessor env |        NA         | Enable parameter assertions
  USE_HSP_ASSERT_DBG_STATE        | PreProcessor env |        NA         | Enable state assertions
  */

/** @addtogroup HSP_CORE
  * @{
  */

/** @addtogroup HSP_CORE_Init
  * @{
  */

/** @defgroup HSP_CORE_Init_Private_Functions HSP Core Private Functions
  * @{
  */
hsp_core_status_t HSP_CORE_LoadFW(hsp_core_handle_t *hmw);

/**
  * @}
  */

/** @addtogroup HSP_CORE_Exported_Functions
  * @{
  */
/** @defgroup HSP_CORE_Exported_Functions_Group1 HSP Init functions
  * @{
  */
/**
  * @brief  Initialize the MW HSP according to the associated handle.
  * @param  hmw   Pointer to a hsp_core_handle_t.
  * @retval HSP_CORE_OK            MW HSP handle has been correctly initialized.
  * @retval HSP_CORE_INVALID_PARAM One of parameter is NULL
  * @retval HSP_CORE_ERROR         MW HSP handle not initialized
  */
hsp_core_status_t HSP_CORE_Init(hsp_core_handle_t *hmw)
{
  HSP_ASSERT_DBG_PARAM((hmw != NULL));

#if defined(USE_HSP_CHECK_PARAM) && (USE_HSP_CHECK_PARAM == 1)
  /* Check whether the HSP handle is valid */
  if (hmw == NULL)
  {
    return HSP_CORE_INVALID_PARAM;
  }
#endif /* USE_HSP_CHECK_PARAM */

  /* Interfaces Initialization ---------------------------------------------------------------------------------------*/
  /* Initialize the BRAM memory interface */
  if (HSP_BRAM_IF_Init(&(hmw->hbram), hmw->hdriver) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }

  /* Load the Firmware & Plugin --------------------------------------------------------------------------------------*/
  /* Call the weak function to load Firmware */
  if (HSP_CORE_LoadFW(hmw) != HSP_CORE_OK)
  {
    return HSP_CORE_ERROR;
  }

#if defined(USE_HSP_PLUGIN) && (USE_HSP_PLUGIN == 1)
  /* Call the weak function to load Plugins */
  if (HSP_CORE_LoadPlugins(hmw) != HSP_CORE_OK)
  {
    return HSP_CORE_ERROR;
  }
#endif /* USE_HSP_PLUGIN */

  /* Change HSP Middleware state */
  hmw->global_state = HSP_CORE_STATE_READY;
 
  /* Boot the HSP */
  if (HSP_HW_IF_Boot(hmw->hdriver, ENABLE_HSP_PERFORMANCE_MONITOR) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  /* Change HSP Middleware state */
  hmw->global_state = HSP_CORE_STATE_IDLE;

  return HSP_CORE_OK;
}

/**
  * @brief  Load the HSP Firmware.
  * @param  hmw  Pointer to a HSP_CORE_HandleTypeDef structure.
  * @retval MW Status
  */
hsp_core_status_t HSP_CORE_LoadFW(hsp_core_handle_t *hmw)
{

  HSP_ASSERT_DBG_PARAM((hmw != NULL));
  HSP_ASSERT_DBG_STATE(hmw->global_state, (uint32_t)HSP_CORE_STATE_RESET);

  hsp_hw_if_firmware_t firmware;

  firmware.p_fw_cram_bin = (const uint32_t *)(&dataCRAMTab[0]);
  firmware.fw_cram_size_in_word = HSP_FW_DATA_CRAM_SIZE;
  firmware.p_fw_dram_bin = (const uint32_t *)(&dataDRAMTab[0]);
  firmware.fw_dram_size_in_word = HSP_FW_DATA_DRAM_SIZE;

  if (HSP_HW_IF_LoadFW(hmw->hdriver, &firmware) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }

  HSP_CHECK_UPDATE_STATE(hmw, global_state, HSP_CORE_STATE_RESET, HSP_CORE_STATE_FW_LOADED);
  ((HSP_HandleTypeDef *)(hmw->hdriver))->global_state = HAL_HSP_STATE_READY_TO_BOOT;
  return HSP_CORE_OK;
}

#if defined(USE_HSP_PLUGIN) && (USE_HSP_PLUGIN == 1)
/**
  * @brief  Load the HSP Plugins
  * @param  hmw  Pointer to a HSP_HandleTypeDef structure.
  * @retval MW Status
  */
hsp_core_status_t HSP_CORE_LoadPlugins(hsp_core_handle_t *hmw)
{
  HSP_ASSERT_DBG_PARAM((hmw != NULL));

  /* Load a set of Plugins */
  for (uint32_t id = 0UL; id < HSP_PLUGIN_ARRAY_SIZE; id++)
  {
    hsp_core_plugin_t plugin = HSP_PLUGIN_ARRAY_NAME[id];
    if (HSP_HW_IF_LoadPlugin(hmw->hdriver, plugin.p_bin, plugin.size_in_word) != HSP_IF_OK)
    {
      return HSP_CORE_ERROR;
    }
  }
  return HSP_CORE_OK;
}
#endif /* USE_HSP_PLUGIN */

/**
  * @brief  DeInitialize the MW HSP handle.
  * @param  hmw  Pointer to a hsp_core_handle_t.
  */
void HSP_CORE_DeInit(hsp_core_handle_t *hmw)
{
  HSP_ASSERT_DBG_PARAM((hmw != NULL));

  /* DeInit the Hardware interface */
  HSP_HW_IF_DeInit(hmw->hdriver);

  /* DeInit the BRAM Memory interface */
  HSP_BRAM_IF_DeInit(&(hmw->hbram));
}

/**
  * @}
  */

/** @addtogroup HSP_CORE_Exported_Functions_Group2
  * @{
  */
/**
  * @brief Configure a TRGO.
  * @param  hmw      Pointer to a hsp_core_handle_t.
  * @param  trgo_id  TRGO ID
  * @param  source   Trigger source
  * @retval HSP_CORE_OK     Success.
  * @retval HSP_CORE_ERROR  Running failed.
  */
hsp_core_status_t HSP_CORE_OUTPUT_SetConfig(hsp_core_handle_t *hmw, hsp_core_trgo_t trgo_id,
                                            hsp_core_trgo_source_t source)
{
  HSP_ASSERT_DBG_PARAM((hmw != NULL));
  HSP_ASSERT_DBG_STATE(hmw->global_state, (uint32_t)HSP_CORE_STATE_IDLE);

  if (HSP_HW_IF_OUTPUT_SetConfig(hmw->hdriver, trgo_id, (uint32_t) source) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }

  return HSP_CORE_OK;
}

/**
  * @brief Enable the HSP Outputs.
  * @param  hmw  Pointer to a hsp_core_handle_t.
  * @retval HSP_CORE_OK     Success.
  * @retval HSP_CORE_ERROR  Running failed.
  */
hsp_core_status_t HSP_CORE_OUTPUT_Enable(hsp_core_handle_t *hmw)
{
  HSP_ASSERT_DBG_PARAM((hmw != NULL));
  HSP_ASSERT_DBG_STATE(hmw->global_state, (uint32_t)HSP_CORE_STATE_IDLE);

  if (HSP_HW_IF_OUTPUT_Enable(hmw->hdriver) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }

  return HSP_CORE_OK;
}

/**
  * @brief Disable the HSP Outputs.
  * @param  hmw  Pointer to a hsp_core_handle_t.
  * @retval HSP_CORE_OK     Success.
  * @retval HSP_CORE_ERROR  Running failed.
  */
hsp_core_status_t HSP_CORE_OUTPUT_Disable(hsp_core_handle_t *hmw)
{
  HSP_ASSERT_DBG_PARAM((hmw != NULL));
  HSP_ASSERT_DBG_STATE(hmw->global_state, (uint32_t)HSP_CORE_STATE_IDLE);

  if (HSP_HW_IF_OUTPUT_Disable(hmw->hdriver) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }

  return HSP_CORE_OK;
}

/**
  * @brief Get the HSP Outputs enabling status.
  * @param  hmw  Pointer to a hsp_core_handle_t.
  * @retval HSP_CORE_OK     Success.
  * @retval HSP_CORE_ERROR  Running failed.
  */
uint32_t HSP_CORE_OUTPUT_IsEnabled(hsp_core_handle_t *hmw)
{
  HSP_ASSERT_DBG_PARAM((hmw != NULL));

  return HSP_HW_IF_OUTPUT_IsEnabled(hmw->hdriver);
}

/**
  * @}
 */

/** @addtogroup HSP_CORE_Exported_Functions_Group3
  * @{
  */
/**
  * @brief Enable the SmartClocking of HSP block.
  * @param  hmw    Pointer to a hsp_core_handle_t.
  * @param  clock  Block clock
  * @retval HSP_CORE_OK     Success.
  * @retval HSP_CORE_ERROR  Running failed.
  */
hsp_core_status_t HSP_CORE_EnableSmartClocking(hsp_core_handle_t *hmw, hsp_core_smart_clocking_t clock)
{
  HSP_ASSERT_DBG_PARAM((hmw != NULL));

  HSP_ASSERT_DBG_STATE(hmw->global_state, (uint32_t)HSP_CORE_STATE_IDLE);

  if (HSP_HW_IF_EnableSmartClocking(hmw->hdriver, (uint32_t)clock) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }

  return HSP_CORE_OK;
}

/**
  * @brief Disable the SmartClocking of HSP block.
  * @param  hmw    Pointer to a hsp_core_handle_t.
  * @param  clock  Block clock
  * @retval HSP_CORE_OK     Success.
  * @retval HSP_CORE_ERROR  Running failed.
  */
hsp_core_status_t HSP_CORE_DisableSmartClocking(hsp_core_handle_t *hmw, hsp_core_smart_clocking_t clock)
{
  HSP_ASSERT_DBG_PARAM((hmw != NULL));

  HSP_ASSERT_DBG_STATE(hmw->global_state, (uint32_t)HSP_CORE_STATE_IDLE);

  if (HSP_HW_IF_DisableSmartClocking(hmw->hdriver, (uint32_t)clock) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }

  return HSP_CORE_OK;
}

/**
  * @brief Get the SmartClocking of HSP block status.
  * @param  hmw    Pointer to a hsp_core_handle_t.
  * @retval HSP_CORE_SMART_CLOCKING_DISABLED  SmartClocking is disabled.
  * @retval HSP_CORE_SMART_CLOCKING_ENABLED   SmartClocking is enabled.
  */
hsp_core_smart_clocking_status_t HSP_CORE_GetSmartClockingStatus(hsp_core_handle_t *hmw,
                                                                 hsp_core_smart_clocking_t clock)
{
  HSP_ASSERT_DBG_PARAM((hmw != NULL));

  return HSP_CORE_SMART_CLOCKING_ENABLED;
}

/**
  * @}
 */

/** @addtogroup HSP_CORE_Exported_Functions_Group4
  * @{
  */
/**
  * @brief Lock the HSP configuration.
  * @param  hmw    Pointer to a hsp_core_handle_t.
  * @retval HSP_CORE_OK     Success.
  * @retval HSP_CORE_ERROR  Running failed.
  */
hsp_core_status_t HSP_CORE_Lock(hsp_core_handle_t *hmw)
{
  HSP_ASSERT_DBG_PARAM((hmw != NULL));
  HSP_ASSERT_DBG_STATE(hmw->global_state, (uint32_t)HSP_CORE_STATE_IDLE);

  if (HSP_HW_IF_Lock(hmw->hdriver) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }

  return HSP_CORE_OK;
}

/**
  * @}
 */

/** @addtogroup HSP_CORE_Exported_Functions_Group5
  * @{
  */
/**
  * @brief Retrieve the Middleware state.
  * @param  hmw    Pointer to a hsp_core_handle_t.
  * @retval HSP_CORE_STATE_RESET 
  * @retval HSP_CORE_STATE_FW_LOADED
  * @retval HSP_CORE_STATE_READY
  * @retval HSP_CORE_STATE_IDLE
  * @retval HSP_CORE_STATE_PROCLIST_RECORDING
  * @retval HSP_CORE_STATE_CNN_ACTIVE
  * @retval HSP_CORE_STATE_FAULT
  */
hsp_core_state_t HSP_CORE_GetState(const hsp_core_handle_t *hmw)
{
  HSP_ASSERT_DBG_PARAM((hmw != NULL));

  return hmw->global_state;
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