/**
  ******************************************************************************
  * @file    hsp_hw_if.h
  * @author  GPM Application Team
  * @brief   Header file for hsp_hw_if.c.
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

#ifndef HSP_HW_IF_H
#define HSP_HW_IF_H

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/* Includes ------------------------------------------------------------------*/
#include "interface.h"
#include "hsp_if_conf.h"

/** @addtogroup HSP_ENGINE
  * @{
  */

/** @defgroup HSP_INTERFACES HSP Interfaces
  * @{
  */

/** @defgroup HSP_HW_IF HSP Hardware Interface
  * @{
  */

/* Exported constants --------------------------------------------------------*/
/** @defgroup HSP_HW_IF_Exported_Constants HSP_HW_IF Exported Constants
  * @{
  */
#if defined(USE_HAL_HSP_CUBE1_LEGACY)
#define hal_hsp_handle_t  HSP_HandleTypeDef
#define hal_hsp_bram_arbitration_t  HAL_HSP_BRAM_ArbitrationTypeDef
#define hal_hsp_boot_config_t  HAL_HSP_Boot_ConfigTypeDef
#define hal_hsp_perf_monitor_t  HAL_HSP_PERF_MONITOR_StateTypeDef
#define hal_hsp_fw_version_t  HAL_HSP_FW_VersionTypeDef
#define hal_hsp_task_t    HAL_HSP_TaskTypeDef
#define hal_hsp_event_sync_t  HAL_HSP_EVENT_SyncTypeDef
#define hal_hsp_event_config_t  HAL_HSP_EVENT_ConfigTypeDef
#define hal_hsp_trgin_source_t  HAL_HSP_TRGIN_SourceTypeDef
#define hal_hsp_trgin_config_t  HAL_HSP_TRGIN_ConfigTypeDef
#define hal_hsp_stream_buffer_config_t  HAL_HSP_STREAM_Buffer_ConfigTypeDef
#define hal_hsp_output_trigger_source_t  HAL_HSP_OUTPUT_TRIGGER_SourceTypeDef
#define hal_hsp_output_trigger_t  HAL_HSP_OUTPUT_TriggerTypeDef
#endif /* USE_HAL_HSP_CUBE1_LEGACY */

#define hsp_hw_if_event_sync_t  hal_hsp_event_sync_t
#define hsp_hw_if_event_config_t  hal_hsp_event_config_t
#define hsp_hw_if_trgin_source_t  hal_hsp_trgin_source_t
#define hsp_hw_if_trgin_config_t  hal_hsp_trgin_config_t
#define hsp_hw_if_stream_buffer_config_t  hal_hsp_stream_buffer_config_t
#define hsp_hw_if_output_trigger_source_t  hal_hsp_output_trigger_source_t
#define hsp_hw_if_output_trigger_t  hal_hsp_output_trigger_t
#define hsp_hw_if_smart_clocking_t  hal_hsp_smart_clocking_t


#define HSP_HW_IF_WRITE_PARAMR0(val)  (*((volatile uint32_t *)HAL_HSP1_PARAMR0_ADDRESS) = (val))
#define HSP_HW_IF_WRITE_PARAMR1(val)  (*((volatile uint32_t *)HAL_HSP1_PARAMR1_ADDRESS) = (val))
#define HSP_HW_IF_WRITE_PARAMR2(val)  (*((volatile uint32_t *)HAL_HSP1_PARAMR2_ADDRESS) = (val))
#define HSP_HW_IF_WRITE_PARAMR3(val)  (*((volatile uint32_t *)HAL_HSP1_PARAMR3_ADDRESS) = (val))
#define HSP_HW_IF_WRITE_PARAMR4(val)  (*((volatile uint32_t *)HAL_HSP1_PARAMR4_ADDRESS) = (val))
#define HSP_HW_IF_WRITE_PARAMR5(val)  (*((volatile uint32_t *)HAL_HSP1_PARAMR5_ADDRESS) = (val))
#define HSP_HW_IF_WRITE_PARAMR6(val)  (*((volatile uint32_t *)HAL_HSP1_PARAMR6_ADDRESS) = (val))
#define HSP_HW_IF_WRITE_PARAMR7(val)  (*((volatile uint32_t *)HAL_HSP1_PARAMR7_ADDRESS) = (val))
#define HSP_HW_IF_WRITE_PARAMR8(val)  (*((volatile uint32_t *)HAL_HSP1_PARAMR8_ADDRESS) = (val))
#define HSP_HW_IF_WRITE_PARAMR9(val)  (*((volatile uint32_t *)HAL_HSP1_PARAMR9_ADDRESS) = (val))
#define HSP_HW_IF_WRITE_PARAMR10(val) (*((volatile uint32_t *)HAL_HSP1_PARAMR10_ADDRESS) = (val))
#define HSP_HW_IF_WRITE_PARAMR11(val) (*((volatile uint32_t *)HAL_HSP1_PARAMR11_ADDRESS) = (val))
#define HSP_HW_IF_WRITE_PARAMR12(val) (*((volatile uint32_t *)HAL_HSP1_PARAMR12_ADDRESS) = (val))
#define HSP_HW_IF_WRITE_PARAMR13(val) (*((volatile uint32_t *)HAL_HSP1_PARAMR13_ADDRESS) = (val))
#define HSP_HW_IF_WRITE_PARAMR14(val) (*((volatile uint32_t *)HAL_HSP1_PARAMR14_ADDRESS) = (val))
#define HSP_HW_IF_WRITE_PARAMR15(val) (*((volatile uint32_t *)HAL_HSP1_PARAMR15_ADDRESS) = (val))
#define HSP_HW_IF_WRITE_DCMDPTR0(val) (*((volatile uint32_t *)HAL_HSP1_DCMDPTR0_ADDRESS) = (val))
#define HSP_HW_IF_WRITE_DCMDPTR1(val) (*((volatile uint32_t *)HAL_HSP1_DCMDPTR1_ADDRESS) = (val))
#define HSP_HW_IF_WRITE_DCMDPTR2(val) (*((volatile uint32_t *)HAL_HSP1_DCMDPTR2_ADDRESS) = (val))
#define HSP_HW_IF_WRITE_DCMDIDR(val)  (*((volatile uint32_t *)HAL_HSP1_DCMDIDR_ADDRESS) = (val))
#define HSP_HW_IF_READ_FWERR()    HAL_READ_FWERR()
#define HSP_HW_IF_READ_H2CSEMR()  HAL_READ_H2CSEMR()
#define HSP_HW_IF_READ_DCMDSR()   HAL_READ_DCMDSR()
#define HSP_HW_IF_READ_CDEGR()    HAL_READ_CDEGR()

#define HSP_HW_IF_SMARTCLOCKING_CTRL    0U
#define HSP_HW_IF_SMARTCLOCKING_SPE     1U
#define HSP_HW_IF_SMARTCLOCKING_MMC     2U
#define HSP_HW_IF_SMARTCLOCKING_HSPDMA  3U

/**
  * @brief Direct Command Parameters
  */
typedef enum
{
  HSP_HW_IF_PARAM0  = HAL_HSP_PARAM_0,
  HSP_HW_IF_PARAM1  = HAL_HSP_PARAM_1,
  HSP_HW_IF_PARAM2  = HAL_HSP_PARAM_2,
  HSP_HW_IF_PARAM3  = HAL_HSP_PARAM_3,
  HSP_HW_IF_PARAM4  = HAL_HSP_PARAM_4,
  HSP_HW_IF_PARAM5  = HAL_HSP_PARAM_5,
  HSP_HW_IF_PARAM6  = HAL_HSP_PARAM_6,
  HSP_HW_IF_PARAM7  = HAL_HSP_PARAM_7,
  HSP_HW_IF_PARAM8  = HAL_HSP_PARAM_8,
  HSP_HW_IF_PARAM9  = HAL_HSP_PARAM_9,
  HSP_HW_IF_PARAM10 = HAL_HSP_PARAM_10,
  HSP_HW_IF_PARAM11 = HAL_HSP_PARAM_11,
  HSP_HW_IF_PARAM12 = HAL_HSP_PARAM_12,
  HSP_HW_IF_PARAM13 = HAL_HSP_PARAM_13,
  HSP_HW_IF_PARAM14 = HAL_HSP_PARAM_14,
  HSP_HW_IF_PARAM15 = HAL_HSP_PARAM_15
} hsp_hw_if_parameter_t;

typedef enum
{
  HSP_HW_IF_POINTER0  = 0U,
  HSP_HW_IF_POINTER1  = 1U,
  HSP_HW_IF_POINTER2  = 2U,
} hsp_hw_if_pointer_t;

#define HSP_HW_IF_LOG2NBP_32    5  /**< Value of Log2(32) for FFT 32 points */
#define HSP_HW_IF_LOG2NBP_64    6  /**< Value of Log2(64) for FFT 64 points */
#define HSP_HW_IF_LOG2NBP_128   7  /**< Value of Log2(128) for FFT 128 points */
#define HSP_HW_IF_LOG2NBP_256   8  /**< Value of Log2(256) for FFT 256 points */
#define HSP_HW_IF_LOG2NBP_512   9  /**< Value of Log2(512) for FFT 512 points */
#define HSP_HW_IF_LOG2NBP_1024  10 /**< Value of Log2(1024) for FFT 1024 points */
#define HSP_HW_IF_LOG2NBP_2048  11 /**< Value of Log2(2048) for FFT 2048 points */
#define HSP_HW_IF_LOG2NBP_4096  12 /**< Value of Log2(4096) for FFT 4096 points */

/** @defgroup HSP_TRGIN_ID  HSP Trigger Input Identifiers
  * @{
  */
#define HSP_HW_IF_TRGIN_0  HAL_HSP_TRGIN_0
#define HSP_HW_IF_TRGIN_1  HAL_HSP_TRGIN_1
#define HSP_HW_IF_TRGIN_2  HAL_HSP_TRGIN_2
#define HSP_HW_IF_TRGIN_3  HAL_HSP_TRGIN_3
#define HSP_HW_IF_TRGIN_4  HAL_HSP_TRGIN_4
#define HSP_HW_IF_TRGIN_5  HAL_HSP_TRGIN_5
#define HSP_HW_IF_TRGIN_6  HAL_HSP_TRGIN_6
#define HSP_HW_IF_TRGIN_7  HAL_HSP_TRGIN_7
#define HSP_HW_IF_TRGIN_8  HAL_HSP_TRGIN_8
#define HSP_HW_IF_TRGIN_9  HAL_HSP_TRGIN_9

#define HSP_HW_IF_STREAM_BUFFER_0  HAL_HSP_STREAM_0
#define HSP_HW_IF_STREAM_BUFFER_1  HAL_HSP_STREAM_1
#define HSP_HW_IF_STREAM_BUFFER_2  HAL_HSP_STREAM_2
#define HSP_HW_IF_STREAM_BUFFER_3  HAL_HSP_STREAM_3

#define HSP_HW_IF_STREAM_BUFFER_SYNC_DISABLE  HAL_HSP_STREAM_SYNC_DISABLE
#define HSP_HW_IF_STREAM_BUFFER_SYNC_ENABLE   HAL_HSP_STREAM_SYNC_ENABLE

/**
  * @brief EVENT IDs
  */
#define HSP_HW_IF_EVENT_1  (1U << 1U)
#define HSP_HW_IF_EVENT_2  (1U << 2U)
#define HSP_HW_IF_EVENT_3  (1U << 3U)
#define HSP_HW_IF_EVENT_4  (1U << 4U)
#define HSP_HW_IF_EVENT_5  (1U << 5U)
#define HSP_HW_IF_EVENT_6  (1U << 6U)
#define HSP_HW_IF_EVENT_7  (1U << 7U)
#define HSP_HW_IF_EVENT_8  (1U << 8U)
#define HSP_HW_IF_EVENT_9  (1U << 9U)
#define HSP_HW_IF_EVENT_10  (1U << 10U)
#define HSP_HW_IF_EVENT_11  (1U << 11U)
#define HSP_HW_IF_EVENT_12  (1U << 12U)
#define HSP_HW_IF_EVENT_13  (1U << 13U)
#define HSP_HW_IF_EVENT_14  (1U << 14U)
#define HSP_HW_IF_EVENT_15  (1U << 15U)
#define HSP_HW_IF_EVENT_16  (1U << 16U)
#define HSP_HW_IF_EVENT_17  (1U << 17U)
#define HSP_HW_IF_EVENT_18  (1U << 18U)
#define HSP_HW_IF_EVENT_19  (1U << 19U)
#define HSP_HW_IF_EVENT_20  (1U << 20U)
#define HSP_HW_IF_EVENT_21  (1U << 21U)
#define HSP_HW_IF_EVENT_22  (1U << 22U)

#define HSP_HW_IF_EVENT_SYNC_DISABLE  HAL_HSP_EVENT_SYNC_DISABLE
#define HSP_HW_IF_EVENT_SYNC_ENABLE  HAL_HSP_EVENT_SYNC_ENABLE

#define HSP_HW_IF_TRGIN_POLARITY_RISING  HAL_HSP_TRGIN_POLARITY_RISING
#define HSP_HW_IF_TRGIN_POLARITY_FALLING  HAL_HSP_TRGIN_POLARITY_FALLING

#define  HSP_HW_IF_STREAM_BUFFER_DIRECTION_IN  HAL_HSP_STREAM_DIRECTION_CPU_TO_HSP
#define  HSP_HW_IF_STREAM_BUFFER_DIRECTION_OUT HAL_HSP_STREAM_DIRECTION_HSP_TO_CPU

#define HSP_HW_IF_TASK_COMPARATOR_0  HAL_HSP_TASK_COMPARATOR_0
#define HSP_HW_IF_TASK_COMPARATOR_1  HAL_HSP_TASK_COMPARATOR_1
#define HSP_HW_IF_TASK_COMPARATOR_2  HAL_HSP_TASK_COMPARATOR_2
#define HSP_HW_IF_TASK_COMPARATOR_3  HAL_HSP_TASK_COMPARATOR_3

#define HSP_HW_IF_OUTPUT_TRIGGER_NONE    HAL_HSP_OUTPUT_TRIGGER_NONE
#define HSP_HW_IF_OUTPUT_TRIGGER_STREAM  HAL_HSP_OUTPUT_TRIGGER_STREAM
#define HSP_HW_IF_OUTPUT_TRIGGER_CORE    HAL_HSP_OUTPUT_TRIGGER_CORE
#define HSP_HW_IF_OUTPUT_TRIGGER_TIMESTAMPCAPTURE  HAL_HSP_OUTPUT_TRIGGER_TIMESTAMPCAPTURE

/**
  * @}
  */

/* Exported macros -----------------------------------------------------------*/
/** @defgroup HSP_HW_IF_Exported_Macros HSP_HW_IF Exported Macros
  * @{
 */

/**
  * @}
  */

/* Exported types ------------------------------------------------------------*/
/** @defgroup  HSP_HW_IF_Exported_Types HSP_HW_IF Exported Types
  * @{
  */
typedef struct
{
  const uint32_t *p_fw_cram_bin;
  uint32_t fw_cram_size_in_word;
  const uint32_t *p_fw_dram_bin;
  uint32_t fw_dram_size_in_word;
} hsp_hw_if_firmware_t;

typedef struct
{
  uint32_t instance;
  hsp_hw_if_stream_buffer_config_t itf;
  hsp_hw_if_event_config_t evt_cfg;
} hsp_hw_if_stream_config_t;

typedef struct
{
  uint32_t instance;
  hsp_hw_if_trgin_config_t itf;
  hsp_hw_if_event_config_t evt_cfg;
} hsp_hw_if_trigger_config_t;

/**
  * @}
  */

/** @defgroup HSP_HW_IF_Driver_Aliases HSP_HW_IF Driver Aliases
  * @{
  */
#define HSP_HW_IF_WriteDirectCommand(hdriver, cmd_id)  \
  HAL_HSP_DIRECT_WriteCommand((hal_hsp_handle_t *)(hdriver), cmd_id)

#define HSP_HW_IF_WriteParameter(hdriver, param_id, value)  \
  HAL_HSP_WriteParameter((hal_hsp_handle_t *)(hdriver), (HAL_HSP_PARAMTypeDef)(param_id), (uint32_t)(value))

#define HSP_HW_IF_WaitEndOfDirectCommand(hdriver)  while (((hal_hsp_handle_t *)(hdriver))->Instance->DCMDSR)

#define HSP_HW_IF_MSGBOX_IsMsgAvailable(hdriver)  HAL_HSP_MSGBOX_IsMsgAvailable((hal_hsp_handle_t *)(hdriver))
#define HSP_HW_IF_MSGBOX_ReleaseSemaphore(hdriver)  HAL_HSP_MSGBOX_ReleaseSemaphore((hal_hsp_handle_t *)(hdriver))


#define HSP_HW_IF_EVENT_GetStatus(hdriver)  HAL_HSP_EVENT_GetStatus((hal_hsp_handle_t *)(hdriver))

#define HSP_HW_IF_SEQ_STREAM_Enable(hdriver)   HAL_HSP_STREAM_Enable((hal_hsp_handle_t *)(hdriver))
#define HSP_HW_IF_SEQ_STREAM_Disable(hdriver)  HAL_HSP_STREAM_Disable((hal_hsp_handle_t *)(hdriver))



/**
  * @}
  */

/* Exported functions --------------------------------------------------------*/
/** @defgroup HSP_HW_IF_Exported_Functions HSP_HW_IF Exported Functions
  * @{
  */
void   HSP_HW_IF_DeInit(void *hdriver);

/* API to manage the FW & Plugin ---------------------------------------------*/
hsp_if_status_t HSP_HW_IF_LoadFW(void *hdriver, const hsp_hw_if_firmware_t *p_fw);
hsp_if_status_t HSP_HW_IF_LoadPlugin(void *hdriver, const uint32_t *p_bin, uint32_t size_word);

/* API to boot the HSP           ---------------------------------------------*/
hsp_if_status_t HSP_HW_IF_Boot(void *hdriver, uint32_t enableCycleCount);

/* API to manage the SmartClocking -------------------------------------------*/
hsp_if_status_t HSP_HW_IF_EnableSmartClocking(void *hdriver, uint32_t clock_id);
hsp_if_status_t HSP_HW_IF_DisableSmartClocking(void *hdriver, uint32_t clock_id);

/* API to manage Direct Command ----------------------------------------------*/
hsp_if_status_t HSP_HW_IF_SendCommand(void *hdriver, uint32_t command_id);

/* API to manage OUTPUT ------------------------------------------------------*/
hsp_if_status_t HSP_HW_IF_OUTPUT_SetConfig(void *hdriver, hsp_hw_if_output_trigger_t trgo_id, uint32_t source);

hsp_if_status_t HSP_HW_IF_OUTPUT_Enable(void *hdriver);
hsp_if_status_t HSP_HW_IF_OUTPUT_Disable(void *hdriver);
if_activation_status_t HSP_HW_IF_OUTPUT_IsEnabled(void *hdriver);

/* API to manage Processing List ---------------------------------------------*/
hsp_if_status_t HSP_HW_IF_SEQ_StartRecord(void *hdriver, uint32_t pl_id);
hsp_if_status_t HSP_HW_IF_SEQ_StopRecord(void *hdriver);
hsp_if_status_t HSP_HW_IF_SEQ_ResetPL(void *hdriver);

hsp_if_status_t HSP_HW_IF_SEQ_RunTask(void *hdriver, uint32_t task_id, uint32_t timeout_ms);

/* API to manage Trigger & Event configuration -------------------------------*/
hsp_if_status_t HSP_HW_IF_EVENT_SetConfig_StreamBuffer(void *hdriver, uint32_t evt_id,
                                                   const hsp_hw_if_stream_config_t *p_config);
hsp_if_status_t HSP_HW_IF_EVENT_SetConfig_Trigger(void *hdriver, uint32_t evt_id,
                                              const hsp_hw_if_trigger_config_t *p_config);
hsp_if_status_t HSP_HW_IF_EVENT_SetConfig_CSEG(void *hdriver, uint32_t evt_id, const hsp_hw_if_event_sync_t *p_sync);
hsp_if_status_t HSP_HW_IF_EVENT_SetConfig_SPE(void *hdriver, uint32_t evt_id, const hsp_hw_if_event_sync_t *p_sync);

hsp_if_status_t HSP_HW_IF_EVENT_Enable(void *hdriver, uint32_t events_mask);
hsp_if_status_t HSP_HW_IF_EVENT_Disable(void *hdriver, uint32_t events_mask);

hsp_if_status_t HSP_HW_IF_SEQ_EVENT_Trig_CSEG(void *hdriver, uint32_t events);


/* API to BRAM Bandwith Access control ---------------------------------------*/
hsp_if_status_t HSP_HW_IF_BRAM_EnableConflictAccessCounter(void *hdriver);
hsp_if_status_t HSP_HW_IF_BRAM_DisableConflictAccessCounter(void *hdriver);
uint32_t HSP_HW_IF_BRAM_GetConflictAccessCounter(void *hdriver);
hsp_if_status_t HSP_HW_IF_BRAM_SetBandwidthArbitration(void *hdriver, uint32_t mode);
uint32_t HSP_HW_IF_BRAM_GetBandwidthArbitration(void *hdriver);

/* API for secure Access -----------------------------------------------------*/
hsp_if_status_t HSP_HW_IF_Lock(void *hdriver);

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

#endif /* HSP_HW_IF_H */
