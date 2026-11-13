/**
  ******************************************************************************
  * @file    hsp_proclist_transform.c
  * @author  MCD Application Team
  * @brief   This file implements the HSP TRANSFORM Processing functions used to
  *          record a processing list
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

/* Includes ------------------------------------------------------------------*/
#include "hsp_proclist.h"
#include "hsp_proclist_transform.h"
#include "hsp_hw_if.h"
#include "hsp_utilities.h"


/** @addtogroup HSP
  * @{
  */

/** @addtogroup STM32_HSP_PROCLIST
  * @{
  */
/* Private defines -----------------------------------------------------------*/
/* Private typedef -----------------------------------------------------------*/
/* Private macros ------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/

/* Exported functions --------------------------------------------------------*/
/** @addtogroup MW_HSP_Exported_Functions
  * @{
  */

/** @addtogroup MW_HSP_PROCLIST_Exported
  * @{
  */

/** @addtogroup MW_HSP_PROCLIST_Exported_Functions_Transform
  * @{
  */
/**
  * @brief FFT transform
  * @param hmw          HSP handle.
  * @param buff         Input and output Buffer address (must be in HSP memory)
  * @param fftSize      FFT size
  * @param log2Nbp      log2(number of FFT point)
  * @param ifftFlag     Inverse FFT flag
  * @param bitrev       Bit reverse flag
  * @param ioType       User iotype information
  * @retval             Core status.
  */
hsp_core_status_t HSP_SEQ_Fft_f32(hsp_core_handle_t *hmw, float32_t *buff, hsp_ftt_lognbp_cmd_t log2Nbp,
                                  uint8_t ifftFlag, uint8_t bitrev, uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr;
  uint32_t fftSize = (1 << log2Nbp);

  if ((fftSize != 32) && (fftSize != 64) && (fftSize != 128) && (fftSize != 256) &&
      (fftSize != 512) && (fftSize != 1024) && (fftSize != 2048) && (fftSize != 4096))
  {
    return HSP_CORE_ERROR;
  }

  /*  Input buffer must be in shared memory */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t) buff, &ouAddr, 0, NULL, 0, NULL, 1) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }

  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ifftFlag);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, (1 << log2Nbp));
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, HAL_HSP_FFT_COMPLEX);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM4, bitrev);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_FFT_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}
  
/**
  * @brief FFT transform
  * @param hhsp         HSP handle.
  * @param buff         Input and output Buffer address (must be in HSP memory)
  * @param log2Nbp      log2(number of FFT point)
  * @param fftSize      FFT size
  * @param ifftFlag     Inverse FFT flag: 0: not inverse, 1: inverse
  * @param bitrev       Bit reverse flag: 0: not reverse, 1 reverse
  * @param fftVariant   Type of FFT: HSP_RFFT_TYPE_1, HSP_RFFT_TYPE_2, HSP_RFFT_TYPE_3
  * @param ioType       User iotype information
  * @retval             Core status.
  */
hsp_core_status_t HSP_SEQ_Rfft_f32(hsp_core_handle_t *hmw, float32_t *buff, hsp_ftt_lognbp_cmd_t log2Nbp,
                                   uint8_t ifftFlag, uint8_t bitrev, hsp_type_rfft_cmd_t fftVariant, 
                                   uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr;
  uint32_t fftSize = (1 << (uint32_t)log2Nbp);
  
  if ((fftSize != 32) && (fftSize != 64) && (fftSize != 128) && (fftSize != 256) &&
      (fftSize != 512) && (fftSize != 1024) && (fftSize != 2048) && (fftSize != 4096))
  {
    return HSP_CORE_ERROR;
  }

  /* Input buffer must be in shared memory */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t) buff, &ouAddr, 0, NULL, 0, NULL, 1) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ifftFlag);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, (1 << (uint32_t)log2Nbp));
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, (uint32_t)fftVariant);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM4, bitrev);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_FFT_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief DCT transform
  * @param hhsp         HSP handle.
  * @param buff         Input and output Buffer address (must be in HSP memory)
  * @param log2Nbp      log2(number of FFT point)
  * @param ioType       User iotype information
  * @retval             Core status.
  */
hsp_core_status_t HSP_SEQ_Dct_f32(hsp_core_handle_t *hmw, float32_t *buff, hsp_ftt_lognbp_cmd_t log2Nbp, uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr;
  uint32_t fftSize = (1 << log2Nbp);
  if ((fftSize != 32) && (fftSize != 64) && (fftSize != 128) && (fftSize != 256) &&
      (fftSize != 512) && (fftSize != 1024) && (fftSize != 2048) && (fftSize != 4096))
  {
    return HSP_CORE_ERROR;
  }

  /* Input buffer must be in shared memory */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t) buff, &ouAddr, 0, NULL, 0, NULL, 1) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }

  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, 0);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, (1 << log2Nbp));
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_DCT_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Inverse DCT transform
  * @param hhsp       HSP handle.
  * @param buff       Input and output Buffer address (must be in HSP memory)
  * @param log2Nbp    log2(number of FFT point)
  * @param ioType     User iotype information
  * @retval           HAL status.
  */
hsp_core_status_t HSP_SEQ_IDct_f32(hsp_core_handle_t *hmw, float32_t *buff, hsp_ftt_lognbp_cmd_t log2Nbp, uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr;
  uint32_t fftSize = (1 << log2Nbp);
  if ((fftSize != 32) && (fftSize != 64) && (fftSize != 128) && (fftSize != 256) &&
      (fftSize != 512) && (fftSize != 1024) && (fftSize != 2048) && (fftSize != 4096))
  {
    return HSP_CORE_ERROR;
  }

  /* Input buffer must be in shared memory */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t) buff, &ouAddr, 0, NULL, 0, NULL, 1) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }

  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, 0);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, (1 << log2Nbp));
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_IDCT_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
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

