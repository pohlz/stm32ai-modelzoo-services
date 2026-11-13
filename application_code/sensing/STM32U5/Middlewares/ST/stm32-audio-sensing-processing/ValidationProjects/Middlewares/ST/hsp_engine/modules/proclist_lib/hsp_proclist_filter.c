/**
  ******************************************************************************
  * @file    hsp_proclist_filter.c
  * @author  MCD Application Team
  * @brief   This file implements the HSP Filter Processing functions used to
  *          record a processing list
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
#include "hsp_proclist_filter.h"
#include "hsp_hw_if.h"
#include "hsp_utilities.h"
#include "hsp_bram.h"

/** @addtogroup HSP
  * @{
  */

/** @addtogroup HSP_PROCLIST
  * @{
  */
/** @addtogroup HSP_MODULES_PROCLIST_FILTER_LIBRARY
  * @{
  */
/* Private defines -----------------------------------------------------------*/
/* Private typedef -----------------------------------------------------------*/
/* Private macros ------------------------------------------------------------*/
#define HSP_CHECK_CMD_SIZE_NULL(hmw, size)
#define HSP_CHECK_ASSERT(hmw, cond)

/* Private variables ---------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/
/* Exported functions --------------------------------------------------------*/
/** @addtogroup HSP_Exported_Functions_ProcList_Filter
  * @{
  */
/**
  * @brief Add FIR function in the current processing list
  * @param hmw         HSP handle.
  * @param inBuff      Input Buffer address
  * @param coefBuff    Coefficients Buffer address
  * @param staBuff     State Buffer
  * @param outBuff     Output Buffer address
  * @param nbSamples   Number of float elements to proceed
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_Fir_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t *coefBuff,
                                  hsp_filter_state_identifier_t staBuff, float32_t *outBuff,
                                  uint32_t nbSamples, uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  uint32_t ouAddr3;
  uint32_t ouAddr4;

   /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t)inBuff, &ouAddr1, (uint32_t)coefBuff, &ouAddr2,
                               (uint32_t)outBuff, &ouAddr3, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  ouAddr4 = (uint32_t)((hsp_hw_if_filter_state_t *)staBuff)->addrHsp;

  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr3);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, ouAddr4);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM4, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_FIR_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Add FIR Decimate function in the current processing list
  * @param hmw         HSP handle.
  * @param inBuff      Input Buffer address
  * @param coefBuff    Coefficients Buffer address
  * @param staBuff     State Buffer
  * @param outBuff     Output Buffer address
  * @param nbSamples   Number of float elements to proceed
  * @param decimFactor Decimation factor
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_FirDecimate_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t *coefBuff,
                                          hsp_filter_state_identifier_t staBuff, float32_t *outBuff,
                                          uint32_t nbSamples,
                                          uint32_t decimFactor, uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  uint32_t ouAddr3;
  uint32_t ouAddr4;

   /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t)inBuff, &ouAddr1, (uint32_t)coefBuff, &ouAddr2,
                               (uint32_t)outBuff, &ouAddr3, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  hsp_hw_if_fir_decimate_filter_state_t *tmp = (hsp_hw_if_fir_decimate_filter_state_t *)staBuff;

  ouAddr4 = (uint32_t)(tmp->addrHsp);
// ouAddr4 = (uint32_t)((hsp_hw_if_fir_decimate_filter_state_t *)staBuff)->addrHsp;

  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr3);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, ouAddr4);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM4, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM5, decimFactor);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_FIRDEC_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Add Biquad cascade df1 function in the current processing list
  * @param hmw         HSP handle.
  * @param inBuff      Input Buffer address
  * @param coefBuff    Coefficients Buffer address
  * @param staBuff     State Buffer
  * @param outBuff     Output Buffer address
  * @param nbSamples   Number of float elements to proceed
  * @param nbStages    Number of stage in filter
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_BiquadCascadeDf1_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t *coefBuff,
                                               hsp_filter_state_identifier_t staBuff, float32_t *outBuff,
                                               uint32_t nbSamples, uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  uint32_t ouAddr3;
  uint32_t ouAddr4;

   /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t)inBuff, &ouAddr1, (uint32_t)coefBuff, &ouAddr2,
                               (uint32_t)outBuff, &ouAddr3, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  ouAddr4 = (uint32_t)((hsp_hw_if_filter_state_t *)staBuff)->addrHsp;

  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr3);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, ouAddr4);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM4, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_BQ_CAS_DF1_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Add Biquad cascade df2 function in the current processing list
  * @param hmw         HSP handle.
  * @param inBuff      Input Buffer address
  * @param coefBuff    Coefficients Buffer address
  * @param staBuff     State Buffer
  * @param outBuff     Output Buffer address
  * @param nbSamples   Number of float elements to proceed
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_BiquadCascadeDf2T_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t *coefBuff,
                                                hsp_filter_state_identifier_t staBuff, float32_t *outBuff,
                                                uint32_t nbSamples, uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  uint32_t ouAddr3;
  uint32_t ouAddr4;

   /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t)inBuff, &ouAddr1, (uint32_t)coefBuff, &ouAddr2,
                               (uint32_t)outBuff, &ouAddr3, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  ouAddr4 = (uint32_t)((hsp_hw_if_filter_state_t *)staBuff)->addrHsp;

  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr3);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, ouAddr4);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM4, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_BQ_CAS_DF2T_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Add CMSIS Convolution function in the current processing list
  * @param hmw         HSP handle.
  * @param inABuff     Input A Buffer address
  * @param inBBuff     Input B Buffer address
  * @param outBuff     Output Buffer address
  * @param sizeA       Number of float elements in vectA
  * @param sizeB       Number of float elements in vectB
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_Conv_f32(hsp_core_handle_t *hmw, float32_t *inABuff, float32_t *inBBuff, float32_t *outBuff,
                                   uint32_t sizeA, uint32_t sizeB, uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  uint32_t ouAddr3;

   /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t)inABuff, &ouAddr1, (uint32_t)inBBuff, &ouAddr2,
                               (uint32_t)outBuff, &ouAddr3, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }

  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr3);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, sizeA);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM4, sizeB);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_CONV_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Add CMSIS Correlate function in the current processing list
  * @param hmw         HSP handle.
  * @param inABuff     Input A Buffer address
  * @param inBBuff     Input B Buffer address
  * @param outBuff     Output Buffer address
  * @param sizeA       Number of float elements in vectA
  * @param sizeB       Number of float elements in vectB
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_Correlate_f32(hsp_core_handle_t *hmw, float32_t *inABuff, float32_t *inBBuff,
                                        float32_t *outBuff, uint32_t sizeA, uint32_t sizeB, uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  uint32_t ouAddr3;

   /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t)inABuff, &ouAddr1, (uint32_t)inBBuff, &ouAddr2,
                               (uint32_t)outBuff, &ouAddr3, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }

  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr3);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, sizeA);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM4, sizeB);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_CORR_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}


/**
  * @brief Add FLTBANK filter function in the current processing list
  * @param hmw         HSP handle.
  * @param spectrCol   Input spectrogram slice of length FFTLen / 2 Buffer address
  * @param startIdx    FLTBANK filter pCoefficients start indexes Buffer address
  * @param idxSize     FLTBANK filter pCoefficients size indexes Buffer address
  * @param coef        FLTBANK filter weights Buffer address
  * @param fltbankCol  Output fltbank energies in each filterbank Buffer address
  * @param nFltbanks   Number of Fltbank bands to generate
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_FltBank_f32(hsp_core_handle_t *hmw, float32_t *spectrCol, uint32_t *startIdx,
                                      uint32_t *idxSize, float32_t *coef, float32_t *fltbankCol,
                                      uint32_t nFltbanks, uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  uint32_t ouAddr3;
  uint32_t ouAddr4;

  /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t) spectrCol, &ouAddr0, (uint32_t) fltbankCol, &ouAddr1,
                               0, NULL, 2) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  if (HSP_UTILITIES_ToBramABAddress(hmw, (uint32_t) startIdx, &ouAddr2) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  if (HSP_UTILITIES_ToBramABAddress(hmw, (uint32_t) idxSize, &ouAddr3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  if (HSP_UTILITIES_ToBramABAddress(hmw, (uint32_t) coef, &ouAddr4) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr0); /* spectrCol */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr1); /* fltbankCol */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr2); /* startIdx */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, ouAddr3); /* idxSize */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM4, ouAddr4); /* coef */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM5, nFltbanks);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_FLTBANK_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

#ifdef __HSP_DMA__
/**
  * @brief Add FLTBANK filter with external coefficients in the current processing list
  * (internal DMA is used to get ext coef in dmaBuffId pingpoing buffer)
  * @param hmw         HSP handle.
  * @param spectrCol   Input spectrogram slice of length FFTLen / 2 Buffer address
  * @param startIdx    FLTBANK filter pCoefficients start indexes Buffer address
  * @param idxSize     FLTBANK filter pCoefficients size indexes Buffer address
  * @param coef        FLTBANK filter weights Buffer address
  * @param fltbankCol  Output fltbank energies in each filterbank Buffer address
  * @param nFltbanks   Number of Fltbank bands to generate
  * @param dmaAdd      FLTBANK DMA Buffer address (must be max filter size x2 for pingpong)
  * @param dmaSize     FLTBANK DMA Buffer size (full DMA buffer size (ping + pong))
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_FltBankExtC_f32(hsp_core_handle_t *hmw, float32_t *spectrCol, uint32_t *startIdx,
                                          uint32_t *idxSize, float32_t *coef, float32_t *fltbankCol,
                                          uint32_t nFltbanks,
                                          float32_t *dmaAdd, uint32_t dmaSize, uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  uint32_t ouAddr3;
  uint32_t ouAddr4;
  uint32_t ouAddr6;

  /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t) spectrCol, &ouAddr0, (uint32_t) fltbankCol, &ouAddr1,
                               0, NULL, 2) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  if (HSP_UTILITIES_ToBramABAddress(hmw, (uint32_t) startIdx, &ouAddr2) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  if (HSP_UTILITIES_ToBramABAddress(hmw, (uint32_t) idxSize, &ouAddr3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  if (HSP_UTILITIES_ToBramABAddress(hmw, (uint32_t) dmaAdd, &ouAddr6) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr0); /* spectrCol */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr1); /* fltbankCol */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr2); /* startIdx */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, ouAddr3); /* idxSize */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM4, (uint32_t) coef);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM5, nFltbanks);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM6, ouAddr6);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM7, dmaSize);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_FLTBANK_EXTC_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}
#endif /* __HSP_DMA__ */

/**
  * @brief Add LMS filter function in the current processing list
  * @param hmw         HSP handle.
  * @param inBuff      Input Buffer address
  * @param coefBuff    Coefficients Buffer address
  * @param staBuff     State Buffer
  * @param outBuff     Output Buffer address
  * @param refBuff     Reference Buffer address
  * @param errBuff     Error Buffer address
  * @param nbSamples   Number of float elements to proceed
  * @param mu          Adaptative factor
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_Lms_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t *coefBuff,
                                  hsp_filter_state_identifier_t staBuff,
                                  float32_t *outBuff, float32_t *refBuff,
                                  float32_t *errBuff, uint32_t nbSamples, float32_t mu, uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  uint32_t ouAddr3;
  uint32_t ouAddr4;
  uint32_t ouAddr5;
  uint32_t ouAddr6;

   /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t)inBuff, &ouAddr1, (uint32_t)coefBuff, &ouAddr2,
                               (uint32_t)outBuff, &ouAddr3, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }

  if (HSP_UTILITIES_ToBramABAddress(hmw, (uint32_t) refBuff, &ouAddr5) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  if (HSP_UTILITIES_ToBramABAddress(hmw, (uint32_t) errBuff, &ouAddr6) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  ouAddr4 = (uint32_t)((hsp_hw_if_filter_state_t *)staBuff)->addrHsp;

  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr3);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, ouAddr4);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM4, ouAddr5);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM5, ouAddr6);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM6, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM7, ((uint32_t) * (uint32_t *)&mu));
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_LMS_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Add IIR Lattice filter function in the current processing list
  * @param hmw         HSP handle.
  * @param inBuff      Input Buffer address
  * @param coeffsk     Coefficients Buffer address
  * @param coeffsv     Coefficients Buffer address
  * @param staBuff     State Buffer
  * @param outBuff     Output Buffer address
  * @param nbSamples   Number of float elements to proceed
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_IirLattice_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t *coeffsk,
                                         float32_t *coeffsv, hsp_filter_state_identifier_t staBuff,
										 float32_t *outBuff,
                                         uint32_t nbSamples, uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  uint32_t ouAddr3;
  uint32_t ouAddr4;
  uint32_t ouAddr5;

   /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t)inBuff, &ouAddr1, (uint32_t)coeffsk, &ouAddr2,
                               (uint32_t)outBuff, &ouAddr3, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }

  if (HSP_UTILITIES_ToBramABAddress(hmw, (uint32_t) coeffsv, &ouAddr5) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }

  ouAddr4 = (uint32_t)((hsp_hw_if_filter_state_t *)staBuff)->addrHsp;

  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr3);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, ouAddr4);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM4, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM5, ouAddr5);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_IIR_LATTICE_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Add IIR DF1 filter function in the current processing list
  * IIR coeffs are stored interleaved and in reversed order if K nb stages:
  * B[k-1] A[k-1] B[k-2] A[k-2] ... B[1] A[1] B[0]
  * @param hmw         HSP handle.
  * @param inBuff      Input Buffer address
  * @param coefBuff    Coefficients Buffer address
  * @param staBuff     State Buffer
  * @param outBuff     Output Buffer address
  * @param nbSamples   Number of float elements to proceed
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_IirDf1_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t *coefBuff,
                                     hsp_filter_state_identifier_t staBuff, float32_t *outBuff,
                                     uint32_t nbSamples, uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  uint32_t ouAddr3;
  uint32_t ouAddr4;

   /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t)inBuff, &ouAddr1, (uint32_t)coefBuff, &ouAddr2,
                               (uint32_t)outBuff, &ouAddr3, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  ouAddr4 = (uint32_t)((hsp_hw_if_filter_state_t *)staBuff)->addrHsp;

  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr3);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, ouAddr4);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM4, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_IIR_DF1_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Add IIR 3p3z filter function in the current processing list
  *        @todo need more description
  * To be updated: IIR coeffs are stored interleaved and in reversed order if K nb stages:
  * To be updated: B[k-1] A[k-1] B[k-2] A[k-2] ... B[1] A[1] B[0]
  * @param hmw         HSP handle.
  * @param inBuff      Input Buffer address
  * @param coefBuff    Coefficients Buffer address
  * @param staBuff     State Buffer
  * @param outBuff     Output Buffer address
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_Iir3p3z_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t *coefBuff,
                                      hsp_filter_state_identifier_t staBuff, float32_t *outBuff, uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  uint32_t ouAddr3;
  uint32_t ouAddr4;

   /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t)inBuff, &ouAddr1, (uint32_t)coefBuff, &ouAddr2,
                               (uint32_t)outBuff, &ouAddr3, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  ouAddr4 = (uint32_t)((hsp_hw_if_filter_state_t *)staBuff)->addrHsp;

  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr3);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, ouAddr4);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_IIR_3P3Z_1S_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Add IIR 2p2z filter function in the current processing list
  *        @todo need more description
  * To be updated: IIR coeffs are stored interleaved and in reversed order if K nb stages:
  * To be updated: B[k-1] A[k-1] B[k-2] A[k-2] ... B[1] A[1] B[0]
  * @param hmw         HSP handle.
  * @param inBuff      Input Buffer address
  * @param coefBuff    Coefficients Buffer address
  * @param staBuff     State Buffer
  * @param outBuff     Output Buffer address
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_Iir2p2z_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t *coefBuff,
                                      hsp_filter_state_identifier_t staBuff, float32_t *outBuff, uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  uint32_t ouAddr3;
  uint32_t ouAddr4;

   /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t)inBuff, &ouAddr1, (uint32_t)coefBuff, &ouAddr2,
                               (uint32_t)outBuff, &ouAddr3, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  ouAddr4 = (uint32_t)((hsp_hw_if_filter_state_t *)staBuff)->addrHsp;

  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr3);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, ouAddr4);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_IIR_2P2Z_1S_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Add Windowing symmetric function in the current processing list
  * @param hmw         HSP handle.
  * @param inABuff     Input Buffer address
  * @param inBBuff     Input window Buffer address
  * @param outBuff     Output Buffer address
  * @param sizeW       Number of float elements in input vectA (= Window size)
  * @param sizeD0      Number of extra dest value pad to 0
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_WinSym_f32(hsp_core_handle_t *hmw, float32_t *inABuff, float32_t *inBBuff,
                                     float32_t *outBuff, uint32_t sizeW, uint32_t sizeD0, uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  uint32_t ouAddr3;

   /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t)inABuff, &ouAddr1, (uint32_t)inBBuff, &ouAddr2,
                               (uint32_t)outBuff, &ouAddr3, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr3);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, sizeW);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM4, sizeD0);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_WINSYM_F32) != HSP_IF_OK)
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

