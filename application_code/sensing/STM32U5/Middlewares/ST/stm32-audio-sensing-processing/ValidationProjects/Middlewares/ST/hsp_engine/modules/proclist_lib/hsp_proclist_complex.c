/**
  ******************************************************************************
  * @file    hsp_proclist_complex.c
  * @author  MCD Application Team
  * @brief   This file implements the HSP COMPLEX Processing functions used to
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
#include "hsp_proclist_complex.h"
#include "hsp_hw_if.h"
#include "hsp_utilities.h"

#define HSP_CHECK_CMD_SIZE_NULL(a, b)
#define HSP_CHECK_ASSERT(a, b)

/** @addtogroup STM32_HSP
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

/** @addtogroup MW_HSP_PROCLIST_Exported_Functions_Complex
  * @{
  */
/**
  * @brief Compute the conjugate of each vector element (vector is a complex interleaved real, img)
  * @param hhsp       HSP handle.
  * @param pSrc       Input Buffer address
  * @param pDst       Output Buffer address
  * @param nbSamples  Number of <b>complex</b> elements to proceed
  * @param ioType     User iotype information
  * @retval           HAL status.
  */
hsp_core_status_t HSP_SEQ_CmplxConj_f32(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pDst,
                                        uint32_t nbSamples, uint32_t ioType)
{
  HSP_CHECK_CMD_SIZE_NULL(hhsp, nbSamples);

  uint32_t encoded = 0;
  uint32_t ouAddr1, ouAddr2;

  /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t) pSrc, &ouAddr1,
                               (uint32_t) pDst, &ouAddr2, 0, NULL, 2) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }

  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, (2*nbSamples));
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_CMPLX_CONJ_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Computes the dot product of two complex vectors
  * @param hhsp       HSP handle.
  * @param pSrcA      Input A Buffer address
  * @param pSrcB      Input B Buffer address
  * @param pDst       Output Buffer address
  * @param nbSamples  Number of <b>complex</b> elements to proceed
  * @param ioType     User iotype information
  * @retval           HAL status.
  */
hsp_core_status_t HSP_SEQ_CmplxDotProd_f32(hsp_core_handle_t *hmw, float32_t *pSrcA, float32_t *pSrcB,
                                           float32_t *pDst, uint32_t nbSamples, uint32_t ioType)
{
  HSP_CHECK_CMD_SIZE_NULL(hhsp, (2*nbSamples));

  uint32_t encoded = 0;
  uint32_t ouAddr1, ouAddr2, ouAddr3;

  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t) pSrcA, &ouAddr1,
                               (uint32_t) pSrcB, &ouAddr2, (uint32_t) pDst, &ouAddr3, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr3);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, (2*nbSamples));
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_CMPLX_DOTPROD_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Vector Floating-point complex magnitude
  * @param hhsp       HSP handle.
  * @param pSrc       Input Buffer address
  * @param pDst       Output Buffer address
  * @param nbSamples  Number of <b>complex</b> elements to proceed
  * @param ioType     User iotype information
  * @retval           HAL status.
  */
hsp_core_status_t HSP_SEQ_CmplxMag_f32(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pDst, uint32_t nbSamples,
                                       uint32_t ioType)
{
  HSP_CHECK_CMD_SIZE_NULL(hhsp, (2*nbSamples));

  uint32_t encoded = 0;
  uint32_t ouAddr1, ouAddr2;

  /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t) pSrc, &ouAddr1,
                               (uint32_t) pDst, &ouAddr2, 0, NULL, 2) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }

  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, (2*nbSamples));
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_CMPLX_MAG_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Vector Vector Floating-point complex magnitude squared
  * @param hhsp       HSP handle.
  * @param pSrc       Input Buffer address
  * @param pDst       Output Buffer address
  * @param nbSamples  Number of <b>complex</b> elements to proceed
  * @param ioType     User iotype information
  * @retval           HAL status.
  */
hsp_core_status_t HSP_SEQ_CmplxMagSquared_f32(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pDst,
                                              uint32_t nbSamples, uint32_t ioType)
{
  HSP_CHECK_CMD_SIZE_NULL(hhsp, (2*nbSamples));

  uint32_t encoded = 0;
  uint32_t ouAddr1, ouAddr2;

  /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t) pSrc, &ouAddr1,
                               (uint32_t) pDst, &ouAddr2, 0, NULL, 2) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }

  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, (2*nbSamples));
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_CMPLX_MAGSQUARED_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Compute the complex multiplication of 2 complex vector element (vector is a complex interleaved real, img)
  * @param hhsp       HSP handle.
  * @param pSrcA      Input A Buffer address
  * @param pSrcB      Input B Buffer address
  * @param pDst       Output Buffer address
  * @param nbSamples  Number of <b>complex</b> elements to proceed
  * @param ioType     User iotype information
  * @retval           HAL status.
  */
hsp_core_status_t HSP_SEQ_CmplxMul_f32(hsp_core_handle_t *hmw, float32_t *pSrcA, float32_t *pSrcB, float32_t *pDst,
                                       uint32_t nbSamples, uint32_t ioType)
{
  HSP_CHECK_CMD_SIZE_NULL(hhsp, (2*nbSamples));

  uint32_t encoded = 0;
  uint32_t ouAddr1, ouAddr2, ouAddr3;

  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t) pSrcA, &ouAddr1,
                               (uint32_t) pSrcB, &ouAddr2, (uint32_t) pDst, &ouAddr3, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr3);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, (2*nbSamples));
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_CMPLX_MUL_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Vector real cmplx mul function (Compute the multiplication of a complex vector by a real vector and generates a complex result)
  * @param hhsp       HSP handle.
  * @param pSrcA      Input A Buffer address
  * @param pSrcB      Input B Buffer address
  * @param pDst       Output Buffer address
  * @param nbSamples  Number of <b>complex</b> elements to proceed
  * @param ioType     User iotype information
  * @retval           HAL status.
  */
hsp_core_status_t HSP_SEQ_CmplxRMul_f32(hsp_core_handle_t *hmw, float32_t *pSrcA, float32_t *pSrcB, float32_t *pDst,
                                        uint32_t nbSamples, uint32_t ioType)
{
  HSP_CHECK_CMD_SIZE_NULL(hhsp, (2*nbSamples));

  uint32_t encoded = 0;
  uint32_t ouAddr1, ouAddr2, ouAddr3;

  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t) pSrcA, &ouAddr1,
                               (uint32_t) pSrcB, &ouAddr2, (uint32_t) pDst, &ouAddr3, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr3);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, (2*nbSamples));
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_CMPLX_RMUL_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Compute the complex multiplication of one complex vector element (vector is a complex interleaved real, img)
  * and an exponential complex signal
  * @param hhsp           HSP handle.
  * @param pSrc           Input Buffer address
  * @param pStart         Start buffer address (I/O: input is start index and output is nextIdx)
  * @param pDst           Output Buffer address
  * @param nbSamples      Number of <b>complex</b> elements to proceed
  * @param step           Step value between 2 exponential number in ROM
  * @param ioType         User iotype information
  * @retval               HAL status.
  */
hsp_core_status_t HSP_SEQ_CmplxMulExp_f32(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pStart,
                                          float32_t *pDst, uint32_t nbSamples, int32_t step, uint32_t ioType)
{
  HSP_CHECK_CMD_SIZE_NULL(hhsp, (2*nbSamples));

  uint32_t encoded = 0;
  uint32_t ouAddr1, ouAddr2, ouAddr3;

  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t) pSrc, &ouAddr1,
                               (uint32_t) pStart, &ouAddr2, (uint32_t) pDst, &ouAddr3, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr3);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, (2*nbSamples));
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM4, step);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_CMPLX_MUL_EXP_F32) != HSP_IF_OK)
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

