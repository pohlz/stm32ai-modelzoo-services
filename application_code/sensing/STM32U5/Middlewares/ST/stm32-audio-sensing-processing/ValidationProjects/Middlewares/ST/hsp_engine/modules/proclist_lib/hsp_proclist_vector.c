/**
  ******************************************************************************
  * @file    hsp_proclist_vector.c
  * @author  MCD Application Team
  * @brief   This file implements the HSP VECTOR Processing functions used to
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
#include "hsp_proclist_vector.h"
#include "hsp_hw_if.h"
#include "hsp_utilities.h"

/** @addtogroup HSP
  * @{
  */

/** @addtogroup HSP_PROCLIST
  * @{
  */
/* Private defines -----------------------------------------------------------*/
/* Private typedef -----------------------------------------------------------*/
/* Private macros ------------------------------------------------------------*/
#define HSP_CHECK_CMD_SIZE_NULL(a, b)
#define HSP_CHECK_ASSERT(a, b)

/* Private variables ---------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/

/* Exported functions --------------------------------------------------------*/
/** @addtogroup HSP_MODULES_PROCLIST_VECTOR_LIBRARY
  * @{
  */

/** @addtogroup HSP_Exported_Functions_ProcList_vector
  * @{
  */
/**
  * @brief Element-wise absolute value of a vector
  * @param hmw        HSP handle.
  * @param pSrc       Input Buffer address
  * @param pDst       Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @param ioType     User iotype information
  * @retval           Status returned.
  */
hsp_core_status_t HSP_SEQ_VectAbs_f32(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pDst,
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
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
  uint32_t cmdid = HSP_CMD_VEC_ABS_F32;
  if (nbSamples == 1)
  {
    cmdid = HSP_CMD_SCA_ABS_F32;
  }

  if (HSP_HW_IF_SendCommand(hmw->hdriver, cmdid) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;

}

/**
  * @brief Performs a vector addition
  * @param hmw          HSP handle.
  * @param pSrcA        Input A Buffer address
  * @param pSrcB        Input B Buffer address
  * @param pDst         Output Buffer address
  * @param nbSamples    Number of float elements to proceed
  * @param ioType       User iotype information
  * @retval             Status returned.
  */
hsp_core_status_t HSP_SEQ_VectAdd_f32(hsp_core_handle_t *hmw, float32_t *pSrcA, float32_t *pSrcB,
                                        float32_t *pDst, uint32_t nbSamples, uint32_t ioType)
{
  HSP_CHECK_CMD_SIZE_NULL(hhsp, nbSamples);
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
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
  uint32_t cmdid = HSP_CMD_VEC_ADD_F32;
  if (nbSamples == 1)
  {
    cmdid = HSP_CMD_SCA_ADD_F32;
  }
  if (HSP_HW_IF_SendCommand(hmw->hdriver, cmdid) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Compute the average value of a vector
  * @param hmw          HSP handle.
  * @param pSrc         Input Buffer address
  * @param pDst         Output Buffer address
  * @param nbSamples    Number of float elements to proceed
  * @param ioType       User iotype information
  * @retval             Status returned.
  */
hsp_core_status_t HSP_SEQ_VectAvg_f32(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pDst, uint32_t nbSamples,
                                        uint32_t ioType)
{
  HSP_CHECK_CMD_SIZE_NULL(hhsp, nbSamples);
  HSP_CHECK_ASSERT(hhsp, (nbSamples > 1));

  uint32_t encoded = 0;
  uint32_t ouAddr1, ouAddr2;

  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t) pSrc, &ouAddr1,
                               (uint32_t) pDst, &ouAddr2, 0, NULL, 2) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }

  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_VEC_AVG_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Perform the copy of a vector
  * @param hmw          HSP handle.
  * @param pSrc         Input Buffer address
  * @param pDst         Output Buffer address
  * @param nbSamples    Number of float elements to proceed
  * @param ioType       User iotype information
  * @retval             Status returned.
  */
hsp_core_status_t HSP_SEQ_VectCopy_f32(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pDst, uint32_t nbSamples,
                                         uint32_t ioType)
{
  HSP_CHECK_CMD_SIZE_NULL(hhsp, nbSamples);

  uint32_t encoded = 0;
  uint32_t ouAddr1, ouAddr2;

  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t) pSrc, &ouAddr1,
                               (uint32_t) pDst, &ouAddr2, 0, NULL, 2) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }

  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
  uint32_t cmdid = HSP_CMD_VEC_COPY;
  if (nbSamples == 1)
  {
    cmdid = HSP_CMD_SCA_SET;
  }
  if (HSP_HW_IF_SendCommand(hmw->hdriver, cmdid) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Perform the cos of a vector
  * @param hmw        HSP handle.
  * @param pSrc       Input Buffer
  * @param pDst       Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @param ioType     User iotype information
  * @retval           Status returned.
  */
hsp_core_status_t HSP_SEQ_VectCos_f32(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pDst, uint32_t nbSamples,
                                        uint32_t ioType)
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
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
  uint32_t cmdid = HSP_CMD_VEC_COS_F32;
  if (nbSamples == 1)
  {
    cmdid = HSP_CMD_SCA_COS_F32;
  }
  if (HSP_HW_IF_SendCommand(hmw->hdriver, cmdid) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Element-wise division of a vector
  * @param hmw        HSP handle.
  * @param pSrcA      Input A Buffer address
  * @param pSrcB      Input B Buffer address
  * @param pDst       Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @param ioType     User iotype information
  * @retval           Status returned.
  */
hsp_core_status_t HSP_SEQ_VectDiv_f32(hsp_core_handle_t *hmw, float32_t *pSrcA, float32_t *pSrcB, float32_t *pDst,
                                        uint32_t nbSamples, uint32_t ioType)
{
  HSP_CHECK_CMD_SIZE_NULL(hhsp, nbSamples);

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
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
  uint32_t cmdid = HSP_CMD_VEC_DIV_F32;
  if (nbSamples == 1)
  {
    cmdid = HSP_CMD_SCA_DIV_F32;
  }
  if (HSP_HW_IF_SendCommand(hmw->hdriver, cmdid) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Perform dot product of a vector
  * @param hhsp        HSP handle.
  * @param pSrcA       Input A Buffer address
  * @param pSrcB       Input B Buffer address
  * @param pDst        Output Buffer address
  * @param nbSamples   Number of float elements to proceed
  * @param ioType      User iotype information
  * @retval            HAL status.
  */
hsp_core_status_t HSP_SEQ_VectDotProd_f32(hsp_core_handle_t *hmw, float32_t *pSrcA, float32_t *pSrcB,
                                            float32_t *pDst, uint32_t nbSamples, uint32_t ioType)
{
  HSP_CHECK_CMD_SIZE_NULL(hhsp, nbSamples);

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
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
  uint32_t cmdid = HSP_CMD_VEC_DOTPROD_F32;
  if (nbSamples == 1)
  {
    cmdid = HSP_CMD_SCA_MUL_F32;
  }
  if (HSP_HW_IF_SendCommand(hmw->hdriver, cmdid) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Search for value and position of the absolute biggest element of a vector
  * @param hhsp      HSP handle.
  * @param pSrc      Input Buffer address
  * @param outVal    Output max_value address
  * @param outIdx    Output max_pos address
  * @param nbSamples Number of float elements to proceed
  * @param ioType    User iotype information
  * @retval          HAL status.
  */
hsp_core_status_t HSP_SEQ_VectAbsmax_f32(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *outVal,
                                           uint32_t *outIdx, uint32_t nbSamples, uint32_t ioType)
{
  HSP_CHECK_CMD_SIZE_NULL(hhsp, nbSamples);
  HSP_CHECK_ASSERT(hhsp, (nbSamples > 1));

  uint32_t encoded = 0;
  uint32_t ouAddr1, ouAddr2, ouAddr3;

  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t) pSrc, &ouAddr1,
                               (uint32_t) outVal, &ouAddr2, (uint32_t) outIdx, &ouAddr3, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr3);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_VEC_ABSMAX_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Search for value and position of the biggest element of a vector
  * @param hmw        HSP handle.
  * @param pSrc       Input Buffer address
  * @param outVal     Output max_value address
  * @param outIdx     Output max_pos address
  * @param nbSamples  Number of float elements to proceed
  * @param ioType     User iotype information
  * @retval           Status returned.
  */
hsp_core_status_t HSP_SEQ_VectMax_f32(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *outVal,
                                        uint32_t *outIdx, uint32_t nbSamples, uint32_t ioType)
{
  HSP_CHECK_CMD_SIZE_NULL(hhsp, nbSamples);
  HSP_CHECK_ASSERT(hhsp, (nbSamples > 1));

  uint32_t encoded = 0;
  uint32_t ouAddr1, ouAddr2, ouAddr3;

  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t) pSrc, &ouAddr1,
                               (uint32_t) outVal, &ouAddr2, (uint32_t) outIdx, &ouAddr3, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr3);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_VEC_MAX_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Search for value and position of the biggest element of a vector
  * @param hmw        HSP handle.
  * @param pSrc       Input Buffer address
  * @param outVal     Output min_value address
  * @param outIdx     Output min_pos address
  * @param nbSamples  Number of float elements to proceed
  * @param ioType     User iotype information
  * @retval           Status returned.
  */
hsp_core_status_t HSP_SEQ_VectMin_f32(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *outVal,
                                        uint32_t *outIdx, uint32_t nbSamples, uint32_t ioType)
{
  HSP_CHECK_CMD_SIZE_NULL(hhsp, nbSamples);
  HSP_CHECK_ASSERT(hhsp, (nbSamples > 1));

  uint32_t encoded = 0;
  uint32_t ouAddr1, ouAddr2, ouAddr3;

  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t) pSrc, &ouAddr1,
                               (uint32_t) outVal, &ouAddr2, (uint32_t) outIdx, &ouAddr3, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr3);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_VEC_MIN_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Element-wise multiplication of a vector
  * @param hmw        HSP handle.
  * @param pSrcA      Input A Buffer address
  * @param pSrcB      Input B Buffer address
  * @param pDst       Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @param ioType     User iotype information
  * @retval           Status returned.
  */
hsp_core_status_t HSP_SEQ_VectMul_f32(hsp_core_handle_t *hmw, float32_t *pSrcA, float32_t *pSrcB,
                                        float32_t *pDst, uint32_t nbSamples, uint32_t ioType)
{
  HSP_CHECK_CMD_SIZE_NULL(hhsp, nbSamples);

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
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
  uint32_t cmdid = HSP_CMD_VEC_MUL_F32;
  if (nbSamples == 1)
  {
    cmdid = HSP_CMD_SCA_MUL_F32;
  }
  if (HSP_HW_IF_SendCommand(hmw->hdriver, cmdid) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Add to each vector element a scalar
  * @param hmw        HSP handle.
  * @param pSrcA      Input A Buffer address
  * @param pSrcB      Offset to add Buffer address
  * @param pDst       Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @param ioType     User iotype information
  * @retval           Status returned.
  */
hsp_core_status_t HSP_SEQ_VectOffset_f32(hsp_core_handle_t *hmw, float32_t *pSrcA, float32_t *pSrcB,
                                           float32_t *pDst, uint32_t nbSamples, uint32_t ioType)
{
  HSP_CHECK_CMD_SIZE_NULL(hhsp, nbSamples);
  HSP_CHECK_ASSERT(hhsp, (nbSamples > 1));

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
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_VEC_OFFSET_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Compute the RMS value of a vector
  * @param hmw        HSP handle.
  * @param pSrc       Input Buffer
  * @param pDst       Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @param ioType     User iotype information
  * @retval           Status returned.
  */
hsp_core_status_t HSP_SEQ_VectRms_f32(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pDst, uint32_t nbSamples,
                                        uint32_t ioType)
{
  HSP_CHECK_CMD_SIZE_NULL(hhsp, nbSamples);
  HSP_CHECK_ASSERT(hhsp, (nbSamples > 1));

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
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_VEC_RMS_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Multiply each vector element by a scalar
  * @param hmw        HSP handle.
  * @param pSrcA      Input A Buffer
  * @param pSrcB      Scale Buffer
  * @param pDst       Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @param ioType     User iotype information
  * @retval           Status returned.
  */
hsp_core_status_t HSP_SEQ_VectScale_f32(hsp_core_handle_t *hmw, float32_t *pSrcA, float32_t *pSrcB,
                                          float32_t *pDst, uint32_t nbSamples, uint32_t ioType)
{
  HSP_CHECK_CMD_SIZE_NULL(hhsp, nbSamples);
  HSP_CHECK_ASSERT(hhsp, (nbSamples > 1));

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
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_VEC_SCALE_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Set value for whole vector
  * @param hmw        HSP handle.
  * @param pSrc       Set Buffer address
  * @param pDst       Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @param ioType     User iotype information
  * @retval           Status returned.
  */
hsp_core_status_t HSP_SEQ_VectSet_f32(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pDst, uint32_t nbSamples,
                                        uint32_t ioType)
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
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
  uint32_t cmdid = HSP_CMD_VEC_SET;
  if (nbSamples == 1)
  {
    cmdid = HSP_CMD_SCA_SET;
  }
  if (HSP_HW_IF_SendCommand(hmw->hdriver, cmdid) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Element-wise square-root of a vector
  * @param hmw        HSP handle.
  * @param pSrc       Input Buffer address
  * @param pDst       Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @param ioType     User iotype information
  * @retval           HAL status.
  */
hsp_core_status_t HSP_SEQ_VectSqrt_f32(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pDst, uint32_t nbSamples,
                                         uint32_t ioType)
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
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
  uint32_t cmdid = HSP_CMD_VEC_SQRT_F32;
  if (nbSamples == 1)
  {
    cmdid = HSP_CMD_SCA_SQRT_F32;
  }
  if (HSP_HW_IF_SendCommand(hmw->hdriver, cmdid) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Perform the sin of a vector
  * @param hmw        HSP handle.
  * @param pSrc       Input Buffer address
  * @param pDst       Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @param ioType     User iotype information
  * @retval           Status returned.
  */
hsp_core_status_t HSP_SEQ_VectSin_f32(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pDst, uint32_t nbSamples,
                                        uint32_t ioType)
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
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
  uint32_t cmdid = HSP_CMD_VEC_SIN_F32;
  if (nbSamples == 1)
  {
    cmdid = HSP_CMD_SCA_SIN_F32;
  }
  if (HSP_HW_IF_SendCommand(hmw->hdriver, cmdid) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Element-wise substraction of a vector
  * @param hmw        HSP handle.
  * @param pSrcA      Input A Buffer address
  * @param pSrcB      Input B Buffer address
  * @param pDst       Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @param ioType     User iotype information
  * @retval           Status returned.
  */
hsp_core_status_t HSP_SEQ_VectSub_f32(hsp_core_handle_t *hmw, float32_t *pSrcA, float32_t *pSrcB, float32_t *pDst,
                                        uint32_t nbSamples, uint32_t ioType)
{
  HSP_CHECK_CMD_SIZE_NULL(hhsp, nbSamples);

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
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
  uint32_t cmdid = HSP_CMD_VEC_SUB_F32;
  if (nbSamples == 1)
  {
    cmdid = HSP_CMD_SCA_SUB_F32;
  }
  if (HSP_HW_IF_SendCommand(hmw->hdriver, cmdid) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Perform the sin,cos of each element of vector
  * @param hmw        HSP handle.
  * @param pSrc       Input Buffer address
  * @param pDst       Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @param ioType     User iotype information
  * @retval           Status returned.
  */
hsp_core_status_t HSP_SEQ_VectSinCos_f32(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pDst,
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
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
  uint32_t cmdid = HSP_CMD_VEC_SINCOS_F32;
  if (nbSamples == 1)
  {
    cmdid = HSP_CMD_SCA_SINCOS_F32;
  }
  if (HSP_HW_IF_SendCommand(hmw->hdriver, cmdid) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Perform the float32 to integer 32bits conversion
  * @param hmw          HSP handle.
  * @param pSrc         Input Buffer address
  * @param pDst         Output Buffer address
  * @param nbSamples    Number of float elements to proceed
  * @param ioType       User iotype information
  * @retval             Status returned.
  */
hsp_core_status_t HSP_SEQ_VectF2I(hsp_core_handle_t *hmw, float32_t *pSrc, int32_t *pDst, uint32_t nbSamples,
                                    uint32_t ioType)
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
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
  uint32_t cmdid = HSP_CMD_VEC_F2I;
  if (nbSamples == 1)
  {
    cmdid = HSP_CMD_SCA_F2I;
  }
  if (HSP_HW_IF_SendCommand(hmw->hdriver, cmdid) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Perform the integer 32bits to float32 conversion
  * @param hmw          HSP handle.
  * @param pSrc         Input Buffer address
  * @param pDst         Output Buffer address
  * @param nbSamples    Number of float elements to proceed
  * @param ioType       User iotype information
  * @retval             Status returned.
  */
hsp_core_status_t HSP_SEQ_VectI2F(hsp_core_handle_t *hmw, int32_t *pSrc, float32_t *pDst, uint32_t nbSamples,
                                    uint32_t ioType)
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
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
  uint32_t cmdid = HSP_CMD_VEC_I2F;
  if (nbSamples == 1)
  {
    cmdid = HSP_CMD_SCA_I2F;
  }
  if (HSP_HW_IF_SendCommand(hmw->hdriver, cmdid) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Perform the float32 to unsigned integer 32bits conversion
  * @param hmw          HSP handle.
  * @param pSrc         Input Buffer address
  * @param pDst         Output Buffer address
  * @param nbSamples    Number of float elements to proceed
  * @param ioType       User iotype information
  * @retval             Status returned.
  */
hsp_core_status_t HSP_SEQ_VectF2U(hsp_core_handle_t *hmw, float32_t *pSrc, uint32_t *pDst, uint32_t nbSamples,
                                    uint32_t ioType)
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
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
  uint32_t cmdid = HSP_CMD_VEC_F2U;
  if (nbSamples == 1)
  {
    cmdid = HSP_CMD_SCA_F2U;
  }
  if (HSP_HW_IF_SendCommand(hmw->hdriver, cmdid) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Perform the unsigned integer 32bits to float32 conversion
  * @param hmw          HSP handle.
  * @param pSrc         Input Buffer address
  * @param pDst         Output Buffer address
  * @param nbSamples    Number of float elements to proceed
  * @param ioType       User iotype information
  * @retval             Status returned.
  */
hsp_core_status_t HSP_SEQ_VectU2F(hsp_core_handle_t *hmw, uint32_t *pSrc, float32_t *pDst, uint32_t nbSamples,
                                    uint32_t ioType)
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
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
  uint32_t cmdid = HSP_CMD_VEC_U2F;
  if (nbSamples == 1)
  {
    cmdid = HSP_CMD_SCA_U2F;
  }
  if (HSP_HW_IF_SendCommand(hmw->hdriver, cmdid) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Perform the signed integer 24bits to float32 conversion
  * @param hmw          HSP handle.
  * @param pSrc         Input Buffer address
  * @param pDst         Output Buffer address
  * @param nbSamples    Number of float elements to proceed
  * @param ioType       User iotype information
  * @retval             Status returned.
  */
hsp_core_status_t HSP_SEQ_Vect24s2F(hsp_core_handle_t *hmw, int32_t *pSrc, float32_t *pDst, uint32_t nbSamples,
                                      uint32_t ioType)
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
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
  uint32_t cmdid = HSP_CMD_VEC_24S2F;
  if (nbSamples == 1)
  {
    cmdid = HSP_CMD_SCA_24S2F;
  }
  if (HSP_HW_IF_SendCommand(hmw->hdriver, cmdid) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Perform the float32 to Q31 conversion
  * @param hmw          HSP handle.
  * @param pSrc         Input Buffer address
  * @param pDst         Output Buffer address
  * @param nbSamples    Number of float elements to proceed
  * @param ioType       User iotype information
  * @retval             HAL status.
  */
hsp_core_status_t HSP_SEQ_VectF2Q31(hsp_core_handle_t *hmw, float32_t *pSrc, int32_t *pDst, uint32_t nbSamples,
                                      uint32_t ioType)
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
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
  uint32_t cmdid = HSP_CMD_VEC_F2Q31;
  if (nbSamples == 1)
  {
    cmdid = HSP_CMD_SCA_F2Q31;
  }
  if (HSP_HW_IF_SendCommand(hmw->hdriver, cmdid) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Perform the Q31 to float32 conversion
  * @param hhsp        HSP handle.
  * @param pSrc        Input Buffer address
  * @param pDst        Output Buffer address
  * @param nbSamples   Number of float elements to proceed
  * @param ioType      User iotype information
  * @retval            HAL status.
  */
hsp_core_status_t HSP_SEQ_VectQ312F(hsp_core_handle_t *hmw, int32_t *pSrc, float32_t *pDst, uint32_t nbSamples,
                                      uint32_t ioType)
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
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
  uint32_t cmdid = HSP_CMD_VEC_Q312F;
  if (nbSamples == 1)
  {
    cmdid = HSP_CMD_SCA_Q312F;
  }
  if (HSP_HW_IF_SendCommand(hmw->hdriver, cmdid) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Perform the float32 to Q15 conversion
  * @param hmw          HSP handle.
  * @param pSrc         Input Buffer address
  * @param pDst         Output Buffer address
  * @param nbSamples    Number of float elements to proceed
  * @param ioType       User iotype information
  * @retval             Status returned.
  */
hsp_core_status_t HSP_SEQ_VectF2Q15(hsp_core_handle_t *hmw, float32_t *pSrc, int32_t *pDst, uint32_t nbSamples,
                                      uint32_t ioType)
{
  HSP_CHECK_CMD_SIZE_NULL(hhsp, nbSamples);
  HSP_CHECK_ASSERT(hhsp, (nbSamples > 1));

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
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
  uint32_t cmdid = HSP_CMD_VEC_F2Q15;
  if (HSP_HW_IF_SendCommand(hmw->hdriver, cmdid) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Perform the Q15 to float32 conversion
  * @param hhsp        HSP handle.
  * @param pSrc        Input Buffer address
  * @param pDst        Output Buffer address
  * @param nbSamples   Number of float elements to proceed
  * @param ioType      User iotype information
  * @retval            HAL status.
  */
hsp_core_status_t HSP_SEQ_VectQ152F(hsp_core_handle_t *hmw, int32_t *pSrc, float32_t *pDst, uint32_t nbSamples,
                                      uint32_t ioType)
{
  HSP_CHECK_CMD_SIZE_NULL(hhsp, nbSamples);
  HSP_CHECK_ASSERT(hhsp, (nbSamples > 1));

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
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
  uint32_t cmdid = HSP_CMD_VEC_Q152F;
  if (HSP_HW_IF_SendCommand(hmw->hdriver, cmdid) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Performs the decimation of a vector
  * @param hmw          HSP handle.
  * @param pSrc         Input Buffer address
  * @param decim        Decimator factor
  * @param pDst         Output Buffer address
  * @param sizeOu       Output vector size
  * @param ioType       User iotype information
  * @retval             Status returned.
  */
hsp_core_status_t HSP_SEQ_VectDecim_f32(hsp_core_handle_t *hmw, float32_t *pSrc, uint32_t decim,
                                          float32_t *pDst, uint32_t sizeOu, uint32_t ioType)
{
  HSP_CHECK_CMD_SIZE_NULL(hhsp, decim);
  HSP_CHECK_CMD_SIZE_NULL(hhsp, sizeOu);
  HSP_CHECK_ASSERT(hhsp, (sizeOu > 1));

  uint32_t encoded = 0;
  uint32_t ouAddr1, ouAddr3;
  ioType |= HSP_SEQ_IOTYPE_IMM_1;

  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t) pSrc, &ouAddr1,
                               0, NULL, (uint32_t) pDst, &ouAddr3, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  /* Second param is not used */
  encoded = encoded & ~HSP_IOTYPE_IMM;
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, decim);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr3);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, sizeOu);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_VEC_DECIM) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Performs the interpolation of a real vector by inserting N-1 zeros between 2 samples
  * @param hmw          HSP handle.
  * @param pSrc         Input Buffer address
  * @param interp       Interpolation factor
  * @param pDst         Output Buffer address
  * @param sizeIn       Input vector size
  * @param ioType       User iotype information
  * @retval             Status returned.
  */
hsp_core_status_t HSP_SEQ_VectZins_f32(hsp_core_handle_t *hmw, float32_t *pSrc, uint32_t interp, float32_t *pDst,
                                         uint32_t sizeIn, uint32_t ioType)
{
  HSP_CHECK_CMD_SIZE_NULL(hhsp, interp);
  HSP_CHECK_CMD_SIZE_NULL(hhsp, sizeIn);
  HSP_CHECK_ASSERT(hhsp, (sizeIn > 1));

  uint32_t encoded = 0;
  uint32_t ouAddr1, ouAddr3;
  ioType |= HSP_SEQ_IOTYPE_IMM_1;

  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t) pSrc, &ouAddr1,
                               0, NULL, (uint32_t) pDst, &ouAddr3, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  /* Second param is not used */
  encoded = encoded & ~HSP_IOTYPE_IMM;
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, interp);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr3);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, sizeIn);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_VEC_ZINS) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Performs atan2 of input vector x and y
  * @param hmw          HSP handle.
  * @param pSrcA        Input A Buffer address (contains x coordinates)
  * @param pSrcB        Input B Buffer address (contains y coordinates)
  * @param pDst         Output Buffer address
  * @param nbSamples    Number of float elements to proceed
  * @param ioType       User iotype information
  * @retval             HAL status.
  */
hsp_core_status_t HSP_SEQ_VectAtan2_f32(hsp_core_handle_t *hmw, float32_t *pSrcA, float32_t *pSrcB,
                                          float32_t *pDst, uint32_t nbSamples, uint32_t ioType)
{
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
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  uint32_t cmdid = HSP_CMD_VEC_ATAN2_F32;
  if (nbSamples == 1)
  {
    cmdid = HSP_CMD_SCA_ATAN2_F32;
  }
  if (HSP_HW_IF_SendCommand(hmw->hdriver, cmdid) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Performs logarithm natural of each element of input vector x
  * @param hmw        HSP handle.
  * @param pSrc       Input Buffer address
  * @param pDst       Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @param ioType     User iotype information
  * @retval           HAL status.
  */
hsp_core_status_t HSP_SEQ_VectLn_f32(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pDst, uint32_t nbSamples,
                                       uint32_t ioType)
{
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
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  uint32_t cmdid = HSP_CMD_VEC_LN_F32;
  if (nbSamples == 1)
  {
    cmdid = HSP_CMD_SCA_LN_F32;
  }
  if (HSP_HW_IF_SendCommand(hmw->hdriver, cmdid) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Performs common logarithm (base 10) of each element of input vector x
  * @param hmw        HSP handle.
  * @param pSrc       Input Buffer address
  * @param pDst       Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @param ioType     User iotype information
  * @retval           HAL status.
  */
hsp_core_status_t HSP_SEQ_VectLog10_f32(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pDst, uint32_t nbSamples,
                                          uint32_t ioType)
{
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
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  uint32_t cmdid = HSP_CMD_VEC_LOG10_F32;
  if (nbSamples == 1)
  {
    cmdid = HSP_CMD_SCA_LOG10_F32;
  }
  if (HSP_HW_IF_SendCommand(hmw->hdriver, cmdid) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Performs exponential of each element of input vector x
  * @param hmw        HSP handle.
  * @param pSrc       Input Buffer address
  * @param pDst       Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @param ioType     User iotype information
  * @retval           HAL status.
  */
hsp_core_status_t HSP_SEQ_VectExp_f32(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pDst, uint32_t nbSamples,
                                        uint32_t ioType)
{
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
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  uint32_t cmdid = HSP_CMD_VEC_EXP_F32;
  if (nbSamples == 1)
  {
    cmdid = HSP_CMD_SCA_EXP_F32;
  }
  if (HSP_HW_IF_SendCommand(hmw->hdriver, cmdid) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Performs powers of ten (exp base 10) of each element of input vector x
  * @param hmw        HSP handle.
  * @param pSrc       Input Buffer address
  * @param pDst       Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @param ioType     User iotype information
  * @retval           HAL status.
  */
hsp_core_status_t HSP_SEQ_VectExp10_f32(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pDst, uint32_t nbSamples,
                                          uint32_t ioType)
{
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
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  uint32_t cmdid = HSP_CMD_VEC_EXP10_F32;
  if (nbSamples == 1)
  {
    cmdid = HSP_CMD_SCA_EXP10_F32;
  }
  if (HSP_HW_IF_SendCommand(hmw->hdriver, cmdid) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Compute the multiplication of one vector and cos ROM
  * and an exponential complex signal
  * @param hhsp         HSP handle.
  * @param pSrc         Input Buffer address
  * @param startBuff    Start buffer address (I/O: input is start index and output is nextIdx)
  * @param pDst         Output Buffer address
  * @param nbSamples    Number of float elements to proceed
  * @param step         Step value between 2 cos number in ROM
  * @param ioType       User iotype information
  * @retval             HAL status.
  */
hsp_core_status_t HSP_SEQ_VectMulCos_f32(hsp_core_handle_t *hmw, float32_t *pSrcA, float32_t *pSrcB, float32_t *pDst,
                                           uint32_t nbSamples, int32_t step, uint32_t ioType)
{
  /* Check command size */
  HSP_CHECK_CMD_SIZE_NULL(hhsp, nbSamples);
  HSP_CHECK_ASSERT(hhsp, (nbSamples > 1));

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
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM4, step);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_VEC_MUL_COS_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Compute the multiplication of one vector and sin ROM
  * and an exponential complex signal
  * @param hmw          HSP handle.
  * @param pSrcId       Input Buffer address
  * @param startBuff    Start buffer address (I/O: input is start index and output is nextIdx)
  * @param pDst         Output Buffer address
  * @param nbSamples    Number of float elements to proceed
  * @param step         Step value between 2 sin number in ROM
  * @param ioType       User iotype information
  * @retval             Status returned.
  */
hsp_core_status_t HSP_SEQ_VectMulSin_f32(hsp_core_handle_t *hmw, float32_t *pSrcA, float32_t *pSrcB, float32_t *pDst,
                                           uint32_t nbSamples, int32_t step, uint32_t ioType)
{
  /* Check command size */
  HSP_CHECK_CMD_SIZE_NULL(hhsp, nbSamples);
  HSP_CHECK_ASSERT(hhsp, (nbSamples > 1));

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
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM4, step);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_VEC_MUL_SIN_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Add to each vector element a immediate scalar value
  * @param hmw          HSP handle.
  * @param pSrc         Input Buffer address
  * @param offset       Immediate value offset to add Buffer address
  * @param pDst         Output Buffer address
  * @param nbSamples    Number of float elements to proceed
  * @param ioType       User iotype information
  * @retval             Status returned.
  */
hsp_core_status_t HSP_SEQ_VectIOffset_f32(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t offset, float32_t *pDst,
                                            uint32_t nbSamples, uint32_t ioType)
{
  /* Check command size */
  HSP_CHECK_CMD_SIZE_NULL(hhsp, nbSamples);
  HSP_CHECK_ASSERT(hhsp, (nbSamples > 1));

  uint32_t encoded = 0;
  uint32_t ouAddr1, ouAddr3;
  ioType |= HSP_SEQ_IOTYPE_IMM_1;

  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t) pSrc, &ouAddr1,
                               0, NULL, (uint32_t) pDst, &ouAddr3, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, *((uint32_t *)&offset));
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr3);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_VEC_OFFSET_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Multiply each vector element by a immediate scalar value
  * @param hmw          HSP handle.
  * @param pSrc         Input Buffer address
  * @param scale        Scale immediate scalar value
  * @param pDst         Output Buffer address
  * @param nbSamples    Number of float elements to proceed
  * @param ioType       User iotype information
  * @retval             Status returned.
  */
hsp_core_status_t HSP_SEQ_VectIScale_f32(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t scale, float32_t *pDst,
                                           uint32_t nbSamples, uint32_t ioType)
{
  /* Check command size */
  HSP_CHECK_CMD_SIZE_NULL(hhsp, nbSamples);
  HSP_CHECK_ASSERT(hhsp, (nbSamples > 1));

  uint32_t encoded = 0;
  uint32_t ouAddr1, ouAddr3;
  ioType |= HSP_SEQ_IOTYPE_IMM_1;

  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t) pSrc, &ouAddr1,
                               0, NULL, (uint32_t) pDst, &ouAddr3, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, *((uint32_t *)&scale));
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr3);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_VEC_SCALE_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Set value for whole vector using immediate value
  * @param hmw          HSP handle.
  * @param value        Value to set in vector
  * @param pDst         Output Buffer address
  * @param nbSamples    Number of float elements to proceed
  * @param ioType       User iotype information
  * @retval             HAL status.
  */
hsp_core_status_t HSP_SEQ_VectISet_f32(hsp_core_handle_t *hmw, float32_t value, float32_t *pDst, uint32_t nbSamples,
                                         uint32_t ioType)
{
  /* Check command size */
  HSP_CHECK_CMD_SIZE_NULL(hhsp, nbSamples);

  uint32_t encoded = 0;
  uint32_t ouAddr2;
  ioType |= HSP_SEQ_IOTYPE_IMM_0;

  /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, 0, NULL,
                               (uint32_t) pDst, &ouAddr2, 0, NULL, 2) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }

  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, *((uint32_t *)&value));
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  uint32_t cmdid = HSP_CMD_VEC_SET;
  if (nbSamples == 1)
  {
    cmdid = HSP_CMD_SCA_SET;
  }

  if (HSP_HW_IF_SendCommand(hmw->hdriver, cmdid) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Comparision of an input vector to two thresholds (LOTH, HITH) with branch option
  * @param hmw          HSP handle.
  * @param pSrc         Input vector Buffer address
  * @param pLim         Scalar buffer address
  * @param pRes         Vector comparison result flag
  * @param nbSamples    Number of samples to proceed
  * @param cmpType      Comparison type (could be an enum in hsp_fw_def: TBD)
  * @param ioType       User iotype information
  * @retval             Status returned.
  */
hsp_core_status_t HSP_SEQ_VectCmp_f32(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pLim, uint32_t *pRes,
                                        uint32_t nbSamples, uint32_t cmpType, uint32_t ioType)
{
  /* Check command size */
  HSP_CHECK_CMD_SIZE_NULL(hhsp, nbSamples);
  HSP_CHECK_CMD_SIZE_NULL(hhsp, cmpType);
  HSP_CHECK_ASSERT(hhsp, (cmpType <= HSP_CMP_TYPE_EQ)); /* Should be the max of all supported types */

  uint32_t encoded = 0;
  uint32_t ouAddr1, ouAddr2, ouAddr3;

  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t) pSrc, &ouAddr1,
                               (uint32_t) pLim, &ouAddr2, (uint32_t) pRes, &ouAddr3, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr3);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM4, cmpType);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_CMPB_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Comparision of an input scalar to two thresholds (LOTH, HITH) and count before branch option
  * @param hmw          HSP handle.
  * @param pSrc         Input value address (float)
  * @param pDst         Res value address (int32_t)
  * @param pLim         Pointer on comparision limit [loTh, hiTh] (float)
  * @param pMax         Pointer on max comparision limit [loMcnt, hiMcnt] (uint32_t)
  * @param pCnt         Pointer on counter struct [locnt, hicnt] (uint32_t)
  * @param ioType       User iotype information
  * @retval             Status returned.
  */
hsp_core_status_t HSP_SEQ_VectCmpCnt_f32(hsp_core_handle_t *hmw, float32_t *pSrc, int32_t *pDst,
                                         hsp_cmp_cnt_lim_t *pLim, hsp_cmp_cnt_cnt_t *pMax,
                                         hsp_cmp_cnt_cnt_t *pCnt, uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1, ouAddr2, ouAddr3;

  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t) pSrc, &ouAddr1,
                               (uint32_t) pDst, &ouAddr2, (uint32_t) pLim, &ouAddr3, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  uint32_t ouAddr4, ouAddr5;
  if (HSP_UTILITIES_ToBramABAddress(hmw, (uint32_t) pMax, &ouAddr4) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  if (HSP_UTILITIES_ToBramABAddress(hmw, (uint32_t) pCnt, &ouAddr5) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr3);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, ouAddr4);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM4, ouAddr5);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_CMPCNT_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}


/**
  * @brief Comparision of an input vector to two thresholds (LOTH, HITH) with branch option + vector saturation
  * @param hmw          HSP handle.
  * @param pSrc         Input buffer address (float)
  * @param pSat         Pointer on saturation limit [loTh, hiTh] (float)
  * @param pDst         Output buffer address (int32_t)
  * @param nbSamples    Number of samples to proceed
  * @param pRes         Res value address (uint32_t, 1 if saturation)
  * @param ioType       User iotype information
  * @retval             Status returned.
  */
hsp_core_status_t HSP_SEQ_VectSat_f32(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pSat, float32_t *pDst,
                                      uint32_t nbSamples, uint32_t *pRes, uint32_t ioType)
{
  /* Check command size */
  HSP_CHECK_CMD_SIZE_NULL(hhsp, nbSamples);

  uint32_t encoded = 0;
  uint32_t ouAddr1, ouAddr2, ouAddr3;

  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t) pSrc, &ouAddr1,
                               (uint32_t) pSat, &ouAddr2, (uint32_t) pDst, &ouAddr3, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  uint32_t ouAddr4;
  if (HSP_UTILITIES_ToBramABAddress(hmw, (uint32_t) pRes, &ouAddr4) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr3);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, ouAddr4);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM4, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_VEC_SAT_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

#ifdef __HSP_DMA__
/** @addtogroup HSP_Exported_Functions_Group12 HSP Memory Transfer  family function
  *  @brief    HSP Memory Transfer functions for processing list
  *
@verbatim
  ==============================================================================
                  ##### HSP Memory Transfer functions #####
  ==============================================================================
  [..]
    This section provides functions allowing to:
    (+) Add memory blocking transfer function from external to internal memory
    (+) Add memory blocking transfer function from internal to external memory
    (+) Add memory blocking transfer function from internal to internal memory
    (+) Add memory transfer function from external to internal memory
    (+) Add memory transfer function from internal to external memory
    (+) Add memory transfer function from internal to internal memory
    (+) Processing List background block transfer command

@endverbatim
  * @{
  */
/**
  * @brief Add memory blocking transfer function from external to internal memory in the current processing list.
  * @param hhsp         HSP handle.
  * @param pSrc         External input buffer pointer
  * @param pDst        Internal output buffer pointer
  * @param nbElems      Number of elements to proceed
  * @param eltFormat    Destination element format to transfer ({HSP_DMA_ELT_FMT_32B 32-bits} {HSP_DMA_ELT_FMT_64B 64-bits} {HSP_DMA_ELT_FMT_U16 uint16_t} {HSP_DMA_ELT_FMT_S16 int16_t})
  * @param iJump        Jump between 2 inputs element read, unit is input element size
  * @param nbIterIn     Number of iterations for input offset application
  * @param offsetIn     Input offset, unit is input element size
  * @param oJump        Jump between 2 outputs element write, unit is output element size
  * @param nbIterOu     Number of iterations for output offset application
  * @param offsetOu     Output offset, unit is output element size
  * @retval             Status returned.
  */
hsp_core_status_t HSP_SEQ_Ext2Int(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pDst, uint32_t nbElems,
                                  uint32_t eltFormat, uint32_t iJump, uint32_t nbIterIn, uint32_t offsetIn,
                                  uint32_t oJump, uint32_t nbIterOu, uint32_t offsetOu)
{
  HSP_CHECK_CMD_SIZE_NULL(hmw, nbElems);
  uint32_t encoded = 0;
  uint32_t ouAddr1, ouAddr2;

  /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, HSP_SEQ_IOTYPE_EXT_0, &encoded, (uint32_t) pSrc, &ouAddr1,
                               (uint32_t) pDst, &ouAddr2, 0, NULL, 2) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }

  /* REG_0  Source buffer. Source buffer type must be EXT_BUFF */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  /* REG_1  Destination buffer. Destination buffer type must be INT_BUFF */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  /* REG_2  Increment of the source buffer */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, iJump);
  /* REG_3  Number of iterations for input offset application */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, nbIterIn);
  /* REG_4  Increment of the destination buffer */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM4, oJump);
  /* REG_5  Number of iterations for output offset application */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM5, nbIterOu);
  /* REG_6  Format of the data in the source buffer and mode */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM6, eltFormat);
  /* REG_7  Number of elements to proceed */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM7, nbElems);
  /* REG_8  Input offset */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM8, offsetIn);
  /* REG_9  Output offset */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM9, offsetOu);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_EXT2INT) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Add memory blocking transfer function from internal to external memory in the current processing list.
  * @param hmw         HSP handle.
  * @param pSrc         Internal input buffer pointer
  * @param pDst        External output buffer pointer
  * @param nbElems      Number of elements to proceed
  * @param eltFormat    Destination element format to transfer ({HSP_DMA_ELT_FMT_32B 32-bits} {HSP_DMA_ELT_FMT_64B 64-bits} {HSP_DMA_ELT_FMT_U16 uint32_t} {HSP_DMA_ELT_FMT_S16 int32_t})
  * @param iJump        Jump between 2 inputs element read, unit is input element size
  * @param nbIterIn     Number of iterations for input offset application
  * @param offsetIn     Input offset, unit is input element size
  * @param oJump        Jump between 2 outputs element write, unit is output element size
  * @param nbIterOu     Number of iterations for output offset application
  * @param offsetOu     Output offset, unit is output element size
  * @retval             Status returned.
  */
hsp_core_status_t HSP_SEQ_Int2Ext(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pDst, uint32_t nbElems,
                                  uint32_t eltFormat, uint32_t iJump, uint32_t nbIterIn, uint32_t offsetIn,
                                  uint32_t oJump, uint32_t nbIterOu, uint32_t offsetOu)
{
  HSP_CHECK_CMD_SIZE_NULL(hmw, nbElems);
  uint32_t encoded = 0;
  uint32_t ouAddr1, ouAddr2;

  /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, HSP_SEQ_IOTYPE_EXT_1, &encoded, (uint32_t) pSrc, &ouAddr1,
                               (uint32_t) pDst, &ouAddr2, 0, NULL, 2) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }

  /* REG_0  Source buffer. Source buffer type must be EXT_BUFF */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  /* REG_1  Destination buffer. Destination buffer type must be INT_BUFF */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  /* REG_2  Increment of the source buffer */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, iJump);
  /* REG_3  Number of iterations for input offset application */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, nbIterIn);
  /* REG_4  Increment of the destination buffer */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM4, oJump);
  /* REG_5  Number of iterations for output offset application */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM5, nbIterOu);
  /* REG_6  Format of the data in the source buffer and mode */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM6, eltFormat);
  /* REG_7  Number of elements to proceed */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM7, nbElems);
  /* REG_8  Input offset */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM8, offsetIn);
  /* REG_9  Output offset */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM9, offsetOu);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_INT2EXT) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Add memory blocking transfer function from internal to internal memory in the current processing list.
  * @param hmw         HSP handle.
  * @param pSrc         Internal input buffer pointer
  * @param pDst        Internal output buffer pointer
  * @param nbElems      Number of elements to proceed
  * @param eltFormat    Destination element format to transfer ({HSP_DMA_ELT_FMT_32B 32-bits}
  * @param iJump        Jump between 2 inputs element read, unit is input element size
  * @param nbIterIn     Number of iterations for input offset application
  * @param offsetIn     Input offset, unit is input element size
  * @param oJump        Jump between 2 outputs element write, unit is output element size
  * @param nbIterOu     Number of iterations for output offset application
  * @param offsetOu     Output offset, unit is output element size
  * @retval             Status returned.
  */
hsp_core_status_t HSP_SEQ_Int2Int(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pDst, uint32_t nbElems,
                                  uint32_t eltFormat, uint32_t iJump, uint32_t nbIterIn, uint32_t offsetIn,
                                  uint32_t oJump, uint32_t nbIterOu, uint32_t offsetOu)
{
  HSP_CHECK_CMD_SIZE_NULL(hmw, nbElems);
  uint32_t encoded = 0;
  uint32_t ouAddr1, ouAddr2;

  /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, HSP_SEQ_IOTYPE_DEFAULT, &encoded, (uint32_t) pSrc, &ouAddr1,
                               (uint32_t) pDst, &ouAddr2, 0, NULL, 2) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }

  /* REG_0  Source buffer. Source buffer type must be EXT_BUFF */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  /* REG_1  Destination buffer. Destination buffer type must be INT_BUFF */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  /* REG_2  Increment of the source buffer */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, iJump);
  /* REG_3  Number of iterations for input offset application */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, nbIterIn);
  /* REG_4  Increment of the destination buffer */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM4, oJump);
  /* REG_5  Number of iterations for output offset application */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM5, nbIterOu);
  /* REG_6  Format of the data in the source buffer and mode */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM6, eltFormat);
  /* REG_7  Number of elements to proceed */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM7, nbElems);
  /* REG_8  Input offset */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM8, offsetIn);
  /* REG_9  Output offset */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM9, offsetOu);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_INT2INT) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Add memory transfer function from external to internal memory in the current processing list.
  * @param hmw         HSP handle.
  * @param pSrc         External input buffer pointer
  * @param pDst        Internal output buffer pointer
  * @param nbElems      Number of elements to proceed
  * @param eltFormat    Destination element format to transfer ({HSP_DMA_ELT_FMT_32B 32-bits} {HSP_DMA_ELT_FMT_64B 64-bits} {HSP_DMA_ELT_FMT_U16 uint32_t} {HSP_DMA_ELT_FMT_S16 int32_t})
  * @param chanId       Channel index ({1 channel 1} {2 channel 2})
  * @param iJump        Jump between 2 inputs element read, unit is input element size
  * @param nbIterIn     Number of iterations for input offset application
  * @param offsetIn     Input offset, unit is input element size
  * @param oJump        Jump between 2 outputs element write, unit is output element size
  * @param nbIterOu     Number of iterations for output offset application
  * @param offsetOu     Output offset, unit is output element size
  * @param nextOffsetIn Next input offset used by bgnd next function (source block increment)
  * @param nextOffsetOu Next output offset used by bgnd next function (destination block increment)
  * @retval             Status returned.
  */
hsp_core_status_t HSP_SEQ_BgndExt2Int(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pDst,
                                      uint32_t nbElems, uint32_t eltFormat, uint32_t chanId, uint32_t iJump,
                                      uint32_t nbIterIn, uint32_t offsetIn, uint32_t oJump, uint32_t nbIterOu,
                                      uint32_t offsetOu, uint32_t nextOffsetIn, uint32_t nextOffsetOu)
{
  HSP_CHECK_CMD_SIZE_NULL(hmw, nbElems);
  uint32_t encoded = 0;
  uint32_t ouAddr1, ouAddr2;

  /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, HSP_SEQ_IOTYPE_EXT_0, &encoded, (uint32_t) pSrc, &ouAddr1,
                               (uint32_t) pDst, &ouAddr2, 0, NULL, 2) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }

  /* REG_0  Source buffer. Source buffer type must be EXT_BUFF */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  /* REG_1  Destination buffer. Destination buffer type must be INT_BUFF */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  /* REG_2  Increment of the source buffer */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, iJump);
  /* REG_3  Number of iterations for input offset application */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, nbIterIn);
  /* REG_4  Increment of the destination buffer */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM4, oJump);
  /* REG_5  Number of iterations for output offset application */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM5, nbIterOu);
  /* REG_6  Format of the data in the source buffer and mode */
  uint32_t tmp32 = eltFormat;
  tmp32 |= chanId;
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM6, tmp32);
  /* REG_7  Number of elements to proceed */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM7, nbElems);
  /* REG_8  Input offset */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM8, offsetIn);
  /* REG_9  Output offset */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM9, offsetOu);
  /* REG_10  Output offset */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM10, nextOffsetIn);
  /* REG_11  Output offset */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM11, nextOffsetOu);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_EXT2INT) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Add memory transfer function from internal to external memory in the current processing list.
  * @param hmw         HSP handle.
  * @param pSrc         Internal input buffer pointer
  * @param pDst        External output buffer pointer
  * @param nbElems      Number of elements to proceed
  * @param eltFormat    Destination element format to transfer ({HSP_DMA_ELT_FMT_32B 32-bits} {HSP_DMA_ELT_FMT_64B 64-bits} {HSP_DMA_ELT_FMT_U16 uint32_t} {HSP_DMA_ELT_FMT_S16 int32_t})
  * @param chanId       Channel index ({1 channel 1} {2 channel 2})
  * @param iJump        Jump between 2 inputs element read, unit is input element size
  * @param nbIterIn     Number of iterations for input offset application
  * @param offsetIn     Input offset, unit is input element size
  * @param oJump        Jump between 2 outputs element write, unit is output element size
  * @param nbIterOu     Number of iterations for output offset application
  * @param offsetOu     Output offset, unit is output element size
  * @param nextOffsetIn Next input offset used by bgnd next function (source block increment)
  * @param nextOffsetOu Next output offset used by bgnd next function (destination block increment)
  * @retval             Status returned.
  */
hsp_core_status_t HSP_SEQ_BgndInt2Ext(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pDst,
                                      uint32_t nbElems, uint32_t eltFormat, uint32_t chanId, uint32_t iJump,
                                      uint32_t nbIterIn, uint32_t offsetIn, uint32_t oJump, uint32_t nbIterOu,
                                      uint32_t offsetOu, uint32_t nextOffsetIn, uint32_t nextOffsetOu)
{
  HSP_CHECK_CMD_SIZE_NULL(hmw, nbElems);
  uint32_t encoded = 0;
  uint32_t ouAddr1, ouAddr2;

  /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, HSP_SEQ_IOTYPE_EXT_1, &encoded, (uint32_t) pSrc, &ouAddr1,
                               (uint32_t) pDst, &ouAddr2, 0, NULL, 2) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }

  /* REG_0  Source buffer. Source buffer type must be EXT_BUFF */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  /* REG_1  Destination buffer. Destination buffer type must be INT_BUFF */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  /* REG_2  Increment of the source buffer */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, iJump);
  /* REG_3  Number of iterations for input offset application */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, nbIterIn);
  /* REG_4  Increment of the destination buffer */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM4, oJump);
  /* REG_5  Number of iterations for output offset application */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM5, nbIterOu);
  /* REG_6  Format of the data in the source buffer and mode */
  uint32_t tmp32 = eltFormat;
  tmp32 |= chanId;
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM6, tmp32);
  /* REG_7  Number of elements to proceed */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM7, nbElems);
  /* REG_8  Input offset */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM8, offsetIn);
  /* REG_9  Output offset */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM9, offsetOu);
  /* REG_10  Output offset */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM10, nextOffsetIn);
  /* REG_11  Output offset */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM11, nextOffsetOu);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_INT2EXT) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Add memory transfer function from internal to internal memory in the current processing list.
  * @param hmw         HSP handle.
  * @param pSrc         Internal input buffer pointer
  * @param pDst        Internal output buffer pointer
  * @param nbElems      Number of elements to proceed
  * @param eltFormat    Destination element format to transfer ({HSP_DMA_ELT_FMT_32B 32-bits}
  * @param chanId       Channel index ({1 channel 1} {2 channel 2})
  * @param iJump        Jump between 2 inputs element read, unit is input element size
  * @param nbIterIn     Number of iterations for input offset application
  * @param offsetIn     Input offset, unit is input element size
  * @param oJump        Jump between 2 outputs element write, unit is output element size
  * @param nbIterOu     Number of iterations for output offset application
  * @param offsetOu     Output offset, unit is output element size
  * @param nextOffsetIn Next input offset used by bgnd next function (source block increment)
  * @param nextOffsetOu Next output offset used by bgnd next function (destination block increment)
  * @retval             Status returned.
  */
hsp_core_status_t HSP_SEQ_BgndInt2Int(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pDst,
                                      uint32_t nbElems, uint32_t eltFormat, uint32_t chanId, uint32_t iJump,
                                      uint32_t nbIterIn, uint32_t offsetIn, uint32_t oJump, uint32_t nbIterOu,
                                      uint32_t offsetOu, uint32_t nextOffsetIn, uint32_t nextOffsetOu)
{
  HSP_CHECK_CMD_SIZE_NULL(hmw, nbElems);
  uint32_t encoded = 0;
  uint32_t ouAddr1, ouAddr2;

  /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, HSP_SEQ_IOTYPE_DEFAULT, &encoded, (uint32_t) pSrc, &ouAddr1,
                               (uint32_t) pDst, &ouAddr2, 0, NULL, 2) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }

  /* REG_0  Source buffer. Source buffer type must be EXT_BUFF */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  /* REG_1  Destination buffer. Destination buffer type must be INT_BUFF */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  /* REG_2  Increment of the source buffer */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, iJump);
  /* REG_3  Number of iterations for input offset application */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, nbIterIn);
  /* REG_4  Increment of the destination buffer */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM4, oJump);
  /* REG_5  Number of iterations for output offset application */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM5, nbIterOu);
  /* REG_6  Format of the data in the source buffer and mode */
  uint32_t tmp32 = eltFormat;
  tmp32 |= chanId;
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM6, tmp32);
  /* REG_7  Number of elements to proceed */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM7, nbElems);
  /* REG_8  Input offset */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM8, offsetIn);
  /* REG_9  Output offset */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM9, offsetOu);
  /* REG_10  Output offset */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM10, nextOffsetIn);
  /* REG_11  Output offset */
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM11, nextOffsetOu);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_INT2INT) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief PL background block transfer command
  * @param hmw         HSP handle.
  * @param cmd          Background command: ({0 next} {1 wait} {2 stop})
  * @param chanId       Background channel ID
  * @retval             Status returned.
  */
hsp_core_status_t HSP_SEQ_BgndCmd(hsp_core_handle_t *hmw, uint32_t cmd, uint32_t chanId)
{
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, cmd);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, chanId);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_DMA_BGND_CMD) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}
#endif /* __HSP_DMA__ */

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
