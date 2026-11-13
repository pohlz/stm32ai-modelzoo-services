/**
  ******************************************************************************
  * @file    hsp_proclist_scalar.c
  * @author  MCD Application Team
  * @brief   This file implements the HSP SCALAR Processing functions used to
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
#include "hsp_proclist_scalar.h"
#include "hsp_hw_if.h"
#include "hsp_utilities.h"

/** @addtogroup HSP
  * @{
  */

/** @addtogroup HSP_PROCLIST
  * @{
  */
/** @addtogroup HSP_MODULES_PROCLIST_SCALAR_LIBRARY
  * @{
  */
/* Private defines -----------------------------------------------------------*/
/* Private typedef -----------------------------------------------------------*/
/* Private macros ------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
#define HSP_CHECK_CMD_SIZE_NULL(hmw, size)
#define HSP_CHECK_ASSERT(hmw, cond)

/* Private function prototypes -----------------------------------------------*/
/* Exported functions --------------------------------------------------------*/

/** @addtogroup HSP_Exported_Functions_ProcList_Scalar
  * @{
  */
/**
  * @brief Perform the Scalar exponential base 10
  * @param hmw         HSP handle.
  * @param inBuff      Input Buffer address
  * @param outBuff     Output Buffer address
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_ScaExp10_f32(hsp_core_handle_t *hmw, uint32_t *inBuff, float32_t *outBuff, uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  
  /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t) inBuff, &ouAddr1,
                               (uint32_t) outBuff, &ouAddr2, 0, NULL, 2) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }

  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_EXP10_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}
  
/**
  * @brief Perform the Scalar exponential
  * @param hmw         HSP handle.
  * @param inBuff      Input Buffer address
  * @param outBuff     Output Buffer address
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_ScaExp_f32(hsp_core_handle_t *hmw, uint32_t *inBuff, float32_t *outBuff, uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  
  /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t) inBuff, &ouAddr1,
                               (uint32_t) outBuff, &ouAddr2, 0, NULL, 2) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }

  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_EXP_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Perform the Scalar logarithm base 10
  * @param hmw         HSP handle.
  * @param inBuff      Input Buffer address
  * @param outBuff     Output Buffer address
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_ScaLog10_f32(hsp_core_handle_t *hmw, uint32_t *inBuff, float32_t *outBuff, uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  
  /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t) inBuff, &ouAddr1,
                               (uint32_t) outBuff, &ouAddr2, 0, NULL, 2) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }

  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_LOG10_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}
  
/**
  * @brief Perform the Scalar logarithm neperien
  * @param hmw         HSP handle.
  * @param inBuff      Input Buffer address
  * @param outBuff     Output Buffer address
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_ScaLn_f32(hsp_core_handle_t *hmw, uint32_t *inBuff, float32_t *outBuff, uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  
  /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t) inBuff, &ouAddr1,
                               (uint32_t) outBuff, &ouAddr2, 0, NULL, 2) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }

  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_LN_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Perform the integer 24bits to float32 conversion
  * @param hmw         HSP handle.
  * @param inBuff      Input Buffer address
  * @param outBuff     Output Buffer address
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_Sca24S2F(hsp_core_handle_t *hmw, int32_t *inBuff, float32_t *outBuff, uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  
  /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t) inBuff, &ouAddr1,
                               (uint32_t) outBuff, &ouAddr2, 0, NULL, 2) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }

  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_24S2F) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Perform the unsigned integer 32bits to float32 conversion
  * @param hmw         HSP handle.
  * @param inBuff      Input Buffer address
  * @param outBuff     Output Buffer address
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_ScaU2F(hsp_core_handle_t *hmw, uint32_t *inBuff, float32_t *outBuff, uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  
  /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t) inBuff, &ouAddr1,
                               (uint32_t) outBuff, &ouAddr2, 0, NULL, 2) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }

  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_U2F) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Perform the float32 to unsigned integer 32bits conversion
  * @param hmw         HSP handle.
  * @param inBuff      Input Buffer address
  * @param outBuff     Output Buffer address
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_ScaF2U(hsp_core_handle_t *hmw, float32_t *inBuff, uint32_t *outBuff, uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  
  /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t) inBuff, &ouAddr1,
                               (uint32_t) outBuff, &ouAddr2, 0, NULL, 2) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }

  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_F2U) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Perform the integer 32bits to float32 conversion
  * @param hmw         HSP handle.
  * @param inBuff      Input Buffer address
  * @param outBuff     Output Buffer address
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_ScaI2F(hsp_core_handle_t *hmw, int32_t *inBuff, float32_t *outBuff, uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  
  /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t) inBuff, &ouAddr1,
                               (uint32_t) outBuff, &ouAddr2, 0, NULL, 2) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }

  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_I2F) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Perform the float32 to integer 32bits conversion
  * @param hmw         HSP handle.
  * @param inBuff      Input Buffer address
  * @param outBuff     Output Buffer address
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_ScaF2I(hsp_core_handle_t *hmw, float32_t *inBuff, int32_t *outBuff, uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  
  /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t) inBuff, &ouAddr1,
                               (uint32_t) outBuff, &ouAddr2, 0, NULL, 2) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }

  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_F2I) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Subtract to one float an scalar float value
  * @param hmw         HSP handle.
  * @param inBuff      Input Buffer address
  * @param inBuffShift Input Value Buffer address
  * @param outBuff     Output Buffer address
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_ScaSub_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t *inBuffSub, float32_t *outBuff,
                                       uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  uint32_t ouAddr3;
   
   /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t)inBuff, &ouAddr1, (uint32_t)inBuffSub, &ouAddr2, (uint32_t)outBuff, &ouAddr3, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
 
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr3);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
 
  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_SUB_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }  
  return HSP_CORE_OK;  
}

/**
  * @brief Square-root of a scalar variable
  * @param hmw        HSP handle.
  * @param inBuff     Input Buffer value
  * @param outBuff    Output Buffer address
  * @param ioType     User iotype information
  * @retval           Core status.
  */
hsp_core_status_t HSP_SEQ_ScaSqrt_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t *outBuff,
                                     uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  
  /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t)inBuff, &ouAddr1,
                               (uint32_t)outBuff, &ouAddr2, 0, NULL, 2) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_SQRT_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Set value for a scalar variable
  * @param hmw        HSP handle.
  * @param inBuff     Input Buffer value
  * @param outBuff    Output Buffer address
  * @param ioType     User iotype information
  * @retval           Core status.
  */
hsp_core_status_t HSP_SEQ_ScaSet_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t *outBuff,
                                     uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  
  /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t)inBuff, &ouAddr1,
                               (uint32_t)outBuff, &ouAddr2, 0, NULL, 2) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_SET) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Multiplication of 2 float scalar
  * @param hmw         HSP handle.
  * @param inABuff     Input A Buffer address
  * @param inBBuff     Input B Buffer address
  * @param outBuff     Output Buffer address
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_ScaMul_f32(hsp_core_handle_t *hmw, float32_t *inABuff, float32_t *inBBuff, float32_t *outBuff,
                                     uint32_t ioType)
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
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
 
  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_MUL_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }  
  return HSP_CORE_OK;  
}
 
/**
  * @brief Division of 2 float scalar
  * @param hmw         HSP handle.
  * @param inABuff     Input A Buffer address
  * @param inBBuff     Input B Buffer address
  * @param outBuff     Output Buffer address
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_ScaDiv_f32(hsp_core_handle_t *hmw, float32_t *inABuff, float32_t *inBBuff, float32_t *outBuff,
                                     uint32_t ioType)
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
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
 
  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_DIV_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }  
  return HSP_CORE_OK;  
}
 
/**
  * @brief Additon of 2 float scalar
  * @param hmw         HSP handle.
  * @param inABuff     Input A Buffer address
  * @param inBBuff     Input B Buffer address
  * @param outBuff     Output Buffer address
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_ScaAdd_f32(hsp_core_handle_t *hmw, float32_t *inABuff, float32_t *inBBuff, float32_t *outBuff,
                                     uint32_t ioType)
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
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
 
  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_ADD_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }  
  return HSP_CORE_OK;  
}

 /**
  * @brief Scalar absolute value of a float value
  * @param hmw        HSP handle.
  * @param inBuff     Input Buffer address
  * @param outBuff    Output Buffer address
  * @param ioType     User iotype information
  * @retval           Core status.
  */
hsp_core_status_t HSP_SEQ_ScaAbs_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t *outBuff, 
                                     uint32_t ioType)										 
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  
  /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t)inBuff, &ouAddr1,
                               (uint32_t)outBuff, &ouAddr2, 0, NULL, 2) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }

  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_ABS_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
} 
  
/**
  * @brief Sine of a float value
  * @param hmw        HSP handle.
  * @param inBuff     Input Buffer address
  * @param outBuff    Output Buffer address
  * @param ioType     User iotype information
  * @retval           Core status.
  */
hsp_core_status_t HSP_SEQ_ScaSin_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t *outBuff, 
                                     uint32_t ioType)											  
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  
  /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t)inBuff, &ouAddr1,
                               (uint32_t)outBuff, &ouAddr2, 0, NULL, 2) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }

  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_SIN_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Cosine of a float value
  * @param hmw        HSP handle.
  * @param inBuff     Input Buffer address
  * @param outBuff    Output Buffer address
  * @param ioType     User iotype information
  * @retval           Core status.
  */
hsp_core_status_t HSP_SEQ_ScaCos_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t *outBuff, 
                                     uint32_t ioType)										 
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  
  /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t)inBuff, &ouAddr1,
                               (uint32_t)outBuff, &ouAddr2, 0, NULL, 2) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }

  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_COS_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Sine and Cosine of a float value
  * @param hmw        HSP handle.
  * @param inBuff     Input Buffer address
  * @param outBuff    Output Buffer address
  * @param ioType     User iotype information
  * @retval           Core status.
  */
hsp_core_status_t HSP_SEQ_ScaSinCos_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t *outBuff,
                                        uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  
  /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t)inBuff, &ouAddr1,
                               (uint32_t)outBuff, &ouAddr2, 0, NULL, 2) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }

  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_SINCOS_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }  
  return HSP_CORE_OK;
}

/**
  * @brief Clarke transform
  * @param hmw        HSP handle.
  * @param inBuff     Input three-phase coordinate Buffer address [Ia, Ib]
  * @param outBuff    Points to output two-phase orthogonal vector axis Buffer address [pIalpha, pIbeta]
  * @param ioType     User iotype information
  * @retval           Core status.
  */
hsp_core_status_t HSP_SEQ_ScaClarke_f32(hsp_core_handle_t *hmw, hsp_i_a_b_t *inBuff,
                                        hsp_i_alpha_beta_t *outBuff, uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  
  /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t)inBuff, &ouAddr1,
                               (uint32_t)outBuff, &ouAddr2, 0, NULL, 2) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }

  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_CLARKE_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }    
  return HSP_CORE_OK;
}

/**
  * @brief Floating-point Inverse Clarke transform
  * @param hmw        HSP handle.
  * @param inBuff     Input two-phase orthogonal vector Buffer address [pIalpha, pIbeta]
  * @param outBuff    Points to output three-phase coordinate Buffer address [Ia, Ib]
  * @param ioType     User iotype information
  * @retval           Core status.
  */
hsp_core_status_t HSP_SEQ_ScaIClarke_f32(hsp_core_handle_t *hmw, hsp_i_alpha_beta_t *inBuff,
                                         hsp_i_a_b_t *outBuff, uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  
  /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t)inBuff, &ouAddr1,
                               (uint32_t)outBuff, &ouAddr2, 0, NULL, 2) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_ICLARKE_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }    

  return HSP_CORE_OK;
}

/**
  * @brief Park transform: transforms stator values alpha and beta, which belong to a stationary qd reference
  * frame, to a rotor flux synchronous reference frame (properly oriented), so as q and d.
  *                   d = alpha *sin(theta) + beta *cos(Theta)
  *                   q = alpha *cos(Theta) - beta *sin(Theta)
  * @param hmw        HSP handle.
  * @param thetaBuff  Input angle rotating frame angular position is in float Buffer
  * @param abBuff     Input stator values alpha and beta [alpha, beta] in float Buffer  [alpha, beta]
  * @param outBuff    Output Stator values q and d [q, d] in float format Buffer
  * @param ioType     User iotype information
  * @retval           Core status.
  */
hsp_core_status_t HSP_SEQ_ScaPark_f32(hsp_core_handle_t *hmw, float32_t *thetaBuff,
                                      hsp_v_alpha_beta_t *abBuff, hsp_v_q_d_t *outBuff, uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  uint32_t ouAddr3;
  
  /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t)thetaBuff, &ouAddr1,
                               (uint32_t)abBuff, &ouAddr2, (uint32_t)outBuff, &ouAddr3, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }

  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr3);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_PARK_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }   
  return HSP_CORE_OK;
}

/**
  * @brief Ipark transform: Transforms stator voltage qVq and qVd, that belong to a rotor flux synchronous rotating
  * frame, to a stationary reference frame, so as to obtain qValpha and qVbeta:
  *                  Valfa = Vq*Cos(theta) + Vd*Sin(theta)
  *                  Vbeta =-Vq*Sin(theta) + Vd*Cos(theta)
  * @param hmw        HSP handle.
  * @param thetaBuff  Input rotating frame angular position in float Buffer
  * @param qdBuff     stator voltage Vq and Vd in float format [q, d] Buffer
  * @param outBuff    Output Stator voltage Valpha and Vbeta [alpha, beta] in float Buffer
  * @param ioType     User iotype information
  * @retval           Core status.
  */
hsp_core_status_t HSP_SEQ_ScaIPark_f32(hsp_core_handle_t *hmw, float32_t *thetaBuff,
                                       hsp_v_q_d_t *qdBuff, hsp_v_alpha_beta_t *outBuff, uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  uint32_t ouAddr3;
  
  /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t)thetaBuff, &ouAddr1,
                               (uint32_t)qdBuff, &ouAddr2, (uint32_t)outBuff, &ouAddr3, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }

  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr3);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_IPARK_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Fast sine of a float value
  * @param hmw        HSP handle.
  * @param inBuff     Input Buffer address
  * @param outBuff    Output Buffer address
  * @param ioType     User iotype information
  * @retval           Core status.
  */
hsp_core_status_t HSP_SEQ_ScaFSin_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t *outBuff,
                                      uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  
  /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t)inBuff, &ouAddr1,
                               (uint32_t)outBuff, &ouAddr2, 0, NULL, 2) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }

  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_FSIN_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Fast cosine of a float value
  * @param hmw        HSP handle.
  * @param inBuff     Input Buffer address
  * @param outBuff    Output Buffer address
  * @param ioType     User iotype information
  * @retval           Core status.
  */
hsp_core_status_t HSP_SEQ_ScaFCos_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t *outBuff,
                                      uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  
  /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t)inBuff, &ouAddr1,
                               (uint32_t)outBuff, &ouAddr2, 0, NULL, 2) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }

  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_FCOS_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }

  return HSP_CORE_OK;
}

/**
  * @brief Performs atan2 of input values x and y
  * @param hmw        HSP handle.
  * @param inABuff    Input A Buffer
  * @param inBBuff    Input B Buffer
  * @param outBuff    Output Buffer address
  * @param ioType     User iotype information
  * @retval           Core status.
  */
hsp_core_status_t HSP_SEQ_ScaAtan2_f32(hsp_core_handle_t *hmw, float32_t *inABuff, float32_t *inBBuff,
                                       float32_t *outBuff, uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  uint32_t ouAddr3;
  
  /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t)inABuff, &ouAddr1,
                               (uint32_t)inBBuff, &ouAddr2, (uint32_t)outBuff, &ouAddr3, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }

  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr3);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_ATAN2_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }  
  return HSP_CORE_OK;
}

/**
  * @brief Performs fast atan2 of input values x and y
  * @param hmw        HSP handle.
  * @param inABuff    Input A Buffer
  * @param inBBuff    Input B Buffer
  * @param outBuff    Output Buffer address
  * @param ioType     User iotype information
  * @retval           Core status.
  */
hsp_core_status_t HSP_SEQ_ScaFAtan2_f32(hsp_core_handle_t *hmw, float32_t *inABuff, float32_t *inBBuff,
                                        float32_t *outBuff, uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  uint32_t ouAddr3;
  
  /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t)inABuff, &ouAddr1,
                               (uint32_t)inBBuff, &ouAddr2, (uint32_t)outBuff, &ouAddr3, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }

  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr3);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_FATAN2_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }  
  return HSP_CORE_OK;
}

/**
  * @brief Add scalaire with immediate value
  * @param hmw        HSP handle.
  * @param inBuff     Input A Buffer
  * @param valueToAdd Immediate value to add
  * @param outBuff    Output Buffer address
  * @param ioType     User iotype information
  * @retval           Core status.
  */
hsp_core_status_t HSP_SEQ_ScaIAdd_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t valueToAdd,
                                      float32_t *outBuff, uint32_t ioType)
{	
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  
  /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, (ioType | HSP_SEQ_IOTYPE_IMM_1), &encoded, (uint32_t)inBuff, &ouAddr1,
                               (uint32_t)&valueToAdd, NULL, (uint32_t)outBuff, &ouAddr2, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }

  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, *((uint32_t *) &valueToAdd));
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_ADD_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Set value for a scalar variable
  * @param hmw        HSP handle.
  * @param valueToSet Immediate value to set
  * @param outBuff    Output Buffer address
  * @param ioType     User iotype information
  * @retval           Core status.
  */
hsp_core_status_t HSP_SEQ_ScaISet_f32(hsp_core_handle_t *hmw, float32_t valueToSet, float32_t *outBuff,
                                      uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  
  /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, (ioType | HSP_SEQ_IOTYPE_IMM_0), &encoded, (uint32_t)&valueToSet, NULL,
                               (uint32_t)outBuff, &ouAddr1, 0, NULL, 2) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, *((uint32_t *) &valueToSet));
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_SET) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}
/**
  * @brief Set value for a integer scalar variable
  * @param hmw        HSP handle.
  * @param valueToSet Input Buffer value
  * @param outBuff    Output Buffer address
  * @param ioType     User iotype information
  * @retval           Core status.
  */
hsp_core_status_t HSP_SEQ_ScaSet_u32(hsp_core_handle_t *hmw, uint32_t *inBuff, uint32_t *outBuff,
                                     uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  
  /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t)inBuff, &ouAddr1,
                               (uint32_t)outBuff, &ouAddr2, 0, NULL, 2) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_SET) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Set value for a scalar integer variable
  * @param hmw        HSP handle.
  * @param valueToSet Immediate value to set
  * @param outBuff    Output Buffer address
  * @param ioType     User iotype information
  * @retval           Core status.
  */
hsp_core_status_t HSP_SEQ_ScaISet_u32(hsp_core_handle_t *hmw, uint32_t valueToSet, uint32_t *outBuff,
                                      uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  
  /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, (ioType | HSP_SEQ_IOTYPE_IMM_0), &encoded, (uint32_t)&valueToSet, NULL,
                               (uint32_t)outBuff, &ouAddr1, 0, NULL, 2) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, valueToSet);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_SET) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Scalar subtraction with immediate value
  * @param hmw        HSP handle.
  * @param inBuff     Input Buffer
  * @param valueToSub Immediate value to sub
  * @param outBuff    Output Buffer address
  * @param ioType     User iotype information
  * @retval           Core status.
  */
hsp_core_status_t HSP_SEQ_ScaISub_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t valueToSub,
                                      float32_t *outBuff, uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  
  /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, (ioType | HSP_SEQ_IOTYPE_IMM_1), &encoded, (uint32_t)inBuff, &ouAddr1,
                               (uint32_t)&valueToSub, NULL, (uint32_t)outBuff, &ouAddr2, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }

  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, *((uint32_t *) &valueToSub));
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_SUB_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Scalar multiplication with immediate value
  * @param hmw         HSP handle.
  * @param inBuff      Input Buffer
  * @param valueToMult Immediate value to multiply
  * @param outBuff     Output Buffer address
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_ScaIMul_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t valueToMult,
                                      float32_t *outBuff, uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  
  /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, (ioType | HSP_SEQ_IOTYPE_IMM_1), &encoded, (uint32_t)inBuff, &ouAddr1,
                               (uint32_t)&valueToMult, NULL, (uint32_t)outBuff, &ouAddr2, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }

  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, *((uint32_t *) &valueToMult));
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_MUL_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }  
  return HSP_CORE_OK;
}


/**
  * @brief Scalar float MAC of 3 scalars
  * @param hmw         HSP handle.
  * @param inABuff     Input A Buffer address
  * @param inBBuff     Input B Buffer address
  * @param outBuff     Output Buffer address
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_ScaMac_f32(hsp_core_handle_t *hmw, float32_t *inABuff, float32_t *inBBuff, float32_t *outBuff,
                                     uint32_t ioType)
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
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
 
  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_MAC_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }  
  return HSP_CORE_OK;  
}

/**
  * @brief Performs pid
  * @param hmw         HSP handle.
  * @param inABuff     Mesured value Buffer
  * @param inBBuff     Set point value Buffer
  * @param cfgBuff     PID config Buffer
  * @param outBuff     Output Buffer address
  * @param satMod      Saturation mode (0 for static, 1 for dynamic)
  * @param satFl       Saturation detected flag index
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_ScaPid_f32(hsp_core_handle_t *hmw, float32_t *inABuff, float32_t *inBBuff,
                                     hsp_pid_buff_cfg_t *cfgBuff, float32_t *outBuff,
                                     uint32_t satMod, uint32_t satfl, uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  uint32_t ouAddr3;
  uint32_t ouAddr4;
  
  /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t)inABuff, &ouAddr1,
                               (uint32_t)inBBuff, &ouAddr2, (uint32_t)cfgBuff, &ouAddr3, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  if (HSP_UTILITIES_ToBramABAddress(hmw, (uint32_t)outBuff, &ouAddr4) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }

  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr3);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, ouAddr4);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM4, (uint32_t)satfl);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM5, (uint32_t)satMod);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_PID_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }  
  return HSP_CORE_OK;
}
 
/**
  * @brief Perform the Q31 to float32 conversion
  * @param hmw         HSP handle.
  * @param inBuff      Input Buffer address
  * @param outBuff     Output Buffer address
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_ScaQ312F(hsp_core_handle_t *hmw, int32_t *inBuff, float32_t *outBuff, uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  
  /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t) inBuff, &ouAddr1,
                               (uint32_t) outBuff, &ouAddr2, 0, NULL, 2) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }

  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_Q312F) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Perform the float32 to Q31 conversion
  * @param hmw         HSP handle.
  * @param inBuff      Input Buffer address
  * @param outBuff     Output Buffer address
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_ScaF2Q31(hsp_core_handle_t *hmw, float32_t *inBuff, int32_t *outBuff, uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  
  /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t) inBuff, &ouAddr1,
                               (uint32_t) outBuff, &ouAddr2, 0, NULL, 2) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }

  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_F2Q31) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Add an immediate value with modulo
  * (*pIn += imm; if (*Pin > (base+size)) *pIn += size;)
  * @param hmw         HSP handle.
  * @param addr        Input Buffer address (uint32_t)
  * @param imm         Immediate value to add (uint32_t)
  * @param base        Base used for modulo (uint32_t)
  * @param size        Size used for modulo (uint32_t)
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_ScaIModAdd_u32(hsp_core_handle_t *hmw, uint32_t *addr, uint32_t imm, uint32_t base,
                                        uint32_t size, uint32_t ioType)
{ 
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t endAddr;
   
  HSP_CHECK_CMD_SIZE_NULL(hmw, size);
  HSP_CHECK_ASSERT(hmw, (size > 1));
   /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, (ioType | HSP_SEQ_IOTYPE_IMM_1), &encoded, (uint32_t)addr, &ouAddr1, 0, NULL, 0, NULL, 1) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  endAddr = base;
  size = size * 4U;
  imm = imm * 4U;
  endAddr += size;
 
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, imm);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, endAddr);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, size);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
 
  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_MADD_U32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }  
  return HSP_CORE_OK;  
}

/**
  * @brief Shift one integer with an immediate scalar value
  * @param hmw         HSP handle.
  * @param inBuff      Input Buffer address 
  * @param imm         Immediate value of shift
  * @param outBuff     Output Buffer address 
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_ScaIShift_i32(hsp_core_handle_t *hmw, int32_t *inBuff, int32_t imm, int32_t *outBuff,
                                        uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
   
   /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, (ioType | HSP_SEQ_IOTYPE_IMM_1), &encoded, (uint32_t)inBuff, &ouAddr1, 0, NULL, (uint32_t)outBuff, &ouAddr2, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
 
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, *((uint32_t *) &imm));
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
 
  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_SHIFT_I32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }  
  return HSP_CORE_OK;  
}

/**
  * @brief Shift one integer with an scalar value
  * @param hmw         HSP handle.
  * @param inBuff      Input Buffer address
  * @param inBuffShift Input Value Buffer address
  * @param outBuff     Output Buffer address
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_ScaShift_i32(hsp_core_handle_t *hmw, int32_t *inBuff, int32_t *inBuffShift, int32_t *outBuff,
                                       uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  uint32_t ouAddr3;
   
   /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t)inBuff, &ouAddr1, (uint32_t)inBuffShift, &ouAddr2, (uint32_t)outBuff, &ouAddr3, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
 
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr3);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
 
  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_SHIFT_I32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }  
  return HSP_CORE_OK;  
}

/**
  * @brief AND one integer with an immediate scalar value
  * @param hmw        HSP handle.
  * @param inBuff      Input Buffer address
  * @param imm         Immediate value to and 
  * @param outBuff     Output Buffer address
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_ScaIAnd_u32(hsp_core_handle_t *hmw, uint32_t *inBuff, uint32_t imm, uint32_t *outBuff,
                                      uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
   
   /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, (ioType | HSP_SEQ_IOTYPE_IMM_1), &encoded, (uint32_t)inBuff, &ouAddr1, 0, NULL, 
                               (uint32_t)outBuff, &ouAddr2, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
 
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, *((uint32_t *) &imm));
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
 
  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_AND_U32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }  
  return HSP_CORE_OK;  
}

/**
  * @brief AND one integer with an scalar value
  * @param hmw        HSP handle.
  * @param inBuff      Input Buffer address
  * @param imm         Input Buffer value to and
  * @param outBuff     Output Buffer address 
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_ScaAnd_u32(hsp_core_handle_t *hmw, uint32_t *inBuff, uint32_t *inBuffAnd, uint32_t *outBuff,
                                     uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  uint32_t ouAddr3;
   
   /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t)inBuff, &ouAddr1, (uint32_t)inBuffAnd, &ouAddr2, 
                               (uint32_t)outBuff, &ouAddr3, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
 
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr3);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
 
  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_AND_U32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }  
  return HSP_CORE_OK;  
}

/**
  * @brief OR one integer with an immediate scalar value
  * @param hmw        HSP handle.
  * @param inBuff      Input Buffer address
  * @param imm         Immediate value to or
  * @param outBuff     Output Buffer address
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_ScaIOr_u32(hsp_core_handle_t *hmw, uint32_t *inBuff, uint32_t imm, uint32_t *outBuff,
                                      uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
   
   /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, (ioType | HSP_SEQ_IOTYPE_IMM_1), &encoded, (uint32_t)inBuff, &ouAddr1, 0, NULL, 
                               (uint32_t)outBuff, &ouAddr2, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
 
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, *((uint32_t *) &imm));
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
 
  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_OR_U32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }  
  return HSP_CORE_OK;  
}

/**
  * @brief OR one integer with an scalar value
  * @param hmw        HSP handle.
  * @param inBuff      Input Buffer address
  * @param imm         Input Buffer value to or
  * @param outBuff     Output Buffer address
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_ScaOr_u32(hsp_core_handle_t *hmw, uint32_t *inBuff, uint32_t *inBuffOr, uint32_t *outBuff,
                                     uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  uint32_t ouAddr3;
   
   /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t)inBuff, &ouAddr1, (uint32_t)inBuffOr, &ouAddr2, 
                               (uint32_t)outBuff, &ouAddr3, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
 
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr3);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
 
  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_OR_U32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }  
  return HSP_CORE_OK;  
}

/**
  * @brief XOR one integer with an immediate scalar value
  * @param hmw        HSP handle.
  * @param inBuff      Input Buffer address
  * @param imm         Immediate value to or
  * @param outBuff     Output Buffer address
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_ScaIXor_u32(hsp_core_handle_t *hmw, uint32_t *inBuff, uint32_t imm, uint32_t *outBuff,
                                      uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
   
   /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, (ioType | HSP_SEQ_IOTYPE_IMM_1), &encoded, (uint32_t)inBuff, &ouAddr1, 0, NULL, 
                               (uint32_t)outBuff, &ouAddr2, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
 
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, *((uint32_t *) &imm));
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
 
  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_XOR_U32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }  
  return HSP_CORE_OK;  
}

/**
  * @brief XOR one integer with an scalar value
  * @param hmw        HSP handle.
  * @param inBuff      Input Buffer address
  * @param imm         Input Buffer value to xor
  * @param outBuff     Output Buffer address
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_ScaXor_u32(hsp_core_handle_t *hmw, uint32_t *inBuff, uint32_t *inBuffXor, uint32_t *outBuff,
                                     uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  uint32_t ouAddr3;
   
   /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t)inBuff, &ouAddr1, (uint32_t)inBuffXor, &ouAddr2, 
                               (uint32_t)outBuff, &ouAddr3, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
 
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr3);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
 
  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_XOR_U32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }  
  return HSP_CORE_OK;  
}

/**
  * @brief NOT one integer
  * @param hmw         HSP handle.
  * @param inBuff      Input Buffer address
  * @param outBuff     Output Buffer address
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_ScaNot_u32(hsp_core_handle_t *hmw, uint32_t *inBuff, uint32_t *outBuff,
                                     uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
   
   /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t)inBuff, &ouAddr1, (uint32_t)outBuff, &ouAddr2, 
                               0, NULL, 2) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
 
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
 
  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_NOT_U32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }  
  return HSP_CORE_OK;  
}

/**
  * @brief Subtract to one integer an immediate scalar value
  * @param hmw         HSP handle.
  * @param inBuff      Input Buffer address
  * @param imm         Input Buffer value to subtract
  * @param outBuff     Output Buffer address
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_ScaISub_i32(hsp_core_handle_t *hmw, int32_t *inBuff, int32_t imm, int32_t *outBuff,
                                     uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
   
   /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, (ioType | HSP_SEQ_IOTYPE_IMM_1), &encoded, (uint32_t)inBuff, &ouAddr1, 0, NULL, 
                               (uint32_t)outBuff, &ouAddr2, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
 
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, *((uint32_t *) &imm));
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
 
  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_SUB_I32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }  
  return HSP_CORE_OK;  
}

/**
  * @brief Subtract to one integer an scalar value
  * @param hmw         HSP handle.
  * @param inBuff      Input Buffer address
  * @param inBuffShift Input Value Buffer address
  * @param outBuff     Output Buffer address
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_ScaSub_i32(hsp_core_handle_t *hmw, int32_t *inBuff, int32_t *inBuffSub, int32_t *outBuff,
                                       uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  uint32_t ouAddr3;
   
   /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t)inBuff, &ouAddr1, (uint32_t)inBuffSub, &ouAddr2, (uint32_t)outBuff, &ouAddr3, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
 
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr3);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
 
  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_SUB_I32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }  
  return HSP_CORE_OK;  
}

/**
  * @brief Multiply to one integer an immediate scalar value
  * @param hmw         HSP handle.
  * @param inBuff      Input Buffer address
  * @param imm         Input Buffer value to multiply
  * @param outBuff     Output Buffer address
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_ScaIMul_i32(hsp_core_handle_t *hmw, int32_t *inBuff, int32_t imm, int32_t *outBuff,
                                      uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
   
   /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, (ioType | HSP_SEQ_IOTYPE_IMM_1), &encoded, (uint32_t)inBuff, &ouAddr1, 0, NULL, 
                               (uint32_t)outBuff, &ouAddr2, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
 
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, *((uint32_t *) &imm));
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
 
  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_MUL_I32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }  
  return HSP_CORE_OK;  
}

/**
  * @brief Multiply to one integer an scalar value
  * @param hmw         HSP handle.
  * @param inBuff      Input Buffer address
  * @param inBuffShift Input Value Buffer address
  * @param outBuff     Output Buffer address
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_ScaMul_i32(hsp_core_handle_t *hmw, int32_t *inBuff, int32_t *inBuffMul, int32_t *outBuff,
                                     uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  uint32_t ouAddr3;
   
   /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t)inBuff, &ouAddr1, (uint32_t)inBuffMul, &ouAddr2, (uint32_t)outBuff, &ouAddr3, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
 
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr3);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
 
  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_MUL_I32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }  
  return HSP_CORE_OK;  
}

/**
  * @brief Add to one integer an immediate scalar value
  * @param hmw         HSP handle.
  * @param inBuff      Input Buffer address
  * @param imm         Input Buffer value to add
  * @param outBuff     Output Buffer address
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_ScaIAdd_i32(hsp_core_handle_t *hmw, int32_t *inBuff, int32_t imm, int32_t *outBuff,
                                      uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
   
   /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, (ioType | HSP_SEQ_IOTYPE_IMM_1), &encoded, (uint32_t)inBuff, &ouAddr1, 0, NULL, 
                               (uint32_t)outBuff, &ouAddr2, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
 
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, *((uint32_t *) &imm));
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
 
  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_ADD_I32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }  
  return HSP_CORE_OK;  
}

/**
  * @brief Add to one integer an scalar value
  * @param hmw         HSP handle.
  * @param inBuff      Input Buffer address
  * @param inBuffShift Input Value Buffer address
  * @param outBuff     Output Buffer address
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_ScaAdd_i32(hsp_core_handle_t *hmw, int32_t *inBuff, int32_t *inBuffAdd, int32_t *outBuff,
                                     uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  uint32_t ouAddr3;
   
   /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t)inBuff, &ouAddr1, (uint32_t)inBuffAdd, &ouAddr2, 
                               (uint32_t)outBuff, &ouAddr3, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
 
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr3);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
 
  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_ADD_I32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }  
  return HSP_CORE_OK;  
}

/**
  * @brief Scalar unsigned increment of a scalar by 1
  * @param hmw         HSP handle.
  * @param inBuff      Input Scalar address
  * @param outBuff     Output Scalar address
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_ScaInc_u32(hsp_core_handle_t *hmw, uint32_t *inBuff, uint32_t *outBuff, uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
   
   /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t)inBuff, &ouAddr1,
                               (uint32_t)outBuff, &ouAddr2, 0, NULL, 2) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
 
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, 1U);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
 
  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_INC_U32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }  
  return HSP_CORE_OK;
}

/**
  * @brief Write a scalar value into a vector in order to build a vector
  * @param hmw         HSP handle.
  * @param pSrc        Input scalar Buffer address
  * @param pIdx        Current index (uint32_t)
  * @param pDst        Output vector Buffer address
  * @param nbSamples   Number of samples to proceed
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_Sca2Vect_f32(hsp_core_handle_t *hmw, float32_t *pSrc, uint32_t *pIdx, float32_t *pDst,
                                      uint32_t nbSamples, uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  uint32_t ouAddr3;
 
  /* Check command size */
  HSP_CHECK_CMD_SIZE_NULL(hhsp, nbSamples);
 
   /* First check and translate if necessary parameter */
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t)pSrc, &ouAddr1, (uint32_t)pIdx, &ouAddr2, 
                               (uint32_t)pDst, &ouAddr3, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  /* Reset current index */
  *pIdx = 0; 
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr3);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, nbSamples);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);
 
  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA2VECT) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }  
  return HSP_CORE_OK;
}

/**
  * @brief Comparision of an input value to two thresholds (LOTH, HITH)
  * @param hmw         HSP handle.
  * @param pSrc        Input buffer address (float)
  * @param pSat        Pointer on saturation limit [loTh, hiTh] (float)
  * @param pDst        Output buffer address (int32_t)
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_ScaSat_f32(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pSat, float32_t *pDst,
                                     uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  uint32_t ouAddr3;

  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t) pSrc, &ouAddr1,
                               (uint32_t) pSat, &ouAddr2, (uint32_t) pDst, &ouAddr3, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr3);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, 1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_SAT_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}


/**
  * @brief Scalar float negate a scalar
  * @param hmw         HSP handle.
  * @param pSrc        Input Scalar address
  * @param pDst        Output Scalar address
  * @param pRes        Res value address (uint32_t, 1 if saturation)
  * @param ioType      User iotype information
  * @retval            Core status.
  */  
hsp_core_status_t HSP_SEQ_ScaNeg_f32(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pDst,
                                     uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ouAddr2;
  uint32_t ouAddr3;
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t) pSrc, &ouAddr1,
                               (uint32_t) pSrc, &ouAddr2, (uint32_t) pDst, &ouAddr3, 3) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, ouAddr3);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SCA_NEG_F32) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Get HSP saturation flag
  * @param hmw         HSP handle.
  * @param pDst        Result flag (uint32_t) address (0x40 for saturation)
  * @param ioType      User iotype information
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_GetSatFlag(hsp_core_handle_t *hmw, uint32_t *pDst, uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr;

  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t) pDst, &ouAddr,
                               (uint32_t) pDst, &ouAddr, 0, NULL, 2) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_GET_SATF) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Clear HSP saturation flag
  * @param hmw         HSP handle.
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_ClrSatFlag(hsp_core_handle_t *hmw)
{
  uint32_t encoded = 0;
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_CLR_SATF) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Counter kernel function to add immediate to global counter index
  * @param hmw         HSP handle.
  * @param cntIdx      Counter index [0,15])
  * @param val         Value to add to counter
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_CounterAdd(hsp_core_handle_t *hmw, uint8_t cntIdx, uint32_t val)
{
  if (val > HAL_HSP_CMD_COUNT_MAX_NB_COUNTIF)
  {
    /* max counter number is reached */
    return HSP_CORE_ERROR;
  }
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, cntIdx);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, val);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, HAL_HSP_CMD_COUNT_MODE_ADD);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, 0);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_COUNT) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Counter kernel function to set immediate to global counter index
  * @param hmw         HSP handle.
  * @param cntIdx      Counter index [0,15])
  * @param val         Value to set to counter
  * @retval            Core status.
  */
hsp_core_status_t HSP_SEQ_CounterSet(hsp_core_handle_t *hmw, uint8_t cntIdx, uint32_t val)
{
  if (val > HAL_HSP_CMD_COUNT_MAX_NB_COUNTIF)
  {
    /* max counter number is reached */
    return HSP_CORE_ERROR;
  }
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, cntIdx);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, val);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, HAL_HSP_CMD_COUNT_MODE_SET);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, 0);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_COUNT) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Add function to send event to trig a processing list
  * @param hhsp          HSP handle.
  * @param listNumber    Processing list number
  * @param itf           Event generated by HDEG (0) or HSEG (1) interface
  * @retval              Core status.
  */
hsp_core_status_t HSP_SEQ_SendEvt(hsp_core_handle_t *hmw, uint32_t listNumber, uint32_t itf)
{
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, listNumber);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, itf);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, 0);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SEND_EVT) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief  Add function to set flag in processing list
  * @param hhsp         HSP handle.
  * @param flag         Flag index
  * @retval             Core status.
  */
hsp_core_status_t HSP_SEQ_SetFlag(hsp_core_handle_t *hmw, uint32_t flag)
{
  if (flag > 31)
  {
    return HSP_CORE_ERROR;
  }
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, flag);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, 0);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SET_FLAG) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }

  return HSP_CORE_OK;
}

/**
  * @brief Add function to generate a trigger pulse on one of the hsp_trg_out[3:0] signals by simply
  * writing into the HSP_HYP_TRGOR register
  * @param hhsp         HSP handle.
  * @param val          Integer value to set (must be in [1,15])
  * @retval             Core status.
  */
hsp_core_status_t HSP_SEQ_SetTrgo(hsp_core_handle_t *hmw, uint32_t val)
{
  if ((val == 0) || (val > 15))
  {
    return HSP_CORE_ERROR;
  }
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, val);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, 0);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SET_TRGO) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }

  return HSP_CORE_OK;
}

/**
  * @brief The SET_GPO function set the specified bitfield of a GPO
  * @param hhsp         HSP handle.
  * @param fieldMark    Mask indicating which bits must be checked
  * @param fieldVal     Field value
  * @retval             Core status.
  */
hsp_core_status_t HSP_SEQ_SetGpo(hsp_core_handle_t *hmw, uint32_t fieldMask, uint32_t fieldVal)
{
  fieldVal = ((fieldVal & fieldMask) << 16); // Put on the GPO field place
  fieldMask = fieldMask << 16; // Put on the GPO field place

  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, fieldMask);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, fieldVal);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, 0);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SET_GPO) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief The SET_BITS function set the specified bitfield of a scalar to the specified value
  * @param hhsp         HSP handle.
  * @param regBuff      Register Buffer address
  * @param fieldMark    Mask indicating which bits must be checked
  * @param fieldVal     Field value
  * @retval             Core status.
  */
hsp_core_status_t HSP_SEQ_SetBits(hsp_core_handle_t *hmw, uint32_t *regBuff, uint32_t fieldMask, uint32_t fieldVal)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ioType = 0;
  fieldVal &= fieldMask;

  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t) regBuff, &ouAddr1,
                               0, NULL, 0, NULL, 1) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, fieldMask);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, fieldVal);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_SET_BITS) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief The WAIT_COND function checks the content of a bit field to a reference value and waits
  * @brief until the condition is false or true. A timeout value forces the function to exit the loop
  * @param hhsp         HSP handle.
  * @param regBuff      Register Buffer address
  * @param fieldMark    Mask indicating which bits must be checked
  * @param fieldVal     Field value
  * @param diffCond     Define the flag position to be set in case of timout (0 for equal, 1 for diff)
  * @param timeout      Timeout value
  * @param timeoutFl    Timeout index
  * @retval             Core status.
  */
hsp_core_status_t HSP_SEQ_WaitCond(hsp_core_handle_t *hmw, uint32_t *regBuff, uint32_t fieldMask,
                                   uint32_t fieldVal, uint32_t diffCond, uint32_t timeout, uint32_t timeoutFl)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1;
  uint32_t ioType = 0;
  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t) regBuff, &ouAddr1,
                               0, NULL, 0, NULL, 1) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, fieldMask);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, fieldVal);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, timeoutFl);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM4, diffCond);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM5, timeout);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_WAIT_COND) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief CRC32 computation
  * @param hhsp           HSP handle.
  * @param pState         Pointer of memory where pState[0] = CRC pState[1] = offset
  * @param pCRC           Pointer on crc computed
  * @param blockSize      Size of block
  * @param posFEOR        Position of the flag indicating when CRC function reaches the end of the ROM.
  * @param posFEOB        Position of the flag indicating when CRC function reaches the end of data block.
  * @param memType        HSP_CRC_CROM: to run CRC on CROM,
  *                       HSP_CRC_DROM: to run CRC on DROM
  * @param ioType         User iotype information
  * @retval               Core status.
  */
hsp_core_status_t HSP_SEQ_Crc32(hsp_core_handle_t *hmw, uint32_t *pState,  uint32_t *pCRC,  uint32_t blockSize,
                                uint32_t posFEOR, uint32_t posFEOB, hsp_crc_mem_type_cmd_t memType, uint32_t ioType)
{
  uint32_t encoded = 0;
  uint32_t ouAddr1, ouAddr2;

  if (HSP_UTILITIES_BuildParam(hmw, ioType, &encoded, (uint32_t) pState, &ouAddr1,
                               (uint32_t) pCRC, &ouAddr2, 0, NULL, 2) == HSP_IF_ERROR)
  {
    return HSP_CORE_ERROR;
  }
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM0, ouAddr1);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM1, ouAddr2);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM2, blockSize);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM3, memType);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM4, posFEOR);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM5, posFEOB);
  HSP_HW_IF_WriteParameter(hmw->hdriver, HSP_HW_IF_PARAM15, encoded);

  if (HSP_HW_IF_SendCommand(hmw->hdriver, HSP_CMD_CRC32) != HSP_IF_OK)
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
