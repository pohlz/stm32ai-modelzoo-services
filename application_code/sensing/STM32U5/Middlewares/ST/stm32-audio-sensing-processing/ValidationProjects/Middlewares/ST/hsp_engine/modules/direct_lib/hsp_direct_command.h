/**
  ******************************************************************************
  * @file    hsp_direct_command.h
  * @author  MCD Application Team
  * @brief   Header file for hsp_direct_command.c
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
#ifndef HSP_DIRECT_COMMAND_H
#define HSP_DIRECT_COMMAND_H

#ifdef __cplusplus
extern "C" {
#endif



/* Includes ------------------------------------------------------------------*/
#include "hsp_conf.h"
#include "hsp_api_def.h"
#include "hsp_hw_if.h"
#include "hsp_fw_def_generic.h"


/** @addtogroup HSP_ENGINE
  * @{
  */

/** @addtogroup HSP_MODULES
  * @{
  */

/** @defgroup HSP_MODULES_DIRECT_COMMAND HSP Modules Direct Command
  * @{
  */

/** @defgroup HSP_MODULES_DIRECT_COMMAND_Private_Macros HSP Modules Direct Private Macros
  * @{
  */
#if defined(USE_HSP_ACC_DIRECT_COMMAND_SYNCHRONOUS)
#define HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw) HSP_HW_IF_WaitEndOfDirectCommand(((hmw)->hdriver));
#else
#define HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw)
#endif /* USE_HSP_ACC_DIRECT_COMMAND_SYNCHRONOUS */
/**
  * @}
  */

/**
  * @defgroup HSP_MODULES_DIRECT_COMMAND_Private_Functions HSP Private Functions
  * @{
  */
void HSP_ACC_Rfft_WriteParam(hsp_core_handle_t *hmw, float32_t *buff, uint32_t Log2nbp, uint8_t ifftFlag,
                             uint8_t bitrev, uint8_t fftVariant);
void HSP_ACC_Fft_WriteParam(hsp_core_handle_t *hmw, float32_t *buff, uint32_t Log2nbp, uint8_t ifftFlag,
                            uint8_t bitrev);
void HSP_ACC_Dct_WriteParam(hsp_core_handle_t *hmw, float32_t *buff, uint32_t log2Nbp);
void HSP_ACC_IDct_WriteParam(hsp_core_handle_t *hmw, float32_t *buff, uint32_t log2Nbp);
void HSP_ACC_VectIOS_WriteParam(hsp_core_handle_t *hmw, uint32_t inBuff, uint32_t outBuff,
                                uint32_t nbSamples);
void HSP_ACC_VectIIOS_WriteParam(hsp_core_handle_t *hmw, uint32_t inABuff, uint32_t inBBuff,
                                 uint32_t outBuff, uint32_t nbSamples);
void HSP_ACC_VectIIVOS_WriteParam(hsp_core_handle_t *hmw, uint32_t inBuff, uint32_t uValue, uint32_t outBuff,
                                  uint32_t nbSamples);
void HSP_ACC_VectIIOSS_WriteParam(hsp_core_handle_t *hmw, uint32_t inABuff, uint32_t inBBuff,
                                  uint32_t outBuff, uint32_t nbSamplesA, uint32_t nbSamplesB);
void HSP_ACC_VectIFVOS_WriteParam(hsp_core_handle_t *hmw, uint32_t inBuff, float32_t fValue, 
                                  uint32_t outBuff, uint32_t nbSamples);
void HSP_ACC_VectIIOOS_WriteParam(hsp_core_handle_t *hmw, uint32_t inABuff, uint32_t inBBuff,
                                 uint32_t outBuffA, uint32_t outBuffB, uint32_t nbSamples);								  
void HSP_ACC_MatInv_WriteParam(hsp_core_handle_t *hmw, uint32_t inBuff, uint32_t outBuff, uint32_t nRows,
                               uint32_t nCols);
void HSP_ACC_MatMult_WriteParam(hsp_core_handle_t *hmw, uint32_t inABuff, uint32_t inBBuff,
                                uint32_t outBuff, uint32_t nRowsA, uint32_t nColsA, uint32_t nRowsB,
                                uint32_t nColsB);
void HSP_ACC_MatTrans_WriteParam(hsp_core_handle_t *hmw, uint32_t inBuff, uint32_t outBuff, uint32_t nRows,
                                 uint32_t nCols);
void HSP_ACC_Filter_WriteParam(hsp_core_handle_t *hmw, uint32_t inBuff, uint32_t coefBBuff, 
                               hsp_filter_state_identifier_t stateId, uint32_t outBuff, uint32_t nbSamples);
void HSP_ACC_FirDecimate_WriteParam(hsp_core_handle_t *hmw, uint32_t inBuff, uint32_t coefBBuff,
                                    hsp_filter_state_identifier_t stateId, uint32_t outBuff, 
                                    uint32_t nbSamples, uint32_t decimFactor);
void HSP_ACC_Conv_WriteParam(hsp_core_handle_t *hmw, uint32_t inABuff, uint32_t inBBuff,
                             uint32_t outBuff, uint32_t sizeA, uint32_t sizeB);
void HSP_ACC_FltBank_WriteParam(hsp_core_handle_t *hmw, uint32_t spectrCol, uint32_t startIdx,
                                uint32_t idxSize, uint32_t coef, uint32_t fltbankCol,
                                uint32_t nFltbanks);
void HSP_ACC_FltBankExtC_WriteParam(hsp_core_handle_t *hmw, uint32_t spectrCol, uint32_t startIdx,
                                    uint32_t idxSize, uint32_t coef, uint32_t fltbankCol, 
                                    uint32_t nFltbanks, uint32_t dmaAdd, uint32_t dmaSize);
void HSP_ACC_Lms_WriteParam(hsp_core_handle_t *hmw, uint32_t inBuff, uint32_t coefBBuff, 
                            hsp_filter_state_identifier_t stateId, uint32_t outBuff, uint32_t refBuff,
                            uint32_t errBuff, uint32_t nbSamples, float32_t mu);
void HSP_ACC_IirLattice_WriteParam(hsp_core_handle_t *hmw, uint32_t inBuff, uint32_t skCoeff,
                                   uint32_t svCoeff, hsp_filter_state_identifier_t stateId, 
                                   uint32_t outBuff, uint32_t nbSamples);
void HSP_ACC_StateBuffer_WriteParam(hsp_core_handle_t *hmw, hsp_filter_state_identifier_t stateId,
                                    uint32_t mulSize, uint32_t addSize, uint32_t buffer);
void HSP_ACC_FirDecimateStateBuffer_WriteParam(hsp_core_handle_t *hmw, 
                                               hsp_filter_state_identifier_t stateId, uint32_t buffer);
void HSP_ACC_Crc32_WriteParam(hsp_core_handle_t *hmw, uint32_t pState, uint32_t pCRC, uint32_t blockSize,
                              uint32_t posFEOR, uint32_t posFEOB, uint32_t memType);

#if defined(__HSP_DMA__)
void HSP_ACC_BackgroundExt2Int_WriteParam(hsp_core_handle_t *hmw, uint32_t inAddr, uint32_t ouAddr,
                                          uint32_t nbElem, uint32_t eltFormat, uint32_t pingpong, 
                                          uint32_t dmaChan, uint32_t perIdx, uint32_t pingPl, uint32_t pongPl);
void HSP_ACC_BackgroundInt2Ext_WriteParam(hsp_core_handle_t *hmw, uint32_t inAddr, uint32_t ouAddr,
                                          uint32_t nbElem, uint32_t eltFormat, uint32_t pingpong,
                                          uint32_t dmaChan, uint32_t perIdx, uint32_t pingPl, uint32_t pongPl);
void HSP_ACC_BackgroundSuspendResumeExt2Int_WriteParam(hsp_core_handle_t *hmw, uint32_t dmaChan,
                                                       uint32_t flag);
#endif /* __HSP_DMA__ */
void HSP_ACC_Counter_WriteParam(hsp_core_handle_t *hmw, uint32_t cntIdx, uint32_t val, uint32_t type);

/**
  * @}
  */
  
/**
  * @defgroup HSP_MODULES_DIRECT_COMMAND_Exported_Macros HSP Modules Direct Command Macros
  * @{
  */

/**
  * @defgroup HSP_MODULES_DIRECT_LIB_Exported_Macros_Transform HSP Modules Direct Transform Command Macros
  * @{
  */
/**
  * @brief Direct RFFT (real) transform function
  * @param hmw        MW handle.
  * @param buff       Input and output Buffer
  * @param log2Nbp    log2(number of FFT point)
  * @param ifftFlag   Inverse FFT flag
  * @param bitrev     Bit reverse flag
  * @param fftVariant Type of FFT: RFFT_TYPE_1, RFFT_TYPE_2, RFFT_TYPE_3
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_Rfft_f32(hsp_core_handle_t *hmw, float32_t *buff, hsp_ftt_lognbp_cmd_t log2Nbp,
                                           uint8_t ifftFlag, uint8_t bitrev, hsp_type_rfft_cmd_t fftVariant)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_RFFT_F32);
  HSP_ACC_Rfft_WriteParam(hmw, buff, (uint32_t)log2Nbp, ifftFlag, bitrev, (uint32_t)fftVariant);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
}


/**
  * @brief Direct FFT (complex) transform function
  * @param hmw        MW handle.
  * @param buff       Input and output Buffer
  * @param log2Nbp    log2(number of FFT point)
  * @param ifftFlag   Inverse FFT flag
  * @param bitrev     Bit reverse flag
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_Fft_f32(hsp_core_handle_t *hmw, float32_t *buff, hsp_ftt_lognbp_cmd_t log2Nbp,
                                          uint8_t ifftFlag, uint8_t bitrev)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_FFT_F32);
  HSP_ACC_Fft_WriteParam(hmw, buff, (uint32_t)log2Nbp, ifftFlag, bitrev);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);  
}

/**
  * @brief Direct DCT transform function
  * @param hmw      MW handle.
  * @param buff     Input and output Buffer
  * @param log2Nbp  log2(number of FFT point)
  * @retval         None
  */
__STATIC_FORCEINLINE void HSP_ACC_Dct_f32(hsp_core_handle_t *hmw, float32_t *buff, hsp_ftt_lognbp_cmd_t log2Nbp)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_DCT_F32);
  HSP_ACC_Dct_WriteParam(hmw, buff, (uint32_t)log2Nbp);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);  
}

/**
  * @brief Direct inverse DCT transform function
  * @param hmw      MW handle.
  * @param buff     Input and output Buffer
  * @param log2Nbp  log2(number of FFT point)
  * @retval         None
  */
__STATIC_FORCEINLINE void HSP_ACC_IDct_f32(hsp_core_handle_t *hmw, float32_t *buff, hsp_ftt_lognbp_cmd_t log2Nbp)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_IDCT_F32);
  HSP_ACC_IDct_WriteParam(hmw, buff, (uint32_t)log2Nbp);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
}

/**
  * @}
  */

/**
  * @defgroup HSP_MODULES_DIRECT_LIB_Exported_Macros_Vector HSP Modules Direct Transform Command Macros
  * @{
  */
/**
  * @brief Direct function to perform element-wise absolute value of a vector
  * @param hmw        MW handle.
  * @param inBuff     Input Buffer address
  * @param outBuff    Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_VectAbs_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t *outBuff,
                                              uint32_t nbSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_ABS_F32);
  HSP_ACC_VectIOS_WriteParam(hmw, (uint32_t)inBuff, (uint32_t)outBuff, nbSamples);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);  
}

/**
  * @brief Direct function to perform element-wise addition of a vector
  * @param hmw        MW handle.
  * @param inABuff    Input A Buffer address
  * @param inBBuff    Input B Buffer address
  * @param outBuff    Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_VectAdd_f32(hsp_core_handle_t *hmw, float32_t *inABuff, float32_t *inBBuff,
                                              float32_t *outBuff, uint32_t nbSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_ADD_F32);
  HSP_ACC_VectIIOS_WriteParam(hmw, (uint32_t)inABuff, (uint32_t)inBBuff, (uint32_t)outBuff, nbSamples);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);   
}

/**
  * @brief Direct function to compute average value of a vector
  * @param hmw        MW handle.
  * @param inBuffId   Input Buffer address
  * @param outBuff    Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_VectAvg_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t *outBuff,
                                              uint32_t nbSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_AVG_F32);
  HSP_ACC_VectIOS_WriteParam(hmw, (uint32_t)inBuff, (uint32_t)outBuff, nbSamples);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);  
}

/**
  * @brief Direct function to perform copy of a float vector
  * @param hmw        MW handle.
  * @param inBuff     Input Buffer address
  * @param outBuff    Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_VectCopy_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t *outBuff,
                                               uint32_t nbSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_COPY);
  HSP_ACC_VectIOS_WriteParam(hmw, (uint32_t)inBuff, (uint32_t)outBuff, nbSamples);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);  
}

/**
  * @brief Direct function to perform copy of a 32-bit int vector
  * @param hmw        MW handle.
  * @param inBuff     Input Buffer address
  * @param outBuff    Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_VectCopy_u32(hsp_core_handle_t *hmw, uint32_t *inBuff, uint32_t *outBuff,
                                               uint32_t nbSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_COPY);
  HSP_ACC_VectIOS_WriteParam(hmw, (uint32_t)inBuff, (uint32_t)outBuff, nbSamples);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);  
}

/**
  * @brief Direct function to perform element-wise cosine of a vector
  * @param hmw        MW handle.
  * @param inBuff     Input Buffer
  * @param outBuff    Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_VectCos_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t *outBuff,
                                              uint32_t nbSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_COS_F32);
  HSP_ACC_VectIOS_WriteParam(hmw, (uint32_t)inBuff, (uint32_t)outBuff, nbSamples);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw); 
}

/**
  * @brief Direct function to perform element-wise division of a vector
  * @param hmw        MW handle.
  * @param inABuff    Input A Buffer address
  * @param inBBuff    Input B Buffer address
  * @param outBuff    Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_VectDiv_f32(hsp_core_handle_t *hmw, float32_t *inABuff, float32_t *inBBuff,
                                              float32_t *outBuff, uint32_t nbSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_DIV_F32);
  HSP_ACC_VectIIOS_WriteParam(hmw, (uint32_t)inABuff, (uint32_t)inBBuff, (uint32_t)outBuff, nbSamples);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);    
}

/**
  * @brief Direct function to perform dot product of a vector
  * @param hmw        MW handle.
  * @param inABuff    Input A Buffer address
  * @param inBBuff    Input B Buffer address
  * @param outBuff    Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_VectDotProd_f32(hsp_core_handle_t *hmw, float32_t *inABuff,
                                                  float32_t *inBBuff, float32_t *outBuff, uint32_t nbSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_DOTPROD_F32);
  HSP_ACC_VectIIOS_WriteParam(hmw, (uint32_t)inABuff, (uint32_t)inBBuff, (uint32_t)outBuff, nbSamples);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);  
}

/**
  * @brief Direct function to search for value and position of the absolute biggest element of a vector
  * @param hmw        MW handle.
  * @param inBuff     Input Buffer address
  * @param outVal     Output max_value address
  * @param outIdx     Output max_pos address
  * @param nbSamples  Number of float elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_VectAbsMax_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t *outVal,
                                                 uint32_t *outIdx, uint32_t nbSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_ABSMAX_F32);
  HSP_ACC_VectIIOS_WriteParam(hmw, (uint32_t)inBuff, (uint32_t)outVal, (uint32_t)outIdx, nbSamples);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
}

/**
  * @brief Direct function to search for value and position of the biggest element of a vector
  * @param hmw        MW handle.
  * @param inBuff     Input Buffer address
  * @param outVal     Output max_value address
  * @param outIdx     Output max_pos address
  * @param nbSamples  Number of float elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_VectMax_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t *outVal,
                                              uint32_t *outIdx, uint32_t nbSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_MAX_F32);
  HSP_ACC_VectIIOS_WriteParam(hmw, (uint32_t)inBuff, (uint32_t)outVal, (uint32_t)outIdx, nbSamples);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
}

/**
  * @brief Direct function to search for value and position of the biggest element of a vector
  * @param hmw        MW handle.
  * @param inBuff     Input Buffer address
  * @param outVal     Output min_value address
  * @param outIdx     Output min_pos address
  * @param nbSamples  Number of float elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_VectMin_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t *outVal,
                                              uint32_t *outIdx, uint32_t nbSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_MIN_F32);
  HSP_ACC_VectIIOS_WriteParam(hmw, (uint32_t)inBuff, (uint32_t)outVal, (uint32_t)outIdx, nbSamples);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
}

/**
  * @brief Direct function to perform element-wise multiplication of a vector
  * @param hmw        MW handle.
  * @param inABuff    Input A Buffer address
  * @param inBBuff    Input B Buffer address
  * @param outBuff    Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_VectMul_f32(hsp_core_handle_t *hmw, float32_t *inABuff, float32_t *inBBuff,
                                              float32_t *outBuff, uint32_t nbSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_MUL_F32);
  HSP_ACC_VectIIOS_WriteParam(hmw, (uint32_t)inABuff, (uint32_t)inBBuff, (uint32_t)outBuff, nbSamples);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
}

/**
  * @brief Direct function to add to each vector element a scalar
  * @param hmw        MW handle.
  * @param inABuff    Input A Buffer address
  * @param inBBuff    Offset to add Buffer address
  * @param outBuff    Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_VectOffset_f32(hsp_core_handle_t *hmw, float32_t *inABuff, float32_t *inBBuff,
                                                 float32_t *outBuff, uint32_t nbSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_OFFSET_F32);
  HSP_ACC_VectIIOS_WriteParam(hmw, (uint32_t)inABuff, (uint32_t)inBBuff, (uint32_t)outBuff, nbSamples);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);  
}

/**
  * @brief Direct function to compute the RMS (Root Mean Square) value of a vector
  * @param hmw        MW handle.
  * @param inBuff     Input Buffer
  * @param outBuff    Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_VectRms_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t *outBuff,
                                              uint32_t nbSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_RMS_F32);
  HSP_ACC_VectIOS_WriteParam(hmw, (uint32_t)inBuff, (uint32_t)outBuff, nbSamples);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);  
}

/**
  * @brief Direct function to multiply each vector element by a scalar
  * @param hmw        MW handle.
  * @param inABuff    Input A Buffer
  * @param inBBuff    Scale Buffer
  * @param outBuff    Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_VectScale_f32(hsp_core_handle_t *hmw, float32_t *inABuff, float32_t *inBBuff,
                                                float32_t *outBuff, uint32_t nbSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_SCALE_F32);
  HSP_ACC_VectIIOS_WriteParam(hmw, (uint32_t)inABuff, (uint32_t)inBBuff, (uint32_t)outBuff, nbSamples);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);    
}

/**
  * @brief Direct function to set value for whole vector
  * @param hmw        MW handle.
  * @param inBuff     Set Buffer address
  * @param outBuff    Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_VectSet_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t *outBuff,
                                              uint32_t nbSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_SET);
  HSP_ACC_VectIOS_WriteParam(hmw, (uint32_t)inBuff, (uint32_t)outBuff, nbSamples);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);   
}

/**
  * @brief Direct function to perform element-wise square-root of a vector
  * @param hmw        MW handle.
  * @param inBuff     Input Buffer address
  * @param outBuff    Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_VectSqrt_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t *outBuff,
                                               uint32_t nbSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_SQRT_F32);
  HSP_ACC_VectIOS_WriteParam(hmw, (uint32_t)inBuff, (uint32_t)outBuff, nbSamples);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);     
}

/**
  * @brief Direct function to perform element-wise sine of a vector
  * @param hmw        MW handle.
  * @param inBuff     Input Buffer address
  * @param outBuff    Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_VectSin_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t *outBuff,
                                              uint32_t nbSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_SIN_F32);
  HSP_ACC_VectIOS_WriteParam(hmw, (uint32_t)inBuff, (uint32_t)outBuff, nbSamples);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);    
}

/**
  * @brief Direct function to perform element-wise subtraction of a vector
  * @param hmw        MW handle.
  * @param inABuff    Input A Buffer address
  * @param inBBuff    Input B Buffer address
  * @param outBuff    Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_VectSub_f32(hsp_core_handle_t *hmw, float32_t *inABuff, float32_t *inBBuff,
                                              float32_t *outBuff, uint32_t nbSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_SUB_F32);
  HSP_ACC_VectIIOS_WriteParam(hmw, (uint32_t)inABuff, (uint32_t)inBBuff, (uint32_t)outBuff, nbSamples);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);  
}

/**
  * @brief Direct function to perform element-wise sine and cosine of each element of vector
  * @param hmw        MW handle.
  * @param inBuff     Input Buffer address
  * @param outBuff    Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_VectSinCos_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t *outBuff,
                                                 uint32_t nbSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_SINCOS_F32);
  HSP_ACC_VectIOS_WriteParam(hmw, (uint32_t)inBuff, (uint32_t)outBuff, nbSamples);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);  
}

/**
  * @brief Direct function to perform element-wise the float32 to integer 32bits conversion
  * @param hmw        MW handle.
  * @param inBuff     Input Buffer address
  * @param outBuff    Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_VectF2I(hsp_core_handle_t *hmw, float32_t *inBuff, int32_t *outBuff,
                                          uint32_t nbSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_F2I);
  HSP_ACC_VectIOS_WriteParam(hmw, (uint32_t)inBuff, (uint32_t)outBuff, nbSamples);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);    
}

/**
  * @brief Direct function to perform element-wise the integer 32bits to float32 conversion
  * @param hmw        MW handle.
  * @param inBuff     Input Buffer address
  * @param outBuff    Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_VectI2F(hsp_core_handle_t *hmw, int32_t *inBuff, float32_t *outBuff,
                                          uint32_t nbSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_I2F);
  HSP_ACC_VectIOS_WriteParam(hmw, (uint32_t)inBuff, (uint32_t)outBuff, nbSamples);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);  
}

/**
  * @brief Direct function to perform element-wise the float32 to unsigned integer 32bits conversion
  * @param hmw        MW handle.
  * @param inBuff     Input Buffer address
  * @param outBuff    Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_VectF2U(hsp_core_handle_t *hmw, float32_t *inBuff, uint32_t *outBuff,
                                          uint32_t nbSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_F2U);
  HSP_ACC_VectIOS_WriteParam(hmw, (uint32_t)inBuff, (uint32_t)outBuff, nbSamples);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
}

/**
  * @brief Direct function to perform element-wise the unsigned integer 32bits to float32 conversion
  * @param hmw        MW handle.
  * @param inBuff     Input Buffer address
  * @param outBuff    Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_VectU2F(hsp_core_handle_t *hmw, uint32_t *inBuff, float32_t *outBuff,
                                          uint32_t nbSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_U2F);
  HSP_ACC_VectIOS_WriteParam(hmw, (uint32_t)inBuff, (uint32_t)outBuff, nbSamples);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
}

/**
  * @brief Direct function to perform element-wise the signed integer 24bits to float32 conversion
  * @param hmw        MW handle.
  * @param inBuff     Input Buffer address
  * @param outBuff    Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_Vect24S2F(hsp_core_handle_t *hmw, int32_t *inBuff, float32_t *outBuff,
                                            uint32_t nbSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_24S2F);
  HSP_ACC_VectIOS_WriteParam(hmw, (uint32_t)inBuff, (uint32_t)outBuff, nbSamples);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
}

/**
  * @brief Direct function to perform the decimation of a vector
  * @param hmw      MW handle.
  * @param inBuff   Input Buffer address
  * @param decim    Decimator factor
  * @param outBuff  Output Buffer address
  * @param sizeOu   Output vector size
  * @retval         None
  */
__STATIC_FORCEINLINE void HSP_ACC_VectDecim_f32(hsp_core_handle_t *hmw, float32_t *inBuff, uint8_t decim,
                                                float32_t *outBuff, uint32_t sizeOu)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_DECIM);
  HSP_ACC_VectIIVOS_WriteParam(hmw, (uint32_t)inBuff, (uint32_t)decim, (uint32_t)outBuff, sizeOu);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
}

/**
  * @brief Direct function to perform the interpolation of a real vector by inserting N-1 zeros between 2 samples
  * @param hmw      MW handle.
  * @param inBuff   Input Buffer address
  * @param interp   Interpolation factor
  * @param outBuff  Output Buffer address
  * @param sizeIn   Input vector size
  * @retval         None
  */
__STATIC_FORCEINLINE void HSP_ACC_VectZIns_f32(hsp_core_handle_t *hmw, float32_t *inBuff, uint8_t interp,
                                               float32_t *outBuff, uint32_t sizeIn)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_ZINS);
  HSP_ACC_VectIIVOS_WriteParam(hmw, (uint32_t)inBuff, (uint32_t)interp, (uint32_t)outBuff, sizeIn);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
}

/**
  * @brief Direct function to perform atan2 of input vector x and y
  * @param hmw        MW handle.
  * @param inABuff    Input A Buffer address
  * @param inBBuff    Input B Buffer address
  * @param outBuff    Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_VectAtan2_f32(hsp_core_handle_t *hmw, float32_t *inABuff, float32_t *inBBuff,
                                                float32_t *outBuff, uint32_t nbSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_ATAN2_F32);
  HSP_ACC_VectIIOS_WriteParam(hmw, (uint32_t)inABuff, (uint32_t)inBBuff, (uint32_t)outBuff, nbSamples);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw); 
}

/**
  * @brief Direct function to perform natural logarithm of each element of input vector x
  * @param hmw        MW handle.
  * @param inBuff     Input Buffer address
  * @param outBuff    Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_VectLn_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t *outBuff,
                                             uint32_t nbSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_LN_F32);
  HSP_ACC_VectIOS_WriteParam(hmw, (uint32_t)inBuff, (uint32_t)outBuff, nbSamples);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);  
}

/**
  * @brief Direct function to perform common logarithm (base 10) of each element of input vector x
  * @param hmw        MW handle.
  * @param inBuff     Input Buffer address
  * @param outBuff    Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_VectLog10_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t *outBuff,
                                                uint32_t nbSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_LOG10_F32);
  HSP_ACC_VectIOS_WriteParam(hmw, (uint32_t)inBuff, (uint32_t)outBuff, nbSamples);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw); 
}

/**
  * @brief Direct function to perform exponential of each element of input vector x
  * @param hmw        MW handle.
  * @param inBuff     Input Buffer address
  * @param outBuff    Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_VectExp_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t *outBuff,
                                              uint32_t nbSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_EXP_F32);
  HSP_ACC_VectIOS_WriteParam(hmw, (uint32_t)inBuff, (uint32_t)outBuff, nbSamples);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);  
}

/**
  * @brief Direct function to perform powers of ten (exp base 10) of each element of input vector x
  * @param hmw        MW handle.
  * @param inBuff     Input Buffer address
  * @param outBuff    Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_VectExp10_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t *outBuff,
                                                uint32_t nbSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_EXP10_F32);
  HSP_ACC_VectIOS_WriteParam(hmw, (uint32_t)inBuff, (uint32_t)outBuff, nbSamples);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);  
}

/**
  * @brief Direct function to compute the multiplication of one vector and cos ROM
  * and an exponential complex signal
  * @param hmw        MW handle.
  * @param inBuff     Input Buffer address
  * @param startBuff  Start buffer address (I/O: input is start index and output is nextIdx)
  * @param outBuff    Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @param step       Step value between 2 cos number in ROM
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_VectMulCos_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t *startBuff,
                                                 float32_t *outBuff, uint32_t nbSamples, uint32_t step)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_MUL_COS_F32);
  HSP_ACC_VectIIOSS_WriteParam(hmw, (uint32_t)inBuff, (uint32_t)startBuff, (uint32_t)outBuff, nbSamples, step);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
}

/**
  * @brief Direct function to compute the multiplication of one vector and sin ROM
  * and an exponential complex signal
  * @param hmw        MW handle.
  * @param inBuff     Input Buffer address
  * @param startBuff  Start buffer address (I/O: input is start index and output is nextIdx)
  * @param outBuff    Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @param step       Step value between 2 sin number in ROM
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_VectMulSin_f32(hsp_core_handle_t *hmw, float32_t *inBuff,
                                                 float32_t *startBuff, float32_t *outBuff, uint32_t nbSamples,
                                                 uint32_t step)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_MUL_SIN_F32);
  HSP_ACC_VectIIOSS_WriteParam(hmw, (uint32_t)inBuff, (uint32_t)startBuff, (uint32_t)outBuff, nbSamples, step);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);   
}

/**
  * @brief Direct function to add to each vector element a immediate scalar value
  * @param hmw        MW handle.
  * @param inBuff     Input Buffer address
  * @param offset     Immediate value offset to add Buffer address
  * @param outBuff    Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_VectIOffset_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t offset,
                                                  float32_t *outBuff, uint32_t nbSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_OFFSET_I_F32);
  HSP_ACC_VectIFVOS_WriteParam(hmw, (uint32_t)inBuff, offset, (uint32_t)outBuff, nbSamples);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw); 
}

/**
  * @brief Direct function to multiply each vector element by a immediate scalar value
  * @param hmw        MW handle.
  * @param inBuff     Input A Buffer address
  * @param scale      Scale immediate scalar value
  * @param outBuff    Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_VectIScale_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t scale,
                                                 float32_t *outBuff, uint32_t nbSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_SCALE_I_F32);
  HSP_ACC_VectIFVOS_WriteParam(hmw, (uint32_t)inBuff, scale, (uint32_t)outBuff, nbSamples);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);   
}

/**
  * @brief Direct function to perform the Q31 to float32 conversion
  * @param hmw        MW handle.
  * @param inBuff     Input Buffer address
  * @param outBuff    Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_VectQ312F(hsp_core_handle_t *hmw, int32_t *inBuff, float32_t *outBuff,
                                            uint32_t nbSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_Q312F);
  HSP_ACC_VectIOS_WriteParam(hmw, (uint32_t)inBuff, (uint32_t)outBuff, nbSamples);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);  
}

/**
  * @brief Direct function to perform the float32 to Q31 conversion
  * @param hmw        MW handle.
  * @param inBuff     Input Buffer address
  * @param outBuff    Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_VectF2Q31(hsp_core_handle_t *hmw, float32_t *inBuff, int32_t *outBuff,
                                            uint32_t nbSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_F2Q31);
  HSP_ACC_VectIOS_WriteParam(hmw, (uint32_t)inBuff, (uint32_t)outBuff, nbSamples);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
}

/**
  * @brief Direct function to perform the Q15 to float32 conversion
  * @param hmw        MW handle.
  * @param inBuff     Input Buffer address
  * @param outBuff    Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_VectQ152F(hsp_core_handle_t *hmw, int32_t *inBuff, float32_t *outBuff,
                                            uint32_t nbSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_Q152F);
  HSP_ACC_VectIOS_WriteParam(hmw, (uint32_t)inBuff, (uint32_t)outBuff, nbSamples);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
}

/**
  * @brief Direct function to perform the float32 to Q15 conversion
  * @param hmw        MW handle.
  * @param inBuff     Input Buffer address
  * @param outBuff    Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_VectF2Q15(hsp_core_handle_t *hmw, float32_t *inBuff, int32_t *outBuff,
                                            uint32_t nbSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_F2Q15);
  HSP_ACC_VectIOS_WriteParam(hmw, (uint32_t)inBuff, (uint32_t)outBuff, nbSamples);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
}

/**
  * @brief Direct function to perform vector saturation
  * @param hmw        MW handle.
  * @param inBuff     Input Buffer address
  * @param inSat      Pointer on saturation limit [low Threshold, high Threshold]
  * @param outBuff    Output Buffer address
  * @param nbSamples  Number of float elements to proceed
  * @param isSat      1 if saturation occured
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_VectSat_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t *inSat,
                                                     float32_t *outBuff, uint32_t nbSamples, uint32_t *isSat)
{
	/* ToDo: assert_param(nbSamples > 1) */
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_SAT_F32);
  HSP_ACC_VectIIOSS_WriteParam(hmw, (uint32_t)inBuff, (uint32_t)inSat, (uint32_t)outBuff, (uint32_t)isSat, nbSamples);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
}

/**
  * @brief Direct function to perform copy of a vector from MEMB to MEMA.
  * The copy operation is independent of the type of buffer being copied
  * @param hmw        MW handle.
  * @param inBuff     Input Buffer address
  * @param outBuff    Output Buffer address
  * @param nbSamples  Number of 32-bits elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_VectMemB2A(hsp_core_handle_t *hmw, uint32_t *inBuff, uint32_t *outBuff,
                                             uint32_t nbSamples)
{
	/* ToDo: assert_param(nbSamples > 1) */
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_MB2A);
  HSP_ACC_VectIOS_WriteParam(hmw, (uint32_t)inBuff, (uint32_t)outBuff, nbSamples);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
}
/**
  * @}
  */

/**
  * @defgroup HSP_MODULES_DIRECT_LIB_Exported_Macros_Complex HSP Modules Direct Complex Command Macros
  * @{
  */
/**
  * @brief Direct function to compute the conjugate of each vector element (vector is a complex interleaved real, img)
  * @param hmw        MW handle.
  * @param inBuff     Input Buffer
  * @param outBuff    Output Buffer
  * @param nbSamples  Number of complex elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_CmplxConj_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t *outBuff,
                                                uint32_t nbSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_CMPLX_CONJ_F32);
  HSP_ACC_VectIOS_WriteParam(hmw, (uint32_t)inBuff, (uint32_t)outBuff, (nbSamples*2));
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);  
}

/**
  * @brief Direct function to compute the dot product of two complex vectors
  * @param hmw        MW handle.
  * @param inABuff    Input A Buffer
  * @param inBBuff    Input B Buffer
  * @param outBuff    Output Buffer
  * @param nbSamples  Number of complex elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_CmplxDotProd_f32(hsp_core_handle_t *hmw, float32_t *inABuff, float32_t *inBBuff,
                                                   float32_t *outBuff, uint32_t nbSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_CMPLX_DOTPROD_F32);
  HSP_ACC_VectIIOS_WriteParam(hmw, (uint32_t)inABuff, (uint32_t)inBBuff, (uint32_t)outBuff, (nbSamples*2));
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);    
}

/**
  * @brief Direct function to compute floating-point complex magnitude of a vector
  * @param hmw        MW handle.
  * @param inBuff     Input Buffer
  * @param outBuff    Output Buffer
  * @param nbSamples  Number of complex elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_CmplxMag_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t *outBuff,
                                               uint32_t nbSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_CMPLX_MAG_F32);
  HSP_ACC_VectIOS_WriteParam(hmw, (uint32_t)inBuff, (uint32_t)outBuff, (nbSamples*2));
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
}

/**
  * @brief Direct function to compute floating-point complex magnitude squared of a vector
  * @param hmw        MW handle.
  * @param inBuff     Input Buffer
  * @param outBuff    Output Buffer
  * @param nbSamples  Number of complex elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_CmplxMagSquared_f32(hsp_core_handle_t *hmw, float32_t *inBuff,
                                                      float32_t *outBuff, uint32_t nbSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_CMPLX_MAGSQUARED_F32);
  HSP_ACC_VectIOS_WriteParam(hmw, (uint32_t)inBuff, (uint32_t)outBuff, (nbSamples*2));
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
}

/**
  * @brief Direct function to compute the complex multiplication of 2 complex vector element
  * (vector is a complex interleaved real, img)
  * @param hmw        MW handle.
  * @param inABuff    Input A Buffer
  * @param inBBuff    Input B Buffer
  * @param outBuff    Output Buffer
  * @param nbSamples  Number of complex elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_CmplxMul_f32(hsp_core_handle_t *hmw, float32_t *inABuff, float32_t *inBBuff,
                                               float32_t *outBuff, uint32_t nbSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_CMPLX_MUL_F32);
  HSP_ACC_VectIIOS_WriteParam(hmw, (uint32_t)inABuff, (uint32_t)inBBuff, (uint32_t)outBuff, (nbSamples*2));
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);  
}

/**
  * @brief Direct function to compute multiplication of a complex vector by a real vector and generates a complex result
  * @param hmw        MW handle.
  * @param inABuff    Input A Buffer
  * @param inBBuff    Input B Buffer
  * @param outBuff    Output Buffer
  * @param nbSamples  Number of complex elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_CmplxRMul_f32(hsp_core_handle_t *hmw, float32_t *inABuff, float32_t *inBBuff,
                                                float32_t *outBuff, uint32_t nbSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_CMPLX_RMUL_F32);
  HSP_ACC_VectIIOS_WriteParam(hmw, (uint32_t)inABuff, (uint32_t)inBBuff, (uint32_t)outBuff, (nbSamples*2));
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw); 
}

/**
  * @brief Direct function to compute complex multiplication of one complex vector element and 
  *        an exponential cmplx signal
  * @param hmw        MW handle.
  * @param inBuff     Input Buffer
  * @param startBuff  Start buffer (I/O: input is start index and output is nextIdx)
  * @param outBuff    Output Buffer
  * @param nbSamples  Number of complex elements to proceed
  * @param step       Step value between 2 exponential number in ROM
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_CmplxMulExp_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t *startBuff,
                                                  float32_t *outBuff, uint32_t nbSamples, uint32_t step)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_CMPLX_MUL_EXP_F32);
  HSP_ACC_VectIIOSS_WriteParam(hmw, (uint32_t)inBuff, (uint32_t)startBuff, (uint32_t)outBuff, (nbSamples*2), step);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);  
}
/**
  * @}
  */

/**
  * @defgroup HSP_MODULES_DIRECT_LIB_Exported_Macros_Matrix HSP Modules Direct Matrix Command Macros
  * @{
  */
/**
  * @brief Direct function to perform matrix absolute value function
  * @param hmw      MW handle.
  * @param inBuff   Input Buffer address
  * @param outBuff  Output Buffer address
  * @param nCols    Matrix columns number
  * @param nRows    Matrix rows number
  * @retval         None
  */
__STATIC_FORCEINLINE void HSP_ACC_MatAbs_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t *outBuff,
                                             uint32_t nRows, uint32_t nCols)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_ABS_F32);
  HSP_ACC_VectIOS_WriteParam(hmw, (uint32_t)inBuff, (uint32_t)outBuff, (nCols)*(nRows));
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);  
}

/**
  * @brief Direct function to perform matrix add function
  * @param hmw      MW handle.
  * @param inABuff  Input A Buffer address
  * @param inBBuff  Input B Buffer address
  * @param outBuff  Output Buffer address
  * @param nCols    Matrix columns number
  * @param nRows    Matrix rows number
  * @retval         None
  */
__STATIC_FORCEINLINE void HSP_ACC_MatAdd_f32(hsp_core_handle_t *hmw, float32_t *inABuff, float32_t *inBBuff,
                                             float32_t *outBuff, uint32_t nRows, uint32_t nCols)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_ADD_F32);
  HSP_ACC_VectIIOS_WriteParam(hmw, (uint32_t)inABuff, (uint32_t)inBBuff, (uint32_t)outBuff, (nCols)*(nRows));
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
}

/**
  * @brief Direct function to add to each matrix element a scalar
  * @param hmw      MW handle.
  * @param inABuff  Input A Buffer address
  * @param inBBuff  Buffer address of offset to add
  * @param outBuff  Output Buffer address
  * @param nCols    Matrix columns number
  * @param nRows    Matrix rows number
  * @retval         None
  */
__STATIC_FORCEINLINE void HSP_ACC_MatOffset_f32(hsp_core_handle_t *hmw, float32_t *inABuff, float32_t *inBBuff,
                                                float32_t *outBuff, uint32_t nRows, uint32_t nCols)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_OFFSET_F32);
  HSP_ACC_VectIIOS_WriteParam(hmw, (uint32_t)inABuff, (uint32_t)inBBuff, (uint32_t)outBuff, (nCols)*(nRows));
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);  
}

/**
  * @brief Direct function to multiply each matrix element by a scalar
  * @param hmw      MW handle.
  * @param inABuff  Input A Buffer address
  * @param inBBuff  Buffer address of scale factor
  * @param outBuff  Output Buffer address
  * @param nCols    Matrix columns number
  * @param nRows    Matrix rows number
  * @retval         None
  */
__STATIC_FORCEINLINE void HSP_ACC_MatScale_f32(hsp_core_handle_t *hmw, float32_t *inABuff, float32_t *inBBuff,
                                               float32_t *outBuff, uint32_t nRows, uint32_t nCols)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_SCALE_F32);
  HSP_ACC_VectIIOS_WriteParam(hmw, (uint32_t)inABuff, (uint32_t)inBBuff, (uint32_t)outBuff, (nCols)*(nRows));
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);   
}

/**
  * @brief Direct function to perform matrix subtraction function
  * @param hmw      MW handle.
  * @param inABuff  Input A Buffer address
  * @param inBBuff  Input B Buffer address
  * @param outBuff  Output Buffer address
  * @param nCols    Matrix columns number
  * @param nRows    Matrix rows number
  * @retval         None
  */
__STATIC_FORCEINLINE void HSP_ACC_MatSub_f32(hsp_core_handle_t *hmw, float32_t *inABuff, float32_t *inBBuff,
                                             float32_t *outBuff, uint32_t nRows, uint32_t nCols)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_SUB_F32);
  HSP_ACC_VectIIOS_WriteParam(hmw, (uint32_t)inABuff, (uint32_t)inBBuff, (uint32_t)outBuff, (nCols)*(nRows));
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);   
}

/**
  * @brief Direct function to perform matrix inverse
  * @param hmw      MW handle.
  * @param inBuff   Input Buffer address
  * @param outBuff  Output Buffer address
  * @param nCols    Matrix columns number
  * @param nRows    Matrix rows number
  * @retval         None
  */
__STATIC_FORCEINLINE void HSP_ACC_MatInv_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t *outBuff,
                                             uint32_t nRows, uint32_t nCols)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_MAT_INV_F32);
  HSP_ACC_MatInv_WriteParam(hmw, (uint32_t)inBuff, (uint32_t)outBuff, nCols, nRows);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);   
}

/**
  * @brief Direct function to execute matrix multiplication function
  * @param hmw      MW handle.
  * @param inABuff  Input A Buffer address
  * @param inBBuff  Input B Buffer address
  * @param outBuff  Output Buffer address
  * @param nColsA   Matrix A columns number
  * @param nRowsA   Matrix A rows number
  * @param nColsB   Matrix B columns number
  * @param nRowsB   Matrix B rows number
  * @retval         None.
  */
__STATIC_FORCEINLINE void HSP_ACC_MatMult_f32(hsp_core_handle_t *hmw, float32_t *inABuff, float32_t *inBBuff,
                                              float32_t *outBuff, uint32_t nRowsA, uint32_t nColsA, 
                                              uint32_t nRowsB, uint32_t nColsB)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_MAT_MUL_F32);
  HSP_ACC_MatMult_WriteParam(hmw, (uint32_t)inABuff, (uint32_t)inBBuff, (uint32_t)outBuff, nRowsA, nColsA,
                                         nRowsB, nColsB);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);  
}

/**
  * @brief Direct function to execute matrix transpose function
  * @param hmw      MW handle.
  * @param inBuff   Input Buffer address
  * @param outBuff  Output Buffer address
  * @param nCols    Matrix columns number
  * @param nRows    Matrix rows number
  * @retval         None.
  */
__STATIC_FORCEINLINE void HSP_ACC_MatTrans_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t *outBuff,
                                               uint32_t nRows, uint32_t nCols)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_MAT_TRANS_F32);
  HSP_ACC_MatTrans_WriteParam(hmw, (uint32_t)inBuff, (uint32_t)outBuff, nCols, nRows);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);    
}
/**
  * @}
  */

/**
  * @defgroup HSP_MODULES_DIRECT_LIB_Exported_Macros_Filter HSP Modules Direct Filter Command Macros
  * @{
  */
/**
  * @brief Execute FIR function
  * @param hmw        MW handle.
  * @param inBuff     Input Buffer address
  * @param coefBuff   Coefficients Buffer address
  * @param stateId    State Buffer
  * @param outBuff    Output Buffer
  * @param nbSamples  Number of float elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_Fir_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t *coefBuff,
                                          hsp_filter_state_identifier_t stateId, float32_t *outBuff,
                                          uint32_t nbSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR((uint32_t)((hsp_hw_if_filter_state_t*)stateId)->dirCmd);
  HSP_ACC_Filter_WriteParam(hmw, (uint32_t)inBuff, (uint32_t)coefBuff, stateId, (uint32_t)outBuff, nbSamples);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw); 
}

/**
  * @brief Execute FIR Decimate function
  * @param hmw          MW handle.
  * @param inBuff       Input Buffer address
  * @param coefBuff     Coefficients Buffer address
  * @param stateId      State Buffer
  * @param outBuff      Output Buffer
  * @param nbSamples    Number of float elements to proceed
  * @param decimFactor  Decimation factor
  * @retval             None
  */
__STATIC_FORCEINLINE void HSP_ACC_FirDecimate_f32(hsp_core_handle_t *hmw, float32_t *inBuff,
                                                  float32_t *coefBuff, hsp_filter_state_identifier_t stateId, 
                                                  float32_t *outBuff, uint32_t nbSamples, 
                                                  uint32_t decimFactor)
{
  HSP_HW_IF_WRITE_DCMDIDR((uint32_t)((hsp_hw_if_fir_decimate_filter_state_t*)stateId)->dirCmd);
  HSP_ACC_FirDecimate_WriteParam(hmw, (uint32_t)inBuff, (uint32_t)coefBuff, stateId, (uint32_t)outBuff,
                                                nbSamples, decimFactor);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
}

/**
  * @brief Execute Biquad cascade df1 function
  * @param hmw        MW handle.
  * @param inBuff     Input Buffer address
  * @param coefBuff   Coefficients Buffer address
  * @param stateId    State Buffer
  * @param outBuff    Output Buffer
  * @param nbSamples  Number of float elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void  HSP_ACC_BiquadCascadeDf1_f32(hsp_core_handle_t *hmw, float32_t *inBuff,
                                                        float32_t *coefBuff, 
                                                        hsp_filter_state_identifier_t stateId, 
                                                        float32_t *outBuff, uint32_t nbSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR((uint32_t)((hsp_hw_if_filter_state_t*)stateId)->dirCmd);
  HSP_ACC_Filter_WriteParam(hmw, (uint32_t)inBuff, (uint32_t)coefBuff, stateId, (uint32_t)outBuff, nbSamples);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw); 
}

/**
  * @brief Execute Biquad cascade df2 function
  * @param hmw        MW handle.
  * @param inBuff     Input Buffer address
  * @param coefBuff   Coefficients Buffer address
  * @param stateId    State Buffer
  * @param outBuff    Output Buffer
  * @param nbSamples  Number of float elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_BiquadCascadeDf2T_f32(hsp_core_handle_t *hmw, float32_t *inBuff,
                                                        float32_t *coefBuff,
                                                        hsp_filter_state_identifier_t stateId,
                                                        float32_t *outBuff, uint32_t nbSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR((uint32_t)((hsp_hw_if_filter_state_t*)stateId)->dirCmd);
  HSP_ACC_Filter_WriteParam(hmw, (uint32_t)inBuff, (uint32_t)coefBuff, stateId, (uint32_t)outBuff, nbSamples);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw); 
}

/**
  * @brief Execute CMSIS Convolution
  * @param hmw      MW handle.
  * @param inABuff  Input A Buffer
  * @param inBBuff  Input B Buffer
  * @param outBuff  Output Buffer
  * @param sizeA    Number of float elements in vectA
  * @param sizeB    Number of float elements in vectB
  * @retval         None
  */
__STATIC_FORCEINLINE void HSP_ACC_Conv_f32(hsp_core_handle_t *hmw, float32_t *inABuff, float32_t *inBBuff,
                                           float32_t *outBuff, uint32_t sizeA, uint32_t sizeB)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_CONV_F32);
  HSP_ACC_VectIIOSS_WriteParam(hmw, (uint32_t)inABuff, (uint32_t)inBBuff, (uint32_t)outBuff, sizeA, sizeB);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
}

/**
  * @brief Execute Correlate
  * @param hmw      MW handle.
  * @param inABuff  Input A Buffer
  * @param inBBuff  Input B Buffer
  * @param outBuff  Output Buffer
  * @param sizeA    Number of float elements in vectA
  * @param sizeB    Number of float elements in vectB
  * @retval         None
  */
__STATIC_FORCEINLINE void HSP_ACC_Correlate_f32(hsp_core_handle_t *hmw, float32_t *inABuff, 
                                                float32_t *inBBuff, float32_t *outBuff,
                                                uint32_t sizeA, uint32_t sizeB)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_CORR_F32);
  HSP_ACC_VectIIOSS_WriteParam(hmw, (uint32_t)inABuff, (uint32_t)inBBuff, (uint32_t)outBuff, sizeA, sizeB);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw); 
}

/**
  * @brief Execute FLTBANK filter
  * @param hmw         MW handle.
  * @param spectrCol   Input spectrogram slice of length FFTLen / 2 Buffer
  * @param startIdx    FLTBANK filter pCoefficients start indexes Buffer
  * @param idxSize     FLTBANK filter pCoefficients size indexes Buffer
  * @param coef        FLTBANK filter weights Buffer
  * @param fltbankCol  Output fltbank energies in each filterbank Buffer
  * @param nFltbanks   Number of Fltbank bands to generate
  * @retval            None
  */
__STATIC_FORCEINLINE void HSP_ACC_FltBank_f32(hsp_core_handle_t *hmw, float32_t *spectrCol,
                                              uint32_t *startIdx, uint32_t *idxSize, float32_t *coef,
                                              float32_t *fltbankCol, uint32_t nFltbanks)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_FLTBANK_F32);
  HSP_ACC_FltBank_WriteParam(hmw, (uint32_t)spectrCol, (uint32_t)startIdx, (uint32_t)idxSize,
                                            (uint32_t)coef, (uint32_t)fltbankCol, (uint32_t)nFltbanks);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw); 
}

#if defined(__HSP_DMA__)
/**
  * @brief Execute FLTBANK filter with external coefficients (internal DMA is used to get ext coef
  *        in dmaBuffId pingpoing buffer)
  * @param hmw         MW handle.
  * @param spectrCol   Input spectrogram slice of length FFTLen / 2 Buffer
  * @param startIdx    FLTBANK filter pCoefficients start indexes Buffer
  * @param idxSize     FLTBANK filter pCoefficients size indexes Buffer
  * @param coef        FLTBANK filter weights Buffer
  * @param fltbankCol  Output fltbank energies in each filterbank Buffer
  * @param nFltbanks   Number of Fltbank bands to generate
  * @param dmaAdd      FLTBANK DMA Buffer address (must be max filter size x2 for pingpong)
  * @param dmaSize     FLTBANK DMA Buffer size (full DMA buffer size (ping + pong))
  * @retval            None
  */
__STATIC_FORCEINLINE void HSP_ACC_FltBankExtC_f32(hsp_core_handle_t *hmw, float32_t *spectrCol,
                                                  uint32_t *startIdx, uint32_t *idxSize, float32_t *coef,
                                                  float32_t *fltbankCol, uint32_t nFltbanks, 
                                                  float32_t *dmaAdd, uint32_t dmaSize)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_FLTBANK_EXTC_F32);
  HSP_ACC_FltBankExtC_WriteParam(hmw, (uint32_t)spectrCol, (uint32_t)startIdx, (uint32_t)idxSize,
                                 (uint32_t)coef, (uint32_t)fltbankCol, (uint32_t)nFltbanks, (uint32_t)dmaAdd,
                                 dmaSize);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
}
#endif /* __HSP_DMA__ */

/**
  * @brief Execute LMS filter function
  * @param hmw        MW handle.
  * @param inBuff     Input Buffer address
  * @param coefBuff   Coefficients Buffer address
  * @param stateId    State Buffer
  * @param outBuff    Output Buffer
  * @param refBuff    Reference Buffer
  * @param errBuff    Error Buffer
  * @param nbSamples  Number of float elements to proceed
  * @param mu         Adaptative factor
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_Lms_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t *coefBuff,
                                          hsp_filter_state_identifier_t stateId, float32_t *outBuff, 
                                          float32_t *refBuff, float32_t *errBuff, uint32_t nbSamples,
                                          float32_t mu)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_LMS_F32);
  HSP_ACC_Lms_WriteParam(hmw, (uint32_t)inBuff, (uint32_t)coefBuff, (uint32_t)stateId, (uint32_t)outBuff,
                                        (uint32_t)refBuff, (uint32_t)errBuff, (uint32_t)nbSamples, mu);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw); 	
}

/**
  * @brief Execute IIR Lattice filter function
  * @param hmw        MW handle.
  * @param inBuff     Input Buffer address
  * @param coeffsk    Coefficients Buffer
  * @param coeffsv    Coefficients Buffer
  * @param stateId    State Buffer
  * @param outBuff    Output Buffer
  * @param nbSamples  Number of float elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_IirLattice_f32(hsp_core_handle_t *hmw, float32_t *inBuff,
                                                float32_t *coeffsk, float32_t *coeffsv, 
                                                hsp_filter_state_identifier_t stateId,
                                                float32_t *outBuff, uint32_t nbSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_IIR_LATTICE_F32);
  HSP_ACC_IirLattice_WriteParam(hmw, (uint32_t)inBuff, (uint32_t)coeffsk, (uint32_t)coeffsv,
                                               (uint32_t)stateId, (uint32_t)outBuff, (uint32_t)nbSamples);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
}

/**
  * @brief IIR DF1 filter function
  * IIR coeffs are stored interleaved and in reversed order if K=nb stages:
  * B[k-1] A[k-1] B[k-2] A[k-2] ... B[1] A[1] B[0]
  * @param hmw        MW handle.
  * @param inBuff     Input Buffer address
  * @param coefBuff   Coefficients Buffer address
  * @param stateId    State Buffer
  * @param outBuff    Output Buffer identifier
  * @param nbSamples  Number of float elements to proceed
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_IirDf1_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t *coefBuff,
                                             hsp_filter_state_identifier_t stateId, float32_t *outBuff, 
                                             uint32_t nbSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_IIR_DF1_F32);
  HSP_ACC_Filter_WriteParam(hmw, (uint32_t)inBuff, (uint32_t)coefBuff, stateId, (uint32_t)outBuff, nbSamples);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
}

/**
  * @brief Windowing symmetric filter function
  * @param hmw        MW handle.
  * @param inBuff     Input Buffer address
  * @param inWin      Input window Buffer address
  * @param outBuff    Output Buffer address
  * @param sizeWin    Number of float elements in buffer input (Window size)
  * @param sizePad0   Number of extra destination value pad to 0
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_WinSym_f32(hsp_core_handle_t *hmw, float32_t *inBuff, float32_t *inWin,
                                             float32_t *outBuff, uint32_t sizeWin, uint32_t sizePad0)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_WINSYM_F32);
  HSP_ACC_VectIIOSS_WriteParam(hmw, (uint32_t)inBuff, (uint32_t)inWin, (uint32_t)outBuff, sizeWin, sizePad0);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
}

/**
  * @}
  */

/**
  * @defgroup HSP_MODULES_DIRECT_LIB_Exported_Macros_Filter_State HSP Modules Direct Filter State Command Macros
  * @{
  */
/**
  * @brief Reset FIR filter state
  * @param hmw        MW handle.
  * @param stateId    Identifier of the filter state to reset (set to 0)
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_FirResetStateBuffer(hsp_core_handle_t *hmw, 
                                                      hsp_filter_state_identifier_t stateId)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_RESET_STATE);
  HSP_ACC_StateBuffer_WriteParam(hmw, stateId, 1, 0, stateId);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
}

/**
  * @brief Reset FIR Decimate filter state
  * @param hmw         MW handle.
  * @param stateId     Identifier of the filter state to reset (set to 0)
  * @retval            None
  */
__STATIC_FORCEINLINE void HSP_ACC_FirDecimateResetStateBuffer(hsp_core_handle_t *hmw, 
                                                              hsp_filter_state_identifier_t stateId)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_RESET_STATE);
  HSP_ACC_FirDecimateStateBuffer_WriteParam(hmw, stateId, stateId);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
}

/**
  * @brief Reset BiquadCascadedf1 filter state
  * @param hmw         MW handle.
  * @param stateId     Identifier of the filter state to reset (set to 0)
  * @retval            None
  */
__STATIC_FORCEINLINE void HSP_ACC_BiquadCascadeDf1ResetStateBuffer(hsp_core_handle_t *hmw,
                                                                   hsp_filter_state_identifier_t stateId)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_RESET_STATE);
  HSP_ACC_StateBuffer_WriteParam(hmw, stateId, 4, 0, stateId);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);   
}

/**
  * @brief Reset BiquadCascadeDf2t filter state
  * @param hmw         MW handle.
  * @param stateId     Identifier of the filter state to reset (set to 0)
  * @retval            None
  */
__STATIC_FORCEINLINE void HSP_ACC_BiquadCascadeDf2TResetStateBuffer(hsp_core_handle_t *hmw,
                                                                    hsp_filter_state_identifier_t stateId)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_RESET_STATE);
  HSP_ACC_StateBuffer_WriteParam(hmw, stateId, 2, 0, stateId);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw); 
}

/**
  * @brief Reset LMS filter state
  * @param hmw         MW handle.
  * @param stateId     Identifier of the filter state to reset (set to 0)
  * @retval            None
  */
__STATIC_FORCEINLINE void HSP_ACC_LmsResetStateBuffer(hsp_core_handle_t *hmw, hsp_filter_state_identifier_t stateId)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_RESET_STATE);
  HSP_ACC_StateBuffer_WriteParam(hmw, stateId, 1, 0, stateId);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
}

/**
  * @brief Reset IIR Lattice filter state
  * @param hmw        MW handle.
  * @param stateId    Identifier of the filter state to reset (set to 0)
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_IirLatticeResetStateBuffer(hsp_core_handle_t *hmw, 
                                                             hsp_filter_state_identifier_t stateId)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_RESET_STATE);
  HSP_ACC_StateBuffer_WriteParam(hmw, stateId, 1, 0, stateId);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
}

/**
  * @brief Reset IIR Df1 filter state
  * @param hmw        MW handle.
  * @param stateId    Identifier of the filter state to reset (set to 0)
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_IirDf1ResetStateBuffer(hsp_core_handle_t *hmw, 
                                                         hsp_filter_state_identifier_t stateId)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_RESET_STATE);
  HSP_ACC_StateBuffer_WriteParam(hmw, stateId, 1, 0, stateId);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
}

/**
  * @brief Reset IIR 3p3z filter state
  * @param hmw        MW handle.
  * @param stateId    Identifier of the filter state to reset (set to 0)
  * @retval           None
 */
__STATIC_FORCEINLINE void HSP_ACC_Iir3P3ZResetStateBuffer(hsp_core_handle_t *hmw, 
                                                          hsp_filter_state_identifier_t stateId)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_RESET_STATE);
  HSP_ACC_StateBuffer_WriteParam(hmw, stateId, 0, 6, stateId);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
}

/**
  * @brief Reset IIR 2p2z filter state
  * @param hmw        MW handle.
  * @param stateId    Identifier of the filter state to reset (set to 0)
  * @retval           None
  */
__STATIC_FORCEINLINE void HSP_ACC_Iir2P2ZResetStateBuffer(hsp_core_handle_t *hmw, 
                                                          hsp_filter_state_identifier_t stateId)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_RESET_STATE);
  HSP_ACC_StateBuffer_WriteParam(hmw, stateId, 0, 4, stateId);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
}

/**
  * @brief Set FIR filter state
  * @param hmw          MW handle.
  * @param stateId      Identifier of the filter state to set
  * @param inBuffer     Float buffer containing value of state  buffer to set.
  *                     Values are stored from oldest to newest
  * @retval             None
  */
__STATIC_FORCEINLINE void HSP_ACC_FirSetStateBuffer(hsp_core_handle_t *hmw, 
                                                    hsp_filter_state_identifier_t stateId,
                                                    float32_t *inBuffer)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_SET_STATE_CR2L);
  HSP_ACC_StateBuffer_WriteParam(hmw, stateId, 1, 0, (uint32_t)inBuffer);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
}

/**
  * @brief Set FIR Decimate filter state
  * @param hmw          MW handle.
  * @param stateId      Identifier of the filter state to set
  * @param inBuffer     Float buffer containing value of state  buffer to set.
  *                     Values are stored from oldest to newest
  * @retval             None
  */
__STATIC_FORCEINLINE void HSP_ACC_FirDecimateSetStateBuffer(hsp_core_handle_t *hmw,
                                                            hsp_filter_state_identifier_t stateId,
                                                            float32_t *inBuffer)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_SET_STATE_CR2L);
  HSP_ACC_FirDecimateStateBuffer_WriteParam(hmw, stateId, (uint32_t)inBuffer);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);     
}

/**
  * @brief Set BiquadCascadedf1 filter state
  * @param hmw          MW handle.
  * @param stateId      Identifier of the filter state to set
  * @param inBuffer     Float buffer containing value of state  buffer to set.
  *                     Values are stored from oldest to newest
  * @retval             None
  */
__STATIC_FORCEINLINE void HSP_ACC_BiquadCascadeDf1SetStateBuffer(hsp_core_handle_t *hmw,
                                                                 hsp_filter_state_identifier_t stateId,
                                                                 float32_t *inBuffer)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_SET_STATE_L2R);
  HSP_ACC_StateBuffer_WriteParam(hmw, stateId, 4, 0, (uint32_t)inBuffer);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);     
}

/**
  * @brief Set BiquadCascadeDf2t filter state
  * @param hmw          MW handle.
  * @param stateId      Identifier of the filter state to set
  * @param inBuffer     Float buffer containing value of state  buffer to set.
  *                     Values are stored from oldest to newest
  * @retval             None
  */
__STATIC_FORCEINLINE void HSP_ACC_BiquadCascadeDf2TSetStateBuffer(hsp_core_handle_t *hmw,
                                                                  hsp_filter_state_identifier_t stateId, 
                                                                  float32_t *inBuffer)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_SET_STATE_L2R);
  HSP_ACC_StateBuffer_WriteParam(hmw, stateId, 2, 0, (uint32_t)inBuffer);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw); 
}

/**
  * @brief Set LMS filter state
  * @param hmw          MW handle.
  * @param stateId      Identifier of the filter state to set
  * @param inBuffer     Float buffer containing value of state  buffer to set.
  *                     Values are stored from oldest to newest
  * @retval             None
  */
__STATIC_FORCEINLINE void HSP_ACC_LmsSetStateBuffer(hsp_core_handle_t *hmw, 
                                                    hsp_filter_state_identifier_t stateId,
                                                    float32_t *inBuffer)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_SET_STATE_CR2L);
  HSP_ACC_StateBuffer_WriteParam(hmw, stateId, 1, 0, (uint32_t)inBuffer);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);  
}

/**
  * @brief Set IIR Lattice filter state
  * @param hmw          MW handle.
  * @param stateId      Identifier of the filter state to set
  * @param inBuffer     Float buffer containing value of state  buffer to set.
  *                     Values are stored from oldest to newest
  * @retval             None
  */
__STATIC_FORCEINLINE void HSP_ACC_IirLatticeSetStateBuffer(hsp_core_handle_t *hmw, 
                                                           hsp_filter_state_identifier_t stateId, 
                                                           float32_t *inBuffer)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_SET_STATE_CR2L);
  HSP_ACC_StateBuffer_WriteParam(hmw, stateId, 1, 0, (uint32_t)inBuffer);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
}

/**
  * @brief Set IIR Df1 filter state
  * @param hmw          MW handle.
  * @param stateId      Identifier of the filter state to set
  * @param inBuffer     Float buffer containing value of state  buffer to set.
  *                     Values are stored from oldest to newest
  * @retval             None
  */
__STATIC_FORCEINLINE void HSP_ACC_IirDf1SetStateBuffer(hsp_core_handle_t *hmw, 
                                                       hsp_filter_state_identifier_t stateId,
                                                       float32_t *inBuffer)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_SET_STATE_CL2R);
  HSP_ACC_StateBuffer_WriteParam(hmw, stateId, 1, 0, (uint32_t)inBuffer);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
}

/**
  * @brief Set IIR 3p3z filter state
  * @param hmw          MW handle.
  * @param stateId      Identifier of the filter state to set
  * @param inBuffer     Float buffer containing value of state  buffer to set.
  *                     Values are stored from oldest to newest
  * @retval             None
  */
__STATIC_FORCEINLINE void HSP_ACC_Iir3P3ZSetStateBuffer(hsp_core_handle_t *hmw, 
                                                                  hsp_filter_state_identifier_t stateId,
                                                                  float32_t *inBuffer)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_SET_STATE_CR2L);
  HSP_ACC_StateBuffer_WriteParam(hmw, stateId, 0, 6, (uint32_t)inBuffer);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
}

/**
  * @brief Set IIR 2p2z filter state
  * @param hmw          MW handle.
  * @param stateId      Identifier of the filter state to set
  * @param inBuffer     Float buffer containing value of state  buffer to set.
  *                     Values are stored from oldest to newest
  * @retval             None
  */
__STATIC_FORCEINLINE void HSP_ACC_Iir2P2ZSetStateBuffer(hsp_core_handle_t *hmw,
                                                        hsp_filter_state_identifier_t stateId,
                                                        float32_t *inBuffer)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_SET_STATE_CR2L);
  HSP_ACC_StateBuffer_WriteParam(hmw, stateId, 0, 4, (uint32_t)inBuffer);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
}

/**
  * @brief Get FIR filter state
  * @param hmw          MW handle.
  * @param stateId      Identifier of the filter state to get
  * @param outBuffer    Float buffer containing value of state  buffer
  * @retval             None
  */
__STATIC_FORCEINLINE void HSP_ACC_FirGetStateBuffer(hsp_core_handle_t *hmw, 
                                                    hsp_filter_state_identifier_t stateId,
                                                    float32_t *outBuffer)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_GET_STATE_CR2L);
  HSP_ACC_StateBuffer_WriteParam(hmw, stateId, 1, 0, (uint32_t)outBuffer);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
}

/**
  * @brief Get FIR Decimate filter state
  * @param hmw          MW handle.
  * @param stateId      Identifier of the filter state to get
  * @param outBuffer    Float buffer containing value of state  buffer
  * @retval             None
  */
__STATIC_FORCEINLINE void HSP_ACC_FirDecimateGetStateBuffer(hsp_core_handle_t *hmw,
                                                            hsp_filter_state_identifier_t stateId,
                                                            float32_t *outBuffer)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_GET_STATE_CR2L);
  HSP_ACC_FirDecimateStateBuffer_WriteParam(hmw, stateId, (uint32_t)outBuffer);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);       
}

/**
  * @brief Get BiquadCascadedf1 filter state
  * @param hmw          MW handle.
  * @param stateId      Identifier of the filter state to get
  * @param outBuffer    Float buffer containing value of state  buffer
  * @retval             None
  */
__STATIC_FORCEINLINE void HSP_ACC_BiquadCascadeDf1GetStateBuffer(hsp_core_handle_t *hmw,
                                                                 hsp_filter_state_identifier_t stateId, 
                                                                 float32_t *outBuffer)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_GET_STATE_L2R);
  HSP_ACC_StateBuffer_WriteParam(hmw, stateId, 4, 0, (uint32_t)outBuffer);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);       
}

/**
  * @brief Get BiquadCascadeDf2t filter state
  * @param hmw          MW handle.
  * @param stateId      Identifier of the filter state to get
  * @param outBuffer    Float buffer containing value of state  buffer
  * @retval             None
  */
__STATIC_FORCEINLINE void HSP_ACC_BiquadCascadeDf2TGetStateBuffer(hsp_core_handle_t *hmw,
                                                                  hsp_filter_state_identifier_t stateId, 
                                                                  float32_t *outBuffer)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_GET_STATE_L2R);
  HSP_ACC_StateBuffer_WriteParam(hmw, stateId, 2, 0, (uint32_t)outBuffer);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw); 
}

/**
  * @brief Get LMS  filter state
  * @param hmw          MW handle.
  * @param stateId      Identifier of the filter state to get
  * @param outBuffer    Float buffer containing value of state  buffer
  * @retval             None
  */
__STATIC_FORCEINLINE void HSP_ACC_LmsGetStateBuffer(hsp_core_handle_t *hmw, 
                                                    hsp_filter_state_identifier_t stateId,
                                                    float32_t *outBuffer)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_GET_STATE_CR2L);
  HSP_ACC_StateBuffer_WriteParam(hmw, stateId, 1, 0, (uint32_t)outBuffer);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);    
}

/**
  * @brief Get IIR Lattice filter state
  * @param hmw          MW handle.
  * @param stateId      Identifier of the filter state to get
  * @param outBuffer    Float buffer containing value of state buffer
  * @retval             None
  */
__STATIC_FORCEINLINE void HSP_ACC_IirLatticeGetStateBuffer(hsp_core_handle_t *hmw, 
                                                           hsp_filter_state_identifier_t stateId,
                                                           float32_t *outBuffer)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_GET_STATE_CR2L);
  HSP_ACC_StateBuffer_WriteParam(hmw, stateId, 1, 0, (uint32_t)outBuffer);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
}

/**
  * @brief Get IIR Df1 filter state
  * @param hmw         MW handle.
  * @param stateId     Identifier of the filter state to get
  * @param outBuffer   Float buffer containing value of state buffer
  * @retval            None
  */
__STATIC_FORCEINLINE void HSP_ACC_IirDf1GetStateBuffer(hsp_core_handle_t *hmw, 
                                                       hsp_filter_state_identifier_t stateId,
                                                       float32_t *outBuffer)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_GET_STATE_CL2R);
  HSP_ACC_StateBuffer_WriteParam(hmw, stateId, 1, 0, (uint32_t)outBuffer);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);  
}

/**
  * @brief Get IIR 3p3z filter state
  * @param hmw          MW handle.
  * @param stateId      Identifier of the filter state to get
  * @param outBuffer    Float buffer containing value of state buffer
  * @retval             None
  */
__STATIC_FORCEINLINE void HSP_ACC_Iir3P3ZGetStateBuffer(hsp_core_handle_t *hmw, 
                                                        hsp_filter_state_identifier_t stateId,
                                                        float32_t *outBuffer)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_GET_STATE_CR2L);
  HSP_ACC_StateBuffer_WriteParam(hmw, stateId, 0, 6, (uint32_t)outBuffer);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw); 
}

/**
  * @brief Get IIR 2p2z  filter state
  * @param hmw          MW handle.
  * @param stateId      Identifier of the filter state to get
  * @param outBuffer    Float buffer containing value of state buffer
  * @retval             None
  */
__STATIC_FORCEINLINE void HSP_ACC_Iir2P2ZGetStateBuffer(hsp_core_handle_t *hmw, 
                                                        hsp_filter_state_identifier_t stateId,
                                                        float32_t *outBuffer)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_GET_STATE_CR2L);
  HSP_ACC_StateBuffer_WriteParam(hmw, stateId, 0, 4, (uint32_t)outBuffer);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);  
}
/**
  * @}
  */

/**
  * @defgroup HSP_MODULES_DIRECT_LIB_Exported_Macros_CRC HSP Modules Direct CRC Command Macros
  * @{
  */
/**
  * @brief Direct CRC32 computation
  * @param hmw            MW handle.
  * @param pState         Pointer of memory where pState[0] = CRC pState[1] = offset
  * @param pCRC           Pointer on crc computed
  * @param blockSize      Size of block
  * @param posFEOR        Position of the flag indicating when CRC function reaches the end of the ROM.
  * @param posFEOB        Position of the flag indicating when CRC function reaches the end of data block.
  * @param memType        HSP_CRC_CROM: to run CRC on CROM,
  *                       HSP_CRC_DROM: to run CRC on DROM
  * @retval               None
  */
__STATIC_FORCEINLINE void HSP_ACC_Crc32(hsp_core_handle_t *hmw, uint32_t *pState, uint32_t *pCRC,
                                        uint32_t blockSize, uint32_t posFEOR, uint32_t posFEOB, hsp_crc_mem_type_cmd_t memType)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_CRC32);
  HSP_ACC_Crc32_WriteParam(hmw, (uint32_t)pState, (uint32_t)pCRC, blockSize, posFEOR, posFEOB, (uint32_t)memType);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
}
/**
  * @}
 */

#if defined(__HSP_DMA__)
  /**
  * @defgroup HSP_MODULES_DIRECT_LIB_Exported_Macros_Background_Transfer HSP Modules Direct Background Transfer Command
  *  Macros
  * @{
  */
/**
  * @brief Direct memory transfer function to program HSP DMA in background mode to transfer data from external to
  *        internal memory
  * @param hmw          MW handle.
  * @param inAddr       Input Buffer Address
  * @param ouAddr       Output Buffer Address
  * @param nbElem       Number of Elements to proceed
  * @param eltFormat    Source element format to transfer ({0 uint16_t} {1 int16_t} {2 32-bits})
  * @param pingpong     Dest buffer is in ping-pong or single buffer
  * @param dmaChan      DMA channel number
  * @param perIdx       Peripheral channel One channel among 8 can be selected.
  *                     Allowed values are from 0 to 7
  * @param pingPl       Ping processing list (only for pingpong option)
  * @param pongPl       Pong processing list (only for pingpong option)
  * @retval             None  
  */
__STATIC_FORCEINLINE void HSP_ACC_BackgroundExt2Int(hsp_core_handle_t *hmw, float32_t *inAddr,
                                                    float32_t *ouAddr, uint32_t nbElem, uint32_t eltFormat,
                                                    uint32_t pingpong, uint32_t dmaChan, uint32_t perIdx,
                                                    uint32_t pingPl, uint32_t pongPl)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_DMA_HW_REQ_START);
  HSP_ACC_BackgroundExt2Int_WriteParam(hmw, (uint32_t)inAddr, (uint32_t)ouAddr, nbElem,
                                       eltFormat, pingpong, dmaChan,
                                       perIdx, pingPl, pongPl);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
}

/**
  * @brief Direct memory transfer function to program HSP DMA in background mode to transfer data from internal to 
  *        external memory
  * @param hmw          MW handle.
  * @param inAddr       Input Buffer Address
  * @param ouAddr       Output Buffer Address
  * @param nbElem       Number of Elements to proceed
  * @param eltFormat    Source element format to transfer ({0 uint16_t} {1 int16_t} {2 32-bits})
  * @param pingpong     Source buffer is in ping-pong or single buffer
  * @param dmaChan      DMA channel number
  * @param perIdx       Peripheral channel One channel among 8 can be selected.
  *                     Allowed values are from 0 to 7
  * @param pingPl       Ping processing list (only for pingpong option)
  * @param pongPl       Pong processing list (only for pingpong option)
  * @retval             None  
  */
__STATIC_FORCEINLINE void HSP_ACC_BackgroundInt2Ext(hsp_core_handle_t *hmw, float32_t *inAddr,
                                                    float32_t *ouAddr, uint32_t nbElem, uint32_t eltFormat,
                                                    uint32_t pingpong, uint32_t dmaChan, uint32_t perIdx,
                                                    uint32_t pingPl, uint32_t pongPl)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_DMA_HW_REQ_START);
  HSP_ACC_BackgroundInt2Ext_WriteParam(hmw, (uint32_t)inAddr, (uint32_t)ouAddr, nbElem,
                                       eltFormat, pingpong, dmaChan,
                                       perIdx, pingPl, pongPl);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
}

/**
  * @brief Direct memory transfer function to suspend/resume background transfer data from external to internal memory
  * @param hmw        MW handle.
  * @param dmaChan    DMA channel number
  * @param flag       Indicate if suspend or resume.
  *                   - Flag=0 suspend transfer
  *                   - Flag=1 resume transfer
  * @retval           None  
  */
__STATIC_FORCEINLINE void HSP_ACC_BackgroundSuspendResumeExt2Int(hsp_core_handle_t *hmw, uint8_t dmaChan,
                                                                 uint32_t flag)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_DMA_HW_REQ_SUSPENDRESUME);
  HSP_ACC_BackgroundSuspendResumeExt2Int_WriteParam((hmw), (uint32_t)dmaChan, (uint32_t)flag);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
}
/**
  * @}
  */
#endif /* __HSPDMA__ */

  /** 
  * @defgroup HSP_MODULES_DIRECT_LIB_Exported_Macros_Functions_Counter HSP Modules Direct Functions Counter Command 
  * Macros
  * @{
  */
/**
  * @brief Counter kernel function to add immediate to global counter index
  * @param hmw        MW handle
  * @param cntIdx     Counter index [0,15])
  * @param val        Value to add to counter
  * @retval           None
  */  
__STATIC_FORCEINLINE void HSP_ACC_CounterAdd(hsp_core_handle_t *hmw, uint32_t cntIdx, uint32_t val)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_COUNT);
  HSP_ACC_Counter_WriteParam((hmw), cntIdx, val, HAL_HSP_CMD_COUNT_MODE_ADD);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
}

/**
  * @brief Counter kernel function to add immediate to global counter index
  * @param hmw        MW handle
  * @param cntIdx     Counter index [0,15])
  * @param val        Value to add to counter
  * @retval           None
  */  
__STATIC_FORCEINLINE void HSP_ACC_CounterSet(hsp_core_handle_t *hmw, uint32_t cntIdx, uint32_t val)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_COUNT);
  HSP_ACC_Counter_WriteParam((hmw), cntIdx, val, HAL_HSP_CMD_COUNT_MODE_SET);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
}


/**
  * @}
  */

  /** 
  * @defgroup HSP_MODULES_DIRECT_LIB_Exported_Macros_Functions_FWERR HSP Modules Direct Functions FW Error register Command 
  * Macros
  * @{
  */
/**
  * @brief Clear FWERR register
  * @param hmw         MW handle
  * @retval            FW error cleared
  */
__STATIC_FORCEINLINE uint32_t HSP_ACC_ClearFWERR(hsp_core_handle_t *hmw)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_CLR_FWERR);

  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
  /* TO BE REWORK !! */
  return (((hal_hsp_handle_t *)(hmw->hdriver))->Instance->PARAMR0);
}

/**
  * @}
  */
  
#ifdef __cplusplus
}
#endif

#endif /* HSP_DIRECT_COMMAND_H */

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
