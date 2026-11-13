/**
  ******************************************************************************
  * @file    hsp_proclist_vector.h
  * @author  MCD Application Team
  * @brief   Header file for hsp_proclist_vector.c
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
#ifndef HSP_PROCLIST_VECTOR_H
#define HSP_PROCLIST_VECTOR_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "hsp_proclist_def.h"


/** @addtogroup HSP
  * @{
  */

/** @defgroup HSP_PROCLIST
  * @{
  */

  
/** @defgroup HSP_MODULES_PROCLIST_VECTOR_LIBRARY
  * @{
  */
  
/* Exported constants --------------------------------------------------------*/
/** @defgroup HSP_MODULES_PROCLIST_VECTOR_Exported_Defines
  * @{
  */

/**
  * @}
  */

/* Exported macros ------------------------------------------------------------*/
/** @defgroup HSP_MODULES_PROCLIST_VECTOR_Exported_Macros
  * @{
  */
/**
  * @}
  */

/* Exported types -------------------------------------------------------------*/
/** @defgroup  HSP_MODULES_PROCLIST_VECTOR_Exported_Types
  * @{
  */
/**
  * @brief  HSP_CMD_CMPCNT configuration fields: [loTh, hiTh, maxCntLo, maxCntHi]
*/
typedef struct
{
  float32_t loTh;     /*!< Low threshold value */
  float32_t hiTh;     /*!< High threshold value */
  uint32_t maxCntLo;  /*!< LO counter value: for number of consecutive value bigger than loTh */
  uint32_t maxCntHi;  /*!< HI counter value: for number of consecutive value lower than hiTh */
} HSP_CmpCntBuffCfgTypeDef;

/**
  * @brief  HSP_CMD_CMPCNT counter fields: [cntLo, cntHi]
  */
typedef struct
{
  uint32_t cntLo; /*!< Low counter value (LO) */
  uint32_t cntHi; /*!< High counter value (HI) */
} HSP_CmpCntBuffCntTypeDef;
/**
  * @}
  */

/* Exported variables ---------------------------------------------------------*/
/** @defgroup  HSP_MODULES_PROCLIST_VECTOR_Exported_Variables
  * @{
  */
/**
  * @}
  */

/* Exported functions ---------------------------------------------------------*/
/** @defgroup HSP_Exported_Functions_ProcList_vector MW HSP Vector family function for Processing List
  * @{
  */

hsp_core_status_t HSP_SEQ_VectAbs_f32(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pDst,
                                        uint32_t nbSamples, uint32_t ioType);
hsp_core_status_t HSP_SEQ_VectAdd_f32(hsp_core_handle_t *hmw, float32_t *pSrcA, float32_t *pSrcB,
                                        float32_t *pDst, uint32_t nbSamples, uint32_t ioType);
hsp_core_status_t HSP_SEQ_VectAvg_f32(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pDst, uint32_t nbSamples,
                                        uint32_t ioType);
hsp_core_status_t HSP_SEQ_VectCopy_f32(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pDst, uint32_t nbSamples,
                                         uint32_t ioType);
hsp_core_status_t HSP_SEQ_VectCos_f32(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pDst, uint32_t nbSamples,
                                        uint32_t ioType);
hsp_core_status_t HSP_SEQ_VectDiv_f32(hsp_core_handle_t *hmw, float32_t *pSrcA, float32_t *pSrcB, float32_t *pDst,
                                        uint32_t nbSamples, uint32_t ioType);
hsp_core_status_t HSP_SEQ_VectDotProd_f32(hsp_core_handle_t *hmw, float32_t *pSrcA, float32_t *pSrcB,
                                            float32_t *pDst, uint32_t nbSamples, uint32_t ioType);
hsp_core_status_t HSP_SEQ_VectAbsmax_f32(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *outVal,
                                           uint32_t *outIdx, uint32_t nbSamples, uint32_t ioType);
hsp_core_status_t HSP_SEQ_VectMax_f32(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *outVal,
                                        uint32_t *outIdx, uint32_t nbSamples, uint32_t ioType);
hsp_core_status_t HSP_SEQ_VectMin_f32(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *outVal,
                                        uint32_t *outIdx, uint32_t nbSamples, uint32_t ioType);
hsp_core_status_t HSP_SEQ_VectMul_f32(hsp_core_handle_t *hmw, float32_t *pSrcA, float32_t *pSrcB,
                                        float32_t *pDst, uint32_t nbSamples, uint32_t ioType);
hsp_core_status_t HSP_SEQ_VectOffset_f32(hsp_core_handle_t *hmw, float32_t *pSrcA, float32_t *pSrcB,
                                           float32_t *pDst, uint32_t nbSamples, uint32_t ioType);
hsp_core_status_t HSP_SEQ_VectRms_f32(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pDst, uint32_t nbSamples,
                                        uint32_t ioType);
hsp_core_status_t HSP_SEQ_VectScale_f32(hsp_core_handle_t *hmw, float32_t *pSrcA, float32_t *pSrcB,
                                          float32_t *pDst, uint32_t nbSamples, uint32_t ioType);
hsp_core_status_t HSP_SEQ_VectSet_f32(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pDst, uint32_t nbSamples,
                                        uint32_t ioType);
hsp_core_status_t HSP_SEQ_VectSqrt_f32(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pDst, uint32_t nbSamples,
                                         uint32_t ioType);
hsp_core_status_t HSP_SEQ_VectSin_f32(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pDst, uint32_t nbSamples,
                                        uint32_t ioType);
hsp_core_status_t HSP_SEQ_VectSub_f32(hsp_core_handle_t *hmw, float32_t *pSrcA, float32_t *pSrcB, float32_t *pDst,
                                        uint32_t nbSamples, uint32_t ioType);
hsp_core_status_t HSP_SEQ_VectSinCos_f32(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pDst,
                                           uint32_t nbSamples, uint32_t ioType);
hsp_core_status_t HSP_SEQ_VectF2I(hsp_core_handle_t *hmw, float32_t *pSrc, int32_t *pDst, uint32_t nbSamples,
                                        uint32_t ioType);
hsp_core_status_t HSP_SEQ_VectI2F(hsp_core_handle_t *hmw, int32_t *pSrc, float32_t *pDst, uint32_t nbSamples,
                                        uint32_t ioType);
hsp_core_status_t HSP_SEQ_VectF2U(hsp_core_handle_t *hmw, float32_t *pSrc, uint32_t *pDst, uint32_t nbSamples,
                                        uint32_t ioType);
hsp_core_status_t HSP_SEQ_VectU2F(hsp_core_handle_t *hmw, uint32_t *pSrc, float32_t *pDst, uint32_t nbSamples,
                                        uint32_t ioType);
hsp_core_status_t HSP_SEQ_Vect24s2F(hsp_core_handle_t *hmw, int32_t *pSrc, float32_t *pDst, uint32_t nbSamples,
                                          uint32_t ioType);
hsp_core_status_t HSP_SEQ_VectF2Q31(hsp_core_handle_t *hmw, float32_t *pSrc, int32_t *pDst, uint32_t nbSamples,
                                          uint32_t ioType);
hsp_core_status_t HSP_SEQ_VectQ312F(hsp_core_handle_t *hmw, int32_t *pSrc, float32_t *pDst, uint32_t nbSamples,
                                          uint32_t ioType);
hsp_core_status_t HSP_SEQ_VectF2Q15(hsp_core_handle_t *hmw, float32_t *pSrc, int32_t *pDst, uint32_t nbSamples,
                                      uint32_t ioType);
hsp_core_status_t HSP_SEQ_VectQ152F(hsp_core_handle_t *hmw, int32_t *pSrc, float32_t *pDst, uint32_t nbSamples,
                                      uint32_t ioType);
hsp_core_status_t HSP_SEQ_VectDecim_f32(hsp_core_handle_t *hmw, float32_t *pSrc, uint32_t decim,
                                          float32_t *pDst, uint32_t sizeOu, uint32_t ioType);
hsp_core_status_t HSP_SEQ_VectZins_f32(hsp_core_handle_t *hmw, float32_t *pSrc, uint32_t interp, float32_t *pDst,
                                         uint32_t sizeIn, uint32_t ioType);
hsp_core_status_t HSP_SEQ_VectAtan2_f32(hsp_core_handle_t *hmw, float32_t *pSrcA, float32_t *pSrcB,
                                          float32_t *pDst, uint32_t nbSamples, uint32_t ioType);
hsp_core_status_t HSP_SEQ_VectLn_f32(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pDst, uint32_t nbSamples,
                                       uint32_t ioType);
hsp_core_status_t HSP_SEQ_VectLog10_f32(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pDst, uint32_t nbSamples,
                                          uint32_t ioType);
hsp_core_status_t HSP_SEQ_VectExp_f32(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pDst, uint32_t nbSamples,
                                        uint32_t ioType);
hsp_core_status_t HSP_SEQ_VectExp10_f32(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pDst, uint32_t nbSamples,
                                          uint32_t ioType);
hsp_core_status_t HSP_SEQ_VectMulCos_f32(hsp_core_handle_t *hmw, float32_t *pSrcA, float32_t *pSrcB, float32_t *pDst,
                                           uint32_t nbSamples, int32_t step, uint32_t ioType);
hsp_core_status_t HSP_SEQ_VectMulSin_f32(hsp_core_handle_t *hmw, float32_t *pSrcA, float32_t *pSrcB, float32_t *pDst,
                                           uint32_t nbSamples, int32_t step, uint32_t ioType);
hsp_core_status_t HSP_SEQ_VectIOffset_f32(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t offset, float32_t *pDst,
                                            uint32_t nbSamples, uint32_t ioType);
hsp_core_status_t HSP_SEQ_VectIScale_f32(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t scale, float32_t *pDst,
                                           uint32_t nbSamples, uint32_t ioType);
hsp_core_status_t HSP_SEQ_VectISet_f32(hsp_core_handle_t *hmw, float32_t value, float32_t *pDst, uint32_t nbSamples,
                                         uint32_t ioType);
hsp_core_status_t HSP_SEQ_VectCmp_f32(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pLim, uint32_t *pRes,
                                        uint32_t nbSamples, uint32_t cmpType, uint32_t ioType);
hsp_core_status_t HSP_SEQ_VectCmpCnt_f32(hsp_core_handle_t *hmw, float32_t *pSrc, int32_t *pDst,
                                         hsp_cmp_cnt_lim_t *pLim, hsp_cmp_cnt_cnt_t *pMax,
                                         hsp_cmp_cnt_cnt_t *pCnt, uint32_t ioType);
hsp_core_status_t HSP_SEQ_VectSat_f32(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pSat, float32_t *pDst,
                                      uint32_t nbSamples, uint32_t *pRes, uint32_t ioType);
#ifdef __HSP_DMA__
hsp_core_status_t HSP_SEQ_Ext2Int(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pDst, uint32_t nbElems,
                                  uint32_t eltFormat, uint32_t iJump, uint32_t nbIterIn, uint32_t offsetIn,
                                  uint32_t oJump, uint32_t nbIterOu, uint32_t offsetOu);
hsp_core_status_t HSP_SEQ_Int2Ext(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pDst, uint32_t nbElems,
                                  uint32_t eltFormat, uint32_t iJump, uint32_t nbIterIn, uint32_t offsetIn,
                                  uint32_t oJump, uint32_t nbIterOu, uint32_t offsetOu);
hsp_core_status_t HSP_SEQ_Int2Int(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pDst, uint32_t nbElems,
                                  uint32_t eltFormat, uint32_t iJump, uint32_t nbIterIn, uint32_t offsetIn,
                                  uint32_t oJump, uint32_t nbIterOu, uint32_t offsetOu);
hsp_core_status_t HSP_SEQ_BgndExt2Int(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pDst,
                                      uint32_t nbElems, uint32_t eltFormat, uint32_t chanId, uint32_t iJump,
                                      uint32_t nbIterIn, uint32_t offsetIn, uint32_t oJump, uint32_t nbIterOu,
                                      uint32_t offsetOu, uint32_t nextOffsetIn, uint32_t nextOffsetOu);
hsp_core_status_t HSP_SEQ_BgndInt2Ext(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pDst,
                                      uint32_t nbElems, uint32_t eltFormat, uint32_t chanId, uint32_t iJump,
                                      uint32_t nbIterIn, uint32_t offsetIn, uint32_t oJump, uint32_t nbIterOu,
                                      uint32_t offsetOu, uint32_t nextOffsetIn, uint32_t nextOffsetOu);
hsp_core_status_t HSP_SEQ_BgndInt2Int(hsp_core_handle_t *hmw, float32_t *pSrc, float32_t *pDst,
                                      uint32_t nbElems, uint32_t eltFormat, uint32_t chanId, uint32_t iJump,
                                      uint32_t nbIterIn, uint32_t offsetIn, uint32_t oJump, uint32_t nbIterOu,
                                      uint32_t offsetOu, uint32_t nextOffsetIn, uint32_t nextOffsetOu);
hsp_core_status_t HSP_SEQ_BgndCmd(hsp_core_handle_t *hmw, uint32_t cmd, uint32_t chanId);
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

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* HSP_PROCLIST_VECTOR_H */
