/**
  ******************************************************************************
  * @file hsp_cnn.c
  * @brief API for HSP CNN functions
  *
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
#include "hsp_conf.h"
#include "hsp_cnn.h"
#include "hsp_def.h"
#include "hsp_hw_if.h"
#include "hsp_bram.h"
#include "hsp_bram_if.h"

/** @addtogroup HSP_ENGINE
  * @{
  */

/** @addtogroup HSP_MODULES
  * @{
  */

/** @addtogroup HSP_MODULES_CNN
  * @{
  */

#define HSP_MEMSET memset

/* Private types -----------------------------------------------------------*/
/** @defgroup HSP_CNN_Private_Types HSP_CNN Private Types
  * @{
  */
typedef uint8_t uint8_mema_t;
typedef int8_t int8_mema_t;
typedef uint8_t uint8_memb_t;
typedef int8_t int8_memb_t;
typedef uint32_t uint32_mema_t;
typedef int32_t int32_mema_t;
typedef uint32_t uint32_memb_t;
typedef int32_t int32_memb_t;
/**
  * @}
  */
/* Private defines -----------------------------------------------------------*/
/** @defgroup HSP_CNN_Private_Defines HSP_HW_CNN Private Defines
  * @{
  */
#define HSP_CNN_CDEG_EVT 27   /**< CDEG event number used for ARM/HSP sync during CNN direct functions */
/**
  * @}
  */

/* Private functions -----------------------------------------------------------*/
static void free_all_ai(hsp_bram_handle_t *hhsp_bram);
static int8_memb_t *alloc_in_memB(hsp_bram_handle_t *hhsp_bram, uint32_t size_in_byte);
static int8_mema_t *alloc_in_memA(hsp_bram_handle_t *hhsp_bram, uint32_t size_in_byte);
static void align_factor_cmsisnn_fast_ch_v3(float32_t in_scale, float32_t out_scale, float32_t wt_scale,
                                            int32_t *p_out_factor, int32_t *p_out_shift);
static void align_factor_cmsisnn_fast_ch_v2(float32_t in_scale, float32_t out_scale, const float32_t *p_wt_scale,
                                            int32_t *p_bias_data, uint16_t ch_im_out, int32_t *p_out);
static void align_factor_cmsisnn_fast_ch(float32_t in_scale, float32_t out_scale,
                                         const float32_t *p_wt_scale, uint16_t ch_im_out, int32_t *p_out);

/** @addtogroup HSP_MODULES_CNN_LIBRARY
  * @{
  */
#ifdef __HSP_DMA__
/**
  * @brief Execute CMSIS CNN Convolution pointwise with coeff fully loaded in memory and
  *        data (input, output) loaded fully
  *
  * @param hmw             HSP handle.
  * @param in_w            Input dimension width
  * @param in_h            Input dimension height
  * @param in_c            Input dimension channel
  * @param ou_w            Output dimension width
  * @param ou_h            Output dimension height
  * @param ou_c            Output dimension channel
  * @param stridex         Stride on X
  * @param stridey         Stride on Y
  * @param *p_input_data   Input data pointer, int8_t data type
  * @param *p_filter_data  Kernel coefficient pointer, int8_t data type
  * @param *p_output_data  Output data pointer, int8_t data type
  * @param *p_bias_data    Bias data pointer, int32_t data type
  * @param in_scale        Input scale
  * @param out_scale       Output scale
  * @param p_wt_scale      Pointer in weight scales (one per output channel)
  * @param off_in          Input offset, int32_t data type
  * @param off_ou          Output offset, int32_t data type
  * @param sat_min         Min sat (Relu), int32_t data type
  * @param sat_max         Max sat (Relu), int32_t data type
  * @retval                Core status.
  * @details
  * Coefficients and data are loaded before running convolution on HSP
  */
hsp_core_status_t HSP_ACC_CnnConvPointwise0_s8(hsp_core_handle_t *hmw,
                                  uint32_t in_w, uint32_t in_h, uint32_t in_c,
                                  uint32_t ou_w, uint32_t ou_h, uint32_t ou_c, uint32_t stridex, uint32_t stridey,
                                  int8_t *p_input_data, int8_t *p_filter_data, int8_t *p_output_data,
                                  int32_t *p_bias_data,
                                  float32_t in_scale, float32_t out_scale, const float32_t *p_wt_scale,
                                  int32_t off_in, int32_t off_ou, int32_t sat_min, int32_t sat_max)
{
  /* First of all, wakeup HSP with Direct command */
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_CNN_CONVPW_I8);

  int8_mema_t *pDst ;
  int8_mema_t *pSrcA;
  int8_memb_t *pSrcB;
  int32_mema_t *pQS;
  int32_mema_t *pBias;
  uint32_t jump_in = 0;
  uint32_t moreChanIn = (0x4U - (in_c & 0x3U)) & 0x3U;

#ifdef USE_HSP_CNN_CRITICAL_SECTION
  /* Save current EVTENR and clear all enable event for ARM/HSP CNN synchro */
  uint32_t evtenrSave = HSP_HW_IF_EVENT_GetStatus(hmw->hsp_id); /* HSPx->EVTENR; */
  HSP_HW_IF_EVENT_Disable(hmw->hsp_id, 0xFFFFFFFFUL); /* HSPx->EVTENR = HSP_CNN_EVTENR_CLR; */
#endif /*  USE_HSP_CNN_CRITICAL_SECTION */

  /* @todo */
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->ITFENR = 0xE0000000;

  if (stridey != 1)
  {
    jump_in = (in_c * in_w);
  }
  /* Allocate and copy input data in MemA */
  if ((pSrcA = (int8_mema_t *)alloc_in_memA(&(hmw->hbram), (in_c * in_w * in_h))) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  /* Allocate output data buffer in MemA */
  if ((pDst = (int8_mema_t *)alloc_in_memA(&(hmw->hbram), (ou_c * ou_w * ou_h))) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  /* Allocate and copy bias data in MemA */
  if ((pBias = (int32_mema_t *)alloc_in_memA(&(hmw->hbram), ou_c * sizeof(uint32_t))) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  /* Allocate and copy kernel data in MemB */
  if ((pSrcB = (int8_memb_t *)alloc_in_memB(&(hmw->hbram), (ou_c * (in_c + moreChanIn)))) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  /* Allocate and copy quantification data in MemA. interleave quant_params */
  if ((pQS = (int32_mema_t *)alloc_in_memA(&(hmw->hbram), (ou_c * sizeof(uint32_t)) * 2)) == NULL)
  {
    return HSP_CORE_ERROR;
  }

  /* Parameter must be written before accessing blocking register */
  HSP_HW_IF_WRITE_PARAMR0(in_w);
  HSP_HW_IF_WRITE_PARAMR1(in_h);
  HSP_HW_IF_WRITE_PARAMR2(in_c);
  HSP_HW_IF_WRITE_PARAMR3(ou_w);
  HSP_HW_IF_WRITE_PARAMR4(ou_h);
  HSP_HW_IF_WRITE_PARAMR5(ou_c);
  HSP_HW_IF_WRITE_PARAMR7(stridex);
  HSP_HW_IF_WRITE_PARAMR8(stridey);
  HSP_HW_IF_WRITE_PARAMR9((uint32_t)off_in);
  HSP_HW_IF_WRITE_PARAMR10((uint32_t)off_ou);
  HSP_HW_IF_WRITE_PARAMR11((uint32_t)sat_min);
  HSP_HW_IF_WRITE_PARAMR12((uint32_t)sat_max);
  HSP_HW_IF_WRITE_PARAMR13((uint32_t)pBias);
  HSP_HW_IF_WRITE_PARAMR14((uint32_t)pQS);
  HSP_HW_IF_WRITE_PARAMR15(HSP_CNN_CFG_MODE_0STEP);

  int8_mema_t *pStartSrc = (int8_mema_t *)pSrcA;
  int8_mema_t *pCurrSrc = pStartSrc;
  int8_mema_t *pStartDst = (int8_mema_t *)pDst;
  int8_mema_t *pCurrDst = pStartDst;

  HSP_HW_IF_WRITE_DCMDPTR0((uint32_t)pSrcA);
  HSP_HW_IF_WRITE_DCMDPTR1((uint32_t)pSrcB);
  HSP_HW_IF_WRITE_DCMDPTR2((uint32_t)pDst);

  int8_t *pTmpOutput = (int8_t *)p_output_data;
  uint32_t inLineSize = (in_c * in_w);
  int8_t *pInput8 = (int8_t *)p_input_data;

  align_factor_cmsisnn_fast_ch(in_scale, out_scale, p_wt_scale, (uint16_t)ou_c, (int32_t *)pQS);

  /* Load all coeffs */
  if (moreChanIn)
  {
    int8_memb_t *pDstCoeffTmp = pSrcB;
    int8_t *pInCoeffTmp = p_filter_data;
    int8_t zeroArray[3] = {0, 0, 0};
    for (uint32_t i = 0; i < ou_c; i++)
    {
      /* For each coeff add moreChanIn 0 coefficient */
      HSP_MEMCPY(pDstCoeffTmp, pInCoeffTmp, in_c);
      pDstCoeffTmp += in_c;
      pInCoeffTmp += in_c;
      /* Add extra 0 coefficients */
      HSP_MEMCPY(pDstCoeffTmp, zeroArray, moreChanIn);
      pDstCoeffTmp += moreChanIn;
    }
  }
  else
  {
    HSP_MEMCPY(pSrcB, p_filter_data, (ou_c * in_c));
  }

  HSP_MEMCPY((int8_t *)pBias, (int8_t *)p_bias_data, (ou_c * sizeof(uint32_t)));
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
  if (HAL_HSP_GetFirmwareError(((HSP_HandleTypeDef *)(hmw->hdriver)))))
  {
    /* Free all memory */
    free_all_ai(&(hmw->hbram));
    return HSP_CORE_ERROR;
  }
  /* Clear sem before send CNN event */
  HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver))));
#ifdef STM32H7P5xx
  __DSB();
#endif  /* STM32H7P5xx */

  /* Load only necessary lines (depending of stride) in circular buffer. */
  for (uint32_t i = 0; i < ou_h; i++)
  {
    HSP_MEMCPY(pCurrSrc, pInput8, (in_w * in_c));
    pInput8 +=  inLineSize + jump_in;
    pCurrSrc +=  inLineSize;
  }
  /* Start working: go */
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->CDEGR = HSP_CNN_CDEG_EVT; /* send CNN event */

  /*  Wait output line  */
  while (HAL_HSP_MSGBOX_IsMsgAvailable(((HSP_HandleTypeDef *)(hmw->hdriver))) == 0U); /* kernel ready */

  /*  Clear sem  */
  HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
  __DSB();
#endif  /* STM32H7P5xx */

  /*  Copy output from HSP to CPU buffer  */
  HSP_MEMCPY(pTmpOutput, pCurrDst, (ou_h * ou_w * ou_c));

#ifdef CNN_EVTENR_SUPPORT
  HSP_HW_IF_EVENT_Enable(evtenrSave); /* Restore previous EVTENR value  */
#endif /*  CNN_EVTENR_SUPPORT  */
  /* Free all memory */
  free_all_ai(&(hmw->hbram));
  return HSP_CORE_OK;
}
#endif /* __HSP_DMA__ */

/**
  * @brief Execute CMSIS CNN Convolution pointwise with coeff fully loaded in memory and
  *        data (input, output) loaded in circular buffer
  *
  * @param hmw                  HSP handle.
  * @param in_w                 Input dimension width
  * @param in_h                 Input dimension height
  * @param in_c                 Input dimension channel
  * @param ou_w                 Output dimension width
  * @param ou_h                 Output dimension height
  * @param ou_c                 Output dimension channel
  * @param stridex              Stride on X
  * @param stridey              Stride on Y
  * @param *p_input_data        Input data pointer, int8_t data type
  * @param *p_filter_data       Kernel coefficient pointer, int8_t data type
  * @param *p_output_data       Output data pointer, int8_t data type
  * @param *p_bias_data         Bias data pointer, int32_t data type
  * @param in_scale             Input scale
  * @param out_scale            Output scale
  * @param p_wt_scale           Pointer in weight scales (one per output channel)
  * @param off_in               Input offset, int32_t data type
  * @param off_ou               Output offset, int32_t data type
  * @param sat_min              Min sat (Relu), int32_t data type
  * @param sat_max              Max sat (Relu), int32_t data type
  * @param nb_line_per_blocks   Nb line to output in buffer
  * @retval                     Core status.
  * @details
  * Coefficients and data are loaded before running convolution on HSP
  * This variant support channel <= 4
  * @retval               None
  */
hsp_core_status_t HSP_ACC_CnnConvPointwise1_s8(hsp_core_handle_t *hmw,
                                  uint32_t in_w, uint32_t in_h, uint32_t in_c,
                                  uint32_t ou_w, uint32_t ou_h, uint32_t ou_c, uint32_t stridex, uint32_t stridey,
                                  int8_t *p_input_data, int8_t *p_filter_data, int8_t *p_output_data,
                                  int32_t *p_bias_data,
                                  float32_t in_scale, float32_t out_scale, const float32_t *p_wt_scale,
                                  int32_t off_in, int32_t off_ou, int32_t sat_min, int32_t sat_max)
{
  /* First of all, wakeup HSP with Direct command */
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_CNN_CONVPW_I8);

  int8_mema_t *pDst ;
  int8_mema_t *pSrcA;
  int8_memb_t *pSrcB;
  int32_mema_t *pBias;
  int32_mema_t *pQS;
  uint32_t jump_in = 0;
  uint32_t moreChanIn = (0x4U - (in_c & 0x3U)) & 0x3U;

#ifdef CNN_EVTENR_SUPPORT
  /*  Save current EVTENR and clear all enable event for ARM/HSP CNN synchro */
  uint32_t evtenrSave = ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->EVTENR;
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->EVTENR = HSP_CNN_EVTENR_CLR;
#endif /* CNN_EVTENR_SUPPORT */

  /* @todo */
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->ITFENR = 0xE0000000;

  if (stridey != 1)
  {
    jump_in = in_c * in_w;
  }

  /* Allocate and copy input data in MemA: 2 input lines in circular buffer  */
  if ((pSrcA = (int8_mema_t *)alloc_in_memA(&(hmw->hbram), ((in_c * in_w)) * 2)) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  /* Allocate output data buffer in MemA: ping pong buffer on destination  */
  if ((pDst = (int8_mema_t *)alloc_in_memA(&(hmw->hbram), ((ou_c * ou_w)) * 2)) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  /* Allocate and copy bias data in MemA */
  if ((pBias = (int32_mema_t *)alloc_in_memA(&(hmw->hbram), ou_c * sizeof(uint32_t))) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  /* Allocate and copy kernel data in MemB */
  if ((pSrcB = (int8_memb_t *)alloc_in_memB(&(hmw->hbram), (ou_c * (in_c + moreChanIn)))) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  /* Allocate and copy quantification data in MemA interleave quant_params */
  if ((pQS = (int32_mema_t *)alloc_in_memA(&(hmw->hbram), (ou_c * sizeof(uint32_t)) * 2)) == NULL)
  {
    return HSP_CORE_ERROR;
  }

  /* Parameter must be written before accessing blocking register */
  HSP_HW_IF_WRITE_PARAMR0(in_w);
  HSP_HW_IF_WRITE_PARAMR1(in_h);
  HSP_HW_IF_WRITE_PARAMR2(in_c);
  HSP_HW_IF_WRITE_PARAMR3(ou_w);
  HSP_HW_IF_WRITE_PARAMR4(ou_h);
  HSP_HW_IF_WRITE_PARAMR5(ou_c);
  HSP_HW_IF_WRITE_PARAMR7(stridex);
  HSP_HW_IF_WRITE_PARAMR8(stridey);
  HSP_HW_IF_WRITE_PARAMR9((uint32_t)off_in);
  HSP_HW_IF_WRITE_PARAMR10((uint32_t)off_ou);
  HSP_HW_IF_WRITE_PARAMR11((uint32_t)sat_min);
  HSP_HW_IF_WRITE_PARAMR12((uint32_t)sat_max);
  HSP_HW_IF_WRITE_PARAMR13((uint32_t)pBias);
  HSP_HW_IF_WRITE_PARAMR14((uint32_t)pQS);
  /* Choose the circular size buffer */
  HSP_HW_IF_WRITE_PARAMR15((2 << HSP_CNN_CFG_NB_IN_LINE_SHIFT) | HSP_CNN_CFG_MODE_1STEP);

  int8_mema_t *pStartSrc = (int8_mema_t *)pSrcA;
  int8_mema_t *pCurrSrc = pStartSrc; /* input */
  int8_mema_t *pStartDst = (int8_mema_t *)pDst; /* pDst */
  int8_mema_t *pCurrDst = pStartDst;
  int8_mema_t *pEndSrc;
  int8_mema_t *pEndDst;

  pEndSrc = pStartSrc + (2 * (in_c * in_w));
  pEndDst = pStartDst + (2 * (ou_c * ou_w));

  HSP_HW_IF_WRITE_DCMDPTR0((uint32_t)pSrcA);
  HSP_HW_IF_WRITE_DCMDPTR1((uint32_t)pSrcB);
  HSP_HW_IF_WRITE_DCMDPTR2((uint32_t)pDst);

  int8_t *pTmpOutput = (int8_t *)p_output_data;
  uint32_t inLineSize = (in_c * in_w);
  uint32_t nbLine = (in_h + (stridey - 1)) / stridey;
  uint32_t outLineSize = (ou_c * ou_w);
  int8_t *pInput8 = (int8_t *)p_input_data;

  align_factor_cmsisnn_fast_ch(in_scale, out_scale, p_wt_scale, (uint16_t)ou_c, (int32_t *)pQS);

  /* Load all coeffs */
  if (moreChanIn)
  {
    int8_memb_t *pDstCoeffTmp = pSrcB;
    int8_t *pInCoeffTmp = p_filter_data;
    int8_t zeroArray[3] = {0, 0, 0};
    for (uint32_t i = 0; i < ou_c; i++)
    {
      /* For each coeff add moreChanIn 0 coefficient */
      HSP_MEMCPY(pDstCoeffTmp, pInCoeffTmp, in_c);
      pDstCoeffTmp += in_c;
      pInCoeffTmp += in_c;
      /* Add extra 0 coefficients */
      HSP_MEMCPY(pDstCoeffTmp, zeroArray, moreChanIn);
      pDstCoeffTmp += moreChanIn;
    }
  }
  else
  {
    HSP_MEMCPY(pSrcB, p_filter_data, (ou_c  * in_c));
  }

  HSP_MEMCPY((int8_t *)pBias, (int8_t *)p_bias_data, (ou_c * sizeof(uint32_t)));

  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
  if (HAL_HSP_GetFirmwareError(((HSP_HandleTypeDef *)(hmw->hdriver))))
  {
    /* Free all CNN allocated memory */
    free_all_ai(&(hmw->hbram));
    return HSP_CORE_ERROR;
  }
  /* Clear sem before send event */
  HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
  __DSB();
#endif  /* STM32H7P5xx */

  /* Load 2 lines in circular buffer with stride_h respect */
  HSP_MEMCPY(pCurrSrc, pInput8, inLineSize);
  pInput8 +=  inLineSize + jump_in;
  pCurrSrc +=  inLineSize;
  HSP_MEMCPY(pCurrSrc, pInput8, inLineSize);
  pInput8 +=  inLineSize + jump_in;  /* Jump over stride */
  pCurrSrc = pStartSrc;

  /* Start working: go */
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->CDEGR = HSP_CNN_CDEG_EVT; /* send CNN event */

  for (uint32_t i = 0; i < nbLine - 1; i++)
  {
    /* Wait output line */
    while (HAL_HSP_MSGBOX_IsMsgAvailable(((HSP_HandleTypeDef *)(hmw->hdriver))) == 0U); /* kernel ready */
    /* Clear sem */
    HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
  __DSB();
#endif  /* STM32H7P5xx */
    /* Go */
    ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->CDEGR = HSP_CNN_CDEG_EVT; /* send CNN event */

    /* Load next line in circular buffer while HSP is working */
    HSP_MEMCPY(pCurrSrc, pInput8, (inLineSize));
    pCurrSrc += inLineSize;
    pInput8 +=  inLineSize;
    /* Jump over stride */
    pInput8 += jump_in;
    if (pCurrSrc == pEndSrc)
    {
      pCurrSrc = pStartSrc;
    }
    /* Copy output while HSP is working */
    HSP_MEMCPY(pTmpOutput, pCurrDst, outLineSize);
    pTmpOutput += outLineSize;
    pCurrDst += outLineSize;
    if (pCurrDst == pEndDst)
    {
      /* Go back to start buffer */
      pCurrDst = pStartDst;
    }
  }
  /* Wait output line */
  while (HAL_HSP_MSGBOX_IsMsgAvailable(((HSP_HandleTypeDef *)(hmw->hdriver))) == 0U); /* kernel ready */
  /* Clear sem */
  HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
  __DSB();
#endif  /* STM32H7P5xx */
  /* Copy output from HSP to CPU buffer */
  HSP_MEMCPY(pTmpOutput, pCurrDst, outLineSize);
#ifdef CNN_EVTENR_SUPPORT
  HSP_HW_IF_EVENT_Enable(evtenrSave); /* Restore previous EVTENR value  */
#endif /* CNN_EVTENR_SUPPORT */

  /* Free all CNN allocated memory */
  free_all_ai(&(hmw->hbram));
  return HSP_CORE_OK;
}

/**
  * @brief Execute CMSIS CNN Convolution pointwise with coeff fully loaded in memory by block and data (input, output)
  *        partially loaded in memory
  *        First loop is on kernel to load all kernels, while HSP working on first channel
  *        Then loop on all other channels
  *
  * @param hmw                  HSP handle.
  * @param in_w                 Input dimension width
  * @param in_h                 Input dimension height
  * @param in_c                 Input dimension channel
  * @param ou_w                 Output dimension width
  * @param ou_h                 Output dimension height
  * @param ou_c                 Output dimension channel
  * @param stridex              Stride on X
  * @param stridey              Stride on Y
  * @param *p_input_data        Input data pointer, int8_t data type
  * @param *p_filter_data       Kernel coefficient pointer, int8_t data type
  * @param *p_output_data       Output data pointer, int8_t data type
  * @param *p_bias_data         Bias data pointer, int32_t data type
  * @param in_scale             Input scale
  * @param out_scale            Output scale
  * @param p_wt_scale           Pointer in weight scales (one per output channel)
  * @param off_in               Input offset, int32_t data type
  * @param off_ou               Output offset, int32_t data type
  * @param sat_min              Min sat (Relu), int32_t data type
  * @param sat_max              Max sat (Relu), int32_t data type
  * @param nb_line_per_blocks   Nb line to output in buffer
  * @retval                     Core status.
  * @details
  * This variant does not support channel <= 4
  */
hsp_core_status_t HSP_ACC_CnnConvPointwise2_s8(hsp_core_handle_t *hmw,
                                  uint32_t in_w, uint32_t in_h, uint32_t in_c,
                                  uint32_t ou_w, uint32_t ou_h, uint32_t ou_c, uint32_t stridex, uint32_t stridey,
                                  int8_t *p_input_data, int8_t *p_filter_data, int8_t *p_output_data,
                                  int32_t *p_bias_data,
                                  float32_t in_scale, float32_t out_scale, const float32_t *p_wt_scale,
                                  int32_t off_in, int32_t off_ou, int32_t sat_min, int32_t sat_max,
                                  uint32_t nb_line_per_blocks)
{
  /* First of all, wakeup HSP with Direct command */
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_CNN_CONVPW_I8);

#ifdef CNN_EVTENR_SUPPORT
  /* Save current EVTENR and clear all enable event for ARM/HSP CNN synchro */
  uint32_t evtenrSave = ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->EVTENR;
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->EVTENR = HSP_CNN_EVTENR_CLR;
#endif /* CNN_EVTENR_SUPPORT */

  /* @todo */
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->ITFENR = 0xE0000000;

  int8_mema_t *pDst ;
  int8_mema_t *pSrcA;
  int8_memb_t *pSrcB;
  int32_mema_t *pBias;
  int32_mema_t *pQS;

  uint32_t nbinLine = 2; /* For circular mode */
  uint32_t circiSize = (in_c * in_w) * nbinLine;
  uint32_t moreChanIn = (0x4U - (in_c & 0x3U)) & 0x3U;

  /* Allocate and copy input data in MemA */
  if ((pSrcA = (int8_mema_t *)alloc_in_memA(&(hmw->hbram), circiSize)) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  /* Allocate output data buffer in MemA */
  if ((pDst = (int8_mema_t *)alloc_in_memA(&(hmw->hbram), (ou_c * ou_w) * nb_line_per_blocks)) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  /* Allocate and copy bias data in MemA */
  if ((pBias = (int32_mema_t *)alloc_in_memA(&(hmw->hbram), ou_c * sizeof(uint32_t))) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  /* Allocate and copy kernel data in MemB */
  if ((pSrcB = (int8_memb_t *)alloc_in_memB(&(hmw->hbram), (ou_c * (in_c + moreChanIn)))) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  /* Allocate and copy quantification data in MemA interleave quant_params */
  if ((pQS = (int32_mema_t *)alloc_in_memA(&(hmw->hbram), (ou_c * sizeof(uint32_t)) * 2)) == NULL)
  {
    return HSP_CORE_ERROR;
  }

  /* Parameter must be written before accessing blocking register */
  HSP_HW_IF_WRITE_PARAMR0(in_w);
  HSP_HW_IF_WRITE_PARAMR1(in_h);
  HSP_HW_IF_WRITE_PARAMR2(in_c);
  HSP_HW_IF_WRITE_PARAMR3(ou_w);
  HSP_HW_IF_WRITE_PARAMR4(ou_h);
  HSP_HW_IF_WRITE_PARAMR5(ou_c);
  HSP_HW_IF_WRITE_PARAMR7(stridex);
  HSP_HW_IF_WRITE_PARAMR8(stridey);
  HSP_HW_IF_WRITE_PARAMR9((uint32_t)off_in);
  HSP_HW_IF_WRITE_PARAMR10((uint32_t)off_ou);
  HSP_HW_IF_WRITE_PARAMR11((uint32_t)sat_min);
  HSP_HW_IF_WRITE_PARAMR12((uint32_t)sat_max);
  HSP_HW_IF_WRITE_PARAMR13((uint32_t)pBias);
  HSP_HW_IF_WRITE_PARAMR14((uint32_t)pQS);
  /* Set circular buffer size and pw cmd mode */
  HSP_HW_IF_WRITE_PARAMR15((nb_line_per_blocks << HSP_CNN_CFG_NB_OU_LINE_SHIFT) |
                           (nbinLine << HSP_CNN_CFG_NB_IN_LINE_SHIFT) | HSP_CNN_CFG_MODE_2STEP);
  HSP_HW_IF_WRITE_DCMDPTR0((uint32_t)pSrcA);
  HSP_HW_IF_WRITE_DCMDPTR1((uint32_t)pSrcB);
  HSP_HW_IF_WRITE_DCMDPTR2((uint32_t)pDst);

  align_factor_cmsisnn_fast_ch(in_scale, out_scale, p_wt_scale, (uint16_t)ou_c, (int32_t *)pQS);

  HSP_MEMCPY((int8_t *)pBias, (int8_t *)p_bias_data, (ou_c * sizeof(uint32_t)));

  /* Load only first channel out coeffs */
  int8_t *pKe = p_filter_data;

  if (moreChanIn)
  {
    int8_memb_t *pDstCoeffTmp = pSrcB;
    int8_t *pInCoeffTmp = p_filter_data;
    int8_t zeroArray[3] = {0, 0, 0};
    /* Add moreChanIn 0 in first set of coefficient */
    HSP_MEMCPY(pDstCoeffTmp, pInCoeffTmp, in_c);
    pDstCoeffTmp += in_c;
    pInCoeffTmp += in_c;
    /* Add extra 0 coefficients */
    HSP_MEMCPY(pDstCoeffTmp, zeroArray, moreChanIn);
    pDstCoeffTmp += moreChanIn;
  }
  else
  {
    HSP_MEMCPY(pSrcB, pKe, in_c);
  }

  /* Load first input tile */
  int8_t *pIn = p_input_data;
  uint32_t inLineSize8 = in_c * in_w; /* 1 input line */
  HSP_MEMCPY(pSrcA, pIn, (inLineSize8));
  pIn += (inLineSize8  * stridey);
  int8_t *pOu = p_output_data;

  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
  if (HAL_HSP_GetFirmwareError(((HSP_HandleTypeDef *)(hmw->hdriver))))
  {
    /* Free all CNN allocated memory */
    free_all_ai(&(hmw->hbram));
    return HSP_CORE_ERROR;
  }
  uint32_t outLineSize8 = ou_c * ou_w; /* 1 output line */
  uint8_t *pKerCur = (uint8_t *)pKe + in_c;
  uint8_t *pSrcbCur = (uint8_t *)pSrcB + in_c;
  uint8_t *pSrcEnd = (uint8_t *)pSrcA + (circiSize);
  uint8_t *pSrcCur = (uint8_t *)pSrcA + (inLineSize8);
  uint8_t *pDstEnd = (uint8_t *)(pDst + (nb_line_per_blocks * outLineSize8));
  uint8_t *pDstCur = (uint8_t *) pDst;
  /* Clear sem */
  HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
  __DSB();
#endif  /* STM32H7P5xx */
  /* Send event through CDEG: WARNING: do we need to disable all other events???  */
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->CDEGR = HSP_CNN_CDEG_EVT; /* send event */

  /* Loop to load all kernels in BRAM */
  if (moreChanIn)
  {
    int8_memb_t *pDstCoeffTmp = pSrcB  + in_c +  moreChanIn;
    int8_t *pInCoeffTmp = (int8_t *)pKerCur;
    int8_t zeroArray[3] = {0, 0, 0};
    for (uint32_t i = 1; i < ou_c; i++)
    {
      /* For each coeff add moreChanIn 0 coefficient */
      HSP_MEMCPY(pDstCoeffTmp, pInCoeffTmp, in_c);
      pDstCoeffTmp += in_c;
      pInCoeffTmp += in_c;
      /* Add extra 0 coefficients */
      HSP_MEMCPY(pDstCoeffTmp, zeroArray, moreChanIn);
      pDstCoeffTmp += moreChanIn;
      /* Wait output channel done */
      while (HAL_HSP_MSGBOX_IsMsgAvailable(((HSP_HandleTypeDef *)(hmw->hdriver))) == 0U); /* kernel ready */
      /* Clear sem */
      HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
      __DSB();
#endif  /* STM32H7P5xx */
      /* Send event through CDEG: WARNING: do we need to disable all other events??? */
      ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->CDEGR = HSP_CNN_CDEG_EVT; /* send event */
    }
  }
  else
  {
    for (uint32_t out_num = 1; out_num < ou_c; out_num++)
    {
      /* Copy kernel block */
      HSP_MEMCPY((int8_t *)pSrcbCur, (int8_t *)pKerCur, in_c);
      pKerCur += in_c;
      pSrcbCur += in_c;
      /* Wait output channel done */
      while (HAL_HSP_MSGBOX_IsMsgAvailable(((HSP_HandleTypeDef *)(hmw->hdriver))) == 0U); /* kernel ready */
      /* Clear sem */
      HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
      __DSB();
#endif  /* STM32H7P5xx */
      /* Send event through CDEG: WARNING: do we need to disable all other events??? */
      ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->CDEGR = HSP_CNN_CDEG_EVT; /* send event */
    }
  }

  for (uint32_t lineIdx = 0; lineIdx < (ou_h - 1); lineIdx++)
  {
    /* Copy input line in circular buffer */
    HSP_MEMCPY((int8_t *)pSrcCur, (int8_t *)pIn, inLineSize8);
    pIn += (inLineSize8 * stridey);
    pSrcCur += inLineSize8;
    if (pSrcCur == pSrcEnd)
    {
      pSrcCur = (uint8_t *)pSrcA;
    }
    /* Wait output line */
    while (HAL_HSP_MSGBOX_IsMsgAvailable(((HSP_HandleTypeDef *)(hmw->hdriver))) == 0U); /* kernel ready */
    /* Clear sem */
    HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
    __DSB();
#endif  /* STM32H7P5xx */
    /* Send event through CDEG: WARNING: do we need to disable all other events??? */
    ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->CDEGR = HSP_CNN_CDEG_EVT; /* send event */
    HSP_MEMCPY((int8_t *)pOu, (int8_t *)pDstCur, outLineSize8);
    pOu += outLineSize8;
    pDstCur += outLineSize8;
    if (pDstCur == pDstEnd)
    {
      pDstCur = (uint8_t *)pDst;
    }
  }
  /* Wait last output line */
  while (HAL_HSP_MSGBOX_IsMsgAvailable(((HSP_HandleTypeDef *)(hmw->hdriver))) == 0U); /* kernel ready */
  /* Clear sem */
  HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
  __DSB();
#endif  /* STM32H7P5xx */
  HSP_MEMCPY((int8_t *)pOu, (int8_t *)pDstCur, outLineSize8);
#ifdef CNN_EVTENR_SUPPORT
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->EVTENR = evtenrSave; /* Restore previous EVTENR value */
#endif /* CNN_EVTENR_SUPPORT */

  /* Free all memory */
  free_all_ai(&(hmw->hbram));
  return HSP_CORE_OK;
}

/**
  * @brief Execute CMSIS CNN Convolution pointwise in 3 steps. Circular input in MEMA and output in MEMB
  *        and ping pong on kernels in MEMB
  *        First step: loop on all input lines with first kernel to get lines from ARM, sync on input line
  *        Second step: loop on all channel (exclude first and last) as classical cio sync by output channel
  *        Third step: finally loop on last channel, sync on output line
  *        HSP send error report via FWERR register
  *        Then play again 3 steps if necessary
  *
  * @param hmw                  HSP handle.
  * @param in_w                 Input dimension width
  * @param in_h                 Input dimension height
  * @param in_c                 Input dimension channel
  * @param ou_w                 Output dimension width
  * @param ou_h                 Output dimension height
  * @param ou_c                 Output dimension channel
  * @param stridex              Stride on X
  * @param stridey              Stride on Y
  * @param *p_input_data        Input data pointer, int8_t data type
  * @param *p_filter_data       Kernel coefficient pointer, int8_t data type
  * @param *p_output_data       Output data pointer, int8_t data type
  * @param *p_bias_data         Bias data pointer, int32_t data type
  * @param in_scale             Input scale
  * @param out_scale            Output scale
  * @param p_wt_scale           Pointer in weight scales (one per output channel)
  * @param off_in               Input offset, int32_t data type
  * @param off_ou               Output offset, int32_t data type
  * @param sat_min              Min sat (Relu), int32_t data type
  * @param sat_max              Max sat (Relu), int32_t data type
  * @param nb_line_per_blocks   Nb line to output in buffer
  * @retval                     Core status.
  * @details
  * This variant does not support channel <= 4
  *
  */
hsp_core_status_t HSP_ACC_CnnConvPointwise3_s8(hsp_core_handle_t *hmw,
                                  uint32_t in_w, uint32_t in_h, uint32_t in_c,
                                  uint32_t ou_w, uint32_t ou_h, uint32_t ou_c, uint32_t stridex, uint32_t stridey,
                                  int8_t *p_input_data, int8_t *p_filter_data, int8_t *p_output_data,
                                  int32_t *p_bias_data,
                                  float32_t in_scale, float32_t out_scale, const float32_t *p_wt_scale,
                                  int32_t off_in, int32_t off_ou, int32_t sat_min, int32_t sat_max,
                                  uint32_t nb_line_per_blocks)
{
  /* First of all, wakeup HSP with Direct command */
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_CNN_CONVPW_I8);

#ifdef CNN_EVTENR_SUPPORT
  /* Save current EVTENR and clear all enable event for ARM/HSP CNN synchro */
  uint32_t evtenrSave = ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->EVTENR;
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->EVTENR = HSP_CNN_EVTENR_CLR;
#endif /* CNN_EVTENR_SUPPORT */

  /* @todo */
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->ITFENR = 0xE0000000;

  int8_memb_t *pDst;
  int8_mema_t *pSrcA;
  int8_memb_t *pSrcB;
  int32_mema_t *pBias;
  int32_mema_t *pQS;
  uint32_t moreChanIn = (0x4U - (in_c & 0x3U)) & 0x3U;

  uint32_t circiSize = nb_line_per_blocks * (in_c * in_w); /* For circular mode */
  /* Allocate and copy input data in MemA */
  if ((pSrcA = (int8_mema_t *)alloc_in_memA(&(hmw->hbram), circiSize)) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  /* Allocate and copy bias data in MemA */
  if ((pBias = (int32_mema_t *)alloc_in_memA(&(hmw->hbram), ou_c * sizeof(uint32_t))) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  /* Allocate and copy kernel data in MemB : Only 2 kernels in ping pong */
  if ((pSrcB = (int8_memb_t *)alloc_in_memB(&(hmw->hbram), ((in_c + moreChanIn) * 2))) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  /* Allocate output data buffer in MemB */
  if ((pDst = (int8_memb_t *)alloc_in_memB(&(hmw->hbram), (ou_c * ou_w) * nb_line_per_blocks)) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  /* Allocate and copy quantification data in MemA interleave quant_params */
  if ((pQS = (int32_mema_t *)alloc_in_memA(&(hmw->hbram), (ou_c * sizeof(uint32_t)) * 2)) == NULL)
  {
    return HSP_CORE_ERROR;
  }

  HSP_HW_IF_WRITE_PARAMR0(in_w);
  HSP_HW_IF_WRITE_PARAMR1(in_h);
  HSP_HW_IF_WRITE_PARAMR2(in_c);
  HSP_HW_IF_WRITE_PARAMR3(ou_w);
  HSP_HW_IF_WRITE_PARAMR4(ou_h);
  HSP_HW_IF_WRITE_PARAMR5(ou_c);
  HSP_HW_IF_WRITE_PARAMR7(stridex);
  HSP_HW_IF_WRITE_PARAMR8(stridey);
  HSP_HW_IF_WRITE_PARAMR9((uint32_t)off_in);
  HSP_HW_IF_WRITE_PARAMR10((uint32_t)off_ou);
  HSP_HW_IF_WRITE_PARAMR11((uint32_t)sat_min);
  HSP_HW_IF_WRITE_PARAMR12((uint32_t)sat_max);
  HSP_HW_IF_WRITE_PARAMR13((uint32_t)pBias);
  HSP_HW_IF_WRITE_PARAMR14((uint32_t)pQS);
  /* Set circular buffer size and pw cmd mode */
  HSP_HW_IF_WRITE_PARAMR15((nb_line_per_blocks << HSP_CNN_CFG_NB_IN_LINE_SHIFT) | HSP_CNN_CFG_MODE_3STEP);
  HSP_HW_IF_WRITE_DCMDPTR0((uint32_t)pSrcA);
  HSP_HW_IF_WRITE_DCMDPTR1((uint32_t)pSrcB);
  HSP_HW_IF_WRITE_DCMDPTR2((uint32_t)pDst);

  align_factor_cmsisnn_fast_ch(in_scale, out_scale, p_wt_scale, (uint16_t)ou_c, (int32_t *)pQS);

  HSP_MEMCPY((int8_t *)pBias, (int8_t *)p_bias_data, (ou_c * sizeof(uint32_t)));
  /* Load only first channel out coeffs */
  int8_t *pKe = p_filter_data;

  int8_t zeroArray[3] = {0, 0, 0};
  if (moreChanIn)
  {
    int8_memb_t *pDstCoeffTmp = pSrcB;
    int8_t *pInCoeffTmp = p_filter_data;
    int8_t zeroArray[3] = {0, 0, 0};
    /* Add moreChanIn 0 in first set of coefficient */
    HSP_MEMCPY(pDstCoeffTmp, pInCoeffTmp, in_c);
    pDstCoeffTmp += in_c;
    pInCoeffTmp += in_c;
    /* Add extra 0 coefficients */
    HSP_MEMCPY(pDstCoeffTmp, zeroArray, moreChanIn);
    pDstCoeffTmp += moreChanIn;
  }
  else
  {
    HSP_MEMCPY(pSrcB, pKe, in_c);
  }
  /* Load first input tile */
  int8_t *pIn = p_input_data;
  uint32_t inLineSize8 = in_c * in_w; /* 1 input line */
  HSP_MEMCPY(pSrcA, pIn, (inLineSize8));
  pIn += (inLineSize8  * stridey);
  int8_t *pOu = p_output_data;

  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
  if (HAL_HSP_GetFirmwareError(((HSP_HandleTypeDef *)(hmw->hdriver))))
  {
    /* Free all CNN allocated memory */
    free_all_ai(&(hmw->hbram));
    return HSP_CORE_ERROR;
  }
  uint32_t outLineSize8 = ou_c * ou_w; /* 1 output line */
  /* Clear sem */
  HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
  __DSB();
#endif  /* STM32H7P5xx */
  uint8_t *pSrcEnd = (uint8_t *)pSrcA + circiSize;
  uint8_t *pSrcCur = (uint8_t *)pSrcA + inLineSize8;
  uint8_t *pDstCur = (uint8_t *)pDst;
  uint8_t *pDstEnd = (uint8_t *)(pDst + (nb_line_per_blocks * ou_c * ou_w));
  uint8_t *pKerCur = (uint8_t *)pKe + in_c;
  uint8_t *pSrcbCur = (uint8_t *)pSrcB + (in_c + moreChanIn);
  uint8_t *pSrcbEnd = (uint8_t *)pSrcB + ((in_c + moreChanIn) * 2);
  uint32_t nbLinesBlck = 0; /* Init with one block for more simple comparison at end of while */
  uint32_t nbLineLoop = nb_line_per_blocks;

  if (pSrcCur == pSrcEnd)
  {
    pSrcCur = (uint8_t *)pSrcA;
  }
  do
  {
    if (nbLinesBlck)
    {
      if (moreChanIn)
      {
        /* Load all coeffs needed for first channel out */
        HSP_MEMCPY((int8_t *)pSrcbCur, (int8_t *)pKe, in_c);
        pKerCur = (uint8_t *)pKe + in_c;
        pSrcbCur += in_c;
        /* Add extra 0 coefficients */
        HSP_MEMCPY((int8_t *)pSrcbCur, (int8_t *)zeroArray, moreChanIn);
        pSrcbCur += moreChanIn;
      }
      else
      {
        /* Load all coeffs needed for first channel out */
        HSP_MEMCPY((int8_t *)pSrcbCur, (int8_t *)pKe, in_c);
        pKerCur = (uint8_t *)pKe + in_c;
        pSrcbCur += in_c;
      }
      if (pSrcbCur == pSrcbEnd)
      {
        pSrcbCur = (uint8_t *)pSrcB;
      }

      /* Copy input line in circular buffer before enter loop */
      HSP_MEMCPY((int8_t *)pSrcCur, (int8_t *)pIn, inLineSize8);
      pIn += (inLineSize8 * stridey);
      pSrcCur += inLineSize8;
      if (pSrcCur == pSrcEnd)
      {
        pSrcCur = (uint8_t *)pSrcA;
      }
    }
    /* Send event through CDEG: WARNING: do we need to disable all other events??? */
    ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->CDEGR = HSP_CNN_CDEG_EVT; /* send event */
    nbLinesBlck += nbLineLoop;
    if (nbLinesBlck > ou_h)
    {
      /* Count how may output lines not done by block */
      nbLinesBlck -= nbLineLoop; /* Remove next block size */
      nbLineLoop = ou_h - nbLinesBlck;
      nbLinesBlck += nbLineLoop;
    }
    /* First step: load all input block */
    for (uint32_t lineIdx = 1; lineIdx < nbLineLoop; lineIdx++)
    {
      /* Copy input line in circular buffer */
      HSP_MEMCPY((int8_t *)pSrcCur, (int8_t *)pIn, inLineSize8);
      pIn += (inLineSize8 * stridey);
      pSrcCur += inLineSize8;
      if (pSrcCur == pSrcEnd)
      {
        pSrcCur = (uint8_t *)pSrcA;
      }
      /* Wait output line */
      while (HAL_HSP_MSGBOX_IsMsgAvailable(((HSP_HandleTypeDef *)(hmw->hdriver))) == 0U); /* kernel ready */
      /* Clear sem */
      HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
  __DSB();
#endif  /* STM32H7P5xx */
      /* Send event through CDEG: WARNING: do we need to disable all other events??? */
      ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->CDEGR = HSP_CNN_CDEG_EVT;
    }
    /* Whole input block is now in BRAM loop on all channels: Loop on each ping pong kernel */
    for (uint32_t out_num = 1; out_num < ou_c; out_num++)
    {
      if (moreChanIn)
      {
        /* Load all coeffs needed for first channel out */
        HSP_MEMCPY((int8_t *)pSrcbCur, (int8_t *)pKerCur, in_c);
        pKerCur += in_c;
        pSrcbCur += in_c;
        /* Add extra 0 coefficients */
        HSP_MEMCPY((int8_t *)pSrcbCur, (int8_t *)zeroArray, moreChanIn);
        pSrcbCur += moreChanIn;
      }
      else
      {
        /* First of all copy kernel block */
        HSP_MEMCPY((int8_t *)pSrcbCur, (int8_t *)pKerCur, in_c);
        pKerCur += in_c;
        pSrcbCur += in_c;
      }
      if (pSrcbCur == pSrcbEnd)
      {
        pSrcbCur = (uint8_t *)pSrcB;
      }
      /* Wait output channel done */
      while (HAL_HSP_MSGBOX_IsMsgAvailable(((HSP_HandleTypeDef *)(hmw->hdriver))) == 0U); /* kernel ready */
      /* Clear sem */
      HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
  __DSB();
#endif  /* STM32H7P5xx */
      /* Send event through CDEG: WARNING: do we need to disable all other events??? */
      ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->CDEGR = HSP_CNN_CDEG_EVT;
    }

    /* Then read output lines in BRAM */
    for (uint32_t lineIdx = 1; lineIdx < nbLineLoop; lineIdx++)
    {
      /* Wait output line */
      while (HAL_HSP_MSGBOX_IsMsgAvailable(((HSP_HandleTypeDef *)(hmw->hdriver))) == 0U); /* kernel ready */
      /* Clear sem */
      HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
  __DSB();
#endif  /* STM32H7P5xx */
      /* Send event through CDEG: WARNING: do we need to disable all other events??? */
      ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->CDEGR = HSP_CNN_CDEG_EVT;
      HSP_MEMCPY((int8_t *)pOu, (int8_t *)pDstCur, outLineSize8);
      pOu += outLineSize8;
      pDstCur += outLineSize8;
      if (pDstCur == pDstEnd)
      {
        pDstCur = (uint8_t *)pDst;
      }
    }
    /* Wait output line */
    while (HAL_HSP_MSGBOX_IsMsgAvailable(((HSP_HandleTypeDef *)(hmw->hdriver))) == 0U); /* kernel ready */
    /* Clear sem */
    HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
  __DSB();
#endif  /* STM32H7P5xx */
    HSP_MEMCPY((int8_t *)pOu, (int8_t *)pDstCur, outLineSize8);
    pOu += outLineSize8;
    pDstCur += outLineSize8;
    if (pDstCur == pDstEnd)
    {
      pDstCur = (uint8_t *)pDst;
    }
  } while (nbLinesBlck < ou_h); /* End of input block loop */
#ifdef CNN_EVTENR_SUPPORT
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->EVTENR = evtenrSave; /* Restore previous EVTENR value */
#endif /* CNN_EVTENR_SUPPORT */

  /* Free all memory */
  free_all_ai(&(hmw->hbram));
  return HSP_CORE_OK;
}

#ifdef __HSP_DMA__
/**
  * @brief Direct CNN conv2D function with padding laft, right, top, bottom
  * (all sizes except WHC: 1x1x1, 1x2x1, 2x1x1, 1x3x1, 3x1x1, 1x4x1)
  * All input in MEMA and full kernels in MEMB, sync with HSP during HSP output lines
  * HSP send error report via FWERR register
  * @param hmw                 HSP handle.
  * @param in_w                Input dimension width
  * @param in_h                Input dimension height
  * @param in_c                Input dimension channel
  * @param k_w                 Kernel dimension width
  * @param k_h                 Kernel dimension height
  * @param ou_w                Output dimension width
  * @param ou_h                Output dimension height
  * @param ou_c                Output dimension channel
  * @param stridex             Stride on X
  * @param stridey             Stride on Y
  * @param *p_input_data       Input data pointer, int8_t data type
  * @param *p_filter_data      Kernel coefficient pointer, int8_t data type
  * @param *p_output_data      Output data pointer, int8_t data type
  * @param *p_bias_data        Bias data pointer, int32_t data type
  * @param in_scale            Input scale
  * @param out_scale           Output scale
  * @param p_wt_scale          Pointer in weight scales (one per output channel)
  * @param off_in              Input offset, int32_t data type
  * @param off_ou              Output offset, int32_t data type
  * @param sat_min             Min sat (Relu), int32_t data type
  * @param sat_max             Max sat (Relu), int32_t data type
  * @param padx_l              Pad on X left
  * @param padx_r              Pad on X right
  * @param pady_t              Pad on Y top
  * @param pady_b              Pad on Y bottom
  * @retval                    Core status.
 */
hsp_core_status_t HSP_ACC_CnnConv2d0_s8(hsp_core_handle_t *hmw,
                           uint32_t in_w, uint32_t in_h, uint32_t in_c, uint32_t k_w, uint32_t k_h,
                           uint32_t ou_w, uint32_t ou_h, uint32_t ou_c, uint32_t stridex, uint32_t stridey,
                           int8_t *p_input_data, int8_t *p_filter_data, int8_t *p_output_data,
                           uint32_t *p_bias_data,
                           float32_t in_scale, float32_t out_scale, const float32_t *p_wt_scale,
                           uint32_t off_in, uint32_t off_ou, uint32_t sat_min, uint32_t sat_max,
                           uint32_t padx_l, uint32_t padx_r, uint32_t pady_t, uint32_t pady_b)
{
  /* ToDo: check param values: HSP_CHECK_ASSERT_NRV(hhsp, (((((k_w * in_c) + 3) / 4) * k_h) > 1)); */
  uint8_memb_t *pSrcB;
  uint8_mema_t *pSrcA;
  uint8_mema_t *pDst;
  uint32_mema_t *pBQ;

  /* First of all, wakeup HSP with Direct command */
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_CNN_CONV2D_I8);

#ifdef CNN_EVTENR_SUPPORT
  /* Save current EVTENR and clear all enable event for ARM/HSP CNN synchro */
  uint32_t evtenrSave = ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->EVTENR;
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->EVTENR = HSP_CNN_EVTENR_CLR;
#endif /* CNN_EVTENR_SUPPORT */

  /* Then update all parameters */
  uint32_t inp_w = in_w + (padx_l + padx_r);
  uint32_t inp_h = in_h + (pady_t + pady_b);
  uint32_t k_c = in_c * ou_c;
  /* Put kernels in MEMB */
  if ((pSrcB = (uint8_memb_t *)alloc_in_memB(&(hmw->hbram), k_w * k_h * k_c)) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  /* Put input image in MEMA */
  if ((pSrcA = (uint8_mema_t *)alloc_in_memA(&(hmw->hbram), inp_w * inp_h * in_c)) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  /* Put output images in MEMA */
  if ((pDst  = (uint8_mema_t *)alloc_in_memA(&(hmw->hbram), ou_w * ou_h * ou_c)) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  /* Put bias in MEMA */
  if ((pBQ  = (uint32_mema_t *)alloc_in_memA(&(hmw->hbram), (3 * ou_c) * sizeof(uint32_t))) == NULL)
  {
    return HSP_CORE_ERROR;
  }

  HSP_HW_IF_WRITE_PARAMR0(inp_w);
  HSP_HW_IF_WRITE_PARAMR1(inp_h);
  HSP_HW_IF_WRITE_PARAMR2(in_c);
  HSP_HW_IF_WRITE_PARAMR3(k_w);
  HSP_HW_IF_WRITE_PARAMR4(k_h);
  HSP_HW_IF_WRITE_PARAMR5(ou_w);
  HSP_HW_IF_WRITE_PARAMR6(ou_h);
  HSP_HW_IF_WRITE_PARAMR7(ou_c);
  HSP_HW_IF_WRITE_PARAMR8(stridex);
  HSP_HW_IF_WRITE_PARAMR9(stridey);
  HSP_HW_IF_WRITE_PARAMR10(off_in);
  HSP_HW_IF_WRITE_PARAMR11(off_ou);
  HSP_HW_IF_WRITE_PARAMR12(sat_min);
  HSP_HW_IF_WRITE_PARAMR13(sat_max);
  HSP_HW_IF_WRITE_PARAMR14((uint32_t) pBQ);
  HSP_HW_IF_WRITE_PARAMR15(HSP_CNN_CFG_MODE_0STEP);

  /* And finally sync with DCMD using DCMDPTR registers */
  HSP_HW_IF_WRITE_DCMDPTR0((uint32_t)pSrcA);
  HSP_HW_IF_WRITE_DCMDPTR1((uint32_t)pSrcB);
  HSP_HW_IF_WRITE_DCMDPTR2((uint32_t)pDst);

  /* Load all coeffs */
  HSP_MEMCPY(pSrcB, p_filter_data, (k_c * k_h * k_w));

  align_factor_cmsisnn_fast_ch_v2(in_scale, out_scale, p_wt_scale, (int32_t *)p_bias_data, (uint16_t)ou_c,
                                  (int32_t *)pBQ);

  /* Load first input tile */
  int8_t padV = (int8_t) off_in;
  padV = -padV;
  uint8_t val = padV;
  uint8_mema_t *inPad = pSrcA;
  /* First lines pad */
  /* Pad one full dest line */
  HSP_MEMSET(inPad, val, (pady_t *(in_w + (padx_l + padx_r)) * in_c));
  inPad += (pady_t *(in_w + (padx_l + padx_r)) * in_c);
  /* Middle lines to pad */
  for (uint32_t i = 0; i < in_h; i++)
  {
    /* First column pad */
    HSP_MEMSET(inPad, val, (padx_l * in_c));
    inPad += (padx_l * in_c);
    /* Copy image line */
    HSP_MEMCPY(inPad, p_input_data, (in_w * in_c));
    inPad += (in_w * in_c);
    p_input_data += (in_w * in_c);
    /* Last column pad */
    HSP_MEMSET(inPad, val, (padx_r * in_c));
    inPad += (padx_r * in_c);
  }
  /* Last lines pad */
  HSP_MEMSET(inPad, val, (pady_b * (in_w + (padx_l + padx_r)) * in_c));

  /* Wait H2CSEMR for HSP done, but check FWERR first */
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
  if (HAL_HSP_GetFirmwareError(((HSP_HandleTypeDef *)(hmw->hdriver))))
  {
    /* Free all CNN allocated memory */
    free_all_ai(&(hmw->hbram));
    return HSP_CORE_ERROR;
  }
  /* Clear sem before send event */
  HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
  __DSB();
#endif  /* STM32H7P5xx */
  /* Send event through CDEG: WARNING: do we need to disable all other events??? */
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->CDEGR = HSP_CNN_CDEG_EVT;
  /* Wait direct command done */
  while (HAL_HSP_MSGBOX_IsMsgAvailable(((HSP_HandleTypeDef *)(hmw->hdriver))) == 0U); /* kernel ready */
  /* Clear sem */
  HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
  __DSB();
#endif  /* STM32H7P5xx */  

  /* Copy result in external memory */
  HSP_MEMCPY(p_output_data, pDst, (ou_c * ou_h * ou_w));

#ifdef CNN_EVTENR_SUPPORT
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->EVTENR = evtenrSave; /* Restore previous EVTENR value */
#endif /* CNN_EVTENR_SUPPORT */

  /* Free all memory */
  free_all_ai(&(hmw->hbram));
  return HSP_CORE_OK;
}
#endif /* __HSP_DMA__ */

/**
  * @brief Direct CNN conv2D function with circular mode with padding laft, right, top, bottom
  *        (all sizes except WHC: 1x1x1, 1x2x1, 2x1x1, 1x3x1, 3x1x1, 1x4x1)
  * Circular input and ping pong output in MEMA and full kernels in MEMB, sync with HSP during HSP output lines
  * HSP send error report via FWERR register
  * @param hmw                 HSP handle.
  * @param in_w                Input dimension width
  * @param in_h                Input dimension height
  * @param in_c                Input dimension channel
  * @param k_w                 Kernel dimension width
  * @param k_h                 Kernel dimension height
  * @param ou_w                Output dimension width
  * @param ou_h                Output dimension height
  * @param ou_c                Output dimension channel
  * @param stridex             Stride on X
  * @param stridey             Stride on Y
  * @param *p_input_data       Input data pointer, int8_t data type
  * @param *p_filter_data      Kernel coefficient pointer, int8_t data type
  * @param *p_output_data      Output data pointer, int8_t data type
  * @param *p_bias_data        Bias data pointer, int32_t data type
  * @param in_scale            Input scale
  * @param out_scale           Output scale
  * @param p_wt_scale          Pointer in weight scales (one per output channel)
  * @param off_in              Input offset, int32_t data type
  * @param off_ou              Output offset, int32_t data type
  * @param sat_min             Min sat (Relu), int32_t data type
  * @param sat_max             Max sat (Relu), int32_t data type
  * @param padx_l              Pad on X left
  * @param padx_r              Pad on X right
  * @param pady_t              Pad on Y top
  * @param pady_b              Pad on Y bottom
  * @param nb_line_per_blocks  Number of output lines per output buffer - 2 (0 is for ping-pong)
  * @retval                    Core status.
  */
hsp_core_status_t HSP_ACC_CnnConv2d1_s8(hsp_core_handle_t *hmw,
                           uint32_t in_w, uint32_t in_h, uint32_t in_c,
                           uint32_t k_w, uint32_t k_h,
                           uint32_t ou_w, uint32_t ou_h, uint32_t ou_c, uint32_t stridex, uint32_t stridey,
                           int8_t *p_input_data, int8_t *p_filter_data, int8_t *p_output_data,
                           uint32_t *p_bias_data,
                           float32_t in_scale, float32_t out_scale, const float32_t *p_wt_scale,
                           uint32_t off_in, uint32_t off_ou, uint32_t sat_min, uint32_t sat_max,
                           uint32_t padx_l, uint32_t padx_r, uint32_t pady_t, uint32_t pady_b,
                           uint32_t nb_line_per_blocks)
{
 /* ToDo: check param values: @todo HSP_CHECK_ASSERT_NRV(hhsp, (((((k_w * in_c) + 3) / 4) * k_h) > 1));*/
  uint8_memb_t *pSrcB;
  uint8_mema_t *pSrcA;
  uint8_mema_t *pDst;
  uint32_mema_t *pBQ;

  /* First of all, wakeup HSP with Direct command */
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_CNN_CONV2D_I8);

#ifdef CNN_EVTENR_SUPPORT
  /* Save current EVTENR and clear all enable event for ARM/HSP CNN synchro */
  uint32_t evtenrSave = ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->EVTENR;
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->EVTENR = HSP_CNN_EVTENR_CLR;
#endif /* CNN_EVTENR_SUPPORT */

  /* ToDo: enable CDEG interface */
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->ITFENR = 0xE0000000;

  uint32_t pad = padx_l + padx_r + pady_t + pady_b;
  /* Then update all parameters */
  uint32_t inp_w = in_w + (padx_l + padx_r);
  uint32_t inp_h = in_h + (pady_t + pady_b);
  uint32_t nbLinePerBlocks = nb_line_per_blocks + 2;
  if (nbLinePerBlocks > ou_h)
  {
    nbLinePerBlocks = ou_h;
  }
  uint32_t k_c = in_c * ou_c;
  uint32_t inLineSize8 = in_c * inp_w; /* 1 input line */
  uint32_t outLineSize8 = ou_c * ou_w; /* 1 output line */
  uint32_t circiLines = (k_h + stridey); /* For circular mode */
  uint32_t circiSize = inLineSize8 * circiLines; /* For circular mode */

  /* Put kernels in MEMB */
  if ((pSrcB = (uint8_memb_t *)alloc_in_memB(&(hmw->hbram), (ou_c * k_h * ((k_w * in_c) + 3)))) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  /* Put input image in MEMA */
  if ((pSrcA = (uint8_mema_t *)alloc_in_memA(&(hmw->hbram), circiSize)) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  /* Put output images in MEMA */
  if ((pDst = (uint8_mema_t *)alloc_in_memA(&(hmw->hbram), nbLinePerBlocks * outLineSize8)) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  /* Put bias in MEMA */
  if ((pBQ = (uint32_mema_t *)alloc_in_memA(&(hmw->hbram), (3 * ou_c) * sizeof(uint32_t))) == NULL)
  {
    return HSP_CORE_ERROR;
  }

  HSP_HW_IF_WRITE_PARAMR0(inp_w);
  HSP_HW_IF_WRITE_PARAMR1(inp_h);
  HSP_HW_IF_WRITE_PARAMR2(in_c);
  HSP_HW_IF_WRITE_PARAMR3(k_w);
  HSP_HW_IF_WRITE_PARAMR4(k_h);
  HSP_HW_IF_WRITE_PARAMR5(ou_w);
  HSP_HW_IF_WRITE_PARAMR6(ou_h);
  HSP_HW_IF_WRITE_PARAMR7(ou_c);
  HSP_HW_IF_WRITE_PARAMR8(stridex);
  HSP_HW_IF_WRITE_PARAMR9(stridey);
  HSP_HW_IF_WRITE_PARAMR10(off_in);
  HSP_HW_IF_WRITE_PARAMR11(off_ou);
  HSP_HW_IF_WRITE_PARAMR12(sat_min);
  HSP_HW_IF_WRITE_PARAMR13(sat_max);
  HSP_HW_IF_WRITE_PARAMR14((uint32_t) pBQ);
  HSP_HW_IF_WRITE_PARAMR15((circiLines << HSP_CNN_CFG_NB_IN_LINE_SHIFT) |
                           (nbLinePerBlocks << HSP_CNN_CFG_NB_OU_LINE_SHIFT) | HSP_CNN_CFG_MODE_1STEP);
  /* And finally sync with DCMD using DCMDPTR registers */
  HSP_HW_IF_WRITE_DCMDPTR0((uint32_t)pSrcA);
  HSP_HW_IF_WRITE_DCMDPTR1((uint32_t)pSrcB);
  HSP_HW_IF_WRITE_DCMDPTR2((uint32_t)pDst);

  /* Load all coeffs */
  if ((((k_w * in_c) & 0x3) == 0) ||
      ((in_c == 1) && (k_w == k_h) && (k_w == 3)) ||
      ((in_c == 3) && (k_w == k_h) && (k_w == 3)))
  {
    HSP_MEMCPY((int8_t *)pSrcB, (int8_t *)p_filter_data, (k_c * k_h * k_w));
  }
  else
  {
    /* Not aligned on 4, add extra kernels to 0 or special sizes */
    uint32_t nbPadK0 = (4 - (k_w * in_c) & 0x3);
    for (uint32_t i = 0; i < (ou_c * k_h); i++)
    {
     HSP_MEMCPY((int8_t *)pSrcB, (int8_t *)p_filter_data, (in_c * k_w));
      p_filter_data += (in_c * k_w);
      pSrcB += (in_c * k_w);
      for (uint32_t kp = 0; kp < nbPadK0; kp++)
      {
        *pSrcB++ = 0;
      }
    }
  }

  align_factor_cmsisnn_fast_ch_v2(in_scale, out_scale, p_wt_scale, (int32_t *)p_bias_data, (uint16_t)ou_c,
                                  (int32_t *)pBQ);

  /* Load first input tile */
  int8_t padV = (int8_t) off_in;
  padV = -padV;
  uint8_t val = padV;
  if (pad)
  {
    /* Need to add padding */
    uint32_t nbPadTop = 0;
    uint8_mema_t *inPad = pSrcA;
    /* First lines pad */
    HSP_MEMSET(inPad, val, (pady_t *((in_w + (padx_l + padx_r)) * in_c)));
    inPad += (pady_t *((in_w + (padx_l + padx_r)) * in_c));
    /* Middle lines to pad */
    uint32_t lineToPad = k_h - pady_t;
    if (lineToPad > in_h)
    {
      nbPadTop = lineToPad - in_h;
      lineToPad =  in_h;
    }
    for (uint32_t i = 0; i < lineToPad; i++)
    {
      /* First column pad */
      HSP_MEMSET(inPad, val, (padx_l * in_c));
      inPad += (padx_l * in_c);
     /* Copy image line */
      HSP_MEMCPY((int8_t *)inPad, (int8_t *)p_input_data, (in_w * in_c));
     inPad += (in_w * in_c);
      p_input_data += (in_w * in_c);
      /* Last column pad */
      HSP_MEMSET(inPad, val, (padx_r * in_c));
      inPad += (padx_r * in_c);
    }
    /* Add bottom pad if necessery */
    if (nbPadTop)
    {
      /* last lines pad */
      HSP_MEMSET(inPad, val, (nbPadTop *((in_w + (padx_l + padx_r)) * in_c)));
      inPad += (nbPadTop *((in_w + (padx_l + padx_r)) * in_c));
    }
  }
  else
  {
    /* No padding */
    HSP_MEMCPY((int8_t *)pSrcA, (int8_t *)p_input_data, (inLineSize8 * k_h));
    p_input_data += (inLineSize8 * k_h);
  }

  /* Wait H2CSEMR for HSP done, but check FWERR first */
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
  if (HAL_HSP_GetFirmwareError(((HSP_HandleTypeDef *)(hmw->hdriver))))
  {
    /* Free all CNN allocated memory */
    free_all_ai(&(hmw->hbram));
    return HSP_CORE_ERROR;
  }
  uint8_t *pSrcEnd = pSrcA + (circiSize);
  uint8_t *pSrcCur = pSrcA + (inLineSize8 * k_h);
  uint8_t *pDstEnd = (uint8_t *)(pDst + (nbLinePerBlocks * outLineSize8));
  uint8_t *pDstCur = (uint8_t *) pDst;
  uint32_t countlines = k_h;
  uint32_t countmax = inp_h - pady_b;
  uint32_t xloop = inLineSize8 - ((padx_l + padx_r) * in_c);
  /* Send event through CDEG: WARNING: do we need to disable all other events??? */
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->CDEGR = HSP_CNN_CDEG_EVT;
  for (uint32_t lineIdx = 0; lineIdx < (ou_h - 1); lineIdx++)
  {
    /* Copy input line in circular buffer */
    for (uint32_t sidx = 0; sidx < stridey; sidx++)
    {
      if (countlines == countmax)
      {
        /* Last lines pad */
        HSP_MEMSET(pSrcCur, val, inLineSize8);
        pSrcCur += inLineSize8;
      }
      else
      {
        countlines++;
        if (pad)
        {
          /* First column pad */
          HSP_MEMSET(pSrcCur, val, (padx_l * in_c));
          pSrcCur += (padx_l * in_c);
          HSP_MEMCPY((int8_t *)pSrcCur, (int8_t *)p_input_data, xloop);
          pSrcCur += xloop;
          p_input_data += xloop;
          /* Last column pad */
          HSP_MEMSET(pSrcCur, val, (padx_r * in_c));
          pSrcCur += (padx_r * in_c);
        }
        else
        {
          HSP_MEMCPY((int8_t *)pSrcCur, (int8_t *)p_input_data, inLineSize8);
          p_input_data += inLineSize8;
          pSrcCur += inLineSize8;
        }
        if (pSrcCur == pSrcEnd)
        {
          pSrcCur = pSrcA;
        }
      }
    }
    /* Wait output line */
    while (HAL_HSP_MSGBOX_IsMsgAvailable(((HSP_HandleTypeDef *)(hmw->hdriver))) == 0U); /* kernel ready */
    /* Clear sem */
    HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
    __DSB();
#endif  /* STM32H7P5xx */
    /* Send event through CDEG: WARNING: do we need to disable all other events??? */
    ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->CDEGR = HSP_CNN_CDEG_EVT;
    HSP_MEMCPY((int8_t *)p_output_data, (int8_t *)pDstCur, outLineSize8);
    p_output_data += outLineSize8;
    pDstCur += outLineSize8;
    if (pDstCur == pDstEnd)
    {
      pDstCur = pDst;
    }
  }
  /* Wait last output line */
  while (HAL_HSP_MSGBOX_IsMsgAvailable(((HSP_HandleTypeDef *)(hmw->hdriver))) == 0U); /* kernel ready */
  /* Clear sem */
  HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
  __DSB();
#endif  /* STM32H7P5xx */  
  HSP_MEMCPY((int8_t *)p_output_data, (int8_t *)pDstCur, outLineSize8);
#ifdef CNN_EVTENR_SUPPORT
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->EVTENR = evtenrSave; /* Restore previous EVTENR value */
#endif /* CNN_EVTENR_SUPPORT */

  /* Free all memory */
  free_all_ai(&(hmw->hbram));
  return HSP_CORE_OK;
}


/**
  * @brief Direct CNN conv2D function with circular mode in 2 steps with padding
  *        (all sizes except WHC: 1x1x1, 1x2x1, 2x1x1, 1x3x1, 3x1x1, 1x4x1, 3x3x1, 3x3x3)
  * Circular input and ping pong output in MEMA and full kernels in MEMB, sync with HSP during HSP output lines
  * Split code in 2 steps
  * First loop is on kernel to load all kernels, while HSP working on first channel
  * Then loop on all other channels
  * HSP send error report via FWERR register
  * @param hmw                 HSP handle.
  * @param in_w                Input dimension width
  * @param in_h                Input dimension height
  * @param in_c                Input dimension channel
  * @param k_w                 Kernel dimension width
  * @param k_h                 Kernel dimension height
  * @param ou_w                Output dimension width
  * @param ou_h                Output dimension height
  * @param ou_c                Output dimension channel
  * @param stridex             Stride on X
  * @param stridey             Stride on Y
  * @param *p_input_data       Input data pointer, int8_t data type
  * @param *p_filter_data      Kernel coefficient pointer, int8_t data type
  * @param *p_output_data      Output data pointer, int8_t data type
  * @param *p_bias_data        Bias data pointer, int32_t data type
  * @param in_scale            Input scale
  * @param out_scale           Output scale
  * @param p_wt_scale          Pointer in weight scales (one per output channel)
  * @param off_in              Input offset, int32_t data type
  * @param off_ou              Output offset, int32_t data type
  * @param sat_min             Min sat (Relu), int32_t data type
  * @param sat_max             Max sat (Relu), int32_t data type
  * @param padx_l              Pad on X left
  * @param padx_r              Pad on X right
  * @param pady_t              Pad on Y top
  * @param pady_b              Pad on Y bottom
  * @param nb_line_per_blocks  Number of output lines per output buffer
   * @retval                   Core status.
 */
hsp_core_status_t HSP_ACC_CnnConv2d2_s8(hsp_core_handle_t *hmw,
                           uint32_t in_w, uint32_t in_h, uint32_t in_c,
                           uint32_t k_w, uint32_t k_h,
                           uint32_t ou_w, uint32_t ou_h, uint32_t ou_c, uint32_t stridex, uint32_t stridey,
                           int8_t *p_input_data, int8_t *p_filter_data, int8_t *p_output_data,
                           uint32_t *p_bias_data,
                           float32_t in_scale, float32_t out_scale, const float32_t *p_wt_scale,
                           uint32_t off_in, uint32_t off_ou, uint32_t sat_min, uint32_t sat_max,
                           uint32_t padx_l, uint32_t padx_r, uint32_t pady_t, uint32_t pady_b,
                           uint32_t nb_line_per_blocks)
{
 /* @ToDo HSP_CHECK_ASSERT_NRV(hhsp, (((((k_w * in_c) + 3) / 4) * k_h) > 1)); */

  uint8_memb_t *pSrcB;
  uint8_mema_t *pSrcA;
  uint8_mema_t *pDst;
  uint32_mema_t *pBQ;

  /* First of all, wakeup HSP with Direct command */
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_CNN_CONV2D_I8);

#ifdef CNN_EVTENR_SUPPORT
  /* Save current EVTENR and clear all enable event for ARM/HSP CNN synchro */
  uint32_t evtenrSave = ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->EVTENR;
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->EVTENR = HSP_CNN_EVTENR_CLR;
#endif /* CNN_EVTENR_SUPPORT */

  /* ToDo: enable CDEG interface */
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->ITFENR = 0xE0000000;

  uint32_t pad = padx_l + padx_r + pady_t + pady_b; /*  Pad option */
  uint32_t inp_w = in_w + (padx_l + padx_r);
  uint32_t inp_h = in_h + (pady_t + pady_b);
  uint32_t nbLinePerBlocks = nb_line_per_blocks + 2;
  if (nbLinePerBlocks > ou_h)
  {
    nbLinePerBlocks = ou_h;
  }
  uint32_t inLineSize8 = in_c * inp_w; /* 1 input line */
  uint32_t outLineSize8 = ou_c * ou_w; /* 1 output line */
  uint32_t circiLines = (k_h + stridey); /* For circular mode */
  uint32_t circiSize = inLineSize8 * circiLines; /* For circular mode */

  /* Put kernels in MEMB */
  if ((pSrcB = (uint8_memb_t *) alloc_in_memB(&(hmw->hbram), (ou_c * k_h * ((k_w * in_c) + 3)))) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  /* Put input image in MEMA */
  if ((pSrcA = (uint8_mema_t *) alloc_in_memA(&(hmw->hbram), circiSize)) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  /* Put output images in MEMA */
  if ((pDst  = (uint8_mema_t *) alloc_in_memA(&(hmw->hbram), nbLinePerBlocks * outLineSize8)) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  /* Put bias in MEMA */
  if ((pBQ  = (uint32_mema_t *)alloc_in_memA(&(hmw->hbram), (3 * ou_c) * sizeof(uint32_t))) == NULL)
  {
    return HSP_CORE_ERROR;
  }

  HSP_HW_IF_WRITE_PARAMR0(inp_w);
  HSP_HW_IF_WRITE_PARAMR1(inp_h);
  HSP_HW_IF_WRITE_PARAMR2(in_c);
  HSP_HW_IF_WRITE_PARAMR3(k_w);
  HSP_HW_IF_WRITE_PARAMR4(k_h);
  HSP_HW_IF_WRITE_PARAMR5(ou_w);
  HSP_HW_IF_WRITE_PARAMR6(ou_h);
  HSP_HW_IF_WRITE_PARAMR7(ou_c);
  HSP_HW_IF_WRITE_PARAMR8(stridex);
  HSP_HW_IF_WRITE_PARAMR9(stridey);
  HSP_HW_IF_WRITE_PARAMR10(off_in);
  HSP_HW_IF_WRITE_PARAMR11(off_ou);
  HSP_HW_IF_WRITE_PARAMR12(sat_min);
  HSP_HW_IF_WRITE_PARAMR13(sat_max);
  HSP_HW_IF_WRITE_PARAMR14((uint32_t) pBQ);
  HSP_HW_IF_WRITE_PARAMR15((circiLines << HSP_CNN_CFG_NB_IN_LINE_SHIFT) |
                           (nbLinePerBlocks << HSP_CNN_CFG_NB_OU_LINE_SHIFT) | HSP_CNN_CFG_MODE_2STEP);
  HSP_HW_IF_WRITE_DCMDPTR0((uint32_t)pSrcA);
  HSP_HW_IF_WRITE_DCMDPTR1((uint32_t)pSrcB);
  HSP_HW_IF_WRITE_DCMDPTR2((uint32_t)pDst);

  /* Load only first channel out coeffs */
  uint32_t nbPadK0 = 0;
  if ((k_w * in_c) & 0x3)
  {
    /* Not aligned on 4, add extra kernels to 0 */
    int8_t *tmpKe = p_filter_data;
    nbPadK0 = (4 - (k_w * in_c) & 0x3);
    for (uint32_t i = 0; i < k_h; i++)
    {
      HSP_MEMCPY((int8_t *)pSrcB, (int8_t *)tmpKe, (k_w * in_c));
      tmpKe += (k_w * in_c);
      pSrcB += (k_w * in_c);
      for (uint32_t kp = 0; kp < nbPadK0; kp++)
      {
        *pSrcB++ = 0;
      }
    }
  }
  else
  {
    HSP_MEMCPY((int8_t *)pSrcB, (int8_t *)p_filter_data, (k_h * k_w * in_c));
    pSrcB += (k_h * k_w * in_c);
  }

  align_factor_cmsisnn_fast_ch_v2(in_scale, out_scale, p_wt_scale, (int32_t *)p_bias_data, (uint16_t)ou_c,
                                  (int32_t *)pBQ);

  /* Load first input tile */
  int8_t padV = (int8_t) off_in;
  padV = -padV;
  uint8_t val = padV;
  if (pad)
  {
    uint32_t nbPadTop = 0;
    uint8_mema_t *inPad = pSrcA;
    /* First lines pad */
    HSP_MEMSET(inPad, val, (pady_t *((in_w + (padx_l + padx_r)) * in_c)));
    inPad += (pady_t *((in_w + (padx_l + padx_r)) * in_c));
    /* Middle lines to pad */
    uint32_t lineToPad = k_h - pady_t;
    if (lineToPad > in_h)
    {
      nbPadTop = lineToPad - in_h;
      lineToPad =  in_h;
    }
    for (uint32_t i = 0; i < lineToPad; i++)
    {
      /* First column pad */
      HSP_MEMSET(inPad, val, (padx_l * in_c));
      inPad += (padx_l * in_c);
      /* Copy image line */
      HSP_MEMCPY((int8_t *)inPad, (int8_t *)p_input_data, (in_w * in_c));
      inPad += (in_w * in_c);
      p_input_data += (in_w * in_c);
      /* Last column pad */
      HSP_MEMSET(inPad, val, (padx_r * in_c));
      inPad += (padx_r * in_c);
    }
    /* Add bottom pad if necessery */
    if (nbPadTop)
    {
      /* last lines pad */
      HSP_MEMSET(inPad, val, (nbPadTop *((in_w + (padx_l + padx_r)) * in_c)));
      inPad += (nbPadTop *((in_w + (padx_l + padx_r)) * in_c));
    }
  }
  else
  {
    HSP_MEMCPY((int8_t *)pSrcA, (int8_t *)p_input_data, (inLineSize8 * k_h));
    p_input_data += (inLineSize8 * k_h);
  }
  /* Wait H2CSEMR for HSP done, but check FWERR first */
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
  if (HAL_HSP_GetFirmwareError(((HSP_HandleTypeDef *)(hmw->hdriver))))
  {
    free_all_ai(&(hmw->hbram));
    return HSP_CORE_ERROR;
  }
  uint32_t countlines = k_h;
  uint32_t countmax = inp_h - pady_b;
  uint32_t xloop = inLineSize8 - ((padx_l + padx_r) * in_c);
  uint8_t *pKerCur = (uint8_t *)(p_filter_data + (k_h * k_w * in_c));
  uint8_t *pSrcbCur = pSrcB;
  uint8_t *pSrcEnd = pSrcA + (circiSize);
  uint8_t *pSrcCur = pSrcA + (inLineSize8 * k_h);
  uint8_t *pDstEnd = (uint8_t *)(pDst + (nbLinePerBlocks * outLineSize8));
  uint8_t *pDstCur = (uint8_t *) pDst;
  /* Clear sem */
  HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
  __DSB();
#endif /* STM32H7P5xx */

  /* Send event through CDEG: WARNING: do we need to disable all other events??? */
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->CDEGR = HSP_CNN_CDEG_EVT;
  /* Loop to load all kernels in BRAM */
  for (uint32_t out_num = 1; out_num < ou_c; out_num++)
  {
    /* Copy kernel block */
    if (nbPadK0)
    {
      /* Not aligned on 4, add extra kernels to 0 */
      for (uint32_t i = 0; i < (k_h); i++)
      {
        HSP_MEMCPY((int8_t *)pSrcbCur, (int8_t *)pKerCur, (in_c * k_w));
        pKerCur += (in_c * k_w);
        pSrcbCur += (in_c * k_w);
        for (uint32_t kp = 0; kp < nbPadK0; kp++)
        {
          *pSrcbCur++ = 0;
        }
      }
    }
    else
    {
      HSP_MEMCPY((int8_t *)pSrcbCur, (int8_t *)pKerCur, (k_w * k_h * in_c));
      pKerCur += (k_w * k_h * in_c);
      pSrcbCur += (k_w * k_h * in_c);
    }
    /* Wait output channel done */
    while (HAL_HSP_MSGBOX_IsMsgAvailable(((HSP_HandleTypeDef *)(hmw->hdriver))) == 0U); /* kernel ready */
    /* Clear sem */
    HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
    __DSB();
#endif /* STM32H7P5xx */

    /* Send event through CDEG: WARNING: do we need to disable all other events??? */
    ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->CDEGR = HSP_CNN_CDEG_EVT;
  }

  for (uint32_t lineIdx = 0; lineIdx < (ou_h - 1); lineIdx++)
  {
    /* Copy input line in circular buffer */
    for (uint32_t sidx = 0; sidx < stridey; sidx++)
    {
      if (countlines == countmax)
      {
        /* Last lines pad */
        HSP_MEMSET(pSrcCur, val, inLineSize8);
        pSrcCur += inLineSize8;
      }
      else
      {
        countlines++;
        if (pad)
        {
          /* First column pad */
          HSP_MEMSET(pSrcCur, val, (padx_l * in_c));
          pSrcCur += (padx_l * in_c);
          HSP_MEMCPY((int8_t *)pSrcCur, (int8_t *)p_input_data, xloop);
          pSrcCur += xloop;
          p_input_data += xloop;
          /* Last column pad */
          HSP_MEMSET(pSrcCur, val, (padx_r * in_c));
          pSrcCur += (padx_r * in_c);
        }
        else
        {
          HSP_MEMCPY((int8_t *)pSrcCur, (int8_t *)p_input_data, inLineSize8);
          p_input_data += inLineSize8;
          pSrcCur += inLineSize8;
        }
        if (pSrcCur == pSrcEnd)
        {
          pSrcCur = pSrcA;
        }
      }
    }
    /* Wait output line */
    while (HAL_HSP_MSGBOX_IsMsgAvailable(((HSP_HandleTypeDef *)(hmw->hdriver))) == 0U); /* kernel ready */
    /* Clear sem */
    HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
    __DSB();
#endif /* STM32H7P5xx */

    /* Send event through CDEG: WARNING: do we need to disable all other events??? */
    ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->CDEGR = HSP_CNN_CDEG_EVT;
    HSP_MEMCPY((int8_t *)p_output_data, (int8_t *)pDstCur, outLineSize8);
    p_output_data += outLineSize8;
    pDstCur += outLineSize8;
    if (pDstCur == pDstEnd)
    {
      pDstCur = pDst;
    }
  }
  /* Wait last output line */
  while (HAL_HSP_MSGBOX_IsMsgAvailable(((HSP_HandleTypeDef *)(hmw->hdriver))) == 0U); /* kernel ready */
  /* Clear sem */
  HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
  __DSB();
#endif  /* STM32H7P5xx */
  HSP_MEMCPY((int8_t *)p_output_data, (int8_t *)pDstCur, outLineSize8);
#ifdef CNN_EVTENR_SUPPORT
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->EVTENR = evtenrSave; /* Restore previous EVTENR value */
#endif /* CNN_EVTENR_SUPPORT */

  /* Free all memory */
  free_all_ai(&(hmw->hbram));
  return HSP_CORE_OK;
}

/**
  * @brief Direct CNN conv2D function with circular mode in 3 steps with padding
  *        (all sizes except: ((k_w * in_c) < 5) WHC: 3x3x1, 3x3x3),
  * Circular input in MEMA and output in MEMB and ping pong on kernels in MEMB, in 3 steps, sync according to step
  * First step, loop on all input lines with first kernel to get lines from ARM, sync on input line
  * Then loop on all channel (exclude first and last) as classical cio sync by output channel
  * And finally loop on last channel, sync on output line
  * Then play again 3 steps if necessary
  * HSP send error report via FWERR register
  * @param hmw                 HSP handle.
  * @param in_w                Input dimension width
  * @param in_h                Input dimension height
  * @param in_c                Input dimension channel
  * @param k_w                 Kernel dimension width
  * @param k_h                 Kernel dimension height
  * @param ou_w                Output dimension width
  * @param ou_h                Output dimension height
  * @param ou_c                Output dimension channel
  * @param stridex             Stride on X
  * @param stridey             Stride on Y
  * @param *p_input_data       Input data pointer, int8_t data type
  * @param *p_filter_data      Kernel coefficient pointer, int8_t data type
  * @param *p_output_data      Output data pointer, int8_t data type
  * @param *p_bias_data        Bias data pointer, int32_t data type
  * @param in_scale            Input scale
  * @param out_scale           Output scale
  * @param p_wt_scale          Pointer in weight scales (one per output channel)
  * @param off_in              Input offset, int32_t data type
  * @param off_ou              Output offset, int32_t data type
  * @param sat_min             Min sat (Relu), int32_t data type
  * @param sat_max             Max sat (Relu), int32_t data type
  * @param padx_l              Pad on X left
  * @param padx_r              Pad on X right
  * @param pady_t              Pad on Y top
  * @param pady_b              Pad on Y bottom
  * @param nb_line_per_blocks  Number of output lines per output buffer
  * @retval                    Core status.
  */
hsp_core_status_t HSP_ACC_CnnConv2d3_s8(hsp_core_handle_t *hmw,
                           uint32_t in_w, uint32_t in_h, uint32_t in_c,
                           uint32_t k_w, uint32_t k_h,
                           uint32_t ou_w, uint32_t ou_h, uint32_t ou_c, uint32_t stridex, uint32_t stridey,
                           int8_t *p_input_data, int8_t *p_filter_data, int8_t *p_output_data,
                           uint32_t *p_bias_data,
                           float32_t in_scale, float32_t out_scale, const float32_t *p_wt_scale,
                           uint32_t off_in, uint32_t off_ou, uint32_t sat_min, uint32_t sat_max,
                           uint32_t padx_l, uint32_t padx_r, uint32_t pady_t, uint32_t pady_b,
                           uint32_t nb_line_per_blocks)
{
  /* @ToDo
  HSP_CHECK_ASSERT_NRV(hhsp, ((((k_w * in_c) + 3) / 4) * k_h));
  HSP_CHECK_ASSERT_NRV(hhsp, ((ou_c) >= 3));
  HSP_CHECK_ASSERT_NRV(hhsp, ((ou_h) >= 3)); */

  uint8_memb_t *pSrcB;
  uint8_mema_t *pSrcA;
  uint8_memb_t *pDst;
  uint32_mema_t *pBQ;

  /* First of all, wakeup HSP with Direct command */
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_CNN_CONV2D_I8);

#ifdef CNN_EVTENR_SUPPORT
  /* Save current EVTENR and clear all enable event for ARM/HSP CNN synchro */
  uint32_t evtenrSave = ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->EVTENR;
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->EVTENR = HSP_CNN_EVTENR_CLR;
#endif /* CNN_EVTENR_SUPPORT */

  /* ToDo: enable CDEG interface */
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->ITFENR = 0xE0000000;

  uint32_t pad = padx_l + padx_r + pady_t + pady_b; /*  Pad option */
  uint32_t inp_w = in_w + (padx_l + padx_r);
  uint32_t inp_h = in_h + (pady_t + pady_b);
  uint32_t nbLineInPerBlocks = (nb_line_per_blocks * stridey) + k_h - stridey; /* For circular mode */
  if (nb_line_per_blocks > ou_h)
  {
    nb_line_per_blocks = ou_h;
  }
  uint32_t inLineSize8 = in_c * inp_w; /* 1 input line */
  uint32_t outLineSize8 = ou_c * ou_w; /* 1 output line */
  uint32_t circiLines = nbLineInPerBlocks; /* For circular mode */
  uint32_t circiSize = inLineSize8 * circiLines; /* For circular mode */

  /* Put kernels in MEMB */
  if ((pSrcB = (uint8_memb_t *) alloc_in_memB(&(hmw->hbram), (2 * k_h * (((k_w * in_c) + 3))))) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  /* Put input image in MEMA */
  if ((pSrcA = (uint8_mema_t *) alloc_in_memA(&(hmw->hbram), circiSize)) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  /* Put output images in MEMA */
  if ((pDst  = (uint8_memb_t *) alloc_in_memB(&(hmw->hbram), nb_line_per_blocks * outLineSize8)) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  /* Put bias in MEMA */
  if ((pBQ  = (uint32_mema_t *)alloc_in_memA(&(hmw->hbram), (3 * ou_c) * sizeof(uint32_t))) == NULL)
  {
    return HSP_CORE_ERROR;
  }

  HSP_HW_IF_WRITE_PARAMR0(inp_w);
  HSP_HW_IF_WRITE_PARAMR1(inp_h);
  HSP_HW_IF_WRITE_PARAMR2(in_c);
  HSP_HW_IF_WRITE_PARAMR3(k_w);
  HSP_HW_IF_WRITE_PARAMR4(k_h);
  HSP_HW_IF_WRITE_PARAMR5(ou_w);
  HSP_HW_IF_WRITE_PARAMR6(ou_h);
  HSP_HW_IF_WRITE_PARAMR7(ou_c);
  HSP_HW_IF_WRITE_PARAMR8(stridex);
  HSP_HW_IF_WRITE_PARAMR9(stridey);
  HSP_HW_IF_WRITE_PARAMR10(off_in);
  HSP_HW_IF_WRITE_PARAMR11(off_ou);
  HSP_HW_IF_WRITE_PARAMR12(sat_min);
  HSP_HW_IF_WRITE_PARAMR13(sat_max);
  HSP_HW_IF_WRITE_PARAMR14((uint32_t) pBQ);
  HSP_HW_IF_WRITE_PARAMR15((circiLines << HSP_CNN_CFG_NB_IN_LINE_SHIFT) |
                           (nb_line_per_blocks << HSP_CNN_CFG_NB_OU_LINE_SHIFT) | HSP_CNN_CFG_MODE_3STEP);
  HSP_HW_IF_WRITE_DCMDPTR0((uint32_t) pSrcA);
  HSP_HW_IF_WRITE_DCMDPTR1((uint32_t) pSrcB);
  HSP_HW_IF_WRITE_DCMDPTR2((uint32_t) pDst);

  /* Load all coeffs needed for first channel out */
  uint32_t nbPadK0 = 0;
  if ((k_w * in_c) & 0x3)
  {
    /* Not aligned on 4, add extra kernels to 0 */
    uint8_memb_t *tmpB = pSrcB;
    int8_t *tmpKe = p_filter_data;
    nbPadK0 = (4 - (k_w * in_c) & 0x3);
    for (uint32_t i = 0; i < (k_h); i++)
    {
      HSP_MEMCPY((int8_t *)tmpB, (int8_t *)tmpKe, (in_c * k_w));
      tmpKe += (in_c * k_w);
      tmpB += (in_c * k_w);
      for (uint32_t kp = 0; kp < nbPadK0; kp++)
      {
        *tmpB++ = 0;
      }
    }
  }
  else
  {
    HSP_MEMCPY((int8_t *)pSrcB, (int8_t *)p_filter_data, (k_w * k_h * in_c));
  }

  align_factor_cmsisnn_fast_ch_v2(in_scale, out_scale, p_wt_scale, (int32_t *)p_bias_data, (uint16_t)ou_c,
                                  (int32_t *)pBQ);

  /* Load first input tile */
  int8_t padV = (int8_t) off_in;
  padV = -padV;
  uint8_t val = padV;
  if (pad)
  {
    uint32_t nbPadTop = 0;
    uint8_mema_t *inPad = pSrcA;
    /* First lines pad */
    HSP_MEMSET(inPad, val, (pady_t *((in_w + (padx_l + padx_r)) * in_c)));
    inPad += (pady_t *((in_w + (padx_l + padx_r)) * in_c));
    /* Middle lines to pad */
    uint32_t lineToPad = k_h - pady_t;
    if (lineToPad > in_h)
    {
      nbPadTop = lineToPad - in_h;
      lineToPad =  in_h;
    }
    for (uint32_t i = 0; i < lineToPad; i++)
    {
      /* First column pad */
      HSP_MEMSET(inPad, val, (padx_l * in_c));
      inPad += (padx_l * in_c);
      /* Copy image line */
      HSP_MEMCPY((int8_t *)inPad, (int8_t *)p_input_data, (in_w * in_c));
      inPad += (in_w * in_c);
      p_input_data += (in_w * in_c);
      /* Last column pad */
      HSP_MEMSET(inPad, val, (padx_r * in_c));
      inPad += (padx_r * in_c);
    }
    /* Add bottom pad if necessery */
    if (nbPadTop)
    {
      /* last lines pad */
      HSP_MEMSET(inPad, val, (nbPadTop *((in_w + (padx_l + padx_r)) * in_c)));
      inPad += (nbPadTop *((in_w + (padx_l + padx_r)) * in_c));
    }
  }
  else
  {
    HSP_MEMCPY((int8_t *)pSrcA, (int8_t *)p_input_data, (inLineSize8 * k_h));
    p_input_data += (inLineSize8 * k_h);
  }

  /* Wait H2CSEMR for HSP done, but check FWERR first */
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
  if (HAL_HSP_GetFirmwareError(((HSP_HandleTypeDef *)(hmw->hdriver))))
  {
    free_all_ai(&(hmw->hbram));
    return HSP_CORE_ERROR;
  }
  /* Clear sem */
  HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
  __DSB();
#endif  /* STM32H7P5xx */

  uint8_t *pSrcEnd = pSrcA + (circiSize);
  uint8_t *pSrcCur = pSrcA + (inLineSize8 * k_h);
  uint8_t *pDstCur = pDst;
  uint8_t *pDstEnd = (uint8_t *)(pDst + (nb_line_per_blocks * ou_c * ou_w));
  uint8_t *pKerCur = (uint8_t *)(p_filter_data + (k_w * k_h * in_c));
  uint8_t *pSrcbCur = pSrcB + (k_h * ((in_c * k_w) + nbPadK0));
  uint8_t *pSrcbEnd = pSrcB + (k_h * ((in_c * k_w) + nbPadK0) * 2);
  uint32_t nbLinesBlck = 0; /* Init with one block for more simple comparison at end of while */
  uint32_t nbLineLoop = nb_line_per_blocks;
  uint32_t countlines = k_h;
  uint32_t countmax = inp_h - pady_b;
  uint32_t xloop = inLineSize8 - ((padx_l + padx_r) * in_c);
  do
  {
    if (nbLinesBlck)
    {
      /* Load all coeffs needed for first channel out */
      pKerCur = (uint8_t *)p_filter_data;
      if (nbPadK0)
      {
        /* Not aligned on 4, add extra kernels to 0 */
        for (uint32_t i = 0; i < (k_h); i++)
        {
          HSP_MEMCPY((int8_t *)pSrcbCur, (int8_t *)pKerCur, (in_c * k_w));
          pKerCur += (in_c * k_w);
          pSrcbCur += (in_c * k_w);
          for (uint32_t kp = 0; kp < nbPadK0; kp++)
          {
            *pSrcbCur++ = 0;
          }
        }
      }
      else
      {
        HSP_MEMCPY((int8_t *)pSrcbCur, (int8_t *)pKerCur, (k_w * k_h * in_c));
        pKerCur += (k_w * k_h * in_c);
        pSrcbCur += (k_w * k_h * in_c);
      }
      if (pSrcbCur == pSrcbEnd)
      {
        pSrcbCur = pSrcB;
      }
      /* Copy input line in circular buffer before enter loop */
      for (uint32_t sidx = 0; sidx < stridey; sidx++)
      {
        if (countlines == countmax)
        {
          /* Last lines pad */
          HSP_MEMSET(pSrcCur, val, inLineSize8);
          pSrcCur += inLineSize8;
        }
        else
        {
          countlines++;
          if (pad)
          {
            /* First column pad */
            HSP_MEMSET(pSrcCur, val, (padx_l * in_c));
            pSrcCur += (padx_l * in_c);
            HSP_MEMCPY((int8_t *)pSrcCur, (int8_t *)p_input_data, xloop);
            pSrcCur += xloop;
            p_input_data += xloop;
            /* Last column pad */
            HSP_MEMSET(pSrcCur, val, (padx_r * in_c));
            pSrcCur += (padx_r * in_c);
          }
          else
          {
            HSP_MEMCPY((int8_t *)pSrcCur, (int8_t *)p_input_data, inLineSize8);
            p_input_data += inLineSize8;
            pSrcCur += inLineSize8;
          }
          if (pSrcCur == pSrcEnd)
          {
            pSrcCur = pSrcA;
          }
        }
      }
    }
    /* Send event through CDEG: WARNING: do we need to disable all other events??? */
    ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->CDEGR = HSP_CNN_CDEG_EVT; /* send event */
    nbLinesBlck += nbLineLoop;
    if (nbLinesBlck > ou_h)
    {
      /* Count how may output lines not done by block */
      nbLinesBlck -= nbLineLoop; /* Remove next block size */
      nbLineLoop = ou_h - nbLinesBlck;
      nbLinesBlck += nbLineLoop;
    }
    /* First step: load all input block */
    for (uint32_t lineIdx = 1; lineIdx < nbLineLoop; lineIdx++)
    {
      /* Copy input line in circular buffer */
      for (uint32_t sidx = 0; sidx < stridey; sidx++)
      {
        if (countlines == countmax)
        {
          /* Last lines pad */
          HSP_MEMSET(pSrcCur, val, inLineSize8);
          pSrcCur += inLineSize8;
        }
        else
        {
          countlines++;
          if (pad)
          {
            /* First column pad */
            HSP_MEMSET(pSrcCur, val, (padx_l * in_c));
            pSrcCur += (padx_l * in_c);
            HSP_MEMCPY((int8_t *)pSrcCur, (int8_t *)p_input_data, xloop);
            pSrcCur += xloop;
            p_input_data += xloop;
            /* Last column pad */
            HSP_MEMSET(pSrcCur, val, (padx_r * in_c));
            pSrcCur += (padx_r * in_c);
          }
          else
          {
            HSP_MEMCPY((int8_t *)pSrcCur, (int8_t *)p_input_data, inLineSize8);
            p_input_data += inLineSize8;
            pSrcCur += inLineSize8;
          }
          if (pSrcCur == pSrcEnd)
          {
            pSrcCur = pSrcA;
          }
        }
      }
      /* Wait output line */
      while (HAL_HSP_MSGBOX_IsMsgAvailable(((HSP_HandleTypeDef *)(hmw->hdriver))) == 0U); /* kernel ready */
      /* Clear sem */
      HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
      __DSB();
#endif  /* STM32H7P5xx */

      /* Send event through CDEG: WARNING: do we need to disable all other events??? */
      ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->CDEGR = HSP_CNN_CDEG_EVT; /* send event */
    }
    /* Whole input block is now in BRAM loop on all channels: Loop on each ping pong kernel */
    for (uint32_t out_num = 1; out_num < ou_c; out_num++)
    {
      /* First of all copy kernel block */
      if (nbPadK0)
      {
        /* Not aligned on 4, add extra kernels to 0 */
        for (uint32_t i = 0; i < (k_h); i++)
        {
          HSP_MEMCPY((int8_t *)pSrcbCur, (int8_t *)pKerCur, (in_c * k_w));
          pKerCur += (in_c * k_w);
          pSrcbCur += (in_c * k_w);
          for (uint32_t kp = 0; kp < nbPadK0; kp++)
          {
            *pSrcbCur++ = 0;
          }
        }
      }
      else
      {
        HSP_MEMCPY((int8_t *)pSrcbCur, (int8_t *)pKerCur, (k_w * k_h * in_c));
        pKerCur += (k_w * k_h * in_c);
        pSrcbCur += (k_w * k_h * in_c);
      }
      if (pSrcbCur == pSrcbEnd)
      {
        pSrcbCur = pSrcB;
      }
      /* Wait output channel done */
      while (HAL_HSP_MSGBOX_IsMsgAvailable(((HSP_HandleTypeDef *)(hmw->hdriver))) == 0U); /* kernel ready */
      /* Clear sem */
      HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
      __DSB();
#endif  /* STM32H7P5xx */

      /* Send event through CDEG: WARNING: do we need to disable all other events??? */
      ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->CDEGR = HSP_CNN_CDEG_EVT; /* send event */
    }

    /* Then read output lines in BRAM */
    for (uint32_t lineIdx = 1; lineIdx < nbLineLoop; lineIdx++)
    {
      /* Wait output line */
      while (HAL_HSP_MSGBOX_IsMsgAvailable(((HSP_HandleTypeDef *)(hmw->hdriver))) == 0U); /* kernel ready */
      /* Clear sem */
      HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
      __DSB();
#endif  /* STM32H7P5xx */

      /* Send event through CDEG: WARNING: do we need to disable all other events??? */
      ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->CDEGR = HSP_CNN_CDEG_EVT; /* send event */
      HSP_MEMCPY((int8_t *)p_output_data, (int8_t *)pDstCur, outLineSize8);
      p_output_data += outLineSize8;
      pDstCur += outLineSize8;
      if (pDstCur == pDstEnd)
      {
        pDstCur = pDst;
      }
    }
    /* Wait output line */
    while (HAL_HSP_MSGBOX_IsMsgAvailable(((HSP_HandleTypeDef *)(hmw->hdriver))) == 0U); /* kernel ready */
    /* Clear sem */
    HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
    __DSB();
#endif  /* STM32H7P5xx */

    HSP_MEMCPY((int8_t *)p_output_data, (int8_t *)pDstCur, outLineSize8);
    p_output_data += outLineSize8;
    pDstCur += outLineSize8;
    if (pDstCur == pDstEnd)
    {
      pDstCur = pDst;
    }
  } while (nbLinesBlck < ou_h); /* End of input block loop */
#ifdef CNN_EVTENR_SUPPORT
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->EVTENR = evtenrSave; /* Restore previous EVTENR value */
#endif /* CNN_EVTENR_SUPPORT */

  /* Free all memory */
  free_all_ai(&(hmw->hbram));
  return HSP_CORE_OK;
}

/**
  * @brief Direct CNN depthwise function with circular mode
  * Circular input and ping pong output in MEMA and full kernels in MEMB, sync with HSP during HSP output lines
  * HSP send error report via FWERR register
  * @param hmw                 HSP handle.
  * @param in_w                Input dimension width
  * @param in_h                Input dimension height
  * @param in_c                Input dimension channel
  * @param k_w                 Kernel dimension width
  * @param k_h                 Kernel dimension height
  * @param ou_w                Output dimension width
  * @param ou_h                Output dimension height
  * @param ou_c                Output dimension channel
  * @param stridex             Stride on X
  * @param stridey             Stride on Y
  * @param *p_input_data       Input data pointer, int8_t data type
  * @param *p_filter_data      Kernel coefficient pointer, int8_t data type
  * @param *p_output_data      Output data pointer, int8_t data type
  * @param *p_bias_data        Bias data pointer, int32_t data type
  * @param in_scale            Input scale
  * @param out_scale           Output scale
  * @param p_wt_scale          Pointer in weight scales (one per output channel)
  * @param off_in              Input offset, int32_t data type
  * @param off_ou              Output offset, int32_t data type
  * @param sat_min             Min sat (Relu), int32_t data type
  * @param sat_max             Max sat (Relu), int32_t data type
  * @param padx_l              Pad on X left
  * @param padx_r              Pad on X right
  * @param pady_t              Pad on Y top
  * @param pady_b              Pad on Y bottom
  * @retval                    Core status.
*/
hsp_core_status_t HSP_ACC_CnnConvDepthwise1_s8(hsp_core_handle_t *hmw,
                                  uint32_t in_w, uint32_t in_h, uint32_t in_c,
                                  uint32_t k_w, uint32_t k_h,
                                  uint32_t ou_w, uint32_t ou_h, uint32_t ou_c,
                                  uint32_t stridex, uint32_t stridey,
                                  int8_t *p_input_data, int8_t *p_filter_data, int8_t *p_output_data,
                                  int32_t *p_bias_data,
                                  float32_t in_scale, float32_t out_scale, const float32_t *p_wt_scale,
                                  int32_t off_in, int32_t off_ou, int32_t sat_min, int32_t sat_max,
                                  uint32_t padx_l, uint32_t padx_r, uint32_t pady_t, uint32_t pady_b)
{
  int8_memb_t *pSrcB;
  int8_mema_t *pSrcA;
  int8_mema_t *pDst;
  int32_mema_t *pBias;
  int32_mema_t *pQuant;

  /* First of all, wakeup HSP with Direct command */
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_CNN_CONVDW_I8);

#ifdef CNN_EVTENR_SUPPORT
  /* Save current EVTENR and clear all enable event for ARM/HSP CNN synchro */
  uint32_t evtenrSave = ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->EVTENR;
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->EVTENR = HSP_CNN_EVTENR_CLR;
#endif /* CNN_EVTENR_SUPPORT */

  /* Then update all parameters */
  uint32_t pad = padx_l + padx_r + pady_t + pady_b;
  uint32_t inp_w = in_w + (padx_l + padx_r);
  uint32_t inp_h = in_h + (pady_t + pady_b);

  /* Then update all parameters */
  HSP_HW_IF_WRITE_PARAMR0(inp_w);
  HSP_HW_IF_WRITE_PARAMR1(inp_h);
  HSP_HW_IF_WRITE_PARAMR2(in_c);
  HSP_HW_IF_WRITE_PARAMR3(k_w);
  HSP_HW_IF_WRITE_PARAMR4(k_h);
  HSP_HW_IF_WRITE_PARAMR5(ou_w);
  HSP_HW_IF_WRITE_PARAMR6(ou_h);
  HSP_HW_IF_WRITE_PARAMR7(ou_c);
  HSP_HW_IF_WRITE_PARAMR8((stridex << 16) | (stridey & 0xff));
  HSP_HW_IF_WRITE_PARAMR9(off_in);
  HSP_HW_IF_WRITE_PARAMR10(off_ou);
  HSP_HW_IF_WRITE_PARAMR11(sat_min);
  HSP_HW_IF_WRITE_PARAMR12(sat_max);

  uint32_t k_c = in_c;
  uint32_t inLineSize8 = in_c * inp_w; /* 1 input line */
  uint32_t outLineSize8 = ou_c * ou_w; /* 1 output line */
  uint32_t circiLines = (k_h + stridey); /* For circular mode */
  uint32_t circiSize = inLineSize8 * circiLines; /* For circular mode */
  uint32_t nbPadK0 = (4 - (in_c & 0x3)) & 0x3;

  if ((pSrcB = (int8_memb_t *)alloc_in_memB(&(hmw->hbram), (in_c  + nbPadK0) * k_h * k_w)) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  if ((pSrcA = (int8_mema_t *)alloc_in_memA(&(hmw->hbram), circiSize)) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  /* Ping pong buffer on destination */
  if ((pDst  = (int8_mema_t *)alloc_in_memA(&(hmw->hbram), (2 * outLineSize8))) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  if ((pBias = (int32_mema_t *)alloc_in_memA(&(hmw->hbram), (ou_c * sizeof(uint32_t)))) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  if ((pQuant = (int32_mema_t *)alloc_in_memA(&(hmw->hbram), (ou_c * sizeof(uint32_t)) * 2)) == NULL)
  {
    return HSP_CORE_ERROR;
  }

  HSP_HW_IF_WRITE_PARAMR13((uint32_t)pBias); /* bias followed by quant */
  HSP_HW_IF_WRITE_PARAMR14((uint32_t)pQuant);
  HSP_HW_IF_WRITE_PARAMR15((circiLines << HSP_CNN_CFG_NB_IN_LINE_SHIFT) | HSP_CNN_CFG_MODE_1STEP);

  /* And finally sync with DCMD using DCMDPTR registers */
  HSP_HW_IF_WRITE_DCMDPTR0((uint32_t)pSrcA);
  HSP_HW_IF_WRITE_DCMDPTR1((uint32_t)pSrcB);
  HSP_HW_IF_WRITE_DCMDPTR2((uint32_t)pDst);

  /* Load all coeffs */
  if ((in_c & 0x3) == 0)
  {
    HSP_MEMCPY(pSrcB, p_filter_data, (k_h * k_w * k_c));
  }
  else
  {
    /* Not aligned on 4, add extra kernels to 0 or special sizes */
    for (uint32_t i = 0; i < k_w * k_h; i++)
    {
      HSP_MEMCPY(pSrcB, p_filter_data, k_c);
      p_filter_data += k_c;
      pSrcB += k_c;
      for (uint32_t kp = 0; kp < nbPadK0; kp++)
      {
        *pSrcB++ = 0;
      }
    }
  }

  /* Now copy biais */
  for (uint32_t i = 0; i < ou_c; i++)
  {
    *pBias++ = p_bias_data[i];
  }

  align_factor_cmsisnn_fast_ch(in_scale, out_scale, p_wt_scale, (uint16_t)ou_c, (int32_t *)pQuant);

  /* Load first input tile */
  int8_t padV = (int8_t) off_in;
  padV = -padV;
  uint8_t val = padV;
  uint8_mema_t *inPad = (uint8_mema_t *)pSrcA;
  uint32_t nbPadTop = 0;

  /* First lines pad */
  if (pad)
  {
    HSP_MEMSET(inPad, val, (pady_t *((in_w + (padx_l + padx_r)) * in_c)));
    inPad += (pady_t *((in_w + (padx_l + padx_r)) * in_c));
  }
  /* Middle lines to pad */
  uint32_t lineToPad = k_h - pady_t;
  if (lineToPad > in_h)
  {
    nbPadTop = lineToPad - in_h;
    lineToPad =  in_h;
  }
  for (uint32_t i = 0; i < lineToPad; i++)
  {
    if (pad)
    {
      /* First column pad */
      HSP_MEMSET(inPad, val, (padx_l * in_c));
      inPad += (padx_l * in_c);
    }
    /* Copy image line */
    HSP_MEMCPY((int8_t *)inPad, (int8_t *)p_input_data, (in_w * in_c));
    inPad += (in_w * in_c);
    p_input_data += (in_w * in_c);

    /* Last column pad */
    if (pad)
    {
      HSP_MEMSET(inPad, val, (padx_r * in_c));
      inPad += (padx_r * in_c);
    }
  }
  /* Add bottom pad if necessery */
  if (nbPadTop)
  {
    /* last lines pad */
    HSP_MEMSET(inPad, val, (nbPadTop *((in_w + (padx_l + padx_r)) * in_c)));
    inPad += (nbPadTop *((in_w + (padx_l + padx_r)) * in_c));
  }
  /* Wait H2CSEMR for HSP done, but check FWERR first */
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
  if (HAL_HSP_GetFirmwareError(((HSP_HandleTypeDef *)(hmw->hdriver))))
  {
    free_all_ai(&(hmw->hbram));
    return HSP_CORE_ERROR;
  }
  uint8_t *pSrcEnd = (uint8_t *)pSrcA + (circiSize);
  uint8_t *pSrcCur = (uint8_t *)pSrcA + (inLineSize8 * k_h);
  uint8_t *pDstEnd = (uint8_t *)(pDst + (2 * outLineSize8));
  uint8_t *pDstCur = (uint8_t *) pDst;
  uint32_t countlines = k_h;
  uint32_t countmax = inp_h - pady_b;
  uint32_t xloop = inLineSize8 - ((padx_l + padx_r) * in_c);

  /* Clear sem */
  HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
  __DSB();
#endif  /* STM32H7P5xx */
  /* Send event through CDEG: WARNING: do we need to disable all other events??? */
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->CDEGR = HSP_CNN_CDEG_EVT; /* send event */

  for (uint32_t lineIdx = 0; lineIdx < (ou_h - 1); lineIdx++)
  {
    /* Copy input line in circular buffer */
    for (uint32_t sidx = 0; sidx <  stridey; sidx++)
    {
      if (countlines == countmax)
      {
        /* Last lines pad */
        HSP_MEMSET(pSrcCur, val, inLineSize8);
        pSrcCur += inLineSize8;
      }
      else
      {
        countlines++;
        if (pad)
        {
          /* First column pad */
          HSP_MEMSET(pSrcCur, val, (padx_l * in_c));
          pSrcCur += (padx_l * in_c);
        }
        HSP_MEMCPY((int8_t *)pSrcCur, (int8_t *)p_input_data, xloop);
        pSrcCur += xloop;
        p_input_data += xloop;
        if (pad)
        {
          /* Last column pad */
          HSP_MEMSET(pSrcCur, val, (padx_r * in_c));
          pSrcCur += (padx_r * in_c);
        }
      }
      if (pSrcCur == pSrcEnd)
      {
        pSrcCur = (uint8_t *)pSrcA;
      }
    }
    /* Wait output line */
    while (HAL_HSP_MSGBOX_IsMsgAvailable(((HSP_HandleTypeDef *)(hmw->hdriver))) == 0U); /* kernel ready */
    /* Clear sem */
    HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
  __DSB();
#endif  /* STM32H7P5xx */
    /* Send event through CDEG: WARNING: do we need to disable all other events??? */
    ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->CDEGR = HSP_CNN_CDEG_EVT;
    HSP_MEMCPY((int8_t *)p_output_data, (int8_t *)pDstCur, outLineSize8);
    p_output_data += outLineSize8;
    pDstCur += outLineSize8;
    if (pDstCur == pDstEnd)
    {
      pDstCur = (uint8_t *)pDst;
    }
  }
  /* Wait last output line */
  while (HAL_HSP_MSGBOX_IsMsgAvailable(((HSP_HandleTypeDef *)(hmw->hdriver))) == 0U); /* kernel ready */
  /* Clear sem */
  HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
  __DSB();
#endif  /* STM32H7P5xx */
  HSP_MEMCPY((int8_t *)p_output_data, (int8_t *)pDstCur, outLineSize8);
#ifdef CNN_EVTENR_SUPPORT
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->EVTENR = evtenrSave; /* Restore previous EVTENR value */
#endif /* CNN_EVTENR_SUPPORT */

  /* Free all memory */
  free_all_ai(&(hmw->hbram));
  return HSP_CORE_OK;
}

/**
  * Does not support nb chan <= 4
  * @brief Direct CNN depthwise function with circular mode in 2 steps with padding
  * Circular input and ping pong output in MEMA and full kernels in MEMB, sync with HSP during HSP output lines
  * Split code in 2 steps
  * First loop is on kernel to load all kernels, while HSP working on first channel
  * Then loop on all other channels
  * HSP send error report via FWERR register
  * @param hmw                 HSP handle.
  * @param in_w                Input dimension width
  * @param in_h                Input dimension height
  * @param in_c                Input dimension channel
  * @param k_w                 Kernel dimension width
  * @param k_h                 Kernel dimension height
  * @param ou_w                Output dimension width
  * @param ou_h                Output dimension height
  * @param ou_c                Output dimension channel
  * @param stridex             Stride on X
  * @param stridey             Stride on Y
  * @param *p_input_data       Input data pointer, int8_t data type
  * @param *p_filter_data      Kernel coefficient pointer, int8_t data type
  * @param *p_output_data      Output data pointer, int8_t data type
  * @param *p_bias_data        Bias data pointer, int32_t data type
  * @param in_scale            Input scale
  * @param out_scale           Output scale
  * @param p_wt_scale          Pointer in weight scales (one per output channel)
  * @param off_in              Input offset, int32_t data type
  * @param off_ou              Output offset, int32_t data type
  * @param sat_min             Min sat (Relu), int32_t data type
  * @param sat_max             Max sat (Relu), int32_t data type
  * @param padx_l              Pad on X left
  * @param padx_r              Pad on X right
  * @param pady_t              Pad on Y top
  * @param pady_b              Pad on Y bottom
  * @param nb_line_per_blocks  Number of output lines per output buffer
  * @retval                    Core status.
  */
hsp_core_status_t HSP_ACC_CnnConvDepthwise2_s8(hsp_core_handle_t *hmw,
                                  uint32_t in_w, uint32_t in_h, uint32_t in_c,
                                  uint32_t k_w, uint32_t k_h,
                                  uint32_t ou_w, uint32_t ou_h, uint32_t ou_c,
                                  uint32_t stridex, uint32_t stridey,
                                  int8_t *p_input_data, int8_t *p_filter_data, int8_t *p_output_data,
                                  int32_t *p_bias_data,
                                  float32_t in_scale, float32_t out_scale, const float32_t *p_wt_scale,
                                  int32_t off_in, int32_t off_ou, int32_t sat_min, int32_t sat_max,
                                  uint32_t padx_l, uint32_t padx_r, uint32_t pady_t, uint32_t pady_b,
                                  uint32_t nb_line_per_blocks)
{
  int8_memb_t *pSrcB;
  int8_mema_t *pSrcA;
  int8_mema_t *pDst;
  int32_mema_t *pBias;
  int32_mema_t *pQuant;

  /* First of all, wakeup HSP with Direct command */
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_CNN_CONVDW_I8);

#ifdef CNN_EVTENR_SUPPORT
  /* Save current EVTENR and clear all enable event for ARM/HSP CNN synchro */
  uint32_t evtenrSave = ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->EVTENR;
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->EVTENR = HSP_CNN_EVTENR_CLR;
#endif /* CNN_EVTENR_SUPPORT */

  /* Then update all parameters */
  uint32_t pad = padx_l + padx_r + pady_t + pady_b;
  uint32_t inp_w = in_w + (padx_l + padx_r);
  uint32_t inp_h = in_h + (pady_t + pady_b);

  /* Then update all parameters */
  HSP_HW_IF_WRITE_PARAMR0(inp_w);
  HSP_HW_IF_WRITE_PARAMR1(inp_h);
  HSP_HW_IF_WRITE_PARAMR2(in_c);
  HSP_HW_IF_WRITE_PARAMR3(k_w);
  HSP_HW_IF_WRITE_PARAMR4(k_h);
  HSP_HW_IF_WRITE_PARAMR5(ou_w);
  HSP_HW_IF_WRITE_PARAMR6(ou_h);
  HSP_HW_IF_WRITE_PARAMR7(ou_c);
  HSP_HW_IF_WRITE_PARAMR8((stridex << 16) | (stridey & 0xff));
  HSP_HW_IF_WRITE_PARAMR9(off_in);
  HSP_HW_IF_WRITE_PARAMR10(off_ou);
  HSP_HW_IF_WRITE_PARAMR11(sat_min);
  HSP_HW_IF_WRITE_PARAMR12(sat_max);

  uint32_t k_c = in_c;
  uint32_t inLineSize8 = in_c * inp_w; /* 1 input line */
  uint32_t outLineSize8 = ou_c * ou_w; /* 1 output line */
  uint32_t circiLines = (k_h + stridey); /* For circular mode */
  uint32_t circiSize = inLineSize8 * circiLines; /* For circular mode */
  uint32_t nbPadK0 = (4 - (in_c & 0x3)) & 0x3;

  if ((pSrcB = (int8_memb_t *)alloc_in_memB(&(hmw->hbram), (in_c  + nbPadK0) * k_h * k_w)) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  /* 2 input lines in circular buffer */
  if ((pSrcA = (int8_mema_t *)alloc_in_memA(&(hmw->hbram), circiSize)) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  /* ping pong buffer on destination */
  if ((pDst  = (int8_mema_t *)alloc_in_memA(&(hmw->hbram), (nb_line_per_blocks * outLineSize8))) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  if ((pBias = (int32_mema_t *)alloc_in_memA(&(hmw->hbram), (ou_c * sizeof(uint32_t)))) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  if ((pQuant = (int32_mema_t *)alloc_in_memA(&(hmw->hbram), (ou_c * sizeof(uint32_t)) * 2)) == NULL)
  {
    return HSP_CORE_ERROR;
  }

  HSP_HW_IF_WRITE_PARAMR13((uint32_t)pBias); /* bias followed by quant */
  HSP_HW_IF_WRITE_PARAMR14((uint32_t)pQuant);
  HSP_HW_IF_WRITE_PARAMR15((circiLines << HSP_CNN_CFG_NB_IN_LINE_SHIFT) |
                           (nb_line_per_blocks << HSP_CNN_CFG_NB_OU_LINE_SHIFT) | HSP_CNN_CFG_MODE_2STEP);

  /* And finally sync with DCMD using DCMDPTR registers */
  HSP_HW_IF_WRITE_DCMDPTR0((uint32_t) pSrcA);
  HSP_HW_IF_WRITE_DCMDPTR1((uint32_t) pSrcB);
  HSP_HW_IF_WRITE_DCMDPTR2((uint32_t) pDst);

  /* Depthwise works  4 by 4 then we need to load (4*k_h*k_w) kernel for first line */
  int8_t *ptmp = (int8_t *)p_filter_data;
  int8_memb_t *ptmpSrcB = (int8_memb_t *)pSrcB;

  if (nbPadK0)
  {

    for (uint32_t j = 0; j < (k_h * k_w); j++)
    {
      HSP_MEMCPY(ptmpSrcB, ptmp, 4);
      ptmpSrcB += (k_c + nbPadK0);
      ptmp += (k_c);
    }
  }
  else
  {
    for (uint32_t j = 0; j < (k_h * k_w); j++)
    {
      ptmpSrcB[0] = ptmp[0];
      ptmpSrcB[1] = ptmp[1];
      ptmpSrcB[2] = ptmp[2];
      ptmpSrcB[3] = ptmp[3];
      ptmpSrcB += k_c;
      ptmp += k_c;
    }
  }
  /* Now copy bias */
  for (uint32_t i = 0; i < ou_c; i++)
  {
    *pBias++ = p_bias_data[i];
  }

  align_factor_cmsisnn_fast_ch(in_scale, out_scale, p_wt_scale, (uint16_t)ou_c, (int32_t *)pQuant);

  /* Load first input tile */
  int8_t padV = (int8_t) off_in;
  padV = -padV;
  uint8_t val = padV;
  uint8_mema_t *inPad = (uint8_mema_t *)pSrcA;
  uint32_t nbPadTop = 0;

  if (pad)
  {
    /* First lines pad */
    HSP_MEMSET(inPad, val, (pady_t *((in_w + (padx_l + padx_r)) * in_c)));
    inPad += (pady_t *((in_w + (padx_l + padx_r)) * in_c));
  }
  /* Middle lines to pad */
  uint32_t lineToPad = k_h - pady_t;
  if (lineToPad > in_h)
  {
    nbPadTop = lineToPad - in_h;
    lineToPad =  in_h;
  }
  for (uint32_t i = 0; i < lineToPad; i++)
  {
    if (pad)
    {
      /* First column pad */
      HSP_MEMSET(inPad, val, (padx_l * in_c));
      inPad += (padx_l * in_c);
    }
    /* Copy image line */
    HSP_MEMCPY((int8_t *)inPad, (int8_t *)p_input_data, (in_w * in_c));
    inPad += (in_w * in_c);
    p_input_data += (in_w * in_c);
    if (pad)
    {
      /* Last column pad */
      HSP_MEMSET(inPad, val, (padx_r * in_c));
      inPad += (padx_r * in_c);
    }
  }
  /* Add bottom pad if necessery */
  if (nbPadTop)
  {
    /* last lines pad */
    HSP_MEMSET(inPad, val, (nbPadTop *((in_w + (padx_l + padx_r)) * in_c)));
    inPad += (nbPadTop *((in_w + (padx_l + padx_r)) * in_c));
  }
  /* Wait H2CSEMR for HSP done, but check FWERR first */
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
  if (HAL_HSP_GetFirmwareError(((HSP_HandleTypeDef *)(hmw->hdriver))))
  {
    free_all_ai(&(hmw->hbram));
    return HSP_CORE_ERROR;
  }
  uint8_t *pSrcEnd = (uint8_t *)pSrcA + (circiSize);
  uint8_t *pSrcCur = (uint8_t *)pSrcA + (inLineSize8 * k_h);
  uint8_t *pDstEnd = (uint8_t *)(pDst + (nb_line_per_blocks * outLineSize8));
  uint8_t *pDstCur = (uint8_t *) pDst;
  uint32_t countlines = k_h;
  uint32_t countmax = inp_h - pady_b;
  uint32_t xloop = inLineSize8 - ((padx_l + padx_r) * in_c);
  uint8_t *pKerCur = (uint8_t *)(p_filter_data + 4);
  uint8_t *pSrcbCur = (uint8_t *)(pSrcB + 4);

  /* Clear sem */
  HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
  __DSB();
#endif /* STM32H7P5xx */

  /* Send event through CDEG: WARNING: do we need to disable all other events??? */
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->CDEGR = HSP_CNN_CDEG_EVT; /* send event */

  /* Loop to load all remaining kernels minus 4 already loaded in BRAM */
  for (uint32_t out_num = 4; out_num < ou_c; out_num += 4)
  {
    /* Copy 4 kernel by 4 kernel */
    int8_t *ptmpKe = (int8_t *)pKerCur;
    int8_memb_t  *ptmpKb = (int8_memb_t *)pSrcbCur;

    if (nbPadK0 && ((out_num + 4) >= ou_c))
    {
      for (uint32_t j = 0; j < (k_h * k_w); j++)
      {
        ptmpKb[0] = ptmpKe[0];
        if (nbPadK0 == 1)
        {
          ptmpKb[1] = ptmpKe[1];
          ptmpKb[2] = ptmpKe[2];
          ptmpKb[3] = 0;
        }
        if (nbPadK0 == 2)
        {
          ptmpKb[1] = ptmpKe[1];
          ptmpKb[2] = 0;
          ptmpKb[3] = 0;
        }
        if (nbPadK0 == 3)
        {
          ptmpKb[1] = 0;
          ptmpKb[2] = 0;
          ptmpKb[3] = 0;
        }
        ptmpKb += (k_c + nbPadK0);
        ptmpKe += k_c;
      }
    }
    else
    {
      for (uint32_t j = 0; j < (k_h * k_w); j++)
      {
        ptmpKb[0] = ptmpKe[0];
        ptmpKb[1] = ptmpKe[1];
        ptmpKb[2] = ptmpKe[2];
        ptmpKb[3] = ptmpKe[3];
        ptmpKb += (k_c + nbPadK0);
        ptmpKe += k_c;
      }
      pKerCur += 4;
      pSrcbCur += 4;
    }
    /* Wait output channel done */
    while (HAL_HSP_MSGBOX_IsMsgAvailable(((HSP_HandleTypeDef *)(hmw->hdriver))) == 0U); /* kernel ready */
    /* Clear sem */
    HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
    __DSB();
#endif /* STM32H7P5xx */

    /* Send event through CDEG: WARNING: do we need to disable all other events??? */
    ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->CDEGR = HSP_CNN_CDEG_EVT;
  }
  for (uint32_t lineIdx = 0; lineIdx < (ou_h - 1); lineIdx++)
  {
    /* Copy input line in circular buffer */
    for (uint32_t sidx = 0; sidx <  stridey; sidx++)
    {
      if (countlines == countmax)
      {
        /* Last lines pad */
        HSP_MEMSET(pSrcCur, val, inLineSize8);
        pSrcCur += inLineSize8;
      }
      else
      {
        countlines++;
        if (pad)
        {
          /* First column pad */
          HSP_MEMSET(pSrcCur, val, (padx_l * in_c));
          pSrcCur += (padx_l * in_c);
        }
        HSP_MEMCPY((int8_t *)pSrcCur, (int8_t *)p_input_data, xloop);
        pSrcCur += xloop;
        p_input_data += xloop;
        if (pad)
        {
          /* Last column pad */
          HSP_MEMSET(pSrcCur, val, (padx_r * in_c));
          pSrcCur += (padx_r * in_c);
        }
      }
      if (pSrcCur == pSrcEnd)
      {
        pSrcCur = (uint8_t *)pSrcA;
      }
    }
    /* Wait output line */
    while (HAL_HSP_MSGBOX_IsMsgAvailable(((HSP_HandleTypeDef *)(hmw->hdriver))) == 0U); /* kernel ready */
    /* Clear sem */
    HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
    __DSB();
#endif /* STM32H7P5xx */

    /* Send event through CDEG: WARNING: do we need to disable all other events??? */
    ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->CDEGR = HSP_CNN_CDEG_EVT;
    HSP_MEMCPY((int8_t *)p_output_data, (int8_t *)pDstCur, outLineSize8);
    p_output_data += outLineSize8;
    pDstCur += outLineSize8;
    if (pDstCur == pDstEnd)
    {
      pDstCur = (uint8_t *)pDst;
    }
  }
  /* Wait last output line */
  while (HAL_HSP_MSGBOX_IsMsgAvailable(((HSP_HandleTypeDef *)(hmw->hdriver))) == 0U); /* kernel ready */
  /* Clear sem */
  HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
  __DSB();
#endif  /* STM32H7P5xx */
  HSP_MEMCPY((int8_t *)p_output_data, (int8_t *)pDstCur, outLineSize8);
#ifdef CNN_EVTENR_SUPPORT
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->EVTENR = evtenrSave; /* Restore previous EVTENR value */
#endif /* CNN_EVTENR_SUPPORT */

  /* Free all memory */
  free_all_ai(&(hmw->hbram));
  return HSP_CORE_OK;
}

/**
  * Does not support nb chan <= 4
  * @brief Direct CNN depthwise function with circular mode in 3 steps with padding
  * Circular input and ping pong output in MEMA and full kernels in MEMB, sync with HSP during HSP output lines
  * Split code in 3 steps
  * First step, loop on all input lines with first kernel to get lines from ARM, sync on input line
  * Then loop on all channel (exclude first and last) as classical cio sync by output channel
  * And finally loop on last channel, sync on output line
  * Then play again 3 steps if necessary
  * HSP send error report via FWERR register
  * @param hmw                 HSP handle.
  * @param in_w                Input dimension width
  * @param in_h                Input dimension height
  * @param in_c                Input dimension channel
  * @param k_w                 Kernel dimension width
  * @param k_h                 Kernel dimension height
  * @param ou_w                Output dimension width
  * @param ou_h                Output dimension height
  * @param ou_c                Output dimension channel
  * @param stridex             Stride on X
  * @param stridey             Stride on Y
  * @param *p_input_data       Input data pointer, int8_t data type
  * @param *p_filter_data      Kernel coefficient pointer, int8_t data type
  * @param *p_output_data      Output data pointer, int8_t data type
  * @param *p_bias_data        Bias data pointer, int32_t data type
  * @param in_scale            Input scale
  * @param out_scale           Output scale
  * @param p_wt_scale          Pointer in weight scales (one per output channel)
  * @param off_in              Input offset, int32_t data type
  * @param off_ou              Output offset, int32_t data type
  * @param sat_min             Min sat (Relu), int32_t data type
  * @param sat_max             Max sat (Relu), int32_t data type
  * @param padx_l              Pad on X left
  * @param padx_r              Pad on X right
  * @param pady_t              Pad on Y top
  * @param pady_b              Pad on Y bottom
  * @param nb_line_per_blocks  Nb line to output in buffer
  * @retval                    Core status.
  */
hsp_core_status_t HSP_ACC_CnnConvDepthwise3_s8(hsp_core_handle_t *hmw,
                                  uint32_t in_w, uint32_t in_h, uint32_t in_c,
                                  uint32_t k_w, uint32_t k_h,
                                  uint32_t ou_w, uint32_t ou_h, uint32_t ou_c,
                                  uint32_t stridex, uint32_t stridey,
                                  int8_t *p_input_data, int8_t *p_filter_data, int8_t *p_output_data,
                                  int32_t *p_bias_data,
                                  float32_t in_scale, float32_t out_scale, const float32_t *p_wt_scale,
                                  int32_t off_in, int32_t off_ou, int32_t sat_min, int32_t sat_max,
                                  uint32_t padx_l, uint32_t padx_r, uint32_t pady_t, uint32_t pady_b,
                                  uint32_t nb_line_per_blocks)
{
  int8_memb_t *pSrcB;
  int8_mema_t *pSrcA;
  int8_memb_t *pDst;
  int32_mema_t *pBias;
  int32_mema_t *pQuant;

  /* First of all, wakeup HSP with Direct command */
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_CNN_CONVDW_I8);

#ifdef CNN_EVTENR_SUPPORT
  /* Save current EVTENR and clear all enable event for ARM/HSP CNN synchro */
  uint32_t evtenrSave = ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->EVTENR;
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->EVTENR = HSP_CNN_EVTENR_CLR;
#endif /* CNN_EVTENR_SUPPORT */

  /* Then update all parameters */
  uint32_t pad = padx_l + padx_r + pady_t + pady_b;
  uint32_t inp_w = in_w + (padx_l + padx_r);
  uint32_t inp_h = in_h + (pady_t + pady_b);

  uint32_t nbLineInPerBlocks = (nb_line_per_blocks * stridey) + k_h - stridey; /* For circular mode */

  /* Then update all parameters */
  HSP_HW_IF_WRITE_PARAMR0(inp_w);
  HSP_HW_IF_WRITE_PARAMR1(inp_h);
  HSP_HW_IF_WRITE_PARAMR2(in_c);
  HSP_HW_IF_WRITE_PARAMR3(k_w);
  HSP_HW_IF_WRITE_PARAMR4(k_h);
  HSP_HW_IF_WRITE_PARAMR5(ou_w);
  HSP_HW_IF_WRITE_PARAMR6(ou_h);
  HSP_HW_IF_WRITE_PARAMR7(ou_c);
  HSP_HW_IF_WRITE_PARAMR8((stridex << 16) | (stridey & 0xff));
  HSP_HW_IF_WRITE_PARAMR9(off_in);
  HSP_HW_IF_WRITE_PARAMR10(off_ou);
  HSP_HW_IF_WRITE_PARAMR11(sat_min);
  HSP_HW_IF_WRITE_PARAMR12(sat_max);

  uint32_t k_c = in_c;
  uint32_t inLineSize8 = in_c * inp_w; /* 1 input line */
  uint32_t outLineSize8 = ou_c * ou_w; /* 1 output line */
  uint32_t circiLines = nbLineInPerBlocks; /* For circular mode */
  uint32_t circiSize = inLineSize8 * circiLines; /* For circular mode */

  uint32_t nbPadK0 = (4 - (in_c & 0x3)) & 0x3;

  if ((pSrcB = (int8_memb_t *)alloc_in_memB(&(hmw->hbram), (4 * k_h * k_w) * 2)) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  /* Nb input lines in buffer */
  if ((pSrcA = (int8_mema_t *)alloc_in_memA(&(hmw->hbram), circiSize)) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  if ((pDst  = (int8_memb_t *)alloc_in_memB(&(hmw->hbram), (outLineSize8 * nb_line_per_blocks))) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  if ((pBias = (int32_mema_t *)alloc_in_memA(&(hmw->hbram), (ou_c * sizeof(uint32_t)))) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  if ((pQuant = (int32_mema_t *)alloc_in_memA(&(hmw->hbram), (ou_c * sizeof(uint32_t)) * 2)) == NULL)
  {
    return HSP_CORE_ERROR;
  }

  HSP_HW_IF_WRITE_PARAMR13((uint32_t)pBias); /* bias followed by quant */
  HSP_HW_IF_WRITE_PARAMR14((uint32_t)pQuant);
  HSP_HW_IF_WRITE_PARAMR15((circiLines << HSP_CNN_CFG_NB_IN_LINE_SHIFT) |
                           (nb_line_per_blocks << HSP_CNN_CFG_NB_OU_LINE_SHIFT) | HSP_CNN_CFG_MODE_3STEP);

  /* And finally sync with DCMD using DCMDPTR registers */
  HSP_HW_IF_WRITE_DCMDPTR0((uint32_t) pSrcA);
  HSP_HW_IF_WRITE_DCMDPTR1((uint32_t) pSrcB);
  HSP_HW_IF_WRITE_DCMDPTR2((uint32_t) pDst);

  /* Depthwise works  4 by 4 then we need to load (4*k_h*k_w) kernel for first line */
  int8_t *ptmp = (int8_t *)p_filter_data;
  int8_memb_t *ptmpSrcB = (int8_memb_t *)pSrcB;

  for (uint32_t j = 0; j < (k_h * k_w); j++)
  {
    ptmpSrcB[0] = ptmp[0];
    ptmpSrcB[1] = ptmp[1];
    ptmpSrcB[2] = ptmp[2];
    ptmpSrcB[3] = ptmp[3];
    ptmpSrcB += 4; /* nb kernel: 4 by 4 */
    ptmp += k_c;
  }

  /* Now copy bias */
  for (uint32_t i = 0; i < ou_c; i++)
  {
    *pBias++ = p_bias_data[i];
  }

  align_factor_cmsisnn_fast_ch(in_scale, out_scale, p_wt_scale, (uint16_t)ou_c, (int32_t *)pQuant);

  /* Load first input tile */
  int8_t padV = (int8_t) off_in;
  padV = -padV;
  uint8_t val = padV;
  uint8_mema_t *inPad = (uint8_mema_t *)pSrcA;
  uint32_t nbPadTop = 0;

  /* First lines pad */
  HSP_MEMSET(inPad, val, (pady_t *((in_w + (padx_l + padx_r)) * in_c)));
  inPad += (pady_t *((in_w + (padx_l + padx_r)) * in_c));
  uint32_t lineToPad = k_h - pady_t;
  if (lineToPad > in_h)
  {
    nbPadTop = lineToPad - in_h;
    lineToPad =  in_h;
  }
  /* Middle lines to pad */
  for (uint32_t i = 0; i < lineToPad; i++)
  {
    /* First column pad */
    if (pad)
    {
      HSP_MEMSET(inPad, val, (padx_l * in_c));
      inPad += (padx_l * in_c);
	}
    /* Copy image line */
    HSP_MEMCPY((int8_t *)inPad, (int8_t *)p_input_data, (in_w * in_c));
    inPad += (in_w * in_c);
    p_input_data += (in_w * in_c);
    /* Last column pad */
    if (pad)
    {
      HSP_MEMSET(inPad, val, (padx_r * in_c));
      inPad += (padx_r * in_c);
	}
  }
  /* Add bottom pad if necessery */
  if (nbPadTop)
  {
    /* last lines pad */
    HSP_MEMSET(inPad, val, (nbPadTop *((in_w + (padx_l + padx_r)) * in_c)));
    inPad += (nbPadTop *((in_w + (padx_l + padx_r)) * in_c));
  }
  /* Wait H2CSEMR for HSP done, but check FWERR first */
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
  if (HAL_HSP_GetFirmwareError(((HSP_HandleTypeDef *)(hmw->hdriver))))
  {
    free_all_ai(&(hmw->hbram));
    return HSP_CORE_ERROR;
  }
  /* Clear sem */
  HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
  __DSB();
#endif /* STM32H7P5xx */


  uint8_t *pSrcEnd = (uint8_t *)pSrcA + (circiSize);
  uint8_t *pSrcCur = (uint8_t *)pSrcA + (inLineSize8 * k_h);
  uint8_t *pDstEnd = (uint8_t *)(pDst + (nb_line_per_blocks * ou_c * ou_w));
  uint8_t *pDstCur = (uint8_t *) pDst;
  uint8_t *pSrcbEnd = (uint8_t *)(pSrcB + (4 * k_h * k_w) * 2); /* Ping Pong coeff buffer */
  uint32_t countlines = k_h;
  uint32_t countmax = inp_h - pady_b;
  uint32_t xloop = inLineSize8 - ((padx_l + padx_r) * in_c);
  uint8_t *pKerCur = (uint8_t *)(p_filter_data + 4);
  uint8_t *pSrcbCur = (uint8_t *)(pSrcB + (k_h * k_w * 4));
  uint32_t nbLinesBlck = 0; /* Init with one block for more simple comparison at end of while */
  uint32_t nbLineLoop = nb_line_per_blocks;

  if (pSrcCur == pSrcEnd)
  {
    pSrcCur = (uint8_t *)pSrcA;
  }

  do
  {
    if (nbLinesBlck)
    {
      pKerCur = (uint8_t *)p_filter_data;
      int8_t *ptmpKe = (int8_t *)p_filter_data;
      int8_memb_t  *ptmpKb = (int8_memb_t *)pSrcbCur;
      for (uint32_t j = 0; j < (k_h * k_w); j++)
      {
        ptmpKb[0] = ptmpKe[0];
        ptmpKb[1] = ptmpKe[1];
        ptmpKb[2] = ptmpKe[2];
        ptmpKb[3] = ptmpKe[3];
        ptmpKb += 4;
        pSrcbCur += 4;
        ptmpKe += k_c;
      }
      pKerCur += 4; /* next 4 channels */
      if (pSrcbCur == pSrcbEnd)
      {
        pSrcbCur = (uint8_t *)pSrcB;
      }
      /* Copy input line in circular buffer before enter loop */
      for (uint32_t sidx = 0; sidx <  stridey; sidx++)
      {
        if (countlines == countmax)
        {
          /* Last lines pad */
          HSP_MEMSET(pSrcCur, val, inLineSize8);
          pSrcCur += inLineSize8;
        }
        else
        {
          countlines++;
          /* First column pad */
          HSP_MEMSET(pSrcCur, val, (padx_l * in_c));
          pSrcCur += (padx_l * in_c);
          HSP_MEMCPY((int8_t *)pSrcCur, (int8_t *)p_input_data, xloop);
          pSrcCur += xloop;
          p_input_data += xloop;
          /* Last column pad */
          HSP_MEMSET(pSrcCur, val, (padx_r * in_c));
          pSrcCur += (padx_r * in_c);
        }
        if (pSrcCur == pSrcEnd)
        {
          pSrcCur = (uint8_t *)pSrcA;
        }
      }
    }
    nbLinesBlck += nbLineLoop;
    if (nbLinesBlck > ou_h)
    {
      /* Count how may output lines not done by block */
      nbLinesBlck -= nbLineLoop; /* Remove next block size */
      nbLineLoop = ou_h - nbLinesBlck;
      nbLinesBlck += nbLineLoop;
    }

    /*Send event through CDEG: WARNING: do we need to disable all other events??? */
    ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->CDEGR = HSP_CNN_CDEG_EVT; /* send event */
    /* First Step: copy all remaining input lines in memory */
    for (uint32_t lineIdx = 1; lineIdx < nbLineLoop; lineIdx++)
    {
      /* Copy input line in circular buffer */
      for (uint32_t sidx = 0; sidx <  stridey; sidx++)
      {
        if (countlines == countmax)
        {
          /* Last lines pad */
          HSP_MEMSET(pSrcCur, val, inLineSize8);
          pSrcCur += inLineSize8;
        }
        else
        {
          countlines++;
          /* First column pad */
          HSP_MEMSET(pSrcCur, val, (padx_l * in_c));
          pSrcCur += (padx_l * in_c);
          HSP_MEMCPY((int8_t *)pSrcCur, (int8_t *)p_input_data, xloop);
          pSrcCur += xloop;
          p_input_data += xloop;
          /* Last column pad */
          HSP_MEMSET(pSrcCur, val, (padx_r * in_c));
          pSrcCur += (padx_r * in_c);
        }
        if (pSrcCur == pSrcEnd)
        {
          pSrcCur = (uint8_t *)pSrcA;
        }
      }
      /* Send ready to FW */
      while (HAL_HSP_MSGBOX_IsMsgAvailable(((HSP_HandleTypeDef *)(hmw->hdriver))) == 0U); /* line ready */
      /* Clear sem */
      HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
      __DSB();
#endif /* STM32H7P5xx */

      /* Send event through CDEG: WARNING: do we need to disable all other events??? */
      ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->CDEGR = HSP_CNN_CDEG_EVT; /* send event */
    }
    /* All Lines are copied: Loop on each ping pong kernel */
    for (uint32_t out_num = 4; out_num < ou_c; out_num += 4)
    {
      /* First of all copy kernel block */
      if (nbPadK0 && ((out_num + 4) >= ou_c))
      {
        int8_t *ptmpKe = (int8_t *)pKerCur;
        int8_memb_t  *ptmpKb = (int8_memb_t *)pSrcbCur;
        for (uint32_t j = 0; j < (k_h * k_w); j++)
        {
          ptmpKb[0] = ptmpKe[0];
          if (nbPadK0 == 1)
          {
            ptmpKb[1] = ptmpKe[1];
            ptmpKb[2] = ptmpKe[2];
            ptmpKb[3] = 0;
          }
          if (nbPadK0 == 2)
          {
            ptmpKb[1] = ptmpKe[1];
            ptmpKb[2] = 0;
            ptmpKb[3] = 0;
          }
          if (nbPadK0 == 3)
          {
            ptmpKb[1] = 0;
            ptmpKb[2] = 0;
            ptmpKb[3] = 0;
          }
          ptmpKb += 4;
          pSrcbCur += 4;
          ptmpKe += k_c;
        }
        pKerCur += (4 - nbPadK0); /* next 4 channels */
      }
      else
      {
        int8_t *ptmpKe = (int8_t *)pKerCur;
        int8_memb_t  *ptmpKb = (int8_memb_t *)pSrcbCur;
        for (uint32_t j = 0; j < (k_h * k_w); j++)
        {
          ptmpKb[0] = ptmpKe[0];
          ptmpKb[1] = ptmpKe[1];
          ptmpKb[2] = ptmpKe[2];
          ptmpKb[3] = ptmpKe[3];
          ptmpKb += 4;
          pSrcbCur += 4;
          ptmpKe += k_c;
        }
        pKerCur += 4; /* next 4 channels */
      }
      if (pSrcbCur == pSrcbEnd)
      {
        pSrcbCur = (uint8_t *)pSrcB;
      }
      /* Wait output channel done */
      while (HAL_HSP_MSGBOX_IsMsgAvailable(((HSP_HandleTypeDef *)(hmw->hdriver))) == 0U); /* kernel ready */
      /* Clear sem */
      HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
      __DSB();
#endif /* STM32H7P5xx */

      /* Send event through CDEG: WARNING: do we need to disable all other events??? */
      ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->CDEGR = HSP_CNN_CDEG_EVT; /* send event */
    }
    /* Now copy output in destination */
    /* Then read output lines in BRAM */
    for (uint32_t lineIdx = 1; lineIdx < nbLineLoop; lineIdx++)
    {
      /* Wait output line */
      while (HAL_HSP_MSGBOX_IsMsgAvailable(((HSP_HandleTypeDef *)(hmw->hdriver))) == 0U); /* kernel ready */
      /* Clear sem */
      HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
      __DSB();
#endif /* STM32H7P5xx */

      /* Send event through CDEG: WARNING: do we need to disable all other events??? */
      ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->CDEGR = HSP_CNN_CDEG_EVT;
      HSP_MEMCPY((int8_t *)p_output_data, (int8_t *)pDstCur, outLineSize8);
      p_output_data += outLineSize8;
      pDstCur += outLineSize8;
      if (pDstCur == pDstEnd)
      {
        pDstCur = (uint8_t *) pDst;
      }
    }
    /* Wait output line */
    while (HAL_HSP_MSGBOX_IsMsgAvailable(((HSP_HandleTypeDef *)(hmw->hdriver))) == 0U); /* kernel ready */
    /* Clear sem */
    HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
    __DSB();
#endif /* STM32H7P5xx */

    HSP_MEMCPY((int8_t *)p_output_data, (int8_t *)pDstCur, outLineSize8);
    p_output_data += outLineSize8;
    pDstCur += outLineSize8;
    if (pDstCur == pDstEnd)
    {
      pDstCur = (uint8_t *)pDst;
    }
  } while (nbLinesBlck < ou_h); /* End of input block loop */

#ifdef CNN_EVTENR_SUPPORT
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->EVTENR = evtenrSave; /* Restore previous EVTENR value */
#endif /* CNN_EVTENR_SUPPORT */
  /* Free all memory */
  free_all_ai(&(hmw->hbram));
  return HSP_CORE_OK;
}

/**
  * @brief CNN fully connected function
  * @param hmw                 HSP handle.
  * @param in_c                Input dimension channel
  * @param ou_c                Output dimension channel
  * @param *p_input_data       Input data pointer, int8_t data type
  * @param *p_filter_data      Kernel coefficient pointer, int8_t data type
  * @param *p_output_data      Output data pointer, int8_t data type
  * @param *p_bias_data        Bias data pointer, int32_t data type
  * @param in_scale            Input scale
  * @param out_scale           Output scale
  * @param p_wt_scale          Pointer in weight scales (one per output channel)
  * @param off_in              Input offset, int32_t data type
  * @param off_ou              Output offset, int32_t data type
  * @param sat_min             Min sat (Relu), int32_t data type
  * @param sat_max             Max sat (Relu), int32_t data type
  * @retval                    Core status.
  */
hsp_core_status_t HSP_ACC_CnnFullyConnected0_s8(hsp_core_handle_t *hmw, uint32_t in_c, uint32_t ou_c,
                                   int8_t *p_input_data, int8_t *p_filter_data, int8_t *p_output_data,
                                   uint32_t *p_bias_data,
                                   float32_t in_scale, float32_t out_scale, float32_t wt_scale,
                                   uint32_t off_in, uint32_t off_ou, uint32_t sat_min, uint32_t sat_max)
{
  uint8_memb_t *pSrcB;
  uint8_mema_t *pSrcA;
  uint8_mema_t *pDst;
  uint32_mema_t *pBQ;

  /* First of all, wakeup HSP with Direct command */
  HSP_HW_IF_WRITE_DCMDIDR((uint32_t) HSP_DIRECT_CMD_CNN_FC_I8);

#ifdef CNN_EVTENR_SUPPORT
  /* Save current EVTENR and clear all enable event for ARM/HSP CNN synchro */
  uint32_t evtenrSave = ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->EVTENR;
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->EVTENR = HSP_CNN_EVTENR_CLR;
#endif /* CNN_EVTENR_SUPPORT */

  uint32_t k_c = in_c * ou_c;
  /* Put kernels in MEMB */
  if ((pSrcB = (uint8_memb_t *) alloc_in_memB(&(hmw->hbram), k_c)) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  /* Put input image in MEMA */
  if ((pSrcA = (uint8_mema_t *) alloc_in_memA(&(hmw->hbram), in_c)) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  /* Put output images in MEMA */
  if ((pDst  = (uint8_mema_t *) alloc_in_memA(&(hmw->hbram), ou_c)) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  /* Put bias in MEMA */
  if ((pBQ  = (uint32_mema_t *)alloc_in_memA(&(hmw->hbram), ou_c * sizeof(uint32_t))) == NULL)
  {
    return HSP_CORE_ERROR;
  }

  int32_t qMul;
  int32_t qShift;
  align_factor_cmsisnn_fast_ch_v3(in_scale, out_scale, wt_scale, (int32_t *)&qMul, (int32_t *)&qShift);

  /* Then update all parameters */
  HSP_HW_IF_WRITE_PARAMR2(in_c);
  HSP_HW_IF_WRITE_PARAMR7(ou_c);
  HSP_HW_IF_WRITE_PARAMR8(qMul);
  HSP_HW_IF_WRITE_PARAMR9(qShift);
  HSP_HW_IF_WRITE_PARAMR10(off_in);
  HSP_HW_IF_WRITE_PARAMR11(off_ou);
  HSP_HW_IF_WRITE_PARAMR12(sat_min);
  HSP_HW_IF_WRITE_PARAMR13(sat_max);
  HSP_HW_IF_WRITE_PARAMR14((uint32_t)pBQ);
  HSP_HW_IF_WRITE_PARAMR15(HSP_CNN_CFG_MODE_0STEP);
  /* And finally sync with DCMD using DCMDPTR registers */
  HSP_HW_IF_WRITE_DCMDPTR0((uint32_t)pSrcA);
  HSP_HW_IF_WRITE_DCMDPTR1((uint32_t)pSrcB);
  HSP_HW_IF_WRITE_DCMDPTR2((uint32_t)pDst);

  /* Load all coeffs */
  uint32_t nbPadK0 = 0;
  if ((in_c) & 0x3)
  {
    /* Not aligned on 4, add extra kernels to 0 */
    uint8_memb_t *tmpB = pSrcB;
    int8_t *tmpKe = p_filter_data;
    nbPadK0 = (4 - (in_c) & 0x3);
    for (uint32_t i = 0; i < ou_c; i++)
    {
      HSP_MEMCPY((int8_t *)tmpB, (int8_t *)tmpKe, (in_c));
      tmpKe += (in_c);
      tmpB += (in_c);
      for (uint32_t kp = 0; kp < nbPadK0; kp++)
      {
        *tmpB++ = 0;
      }
    }
  }
  else
  {
    HSP_MEMCPY((int8_t *)pSrcB, (int8_t *)p_filter_data, (k_c));
  }

  /* Now copy bias */
  HSP_MEMCPY((int8_t *)pBQ, (int8_t *)p_bias_data, (ou_c * 4));
  /* Load first input tile */
  HSP_MEMCPY((int8_t *)pSrcA, (int8_t *)p_input_data, (in_c));

  /* Wait H2CSEMR for HSP done, but check FWERR first */
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
  if (HAL_HSP_GetFirmwareError(((HSP_HandleTypeDef *)(hmw->hdriver))))
  {
    free_all_ai(&(hmw->hbram));
    return HSP_CORE_ERROR;
  }
  /* Clear sem before send event */
  HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
  __DSB();
#endif /* STM32H7P5xx */

  /* Send event through CDEG: WARNING: do we need to disable all other events??? */
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->CDEGR = HSP_CNN_CDEG_EVT; /* send event */
  /* Wait direct command done */
  while (HAL_HSP_MSGBOX_IsMsgAvailable(((HSP_HandleTypeDef *)(hmw->hdriver))) == 0U);
  /* Clear sem */
  HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
  __DSB();
#endif  /* STM32H7P5xx */
  /* Copy result in external memory */
  HSP_MEMCPY((int8_t *)p_output_data, (int8_t *)pDst, (ou_c));

#ifdef CNN_EVTENR_SUPPORT
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->EVTENR = evtenrSave; /* Restore previous EVTENR value */
#endif /* CNN_EVTENR_SUPPORT */

  /* Free all memory */
  free_all_ai(&(hmw->hbram));
  return HSP_CORE_OK;
}

/**
  * @brief CNN fully connected function, with background kernel transfer
  * @param hmw                 HSP handle.
  * @param in_c                Input dimension channel
  * @param ou_c                Output dimension channel
  * @param *p_input_data       Input data pointer, int8_t data type
  * @param *p_filter_data      Kernel coefficient pointer, int8_t data type
  * @param *p_output_data      Output data pointer, int8_t data type
  * @param *p_bias_data        Bias data pointer, int32_t data type
  * @param in_scale            Input scale
  * @param out_scale           Output scale
  * @param p_wt_scale          Pointer in weight scales (one per output channel)
  * @param off_in              Input offset, int32_t data type
  * @param off_ou              Output offset, int32_t data type
  * @param sat_min             Min sat (Relu), int32_t data type
  * @param sat_max             Max sat (Relu), int32_t data type
  * @retval                    Core status.
  */
hsp_core_status_t HSP_ACC_CnnFullyConnected1_s8(hsp_core_handle_t *hmw, uint32_t in_c, uint32_t ou_c,
                                   int8_t *p_input_data, int8_t *p_filter_data, int8_t *p_output_data,
                                   uint32_t *p_bias_data,
                                   float32_t in_scale, float32_t out_scale, float32_t wt_scale,
                                   uint32_t off_in, uint32_t off_ou, uint32_t sat_min, uint32_t sat_max)
{
  uint8_memb_t *pSrcB;
  uint8_mema_t *pSrcA;
  uint8_mema_t *pDst;
  uint32_mema_t *pBQ;

  /* First of all, wakeup HSP with Direct command */
  HSP_HW_IF_WRITE_DCMDIDR((uint32_t) HSP_DIRECT_CMD_CNN_FC_I8);

#ifdef CNN_EVTENR_SUPPORT
  /* Save current EVTENR and clear all enable event for ARM/HSP CNN synchro */
  uint32_t evtenrSave = ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->EVTENR;
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->EVTENR = HSP_CNN_EVTENR_CLR;
#endif /* CNN_EVTENR_SUPPORT */

  /* Force BARB here for the moment */
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->CAPCR = HSP_CAPCR_CCNTREN;
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->CR |= (2 << HSP_CR_BARB_Pos);

  /* Put kernels in MEMB */
  if ((pSrcB = (uint8_memb_t *)alloc_in_memB(&(hmw->hbram), in_c * 2)) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  /* Put input image in MEMA */
  if ((pSrcA = (uint8_mema_t *)alloc_in_memA(&(hmw->hbram), in_c)) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  /* Put output images in MEMA */
  if ((pDst  = (uint8_mema_t *)alloc_in_memA(&(hmw->hbram), ou_c)) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  /* Put bias in MEMA */
  if ((pBQ  = (uint32_mema_t *)alloc_in_memA(&(hmw->hbram), ou_c * sizeof(uint32_t))) == NULL)
  {
    return HSP_CORE_ERROR;
  }

  int32_t qMul;
  int32_t qShift;
  align_factor_cmsisnn_fast_ch_v3(in_scale, out_scale, wt_scale, (int32_t *)&qMul, (int32_t *)&qShift);

  /* Then update all parameters */
  HSP_HW_IF_WRITE_PARAMR2(in_c);
  HSP_HW_IF_WRITE_PARAMR7(ou_c);
  HSP_HW_IF_WRITE_PARAMR8(qMul);
  HSP_HW_IF_WRITE_PARAMR9(qShift);
  HSP_HW_IF_WRITE_PARAMR10(off_in);
  HSP_HW_IF_WRITE_PARAMR11(off_ou);
  HSP_HW_IF_WRITE_PARAMR12(sat_min);
  HSP_HW_IF_WRITE_PARAMR13(sat_max);
  HSP_HW_IF_WRITE_PARAMR14((uint32_t)pBQ);
  HSP_HW_IF_WRITE_PARAMR15(HSP_CNN_CFG_MODE_FCPP);
  /* And finally sync with DCMD using DCMDPTR registers */
  HSP_HW_IF_WRITE_DCMDPTR0((uint32_t)pSrcA);
  HSP_HW_IF_WRITE_DCMDPTR1((uint32_t)pSrcB);
  HSP_HW_IF_WRITE_DCMDPTR2((uint32_t)pDst);

  /* Load all coeffs */
  uint32_t nbPadK0 = 0;
  if ((in_c) & 0x3)
  {
    /* Not aligned on 4, add extra kernels to 0 */
    uint8_memb_t *tmpB = pSrcB;
    int8_t *tmpKe = p_filter_data;
    nbPadK0 = (4 - (in_c) & 0x3);
    HSP_MEMCPY((int8_t *)tmpB, (int8_t *)tmpKe, (in_c));
    tmpKe += (in_c);
    tmpB += (in_c);
    for (uint32_t kp = 0; kp < nbPadK0; kp++)
    {
      *tmpB++ = 0;
    }
  }
  else
  {
    HSP_MEMCPY((int8_t *)pSrcB, (int8_t *)p_filter_data, (in_c));
  }
  /* Now copy bias */
  HSP_MEMCPY((int8_t *)pBQ, (int8_t *)p_bias_data, (ou_c * 4));
  /* Load first input tile */
  HSP_MEMCPY((int8_t *)pSrcA, (int8_t *)p_input_data, (in_c));

  /* Wait H2CSEMR for HSP done, but check FWERR first */
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
  if (HAL_HSP_GetFirmwareError(((HSP_HandleTypeDef *)(hmw->hdriver))))
  {
    free_all_ai(&(hmw->hbram));
    return HSP_CORE_ERROR;
  }
  uint8_t *pKerCur = (uint8_t *)(p_filter_data + (in_c));
  uint8_t *pSrcbCur = pSrcB + (in_c + nbPadK0);
  uint8_t *pSrcbEnd = pSrcB + ((in_c + nbPadK0) * 2);
  /* Clear sem before send event */
  HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
  __DSB();
#endif /* STM32H7P5xx */

  /* Send event through CDEG: WARNING: do we need to disable all other events??? */
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->CDEGR = HSP_CNN_CDEG_EVT;
  /* Loop to load all kernels in BRAM */
  for (uint32_t out_num = 1; out_num < ou_c; out_num++)
  {
    /* Copy kernel block */
    if (nbPadK0)
    {
      /* Not aligned on 4, add extra kernels to 0 */
      HSP_MEMCPY((int8_t *)pSrcbCur, (int8_t *)pKerCur, (in_c));
      pKerCur += (in_c);
      pSrcbCur += (in_c);
      for (uint32_t kp = 0; kp < nbPadK0; kp++)
      {
        *pSrcbCur++ = 0;
      }
    }
    else
    {
      HSP_MEMCPY((int8_t *)pSrcbCur, (int8_t *)pKerCur, in_c);
      pKerCur += in_c;
      pSrcbCur += in_c;
    }
    if (pSrcbCur == pSrcbEnd)
    {
      pSrcbCur = pSrcB;
    }
    /* Wait output channel done */
    while (HAL_HSP_MSGBOX_IsMsgAvailable(((HSP_HandleTypeDef *)(hmw->hdriver))) == 0U); /* kernel ready */
    /* Clear sem */
    HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
    __DSB();
#endif /* STM32H7P5xx */

    /* Send event through CDEG: WARNING: do we need to disable all other events??? */
    ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->CDEGR = HSP_CNN_CDEG_EVT; /* send event */
  }
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->CR &= (~HSP_CR_BARB_Msk); /* Clear BARB */
  /* Wait last output line */
  while (HAL_HSP_MSGBOX_IsMsgAvailable(((HSP_HandleTypeDef *)(hmw->hdriver))) == 0U); /* kernel ready */
  /* Clear sem */
  HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
  __DSB();
#endif  /* STM32H7P5xx */

  /* Copy result in external memory */
  HSP_MEMCPY((int8_t *)p_output_data, (int8_t *)pDst, (ou_c));
#ifdef CNN_EVTENR_SUPPORT
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->EVTENR = evtenrSave; /* Restore previous EVTENR value */
#endif /* CNN_EVTENR_SUPPORT */
  /* Free all memory */
  free_all_ai(&(hmw->hbram));
  return HSP_CORE_OK;
}

/**
  * @brief Direct CNN Pooling function
  * HSP send error report via FWERR register
  * @param hmw                  HSP handle.
  * @param in_w                 Input dimension width
  * @param in_h                 Input dimension height
  * @param in_c                 Input dimension channel
  * @param k_w                  Kernel dimension width
  * @param k_h                  Kernel dimension height
  * @param ou_w                 Output dimension width
  * @param ou_h                 Output dimension height
  * @param ou_c                 Output dimension channel
  * @param stridex              Stride on X
  * @param stridey              Stride on Y
  * @param *p_input_data        Input data pointer, int8_t data type
  * @param *p_output_data       Output data pointer, int8_t data type
  * @param sat_min              Min sat (Relu), int32_t data type
  * @param sat_max              Max sat (Relu), int32_t data type
  * @param pool_type            Pooling type, 1 is avg, 0 is max
  * @retval                     Core status.
 */
hsp_core_status_t HSP_ACC_CnnPool0_s8(hsp_core_handle_t *hmw,
                         uint32_t in_w, uint32_t in_h, uint32_t in_c, uint32_t k_w, uint32_t k_h,
						 uint32_t ou_w, uint32_t ou_h, uint32_t ou_c, uint32_t stridex, uint32_t stridey,
						 int8_t *p_input_data, int8_t *p_output_data,
                         uint32_t sat_min, uint32_t sat_max, uint32_t pool_type)
{
  uint8_mema_t *pSrcA;
  uint8_memb_t *pDst;

  /* First of all, wakeup HSP with Direct command */
  HSP_HW_IF_WRITE_DCMDIDR((uint32_t) HSP_DIRECT_CMD_CNN_POOL_I8);

#ifdef CNN_EVTENR_SUPPORT
  /* Save current EVTENR and clear all enable event for ARM/HSP CNN synchro */
  uint32_t evtenrSave = ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->EVTENR;
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->EVTENR = HSP_CNN_EVTENR_CLR;
#endif /* CNN_EVTENR_SUPPORT */

  uint32_t circiLines = (k_h + stridey); /* For circular mode */
  uint32_t nbLinePerBlocks = ou_h;
  circiLines = in_h;
  /* Put input image in MEMA */
  if ((pSrcA = (uint8_mema_t *)alloc_in_memA(&(hmw->hbram), in_w * in_h * in_c)) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  /* Put output images in MEMA */
  if ((pDst  = (uint8_memb_t *)alloc_in_memB(&(hmw->hbram), ou_w * ou_h * ou_c)) == NULL)
  {
    return HSP_CORE_ERROR;
  }

  /* Then update all parameters */
  HSP_HW_IF_WRITE_PARAMR0(in_w);
  HSP_HW_IF_WRITE_PARAMR1(in_h);
  HSP_HW_IF_WRITE_PARAMR2(in_c);
  HSP_HW_IF_WRITE_PARAMR3(k_w);
  HSP_HW_IF_WRITE_PARAMR4(k_h);
  HSP_HW_IF_WRITE_PARAMR5(ou_w);
  HSP_HW_IF_WRITE_PARAMR6(ou_h);
  HSP_HW_IF_WRITE_PARAMR7(ou_c);
  HSP_HW_IF_WRITE_PARAMR8(stridex);
  HSP_HW_IF_WRITE_PARAMR9(stridey);
  HSP_HW_IF_WRITE_PARAMR12(sat_min);
  HSP_HW_IF_WRITE_PARAMR13(sat_max);
  HSP_HW_IF_WRITE_PARAMR14(pool_type); /* type; */
  HSP_HW_IF_WRITE_PARAMR15((circiLines << HSP_CNN_CFG_NB_IN_LINE_SHIFT) |
                           (nbLinePerBlocks << HSP_CNN_CFG_NB_OU_LINE_SHIFT));
  /* And finally sync with DCMD using DCMDPTR registers */
  HSP_HW_IF_WRITE_DCMDPTR0((uint32_t)pSrcA);
  HSP_HW_IF_WRITE_DCMDPTR2((uint32_t)pDst);

  HSP_MEMCPY((int8_t *)pSrcA, (int8_t *)p_input_data, (in_c * in_h * in_w));

  /* Wait H2CSEMR for HSP done, but check FWERR first */
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
  if (HAL_HSP_GetFirmwareError(((HSP_HandleTypeDef *)(hmw->hdriver))))
  {
    free_all_ai(&(hmw->hbram));
    return HSP_CORE_ERROR;
  }
  /* Clear sem before send event */
  HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
  __DSB();
#endif /* STM32H7P5xx */

  /* Send event through CDEG: WARNING: do we need to disable all other events??? */
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->CDEGR = HSP_CNN_CDEG_EVT;
  /* Wait direct command done */
  while (HAL_HSP_MSGBOX_IsMsgAvailable(((HSP_HandleTypeDef *)(hmw->hdriver))) == 0U);
  /* Clear sem */
  HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
  __DSB();
#endif  /* STM32H7P5xx */  

  /* Copy result in external memory */
  HSP_MEMCPY((int8_t *)p_output_data, (int8_t *)pDst, (ou_c * ou_h * ou_w));
#ifdef CNN_EVTENR_SUPPORT
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->EVTENR = evtenrSave; /* Restore previous EVTENR value */
#endif /* CNN_EVTENR_SUPPORT */

  /* Free all memory */
  free_all_ai(&(hmw->hbram));
  return HSP_CORE_OK;
}

/**
  * @brief Direct CNN Pooling function with circular in and ping pong on nbOlines out.
  *        For Avg pool, padding is forbidden
  * HSP send error report via FWERR register
  * @param hmw                  HSP handle.
  * @param in_w                 Input dimension width
  * @param in_h                 Input dimension height
  * @param in_c                 Input dimension channel
  * @param k_w                  Kernel dimension width
  * @param k_h                  Kernel dimension height
  * @param ou_w                 Output dimension width
  * @param ou_h                 Output dimension height
  * @param ou_c                 Output dimension channel
  * @param stridex              Stride on X
  * @param stridey              Stride on Y
  * @param *p_input_data        Input data pointer, int8_t data type
  * @param *p_output_data       Output data pointer, int8_t data type
  * @param sat_min              Min sat (Relu), int32_t data type
  * @param sat_max              Max sat (Relu), int32_t data type
  * @param padx_l               Pad on X left (only supported for Maxpool)
  * @param padx_r               Pad on X right (only supported for Maxpool)
  * @param pady_t               Pad on Y top (only supported for Maxpool)
  * @param pady_b               Pad on Y bottom (only supported for Maxpool)
  * @param pool_type            0: max pool 1: avg pool
  * @param nb_line_per_blocks   Number of output lines per block
  * @retval                     Core status.
 */
hsp_core_status_t HSP_ACC_CnnPool1_s8(hsp_core_handle_t *hmw,
                         uint32_t in_w, uint32_t in_h, uint32_t in_c,
                         uint32_t k_w, uint32_t k_h,
                         uint32_t ou_w, uint32_t ou_h, uint32_t ou_c,
                         uint32_t stridex, uint32_t stridey,
                         int8_t *p_input_data, int8_t *p_output_data, uint32_t sat_min, uint32_t sat_max,
                         uint32_t padx_l, uint32_t padx_r, uint32_t pady_t, uint32_t pady_b,
                         uint32_t pool_type, uint32_t nb_line_per_blocks)
{
  uint8_mema_t *pSrcA;
  uint8_memb_t *pDst;

  /* ToDo:
  HSP_CHECK_ASSERT_NRV(hhsp, (nbOlines > 1));
  HSP_CHECK_ASSERT_NRV(hhsp, (nbOlines <= ou_h));
  Check padding = 0 for AVG POOL !!! */

  /* First of all, wakeup HSP with Direct command */
  HSP_HW_IF_WRITE_DCMDIDR((uint32_t) HSP_DIRECT_CMD_CNN_POOL_I8);

#ifdef CNN_EVTENR_SUPPORT
  /* Save current EVTENR and clear all enable event for ARM/HSP CNN synchro */
  uint32_t evtenrSave = ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->EVTENR;
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->EVTENR = HSP_CNN_EVTENR_CLR;
#endif /* CNN_EVTENR_SUPPORT */

  /* Then update all parameters */
  uint32_t pad = padx_l + padx_r + pady_t + pady_b; /* Pad option */
  uint32_t inp_w = in_w + (padx_l + padx_r);
  uint32_t inp_h = in_h + (pady_t + pady_b);

  uint32_t inLineSize8 = in_c * inp_w; /* 1 input line */
  uint32_t outLineSize8 = ou_c * ou_w; /* 1 output line */
  uint32_t circiLines = (k_h + stridey); /* For circular mode */
  uint32_t nbLinePerBlocks = nb_line_per_blocks;
  if (circiLines > inp_h)
  {
    circiLines = inp_h;
  }
  uint32_t circiSize = inLineSize8 * circiLines; /* For circular mode */
  /* Put input image in MEMA */
  if ((pSrcA = (uint8_mema_t *)alloc_in_memA(&(hmw->hbram), circiSize)) == NULL)
  {
    return HSP_CORE_ERROR;
  }
  /* Put output images in MEMA */
  if ((pDst  = (uint8_memb_t *)alloc_in_memB(&(hmw->hbram), nb_line_per_blocks * outLineSize8)) == NULL)
  {
    return HSP_CORE_ERROR;
  }

  HSP_HW_IF_WRITE_PARAMR0(inp_w);
  HSP_HW_IF_WRITE_PARAMR1(inp_h);
  HSP_HW_IF_WRITE_PARAMR2(in_c);
  HSP_HW_IF_WRITE_PARAMR3(k_w);
  HSP_HW_IF_WRITE_PARAMR4(k_h);
  HSP_HW_IF_WRITE_PARAMR5(ou_w);
  HSP_HW_IF_WRITE_PARAMR6(ou_h);
  HSP_HW_IF_WRITE_PARAMR7(ou_c);
  HSP_HW_IF_WRITE_PARAMR8(stridex);
  HSP_HW_IF_WRITE_PARAMR9(stridey);
  HSP_HW_IF_WRITE_PARAMR12(sat_min);
  HSP_HW_IF_WRITE_PARAMR13(sat_max);
  HSP_HW_IF_WRITE_PARAMR14(pool_type); /* type; */
  HSP_HW_IF_WRITE_PARAMR15((circiLines << HSP_CNN_CFG_NB_IN_LINE_SHIFT) |
                           (nbLinePerBlocks << HSP_CNN_CFG_NB_OU_LINE_SHIFT) | HSP_CNN_CFG_MODE_PCIRC);
  /* And finally sync with DCMD using DCMDPTR registers */
  HSP_HW_IF_WRITE_DCMDPTR0((uint32_t)pSrcA);
  HSP_HW_IF_WRITE_DCMDPTR2((uint32_t)pDst);

  /* Load first input tile */
  uint8_t val = 0x80;
  if (pad)
  {
    uint8_mema_t *inPad = pSrcA;
    uint32_t nbPadTop = 0;
   /* First lines pad */
    HSP_MEMSET(inPad, val, (pady_t *((in_w + (padx_l + padx_r)) * in_c)));
    inPad += (pady_t *((in_w + (padx_l + padx_r)) * in_c));
    /* Middle lines to pad */
    uint32_t lineToPad = k_h - pady_t;
    if (lineToPad > in_h)
    {
      nbPadTop = lineToPad - in_h;
      lineToPad =  in_h;
    }    
    for (uint32_t i = 0; i < lineToPad; i++)
    {
      /* First column pad */
      HSP_MEMSET(inPad, val, (padx_l * in_c));
      inPad += (padx_l * in_c);
      /* Copy image line */
      HSP_MEMCPY((int8_t *)inPad, (int8_t *)p_input_data, (in_w * in_c));
      inPad += (in_w * in_c);
      p_input_data += (in_w * in_c);
      /* Last column pad */
      HSP_MEMSET(inPad, val, (padx_r * in_c));
      inPad += (padx_r * in_c);
    }
    /* Add bottom pad if necessery */
    if (nbPadTop)
    {
      /* last lines pad */
      HSP_MEMSET(inPad, val, (nbPadTop *((in_w + (padx_l + padx_r)) * in_c)));
      inPad += (nbPadTop *((in_w + (padx_l + padx_r)) * in_c));
    }
  }
  else
  {
    HSP_MEMCPY((int8_t *)pSrcA, (int8_t *)p_input_data, (inLineSize8 * k_h));
    p_input_data += (inLineSize8 * k_h);
  }
  /* Wait H2CSEMR for HSP done, but check FWERR first */
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(hmw);
  if (HAL_HSP_GetFirmwareError(((HSP_HandleTypeDef *)(hmw->hdriver))))
  {
    free_all_ai(&(hmw->hbram));
    return HSP_CORE_ERROR;
  }
  uint8_t *pSrcEnd = pSrcA + (circiSize);
  uint8_t *pSrcCur = pSrcA + (inLineSize8 * k_h);
  uint8_t *pDstEnd = (uint8_t *)(pDst + (nbLinePerBlocks * outLineSize8));
  uint8_t *pDstCur = (uint8_t *) pDst;
  uint32_t countlines = k_h;
  uint32_t countmax = inp_h - pady_b;
  uint32_t xloop = inLineSize8 - ((padx_l + padx_r) * in_c);
  /* Send event through CDEG: WARNING: do we need to disable all other events??? */
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->CDEGR = HSP_CNN_CDEG_EVT;

  for (uint32_t lineIdx = 0; lineIdx < (ou_h - 1); lineIdx++)
  {
    /* Copy input line in circular buffer */
    for (uint32_t sidx = 0; sidx < stridey; sidx++)
    {
      if (countlines == countmax)
      {
        /* Last lines pad */
        HSP_MEMSET(pSrcCur, val, inLineSize8);
        pSrcCur += inLineSize8;
      }
      else
      {
        countlines++;
        if (pad)
        {
          /* First column pad */
          HSP_MEMSET(pSrcCur, val, (padx_l * in_c));
          pSrcCur += (padx_l * in_c);
          HSP_MEMCPY((int8_t *)pSrcCur, (int8_t *)p_input_data, xloop);
          pSrcCur += xloop;
          p_input_data += xloop;
          /* Last column pad */
          HSP_MEMSET(pSrcCur, val, (padx_r * in_c));
          pSrcCur += (padx_r * in_c);
        }
        else
        {
          HSP_MEMCPY((int8_t *)pSrcCur, (int8_t *)p_input_data, inLineSize8);
          p_input_data += inLineSize8;
          pSrcCur += inLineSize8;
        }
      }
      if (pSrcCur == pSrcEnd)
      {
        pSrcCur = pSrcA;
      }	  
    }
    /* Wait output line */
    while (HAL_HSP_MSGBOX_IsMsgAvailable(((HSP_HandleTypeDef *)(hmw->hdriver))) == 0U);
    /* Clear sem */
    HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
    __DSB();
#endif /* STM32H7P5xx */

    /* Send event through CDEG: WARNING: do we need to disable all other events??? */
    ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->CDEGR = HSP_CNN_CDEG_EVT;
    HSP_MEMCPY((int8_t *)p_output_data, (int8_t *)pDstCur, outLineSize8);
    p_output_data += outLineSize8;
    pDstCur += outLineSize8;
    if (pDstCur == pDstEnd)
    {
      pDstCur = pDst;
    }
  }
  /* Wait last output line */
  while (HAL_HSP_MSGBOX_IsMsgAvailable(((HSP_HandleTypeDef *)(hmw->hdriver))) == 0U); /* kernel ready */

  HSP_MEMCPY((int8_t *)p_output_data, (int8_t *)pDstCur, outLineSize8);

  /* Clear sem */
  HAL_HSP_MSGBOX_ReleaseSemaphore(((HSP_HandleTypeDef *)(hmw->hdriver)));
#ifdef STM32H7P5xx
  __DSB();
#endif  /* STM32H7P5xx */  

#ifdef CNN_EVTENR_SUPPORT
  ((HSP_HandleTypeDef *)(hmw->hdriver))->Instance->EVTENR = evtenrSave; /* Restore previous EVTENR value */
#endif /* CNN_EVTENR_SUPPORT */

  /* Free all memory */
  free_all_ai(&(hmw->hbram));
  return HSP_CORE_OK;
}

/**
  * @}
  */
/**
  * @addtogroup HSP_MODULES_CNN_Private_Functions
  * @{
  */
/**
  * @brief compute integer factor and shift to get results in Integer arithmetic
  *        with a per channel dimension using cmsis format (Shift)
  *        factor and shift are interleaved in output
  * @param   in_scale      input scale
  * @param   out_scale     output scale
  * @param   p_wt_scale    pointer in weight scales (one per output channel)
  * @param   ch_im_out     number of output channels
  * @param   p_out_factor  pointer on output scale factors
  * @param   p_out_r_shift pointer on output shift values
  * @param   p_out         pointer on output: scale factor and shift are interleaved
  * @return nothing
  */
static void align_factor_cmsisnn_fast_ch(float32_t in_scale, float32_t out_scale,
                                         const float32_t *p_wt_scale, uint16_t ch_im_out, int32_t *p_out)
{
  for (uint32_t i = 0, j = 0; i < ch_im_out; i++, j += 2)
  {
    float32_t   out_align_factor = (p_wt_scale[i] * in_scale / out_scale);
    uint32_t out_align_factor_u32 = *((uint32_t *) &out_align_factor);
    int32_t OutRShift;
    int32_t OutFactor;
    OutRShift = 126 - ((out_align_factor_u32 & 0x7f800000) >> 23);
    OutFactor = ((out_align_factor_u32 & 0x7FFFFF) + 0x800000);
    if (out_align_factor_u32 & 0x80000000)
    {
      OutFactor = -OutFactor;
    }
    if (OutRShift > 31)
    {
      OutFactor = 0;
      OutRShift = 0;
    }
    OutFactor <<= 7;
    p_out[j] = OutFactor;
    p_out[j + 1] = -OutRShift;
  }
}

/**
  * @brief compute integer factor and shift to get results in Integer arithmetic
  *        with a per channel dimension using cmsis format (Shift).
  *        factor, shift and bias are interleaved in output
  * @param   in_scale      input scale
  * @param   out_scale     output scale
  * @param   p_wt_scale    pointer in weight scales (one per output channel)
  * @param   p_bias_data   pointer in bias data (one per output channel)
  * @param   ch_im_out     number of output channels
  * @param   p_out_factor  pointer on output scale factors
  * @param   p_out_r_shift pointer on output shift values
  * @param   p_out         pointer on output: scale factor and shift are interleaved
  * @return nothing
  */
static void align_factor_cmsisnn_fast_ch_v2(float32_t in_scale, float32_t out_scale, const float32_t *p_wt_scale,
                                            int32_t *p_bias_data, uint16_t ch_im_out, int32_t *p_out)
{
  for (uint32_t i = 0, j = 0; i < ch_im_out; i++, j += 3)
  {
    float32_t   out_align_factor = (p_wt_scale[i] * in_scale / out_scale);
    uint32_t out_align_factor_u32 = *((uint32_t *) &out_align_factor);
    int32_t OutRShift;
    int32_t OutFactor;
    OutRShift = 126 - ((out_align_factor_u32 & 0x7f800000) >> 23);
    OutFactor = ((out_align_factor_u32 & 0x7FFFFF) + 0x800000);
    if (out_align_factor_u32 & 0x80000000)
    {
      OutFactor = -OutFactor;
    }
    if (OutRShift > 31)
    {
      OutFactor = 0;
      OutRShift = 0;
    }
    OutFactor <<= 7;
    p_out[j] = p_bias_data[i];
    p_out[j + 1] = OutFactor;
    p_out[j + 2] = -OutRShift;
  }
}

/**
  * @brief compute only one integer factor and shift to get results in Integer arithmetic
  *         using cmsis format (Shift).
  * @param   in_scale      input scale
  * @param   out_scale     output scale
  * @param   p_wt_scale    weight scales
  * @param   p_out_factor  pointer on output scale factors
  * @param   p_out_shift   pointer on output shift values
  * @param   p_out         pointer on output: scale factor and shift are interleaved
  * @return nothing
  */
static void align_factor_cmsisnn_fast_ch_v3(float32_t in_scale, float32_t out_scale, float32_t wt_scale,
                                            int32_t *p_out_factor, int32_t *p_out_shift)
{

  float32_t   out_align_factor = (wt_scale * in_scale / out_scale);
  uint32_t out_align_factor_u32 = *((uint32_t *) &out_align_factor);
  int32_t OutRShift;
  int32_t OutFactor;
  OutRShift = 126 - ((out_align_factor_u32 & 0x7f800000) >> 23);
  OutFactor = ((out_align_factor_u32 & 0x7FFFFF) + 0x800000);
  if (out_align_factor_u32 & 0x80000000)
  {
    OutFactor = -OutFactor;
  }
  if (OutRShift > 31)
  {
    OutFactor = 0;
    OutRShift = 0;
  }
  OutFactor <<= 7;
  *p_out_factor = OutFactor;
  *p_out_shift = -OutRShift;
}

/**
  * @brief memory allocation in memA in AI area
  * @param   size_in_byte size in byte of allocation
  * @return  address in memA if allocation succeed, 0 if allocation failed
  */
static int8_mema_t *alloc_in_memA(hsp_bram_handle_t *hhsp_bram, uint32_t size_in_byte)
{
  uint32_t addr;
  uint32_t sizeInWord = (size_in_byte + (sizeof(uint32_t) - 1)) / sizeof(uint32_t); /* Align on Word */

  /* First check if AI area is defined */
  if ((HSP_BRAM_AI_SIZE == 0U) || (sizeInWord > hhsp_bram->maxSizeToAllocateA))
  {
    return (NULL);
  }

  /* Update the current A shared offset */
  hhsp_bram->currentSharedOffsetA += sizeInWord; /* currentSharedOffset is calculate in word */
  /* Next free A address */
  addr = (uint32_t)(hhsp_bram->currentSharedAddrA);
  hhsp_bram->currentSharedAddrA += (sizeInWord * sizeof(uint32_t));
  /* Calculate shared remaining size */
  hhsp_bram->maxSizeToAllocateA -= sizeInWord;

  return ((int8_mema_t *)addr);
}

/**
  * @brief memory allocation in memB in AI area
  * @param   size_in_byte size in byte of allocation
  * @return  address in memB if allocation succeed, 0 if allocation failed
  */
static int8_memb_t *alloc_in_memB(hsp_bram_handle_t *hhsp_bram, uint32_t size_in_byte)
{
  uint32_t addr;
  uint32_t sizeInWord = (size_in_byte + (sizeof(uint32_t) - 1)) / sizeof(uint32_t); /* Align on Word */

  /* First check if AI area is defined */
  if ((HSP_BRAM_AI_SIZE == 0) || (sizeInWord > hhsp_bram->maxSizeToAllocateB))
  {
    return (NULL);
  }

  /* Update the current A shared offset */
  hhsp_bram->currentSharedOffsetB += sizeInWord; /* currentSharedOffset is calculate in word */
  /* Next free A address */
  addr = (uint32_t)(hhsp_bram->currentSharedAddrB);
  hhsp_bram->currentSharedAddrB += (sizeInWord * sizeof(uint32_t));
  /* Calculate shared remaining size */
  hhsp_bram->maxSizeToAllocateB -= sizeInWord;

  return ((int8_memb_t *)addr);
}

/**
  * @brief memory allocation in memB in AI area
  * @param   size_in_byte size in byte of allocation
  * @return  address in memB if allocation succeed, 0 if allocation failed
  */
static void free_all_ai(hsp_bram_handle_t *hhsp_bram)
{
  hhsp_bram->currentSharedOffsetA = 0;
  hhsp_bram->currentSharedOffsetB = 0;
  hhsp_bram->currentSharedAddrA = hhsp_bram->baseSharedAddrA;
  hhsp_bram->currentSharedAddrB = hhsp_bram->baseSharedAddrB;
  hhsp_bram->maxSizeToAllocateA = HSP_BRAM_AI_SIZE;
  hhsp_bram->maxSizeToAllocateB = HSP_BRAM_AI_SIZE;
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
