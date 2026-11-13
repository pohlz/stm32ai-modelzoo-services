/**
  ******************************************************************************
  * @file    preproc_dpu.c
  * @author  MCD Application Team
  * @brief   This file is implementing pre-processing functions that are making
  * 		 use of Audio pre-processing libraries
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2023 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */


/* Includes ------------------------------------------------------------------*/
#include "preproc_dpu.h"
#include "main.h"

#ifdef USE_HSP
#include "hsp_cnn.h"
#include "hsp_direct_command.h"
#include "hsp_bram.h"
#endif

#include "cpu_stats.h"
#include "ai_device_adaptor.h"
      
/* Private defines ---------------------------------------------------------- */
#define AI_DPU_G_TO_MS_2 (9.8F)
#define AI_LSB_16B       (1.0F/32768)

/* External functions --------------------------------------------------------*/
#ifdef USE_HSP
#include "hsp_memcpy.h"
extern hsp_core_handle_t hmw;
#define memcpy(dst, src, size)  stm32_hsp_memcpy((int8_t *) dst, (int8_t *) src, size)
#endif

/* External functions --------------------------------------------------------*/
/**
 * @brief initializes preprocessing for Motion
 * @param nb_3_axis_sample: number of 3D acceleration samples
 * @retval DPU_OK if success, DPU_ERROR else
 */
DPU_StatusTypeDef MotionPreProc_DPUInit(MotionPreProcCtx_t *pxCtx)
{
  LogInfo("\n\rPreprocessing\n\r");
  LogInfo(SEPARATION_LINE);
#if   (CTRL_AI_FFT            != CTRL_X_CUBE_AI_PREPROC )
  #if   (CTRL_AI_BYPASS          == CTRL_X_CUBE_AI_PREPROC )
    LogInfo("Bypass\n\r");
  #elif ( CTRL_AI_GRAV_ROT       == CTRL_X_CUBE_AI_PREPROC )
    LogInfo("Gravity Rotation\n\r");
  #elif ( CTRL_AI_GRAV_ROT_SUPPR == CTRL_X_CUBE_AI_PREPROC )
    LogInfo("Gravity Rotation & Suppress \n\r");
  #elif ( CTRL_AI_SCALING        == CTRL_X_CUBE_AI_PREPROC )
    LogInfo("Scaling Only\n\r");
  #endif
    LogInfo("- sampling frequency : %lu Hz\n\r",(uint32_t) (CTRL_X_CUBE_AI_SENSOR_ODR));
    LogInfo("- processing period  : %.3f s\n\r",((float)  pxCtx->in_height) / CTRL_X_CUBE_AI_SENSOR_ODR);
#elif (CTRL_AI_FFT            == CTRL_X_CUBE_AI_PREPROC )
  /* Init RFFT */
  arm_rfft_fast_init_f32(&pxCtx->S_Rfft, CTRL_X_CUBE_AI_SPECTROGRAM_NFFT);
  /* Init Spectrogram */
  pxCtx->FFTLen                   = CTRL_X_CUBE_AI_SPECTROGRAM_NFFT;
  LogInfo("3D FFT %u x 3 \n\r",CTRL_X_CUBE_AI_SPECTROGRAM_NFFT/2 );
  LogInfo("- sampling freq : %lu Hz\n\r",(uint32_t)CTRL_X_CUBE_AI_SENSOR_ODR);
  LogInfo("- acq period    : %lu ms\n\r",CTRL_X_CUBE_AI_ACQ_LENGTH_MS);
#endif  
  
  return DPU_OK;
}

/**
 * @brief preprocessing for Motion
 * @param p_in pointer to input frame
 * @param p_out pointer to output frame
 * @param nb_3_axis_sample: number of 3D acceleration samples
 * @retval DPU_OK if success, DPU_ERROR else
 */
#include "audio_din.h"
#include "dsp/transform_functions.h"

DPU_StatusTypeDef MotionPreProc_DPU (MotionPreProcCtx_t *pxCtx, int16_t *p_in, float *p_out)
{
  port_dwt_reset();
#if (CTRL_AI_FFT            == CTRL_X_CUBE_AI_PREPROC )  
  audio_is16of32_pad(p_in, pxCtx->pSpectrScratchBuffer1,0, pxCtx->in_height, 0);
  arm_rfft_fast_f32(&pxCtx->S_Rfft,pxCtx->pSpectrScratchBuffer1, pxCtx->pSpectrScratchBuffer2, 0);
  arm_cmplx_mag_f32(pxCtx->pSpectrScratchBuffer2,p_out,pxCtx->in_height/2);
 
  audio_is16of32_pad(p_in + pxCtx->in_height, pxCtx->pSpectrScratchBuffer1,0, pxCtx->in_height, 0);
  arm_rfft_fast_f32(&pxCtx->S_Rfft,pxCtx->pSpectrScratchBuffer1, pxCtx->pSpectrScratchBuffer2, 0);
  arm_cmplx_mag_f32(pxCtx->pSpectrScratchBuffer2,p_out + pxCtx->in_height/2 ,pxCtx->in_height/2);
 
  audio_is16of32_pad(p_in + 2 * pxCtx->in_height, pxCtx->pSpectrScratchBuffer1,0, pxCtx->in_height, 0);
  arm_rfft_fast_f32(&pxCtx->S_Rfft,pxCtx->pSpectrScratchBuffer1, pxCtx->pSpectrScratchBuffer2, 0);
  arm_cmplx_mag_f32(pxCtx->pSpectrScratchBuffer2,p_out + pxCtx->in_height ,pxCtx->in_height/2);

  float scale = 1/(float)(pxCtx->FFTLen);
  int nb_loop = pxCtx->in_height*pxCtx->in_width;
  for (int i= 0 ; i < nb_loop ; i ++ )
  {
    *p_out++ *= scale ;
  }
#elif ( CTRL_AI_GRAV_ROT_SUPPR == CTRL_X_CUBE_AI_PREPROC ||  CTRL_AI_GRAV_ROT == CTRL_X_CUBE_AI_PREPROC)
  float buff [ 24 * 3 ];
  float scale = CTRL_X_CUBE_AI_SENSOR_FS /** AI_LSB_16B*/ * AI_DPU_G_TO_MS_2 ;
  acceleration_3D_t gravIn;
  acceleration_3D_t gravOut;
  audio_is16of32_pad(p_in, buff,0, pxCtx->in_height * pxCtx->in_width, 0);
  arm_scale_f32 (buff, scale , buff , pxCtx->in_height * pxCtx->in_width);
  for (int i=0 ; i < pxCtx->in_height; i++)  {
    gravIn.AccX = buff[i ];
    gravIn.AccY = buff[i + pxCtx->in_height];
    gravIn.AccZ = buff[i + 2 * pxCtx->in_height];
#if CTRL_X_CUBE_AI_PREPROC==CTRL_AI_GRAV_ROT_SUPPR
    gravOut = gravity_suppress_rotate (&gravIn);
#elif CTRL_X_CUBE_AI_PREPROC==CTRL_AI_GRAV_ROT
    gravOut = gravity_rotate (&gravIn);
#endif
    *p_out++ = gravOut.AccX;
    *p_out++ = gravOut.AccY;
    *p_out++ = gravOut.AccZ;
  }
#elif CTRL_X_CUBE_AI_PREPROC==CTRL_AI_SCALING
#else /* bypass */
#endif
  
  time_stats_store(TIME_STAT_PRE_PROC,port_dwt_get_cycles()*1000.0F/port_hal_get_cpu_freq());
  return DPU_OK;
}

/**
 * @brief initializes preprocessing for Audio
 * @param pxCtx pointer to preprocessing context
 * @retval DPU_OK if success, DPU_ERROR else
 */
DPU_StatusTypeDef AudioPreProc_DPUInit(AudioPreProcCtx_t *pxCtx)
{
  uint32_t pad;

  assert_param( CTRL_X_CUBE_AI_SPECTROGRAM_NFFT >= CTRL_X_CUBE_AI_SPECTROGRAM_WINDOW_LENGTH );
  assert_param( CTRL_X_CUBE_AI_SPECTROGRAM_NFFT >= CTRL_X_CUBE_AI_SPECTROGRAM_NMEL);

  pxCtx->out_Q_inv_scale = 0.0F;
  pxCtx->out_Q_offset  = 0;

  /* Init RFFT */
#ifndef USE_HSP
  arm_rfft_fast_init_f32(&pxCtx->S_Rfft, CTRL_X_CUBE_AI_SPECTROGRAM_NFFT);

  /* Init Spectrogram */
  pxCtx->S_Spectr.pRfft                    = &pxCtx->S_Rfft;
#endif
  pxCtx->S_Spectr.Type                     = CTRL_X_CUBE_AI_SPECTROGRAM_TYPE;
#ifdef USE_HSP
  pxCtx->S_Spectr.pWindow                  = (float32_t *)HSP_BRAM_Malloc(&hmw, CTRL_X_CUBE_AI_SPECTROGRAM_WINDOW_LENGTH/2, HSP_BRAM_ALLOCATION_DEFAULT);
#else
  pxCtx->S_Spectr.pWindow                  = (float32_t *) CTRL_X_CUBE_AI_SPECTROGRAM_WIN;
#endif
  pxCtx->S_Spectr.SampRate                 = CTRL_X_CUBE_AI_SENSOR_ODR;
  pxCtx->S_Spectr.FrameLen                 = CTRL_X_CUBE_AI_SPECTROGRAM_WINDOW_LENGTH;
  pxCtx->S_Spectr.FFTLen                   = CTRL_X_CUBE_AI_SPECTROGRAM_NFFT;

#ifdef USE_HSP
  pxCtx->S_Spectr.pScratch1                = (float32_t *) HSP_BRAM_Malloc(&hmw, CTRL_X_CUBE_AI_SPECTROGRAM_NFFT, HSP_BRAM_ALLOCATION_DEFAULT);
  pxCtx->S_Spectr.pScratch2                = (float32_t *) HSP_BRAM_Malloc(&hmw, CTRL_X_CUBE_AI_SPECTROGRAM_NFFT, HSP_BRAM_ALLOCATION_DEFAULT);
#else
  pxCtx->S_Spectr.pScratch1                = pxCtx->pSpectrScratchBuffer1;
  pxCtx->S_Spectr.pScratch2                = pxCtx->pSpectrScratchBuffer2;
#endif
  
  pad                                      = CTRL_X_CUBE_AI_SPECTROGRAM_NFFT - CTRL_X_CUBE_AI_SPECTROGRAM_WINDOW_LENGTH;
  pxCtx->S_Spectr.pad_left                 = pad/2;
  pxCtx->S_Spectr.pad_right                = pad/2 + (pad & 0x1);

  /* Init mel filterbank */
#ifdef USE_HSP
  pxCtx->S_MelFilter.pStartIndices         = (uint32_t *)  HSP_BRAM_Malloc(&hmw, CTRL_X_CUBE_AI_SPECTROGRAM_NMEL, HSP_BRAM_ALLOCATION_DEFAULT);
  pxCtx->S_MelFilter.pBandLens             = (uint32_t *)  HSP_BRAM_Malloc(&hmw, CTRL_X_CUBE_AI_SPECTROGRAM_NMEL, HSP_BRAM_ALLOCATION_DEFAULT);
  pxCtx->S_MelFilter.pCoefficients         = (float32_t *) HSP_BRAM_Malloc(&hmw, 462, HSP_BRAM_ALLOCATION_DEFAULT);
#else
  pxCtx->S_MelFilter.pStartIndices         = (uint32_t *)  CTRL_X_CUBE_AI_SPECTROGRAM_MEL_START_IDX;
  pxCtx->S_MelFilter.pCoefficients         = (float32_t *) CTRL_X_CUBE_AI_SPECTROGRAM_MEL_LUT;
#endif
  pxCtx->S_MelFilter.pStopIndices          = (uint32_t *)  CTRL_X_CUBE_AI_SPECTROGRAM_MEL_STOP_IDX;
  pxCtx->S_MelFilter.NumMels               = CTRL_X_CUBE_AI_SPECTROGRAM_NMEL;
  pxCtx->S_MelFilter.FFTLen                = CTRL_X_CUBE_AI_SPECTROGRAM_NFFT;
  pxCtx->S_MelFilter.SampRate              = (uint32_t) CTRL_X_CUBE_AI_SENSOR_ODR;
  pxCtx->S_MelFilter.FMin                  = (float32_t) CTRL_X_CUBE_AI_SPECTROGRAM_FMIN;
  pxCtx->S_MelFilter.FMax                  = (float32_t) CTRL_X_CUBE_AI_SPECTROGRAM_FMAX;
  pxCtx->S_MelFilter.Formula               = CTRL_X_CUBE_AI_SPECTROGRAM_FORMULA;
  pxCtx->S_MelFilter.Normalize             = CTRL_X_CUBE_AI_SPECTROGRAM_NORMALIZE;
  pxCtx->S_MelFilter.Mel2F                 = 1U;

  /* Init MelSpectrogram */
  pxCtx->S_MelSpectr.SpectrogramConf       = &pxCtx->S_Spectr;
  pxCtx->S_MelSpectr.MelFilter             = &pxCtx->S_MelFilter;

  /* Init LogMelSpectrogram */
  pxCtx->S_LogMelSpectr.MelSpectrogramConf = &pxCtx->S_MelSpectr;
  pxCtx->S_LogMelSpectr.LogFormula         = CTRL_X_CUBE_AI_SPECTROGRAM_LOG_FORMULA;
  pxCtx->S_LogMelSpectr.Ref                = 1.0f;
  pxCtx->S_LogMelSpectr.TopdB              = HUGE_VALF;

#ifdef USE_HSP
    memcpy(pxCtx->S_Spectr.pWindow, CTRL_X_CUBE_AI_SPECTROGRAM_WIN, 4 * CTRL_X_CUBE_AI_SPECTROGRAM_WINDOW_LENGTH/2);
    memcpy(pxCtx->S_MelFilter.pStartIndices, CTRL_X_CUBE_AI_SPECTROGRAM_MEL_START_IDX, 4 * CTRL_X_CUBE_AI_SPECTROGRAM_NMEL);
    memcpy(pxCtx->S_MelFilter.pCoefficients, CTRL_X_CUBE_AI_SPECTROGRAM_MEL_LUT, 4 * 462);
    for(int i = 0; i < CTRL_X_CUBE_AI_SPECTROGRAM_NMEL; i++)
    {
        pxCtx->S_MelFilter.pBandLens[i] = ((uint32_t *)  CTRL_X_CUBE_AI_SPECTROGRAM_MEL_STOP_IDX)[i] - ((uint32_t *)  CTRL_X_CUBE_AI_SPECTROGRAM_MEL_START_IDX)[i] + 1;  
    }
#endif
    
  LogInfo("MEL spectrogram %d mel x %d col\n\r",CTRL_X_CUBE_AI_SPECTROGRAM_NMEL,CTRL_X_CUBE_AI_SPECTROGRAM_COL);
  LogInfo("- sampling freq : %u Hz\n\r",(uint32_t)CTRL_X_CUBE_AI_SENSOR_ODR);
  LogInfo("- acq period    : %u ms\n\r",(uint32_t)CTRL_X_CUBE_AI_ACQ_LENGTH_MS);
  LogInfo("- window length : %u samples\n\r",CTRL_X_CUBE_AI_SPECTROGRAM_WINDOW_LENGTH);
  LogInfo("- hop length    : %u samples\n\r",CTRL_X_CUBE_AI_SPECTROGRAM_HOP_LENGTH);

  return DPU_OK;
}

/**
 * @brief spectral preprocessing Audio samples
 * @param pxCtx pointer to preprocessing context
 * @param pDataIn pointer to audio impit samples
 * @param p_spectro pointer to output spectrgramme
 * @retval DPU_OK if success, DPU_ERROR else
 */
DPU_StatusTypeDef AudioPreProc_DPU(AudioPreProcCtx_t * pxCtx, uint8_t *pDataIn, int8_t *p_spectro)
{
  assert_param( NULL != pxCtx );
  assert_param( SPECTROGRAM_LOG_MEL == pxCtx->type );
  assert_param( CTRL_X_CUBE_AI_SPECTROGRAM_NMEL == pxCtx->S_MelFilter.NumMels);

  int16_t *p_in;
  static int8_t  out[CTRL_X_CUBE_AI_SPECTROGRAM_NMEL];

  port_dwt_reset();

#ifdef USE_HSP
    memcpy(pxCtx->S_Spectr.pWindow, CTRL_X_CUBE_AI_SPECTROGRAM_WIN, 4 * CTRL_X_CUBE_AI_SPECTROGRAM_WINDOW_LENGTH/2);
    memcpy(pxCtx->S_MelFilter.pStartIndices, CTRL_X_CUBE_AI_SPECTROGRAM_MEL_START_IDX, 4 * CTRL_X_CUBE_AI_SPECTROGRAM_NMEL);
    memcpy(pxCtx->S_MelFilter.pCoefficients, CTRL_X_CUBE_AI_SPECTROGRAM_MEL_LUT, 4 * 462);
    for(int i = 0; i < CTRL_X_CUBE_AI_SPECTROGRAM_NMEL; i++)
    {
        pxCtx->S_MelFilter.pBandLens[i] = ((uint32_t *)  CTRL_X_CUBE_AI_SPECTROGRAM_MEL_STOP_IDX)[i] - ((uint32_t *)  CTRL_X_CUBE_AI_SPECTROGRAM_MEL_START_IDX)[i] + 1;  
    }
#endif  
  
  /* Create a quantized Mel-scaled spectrogram column */
  for (uint32_t i = 0; i < CTRL_X_CUBE_AI_SPECTROGRAM_COL; i++ )
  {
    p_in = (int16_t *)pDataIn + CTRL_X_CUBE_AI_SPECTROGRAM_HOP_LENGTH * i;
    
    LogMelSpectrogramColumn_q15_Q8(&pxCtx->S_LogMelSpectr, p_in,out,pxCtx->out_Q_offset,pxCtx->out_Q_inv_scale);
    
#ifdef NO_TRANSPOSE
    memcpy(p_spectro + i *  pxCtx->S_MelFilter.NumMels, out, pxCtx->S_MelFilter.NumMels);
#else
    /* transpose */
    for (uint32_t j=0 ; j < pxCtx->S_MelFilter.NumMels ; j++ )
    {
      p_spectro[i+CTRL_X_CUBE_AI_SPECTROGRAM_COL*j]= out[j];
    }
#endif
    
  }
  time_stats_store(TIME_STAT_PRE_PROC,port_dwt_get_cycles()*1000.0F/port_hal_get_cpu_freq());
  return DPU_OK;
}
