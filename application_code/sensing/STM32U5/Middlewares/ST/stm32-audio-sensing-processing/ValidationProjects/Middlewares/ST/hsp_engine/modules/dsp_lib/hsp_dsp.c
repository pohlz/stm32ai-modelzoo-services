/**
  ******************************************************************************************
  * @file hsp_dsp.c
  * @brief API for HSP DSP functions
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
#include "hsp_dsp.h"
#include "hsp_bram.h"
#include "hsp_hw_if.h"
#include "hsp_utilities.h"

#define HSP_CHECK_CMD_SIZE_NULL(hmw, size)
#define HSP_CHECK_ASSERT(hmw, cond)

/** @addtogroup HSP
  * @{
  */

/** @addtogroup HSP_MODULES_DSP
  * @{
  */

/** @defgroup HSP_MODULES_DSP_Private_Variables
  * @{
  */
hsp_core_handle_t *stm32_cmsis_handle;
/**
  * @}
  */

/** @addtogroup HSP_MODULES_DIRECT_COMMAND_Exported_Functions
  * @{
  */
stm32_hsp_status stm32_hsp_init(hsp_core_handle_t *hmw)
{
  if (hmw == NULL)
  {
    return STM32_HSP_MATH_ARGUMENT_ERROR;
  }
  stm32_cmsis_handle = hmw;
  return STM32_HSP_MATH_SUCCESS;
}

/**
  * @private
  * @brief         Initialization function for the 32pt floating-point real FFT.
  * @param         S  points to an arm_rfft_fast_instance_f32 structure
  * @return        execution status
  *                  - \ref ARM_MATH_SUCCESS        : Operation successful
  *                  - \ref ARM_MATH_ARGUMENT_ERROR : an error is detected
  */
stm32_hsp_status stm32_hsp_rfft_fast_init_f32(stm32_hsp_rfft_fast_instance_f32 *S, uint16_t fftLen)
{
  S->pTwiddleRFFT = NULL;
  S->Sint.pTwiddle = NULL;
  S->Sint.pBitRevTable = NULL;
  S->Sint.bitRevLength = 0;
#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
  S->Sint.rearranged_twiddle_tab_stride1_arr = NULL;
  S->Sint.rearranged_twiddle_tab_stride2_arr = NULL;
  S->Sint.rearranged_twiddle_tab_stride3_arr = NULL;
  S->Sint.rearranged_twiddle_stride1 = NULL;
  S->Sint.rearranged_twiddle_stride2 = NULL;
  S->Sint.rearranged_twiddle_stride3 = NULL;
#endif
  S->fftLenRFFT = fftLen;
  /* Set Calculate LOG2NBP according RFFT size */
  if (fftLen == 64)
  { 
    S->Sint.fftLen = HSP_LOG2NBP_64;
  }
  else if (fftLen == 128)
  {
    S->Sint.fftLen = HSP_LOG2NBP_128;
  }
  else if (fftLen == 256)
  { 
    S->Sint.fftLen = HSP_LOG2NBP_256; 
  }
  else if (fftLen == 512) 
  { 
    S->Sint.fftLen = HSP_LOG2NBP_512;
  }
  else if (fftLen == 1024)
  {
    S->Sint.fftLen = HSP_LOG2NBP_1024; 
  }
  else if (fftLen == 2048)
  {
    S->Sint.fftLen = HSP_LOG2NBP_2048; 
  }
  else if (fftLen == 4096) 
  { 
    S->Sint.fftLen = HSP_LOG2NBP_4096;
  }
  else
  {
    return STM32_HSP_MATH_ARGUMENT_ERROR;
  }
  return STM32_HSP_MATH_SUCCESS;
}

/**
  * @brief         Initialization function for the cfft f32 function
  * @param S       points to an instance of the floating-point CFFT structure
  * @param         fftLen         fft length (number of complex samples)
  * @return        execution status
  *                  - \ref ARM_MATH_SUCCESS        : Operation successful
  *                  - \ref ARM_MATH_ARGUMENT_ERROR : an error is detected
  *
  * @par          Use of this function is mandatory only for the MVE version of the FFT.
  *               Other versions can still initialize directly the data structure using
  *               variables declared in arm_const_structs.h
  */
stm32_hsp_status stm32_hsp_cfft_init_f32(stm32_hsp_cfft_instance_f32 *S, uint16_t fftLen)
{
  S->pTwiddle = NULL;
  S->pBitRevTable = NULL;
  S->bitRevLength = 0;
#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
  S->rearranged_twiddle_tab_stride1_arr = NULL;
  S->rearranged_twiddle_tab_stride2_arr = NULL;
  S->rearranged_twiddle_tab_stride3_arr = NULL;
  S->rearranged_twiddle_stride1 = NULL;
  S->rearranged_twiddle_stride2 = NULL;
  S->rearranged_twiddle_stride3 = NULL;
#endif
  /* Set Calculate LOG2NBP according RFFT size */
  if (fftLen == 32)
  { 
    S->fftLen = HSP_LOG2NBP_32;
  }
  else if (fftLen == 64)
  { 
    S->fftLen = HSP_LOG2NBP_64;
  }
  else if (fftLen == 128)
  {
    S->fftLen = HSP_LOG2NBP_128;
  }
  else if (fftLen == 256)
  { 
    S->fftLen = HSP_LOG2NBP_256; 
  }
  else if (fftLen == 512) 
  { 
    S->fftLen = HSP_LOG2NBP_512;
  }
  else if (fftLen == 1024)
  {
    S->fftLen = HSP_LOG2NBP_1024; 
  }
  else if (fftLen == 2048)
  {
    S->fftLen = HSP_LOG2NBP_2048; 
  }
  else if (fftLen == 4096) 
  { 
    S->fftLen = HSP_LOG2NBP_4096;
  }
  else
  {
    return STM32_HSP_MATH_ARGUMENT_ERROR;
  }  
  return STM32_HSP_MATH_SUCCESS;
}


/**
  * @brief  Initialization function for the floating-point FIR filter.
  * @param     S          points to an instance of the floating-point FIR filter structure.
  * @param     numTaps    Number of filter coefficients in the filter.
  * @param     pCoeffs    points to the filter coefficients.
  * @param     pState     points to the state buffer.
  * @param     blockSize  number of samples that are processed at a time.
  * @return               None
  */
void stm32_hsp_fir_init_f32(stm32_hsp_fir_instance_f32 *S, uint16_t numTaps, const float32_t *pCoeffs,
                            float32_t *pState, uint32_t blockSize)
{  
  hsp_filter_state_identifier_t state = HSP_BRAM_MallocStateBuffer_Fir(stm32_cmsis_handle, numTaps, blockSize,
                                                                       HSP_BRAM_ALLOCATION_DEFAULT);
  /* How to return error if create state does not succeed ? */
  pState[0] = (float32_t) * ((float32_t *)&state);
  S->pState = pState;
  S->numTaps = numTaps;
  S->pCoeffs = pCoeffs;
}

/**
  * @brief         Initialization function for the floating-point FIR decimator.
  * @param     S          points to an instance of the floating-point FIR decimator structure
  * @param     numTaps    number of coefficients in the filter
  * @param     M          decimation factor
  * @param     pCoeffs    points to the filter coefficients
  * @param     pState     points to the state buffer
  * @param     blockSize  number of input samples to process per call
  * @return        execution status
  *                  - \ref ARM_MATH_SUCCESS      : Operation successful
  *                  - \ref ARM_MATH_LENGTH_ERROR : <code>blockSize</code> is not a multiple of <code>M</code>
  *
  * @par           Details
  *                  <code>pCoeffs</code> points to the array of filter coefficients stored in time reversed order:
  * <pre>
  *    {b[numTaps-1], b[numTaps-2], b[N-2], ..., b[1], b[0]}
  * </pre>
  * @par
  *                  <code>pState</code> points to the array of state variables.
  *                  <code>pState</code> is of length <code>numTaps+blockSize-1</code> words where <code>blockSize</code> is the number of input samples passed to <code>arm_fir_decimate_f32()</code>.
  *                  <code>M</code> is the decimation factor.
  */
stm32_hsp_status stm32_hsp_fir_decimate_init_f32(stm32_hsp_fir_decimate_instance_f32 *S, uint16_t numTaps, uint8_t M,
                                                 const float32_t *pCoeffs, float32_t *pState, uint32_t blockSize)
{
  /* The size of the input block must be a multiple of the decimation factor */
  if ((blockSize % M) != 0U)
  {
    return STM32_HSP_MATH_LENGTH_ERROR;
  }

  hsp_filter_state_identifier_t state = HSP_BRAM_MallocStateBuffer_FirDecimate(stm32_cmsis_handle, numTaps, blockSize,
                                                                               M, HSP_BRAM_ALLOCATION_DEFAULT);
  
  if (state == (uint32_t)NULL)
  {
    return (STM32_HSP_MATH_LENGTH_ERROR);
  }

  /* Assign filter taps */
  S->numTaps = numTaps;

  /* Assign coefficient pointer */
  S->pCoeffs = pCoeffs;

  /* Assign state pointer */
  pState[0] = (float32_t) * ((float32_t *)&state);
  S->pState = pState;

  /* Assign Decimation Factor */
  S->M = M;

  return STM32_HSP_MATH_SUCCESS;
}

/**
  * @brief  Initialization function for the floating-point Biquad cascade filter.
  * @param     S          points to an instance of the floating-point Biquad cascade structure.
  * @param     numStages  number of 2nd order stages in the filter.
  * @param     pCoeffs    points to the filter coefficients.
  * @param     pCoeffsMod points to the modified filter coefficients (only MVE version).
  * @param     pState     points to the state buffer.
  * @return               None
  */
void stm32_hsp_biquad_cascade_df1_init_f32(stm32_hsp_biquad_casd_df1_inst_f32 *S, uint8_t numStages,
                                           const float32_t *pCoeffs,
                                           float32_t *pState)
{
  hsp_filter_state_identifier_t state = HSP_BRAM_MallocStateBuffer_BiquadCascadeDf1(stm32_cmsis_handle, numStages,
                                                                                    HSP_BRAM_ALLOCATION_DEFAULT);

  /* How to return error if create state does not succeed ? */

  /* Assign filter stages */
  S->numStages = numStages;

  /* Assign coefficient pointer */
  S->pCoeffs = pCoeffs;

  /* Assign state pointer */
  pState[0] = (float32_t) * ((float32_t *)&state);
  S->pState = pState;
}

/**
  * @brief         Initialization function for the floating-point transposed direct form II Biquad cascade filter.
  * @param     S           points to an instance of the filter data structure.
  * @param     numStages   number of 2nd order stages in the filter.
  * @param     pCoeffs     points to the filter coefficients.
  * @param     pState      points to the state buffer.
  * @return        none
  *  @par           Coefficient and State Ordering
  *                The coefficients are stored in the array <code>pCoeffs</code> in the following order
  * <pre>
  *     {b10, b11, b12, a11, a12, b20, b21, b22, a21, a22, ...}
  * </pre>
  * @par
  *                  where <code>b1x</code> and <code>a1x</code> are the coefficients for the first stage,
  *                  <code>b2x</code> and <code>a2x</code> are the coefficients for the second stage,
  *                  and so on.  The <code>pCoeffs</code> array contains a total of <code>5*numStages</code> values.
  *                  and it must be initialized using the function
  *                  arm_biquad_cascade_df2T_compute_coefs_f32 which is taking the
  *                  standard array coefficient as parameters.
  * @par
  *                  The <code>pState</code> is a pointer to state array.
  *                  Each Biquad stage has 2 state variables <code>d1,</code> and <code>d2</code>.
  *                  The 2 state variables for stage 1 are first, then the 2 state variables for stage 2, and so on.
  *                  The state array has a total length of <code>2*numStages</code> values.
  *                  The state variables are updated after each block of data is processed; the coefficients are untouched.
  */
void stm32_hsp_biquad_cascade_df2T_init_f32(stm32_hsp_biquad_cascade_df2T_instance_f32 *S, uint8_t numStages,
                                            const float32_t *pCoeffs, float32_t *pState)
{
  hsp_filter_state_identifier_t state = HSP_BRAM_MallocStateBuffer_BiquadCascadeDf2t(stm32_cmsis_handle, numStages,
                                                                                     HSP_BRAM_ALLOCATION_DEFAULT);
  /* How to return error if create state does not succeed ? */

  /* Assign filter stages */
  S->numStages = numStages;

  /* Assign coefficient pointer */
  S->pCoeffs = pCoeffs;

  /* Assign state pointer */
  pState[0] = (float32_t) * ((float32_t *)&state);
  S->pState = pState;
}

/**
  * @brief     Initialization function for floating-point LMS filter.
  * @param     S          points to an instance of the floating-point LMS filter structure
  * @param     numTaps    number of filter coefficients
  * @param     pCoeffs    points to coefficient buffer
  * @param     pState     points to state buffer
  * @param     mu         step size that controls filter coefficient updates
  * @param     blockSize  number of samples to process
  * @return    none
  * @par       Details
  *            pCoeffs points to the array of filter coefficients stored in time reversed order:
  *            {b[numTaps-1], b[numTaps-2], b[N-2], ..., b[1], b[0]}
  *            The initial filter coefficients serve as a starting point for the adaptive filter.
  *            pState  points to an array of length numTaps+blockSize-1 samples,
  *            where blockSize is the number of input samples processed by each call to arm_lms_f32().
  */
/* @todo  CHECKER POUR VOIR SI LES COEFF SONT EN SENS INVERSE */
void stm32_hsp_lms_init_f32(stm32_hsp_lms_instance_f32 *S, uint16_t numTaps, float32_t *pCoeffs, float32_t *pState,
                            float32_t mu, uint32_t blockSize)
{
  hsp_filter_state_identifier_t state = HSP_BRAM_MallocStateBuffer_Lms(stm32_cmsis_handle, numTaps,
                                                                       HSP_BRAM_ALLOCATION_DEFAULT);
  /*  How to return error if create state does not succeed ? */

  /* Assign filter taps */
  S->numTaps = numTaps;

  /* Assign coefficient pointer */
  S->pCoeffs = pCoeffs;

  /* Assign state pointer */
  pState[0] = (float32_t) * ((float32_t *)&state);
  S->pState = pState;

  /* Assign Step size value */
  S->mu = mu;
}

/**
  * @brief         Initialization function for the floating-point IIR lattice filter.
  * @param     S          points to an instance of the floating-point IIR lattice structure
  * @param     numStages  number of stages in the filter
  * @param     pkCoeffs   points to reflection coefficient buffer.  The array is of length numStages
  * @param     pvCoeffs   points to ladder coefficient buffer.  The array is of length numStages+1
  * @param     pState     points to state buffer.  The array is of length numStages+blockSize
  * @param     blockSize  number of samples to process
  * @return        none
  */
void stm32_hsp_iir_lattice_init_f32(stm32_hsp_iir_lattice_instance_f32 *S, uint16_t numStages, float32_t *pkCoeffs,
                                    float32_t *pvCoeffs, float32_t *pState, uint32_t blockSize)
{
  hsp_filter_state_identifier_t state = HSP_BRAM_MallocStateBuffer_IirLattice(stm32_cmsis_handle, numStages,
                                                                              HSP_BRAM_ALLOCATION_DEFAULT);
  /* How to return error if create state does not succeed ? */

  /* Assign filter taps */
  S->numStages = numStages;

  /* Assign reflection coefficient pointer */
  S->pkCoeffs = pkCoeffs;

  /* Assign ladder coefficient pointer */
  S->pvCoeffs = pvCoeffs;

  /* Assign state pointer */
  pState[0] = (float32_t) * ((float32_t *)&state);
  S->pState = pState;
}

/**
  * @brief  Floating-point matrix initialization.
  * @param     S         points to an instance of the floating-point matrix structure.
  * @param     nRows     number of rows in the matrix.
  * @param     nColumns  number of columns in the matrix.
  * @param     pData     points to the matrix data array.
  * @return               None
  */
void stm32_hsp_mat_init_f32(stm32_hsp_matrix_instance_f32 *S, uint16_t nRows, uint16_t nColumns, float32_t *pData)
{
  /* Assign Number of Rows */
  S->numRows = nRows;

  /* Assign Number of Columns */
  S->numCols = nColumns;

  /* Assign Data pointer */
  S->pData = pData;
}