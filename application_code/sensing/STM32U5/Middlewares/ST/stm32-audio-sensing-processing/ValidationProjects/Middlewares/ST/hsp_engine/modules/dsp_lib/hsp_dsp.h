/**
  ******************************************************************************
  * @file hsp_dsp.h
  * @brief API for STM32 HSP DSP function
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

#ifndef HSP_DSP_H
#define HSP_DSP_H

/* Includes ------------------------------------------------------------------*/
#include "hsp_direct_command.h"


/** @addtogroup HSP_ENGINE
  * @{
  */

/** @addtogroup HSP_MODULES
  * @{
  */

/** @defgroup HSP_MODULES_DSP HSP Modules DSP
  * @{
  */
/** @defgroup HSP_MODULES_DSP_Private_Variable HSP Modules DSP Private variable
  * @{
  */
extern hsp_core_handle_t *stm32_cmsis_handle;
/**
  * @}
  */
/** @defgroup HSP_MODULES_DSP_Exported_Types HSP Modules DSP Exported Types
  * @{
  */
/* Exported types ------------------------------------------------------------*/
/**
  * @brief Error status returned by some functions in the library.
  */
typedef enum
{
  STM32_HSP_MATH_SUCCESS                 =  0,        /**< No error */
  STM32_HSP_MATH_ARGUMENT_ERROR          = -1,        /**< One or more arguments are incorrect */
  STM32_HSP_MATH_LENGTH_ERROR            = -2,        /**< Length of data buffer is incorrect */
  STM32_HSP_MATH_SIZE_MISMATCH           = -3,        /**< Size of matrices is not compatible with the operation */
  STM32_HSP_MATH_NANINF                  = -4,        /**< Not-a-number (NaN) or infinity is generated */
  STM32_HSP_MATH_SINGULAR                = -5,        /**< Input matrix is singular and cannot be inverted */
  STM32_HSP_MATH_TEST_FAILURE            = -6,        /**< Test Failed */
  STM32_HSP_MATH_DECOMPOSITION_FAILURE   = -7         /**< Decomposition Failed */
} stm32_hsp_status;

/**
  * @brief Instance structure for the floating-point CFFT/CIFFT function.
  */
typedef struct
{
  uint16_t fftLen;                   /**< length of the FFT. */
  const float32_t *pTwiddle;         /**< points to the Twiddle factor table. */
  const uint16_t *pBitRevTable;      /**< points to the bit reversal table. */
  uint16_t bitRevLength;             /**< bit reversal table length. */
#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
  const uint32_t *rearranged_twiddle_tab_stride1_arr;        /**< Per stage reordered twiddle pointer (offset 1) */
  const uint32_t *rearranged_twiddle_tab_stride2_arr;        /**< Per stage reordered twiddle pointer (offset 2) */
  const uint32_t *rearranged_twiddle_tab_stride3_arr;        /**< Per stage reordered twiddle pointer (offset 3) */
  const float32_t *rearranged_twiddle_stride1; /**< reordered twiddle offset 1 storage */
  const float32_t *rearranged_twiddle_stride2; /**< reordered twiddle offset 2 storage */
  const float32_t *rearranged_twiddle_stride3;
#endif
} stm32_hsp_cfft_instance_f32;

/**
  * @brief Instance structure for the floating-point RFFT/RIFFT function.
  */
typedef struct
{
  stm32_hsp_cfft_instance_f32 Sint;     /**< Internal CFFT structure. */
  uint16_t fftLenRFFT;            /**< length of the real sequence */
  const float32_t *pTwiddleRFFT;  /**< Twiddle factors real stage  */
} stm32_hsp_rfft_fast_instance_f32 ;


/**
  * @brief Instance structure for the floating-point FIR filter.
  */
typedef struct
{
  uint16_t numTaps;     /**< number of filter coefficients in the filter. */
  float32_t *pState;    /**< points to the state variable array. The array is of length 1. */
  const float32_t *pCoeffs;   /**< points to the coefficient array. The array is of length 1. */
} stm32_hsp_fir_instance_f32;

/**
  @brief Instance structure for floating-point FIR decimator.
  */
typedef struct
{
  uint8_t M;                  /**< decimation factor. */
  uint16_t numTaps;           /**< number of coefficients in the filter. */
  const float32_t *pCoeffs;   /**< points to the coefficient array. The array is of length numTaps.*/
  float32_t *pState;          /**< points to the state variable array. The array is of length 1. */
} stm32_hsp_fir_decimate_instance_f32;

/**
  * @brief Instance structure for the floating-point Biquad cascade filter.
  */
typedef struct
{
  uint32_t numStages;            /**< number of 2nd order stages in the filter.  Overall order is 2*numStages. */
  float32_t *pState;             /**< Points to the array of state coefficients.  The array is of length 1. */
  const float32_t *pCoeffs;      /**< Points to the array of coefficients.  The array is of length 5*numStages. */
} stm32_hsp_biquad_casd_df1_inst_f32;

/**
  * @brief Instance structure for the floating-point transposed direct form II Biquad cascade filter.
  */
typedef struct
{
  uint8_t numStages;              /**< number of 2nd order stages in the filter.  Overall order is 2*numStages. */
  float32_t *pState;              /**< points to the array of state coefficients.  The array is of length 1. */
  const float32_t *pCoeffs;       /**< points to the array of coefficients.  The array is of length 5*numStages. */
} stm32_hsp_biquad_cascade_df2T_instance_f32;


/**
  * @brief Instance structure for the floating-point LMS filter.
  */
typedef struct
{
  uint16_t numTaps;    /**< number of coefficients in the filter. */
  float32_t *pState;   /**< points to the state variable array. The array is of length 1. */
  float32_t *pCoeffs;  /**< points to the coefficient array. The array is of length numTaps. */
  float32_t mu;        /**< step size that controls filter coefficient updates. */
} stm32_hsp_lms_instance_f32;

/**
  * @brief Instance structure for the floating-point IIR lattice filter.
  */
typedef struct
{
  uint16_t numStages;                  /**< number of stages in the filter. */
  float32_t *pState;                   /**< points to the state variable array. The array is of length 1. */
  float32_t *pkCoeffs;                 /**< points to the reflection coefficient array. The array is of length numStages. */
  float32_t *pvCoeffs;                 /**< points to the ladder coefficient array. The array is of length numStages+1. */
} stm32_hsp_iir_lattice_instance_f32;

/**
  * @brief Instance structure for the floating-point matrix structure.
  */
typedef struct
{
  uint16_t numRows;     /**< number of rows of the matrix.     */
  uint16_t numCols;     /**< number of columns of the matrix.  */
  float32_t *pData;     /**< points to the data of the matrix. */
} stm32_hsp_matrix_instance_f32;

/**
  * @}
  */

/** @defgroup HSP_MODULES_DSP_Exported_Functions HSP Modules DSP Exported Functions
  * @{
  */
/** @defgroup HSP_MODULES_DSP_Transform_Functions HSP Modules DSP Transform Functions
  * @{
  */
/** @defgroup HSP_CMSIS_TRANSFORM_Macros HSP CMSIS Transform macro
  * @brief    HSP Transform CMSIS macro
  * @{
  */
/**
  * @brief     Processing function for the floating-point real FFT.
  * @param     rfft_instance <b>[stm32_hsp_rfft_fast_instance_f32 *]</b>     points to an stm32_hsp_rfft_fast_instance_f32 structure
  * @param     input         <b>[float32_t *]</b>     points to input buffer. Processing occurs in-place
  * @param     output        <b>[float32_t *]</b>     PARAMETER NOT USED because processing occurs in-place
  * @param     ifftFlag      <b>[uint8_t]</b>
  *                - value = 0: RFFT
  *                - value = 1: RIFFT
  * @return               None
  */
__STATIC_FORCEINLINE void stm32_hsp_rfft_fast_f32(stm32_hsp_rfft_fast_instance_f32 *rfft_instance, float32_t *input, 
                                                  float32_t *output, uint8_t ifftFlag)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_RFFT_F32);
  HSP_ACC_Rfft_WriteParam(stm32_cmsis_handle, input, (rfft_instance)->Sint.fftLen, ifftFlag, 1, HSP_RFFT_TYPE_2);  
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(stm32_cmsis_handle);
}

/**
  * @brief     Processing function for the floating-point complex FFT.
  * @param     cfft_instance <b>[stm32_hsp_cfft_instance_f32 *]</b>  points to an instance of the floating-point CFFT structure
  * @param     input         <b>[float32_t *]</b>                    points to the complex data buffer of size <code>2*fftLen</code>. Processing occurs in-place
  * @param     ifftFlag      <b>[uint8_t]</b>                        flag that selects transform direction
  *                  - value = 0: forward transform
  *                  - value = 1: inverse transform
  * @param     bitReverseFlag <b>[uint8_t]</b>                       flag that enables / disables bit reversal of output
  *                  - value = 0: disables bit reversal of output
  *                  - value = 1: enables bit reversal of output
  * @return               None
  */
__STATIC_FORCEINLINE void stm32_hsp_cfft_f32(stm32_hsp_cfft_instance_f32 *cfft_instance, float32_t *input, uint8_t ifftFlag,
                                             uint8_t bitReverseFlag)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_FFT_F32);
  HSP_ACC_Fft_WriteParam(stm32_cmsis_handle, input, (cfft_instance)->fftLen, ifftFlag, bitReverseFlag);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(stm32_cmsis_handle);  
}
/**
  * @}
  */

/** @defgroup HSP_MODULES_DSP_Filter_Functions HSP Modules DSP Filter Functions
  * @{
  */
/** @defgroup HSP_CMSIS_FILTER_Macros HSP CMSIS Filter macro
  * @brief    HSP Filter CMSIS functions
  * @{
  */
/**
  * @brief Processing function for the floating-point FIR filter.
  * @param  fir_instance  <b>[stm32_hsp_fir_instance_f32 *]</b>  points to an instance of the floating-point FIR structure.
  *                                                                  Coeff are stored in reverse order than ARM CMSIS
  * @param  pSrc          <b>[float32_t *]</b>               points to the block of input data.
  * @param  pDst          <b>[float32_t *]</b>               points to the block of output data.
  * @param  blockSize     <b>[uint32_t]</b>                  number of samples to process.
  * @return               None
  */
__STATIC_FORCEINLINE void stm32_hsp_fir_f32(stm32_hsp_fir_instance_f32 *fir_instance, float32_t *pSrc, float32_t *pDst, uint32_t blockSize)
{
  HSP_HW_IF_WRITE_DCMDIDR((uint32_t)((hsp_hw_if_filter_state_t*)((uint32_t)*((uint32_t *)(&(fir_instance)->pState[0]))))->dirCmd);
  HSP_ACC_Filter_WriteParam(stm32_cmsis_handle, (uint32_t)pSrc, (uint32_t)((fir_instance)->pCoeffs), ((hsp_filter_state_identifier_t)*((uint32_t *)(&(fir_instance)->pState[0]))), (uint32_t)pDst, blockSize);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(stm32_cmsis_handle);  
}

/**
  * @brief Processing function for floating-point FIR decimator.
  * @param fir_instance <b>[stm32_hsp_fir_decimate_instance_f32 *]</b>  points to an instance of the floating-point FIR decimator structure
  * @param pSrc         <b>[float32_t *]</b>                            points to the block of input data
  * @param pDst        <b>[float32_t *]</b>                            points to the block of output data
  * @param blockSize    <b>[uint32_t]</b>                               number of samples to process
  * @return               None
  */
__STATIC_FORCEINLINE void stm32_hsp_fir_decimate_f32(stm32_hsp_fir_decimate_instance_f32 *fir_instance, float32_t *pSrc, float32_t *pDst, uint32_t blockSize)
{
  HSP_HW_IF_WRITE_DCMDIDR((uint32_t)((hsp_hw_if_fir_decimate_filter_state_t*)((uint32_t)*((uint32_t *)(&(fir_instance)->pState[0]))))->dirCmd);
  HSP_ACC_FirDecimate_WriteParam(stm32_cmsis_handle, (uint32_t)pSrc, (uint32_t)((fir_instance)->pCoeffs),
					   ((hsp_filter_state_identifier_t)*((uint32_t *)(&(fir_instance)->pState[0]))), (uint32_t)pDst, blockSize, (uint32_t)((fir_instance)->M));
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(stm32_cmsis_handle);  
}

/**
  * @brief Processing function for the floating-point Biquad cascade filter.
  * @param biq_instance <b>[stm32_hsp_biquad_casd_df1_inst_f32 *]</b>  points to an instance of the floating-point Biquad cascade structure.
  * @param pSrc         <b>[float32_t *]</b>                           points to the block of input data.
  * @param pDst        <b>[float32_t *]</b>                           points to the block of output data.
  * @param blockSize    <b>[uint32_t]</b>                              number of samples to process.
  * @return               None
  */
__STATIC_FORCEINLINE void stm32_hsp_biquad_cascade_df1_f32(stm32_hsp_biquad_casd_df1_inst_f32 *biq_instance, float32_t *pSrc, float32_t *pDst, uint32_t blockSize)
{
  HSP_HW_IF_WRITE_DCMDIDR((uint32_t)((hsp_hw_if_filter_state_t*)((uint32_t)*((uint32_t *)(&(biq_instance)->pState[0]))))->dirCmd);
  HSP_ACC_Filter_WriteParam(stm32_cmsis_handle, (uint32_t)pSrc, (uint32_t)((biq_instance)->pCoeffs), ((hsp_filter_state_identifier_t)*((uint32_t *)(&(biq_instance)->pState[0]))), (uint32_t)pDst, blockSize);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(stm32_cmsis_handle);
}

/**
  * @brief Processing function for the floating-point transposed direct form II Biquad cascade filter.
  * @param biq_instance <b>[stm32_hsp_biquad_cascade_df2T_instance_f32 *]</b> points to an instance of the filter data structure
  * @param pSrc         <b>[float32_t *]</b>                                  points to the block of input data
  * @param pDst        <b>[float32_t *]</b>                                  points to the block of output data
  * @param blockSize    <b>[uint32_t]</b>                                     number of samples to process.
  * @return               None
  */
__STATIC_FORCEINLINE void stm32_hsp_biquad_cascade_df2T_f32(stm32_hsp_biquad_cascade_df2T_instance_f32 *biq_instance, float32_t *pSrc, float32_t *pDst, uint32_t blockSize)
{
  HSP_HW_IF_WRITE_DCMDIDR((uint32_t)((hsp_hw_if_filter_state_t*)((uint32_t)*((uint32_t *)(&(biq_instance)->pState[0]))))->dirCmd);
  HSP_ACC_Filter_WriteParam(stm32_cmsis_handle, (uint32_t)pSrc, (uint32_t)((biq_instance)->pCoeffs), ((hsp_filter_state_identifier_t)*((uint32_t *)(&(biq_instance)->pState[0]))), (uint32_t)pDst, blockSize);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(stm32_cmsis_handle);  
}

/**
  * @brief Convolution of floating-point sequences.
  * @param pSrcA   <b>[float32_t *]</b>   points to the first input sequence.
  * @param srcALen <b>[uint32_t]</b>      length of the first input sequence.
  * @param pSrcB   <b>[float32_t *]</b>   points to the second input sequence.
  * @param srcBLen <b>[uint32_t]</b>      length of the second input sequence.
  * @param  pDst   <b>[float32_t *]</b>   points to the location where the output result is written.   points to the location where the output result is written.
  *                                           Length srcALen+srcBLen-1.
  * @return               None
  */
__STATIC_FORCEINLINE void stm32_hsp_conv_f32(float32_t *pSrcA, uint32_t srcALen, float32_t *pSrcB, uint32_t srcBLen, float32_t *pDst)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_CONV_F32);
  HSP_ACC_VectIIOSS_WriteParam(stm32_cmsis_handle, (uint32_t)pSrcA, (uint32_t)pSrcB, (uint32_t)pDst, srcALen, srcBLen);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(stm32_cmsis_handle);  
}

/**
  * @brief Correlation of floating-point sequences.
  * @param pSrcA   <b>[float32_t *]</b>   points to the first input sequence.
  * @param srcALen <b>[uint32_t]</b>      length of the first input sequence.
  * @param pSrcB  <b>[float32_t *]</b>   points to the second input sequence.
  * @param srcBLen <b>[uint32_t]</b>      length of the second input sequence.
  * @param pDst   <b>[float32_t *]</b>   points to the location where the output result is written.
  *                                           points to the block of output data  Length 2 * max(srcALen, srcBLen) - 1.
  * @return               None
  */
__STATIC_FORCEINLINE void stm32_hsp_correlate_f32(float32_t *pSrcA, uint32_t srcALen, float32_t *pSrcB, uint32_t srcBLen, float32_t *pDst)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_CORR_F32);
  HSP_ACC_VectIIOSS_WriteParam(stm32_cmsis_handle, (uint32_t)pSrcA, (uint32_t)pSrcB, (uint32_t)pDst, srcALen, srcBLen);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(stm32_cmsis_handle);  
}

/**
  * @brief Processing function for floating-point LMS filter.
  * @param  lms_instance <b>[stm32_hsp_lms_instance_f32 *]</b>  points to an instance of the floating-point LMS filter structure.
  * @param  pSrc         <b>[float32_t *]</b>                   points to the block of input data.
  * @param  pRef         <b>[float32_t *]</b>                   points to the block of reference data.
  * @param  pOut         <b>[float32_t *]</b>                   points to the block of output data.
  * @param  pErr         <b>[float32_t *]</b>                   points to the block of error data.
  * @param  blockSize    <b>[uint32_t]</b>                      number of samples to process.
  * @return               None
  */
__STATIC_FORCEINLINE void stm32_hsp_lms_f32(stm32_hsp_lms_instance_f32 *lms_instance, float32_t *pSrc, float32_t *pRef, float32_t *pOut, float32_t *pErr, uint32_t blockSize)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_LMS_F32);
  HSP_ACC_Lms_WriteParam(stm32_cmsis_handle, (uint32_t)pSrc, (uint32_t)((lms_instance)->pCoeffs), 
                        ((hsp_filter_state_identifier_t)*((uint32_t *)(&(lms_instance)->pState[0]))), (uint32_t)pOut,
                        (uint32_t)pRef, (uint32_t)pErr, blockSize, ((lms_instance)->mu));
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(stm32_cmsis_handle);
}

/**
  * @brief Processing function for the floating-point IIR lattice filter.
  * @param  iir_instance  <b>[stm32_hsp_iir_lattice_instance_f32 *]</b>  points to an instance of the floating-point IIR lattice structure.
  * @param  pSrc          <b>[float32_t *]</b>                           points to the block of input data.
  * @param  pDst          <b>[float32_t *]</b>                           points to the block of output data.
  * @param  blockSize     <b>[uint32_t]</b>                              number of samples to process.
  * @Todo:  stm32_hsp_iir_lattice_f32: a l'init on passe les coeff dont a besoin et pas les coeff de ARM .. A discuter
  * @return               None
  */
__STATIC_FORCEINLINE void stm32_hsp_iir_lattice_f32(stm32_hsp_iir_lattice_instance_f32 *iir_instance, float32_t *pSrc, float32_t *pDst, uint32_t blockSize)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_IIR_LATTICE_F32);
  HSP_ACC_IirLattice_WriteParam(stm32_cmsis_handle, (uint32_t)pSrc, (uint32_t)((iir_instance)->pkCoeffs), 
                               (uint32_t)((iir_instance)->pvCoeffs), 
							   ((uint32_t)*((uint32_t *)(&(iir_instance)->pState[0]))), (uint32_t)pDst, blockSize);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(stm32_cmsis_handle);
}
/**
  * @}
  */

/** @defgroup HSP_MODULES_DSP_Complex_Functions HSP Modules DSP Complex Functions
  * @{
  */
/** @defgroup HSP_CMSIS_COMPLEX_Macros HSP CMSIS Complex macro
  * @brief    HSP Complex CMSIS functions
  * @{
  */
/**
  * @brief  Floating-point complex conjugate.
  * @param  pSrc        <b>[float32_t *]</b>       points to the input vector
  * @param  pDst        <b>[float32_t *]</b>       points to the output vector
  * @param  numSamples  <b>[uint32_t]</b>          number of complex samples in each vector
  * @return               None
  */
__STATIC_FORCEINLINE void stm32_hsp_cmplx_conj_f32(float32_t *pSrc, float32_t *pDst, uint32_t numSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_CMPLX_CONJ_F32);
  HSP_ACC_VectIOS_WriteParam(stm32_cmsis_handle, (uint32_t)pSrc, (uint32_t)pDst, (numSamples*2));
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(stm32_cmsis_handle);  
}

/**
  * @brief  Floating-point complex dot product
  * @param  pSrcA       <b>[float32_t *]</b>  points to the first input vector
  * @param  pSrcB       <b>[float32_t *]</b>  points to the second input vector
  * @param  numSamples  <b>[uint32_t]</b>     number of complex samples in each vector
  * @param  realResult  <b>[float32_t *]</b>  real part of the result returned here
  * @param  imagResult  <b>[float32_t *]</b>  imaginary part of the result returned here
  * @return               None
  */
__STATIC_FORCEINLINE void stm32_hsp_cmplx_dot_prod_f32(float32_t *pSrcA, float32_t *pSrcB, uint32_t numSamples, float32_t *realResult, float32_t *imagResult)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_CMPLX_CMSIS_DOTPROD_F32);
  HSP_ACC_VectIIOOS_WriteParam(stm32_cmsis_handle, (uint32_t)pSrcA, (uint32_t)pSrcB, (uint32_t)realResult, (uint32_t)imagResult, (numSamples*2));
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(stm32_cmsis_handle);
}

/**
  * @brief  Floating-point complex magnitude
  * @param  pSrc       <b>[float32_t *]</b> points to the complex input vector
  * @param  pDst       <b>[float32_t *]</b> points to the real output vector
  * @param  numSamples <b>[uint32_t]</b>    number of complex samples in the input vector
  * @return               None
  */
__STATIC_FORCEINLINE void stm32_hsp_cmplx_mag_f32(float32_t *pSrc, float32_t *pDst, uint32_t numSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_CMPLX_MAG_F32);
  HSP_ACC_VectIOS_WriteParam(stm32_cmsis_handle, (uint32_t)pSrc, (uint32_t)pDst, (numSamples*2));
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(stm32_cmsis_handle); 
}

/**
  * @brief  Floating-point complex magnitude squared
  * @param  pSrc       <b>[float32_t *]</b>  points to the complex input vector
  * @param  pDst       <b>[float32_t *]</b>  points to the real output vector
  * @param  numSamples <b>[uint32_t]</b>     number of complex samples in the input vector
  * @return               None
  */
__STATIC_FORCEINLINE void stm32_hsp_cmplx_mag_squared_f32(float32_t *pSrc, float32_t *pDst, uint32_t numSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_CMPLX_MAGSQUARED_F32);
  HSP_ACC_VectIOS_WriteParam(stm32_cmsis_handle, (uint32_t)pSrc, (uint32_t)pDst, (numSamples*2));
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(stm32_cmsis_handle);  
}

/**
  * @brief  Floating-point complex-by-complex multiplication
  * @param  pSrcA       <b>[float32_t *]</b> points to the first input vector
  * @param  pSrcB       <b>[float32_t *]</b> points to the second input vector
  * @param  pDst        <b>[float32_t *]</b> points to the output vector
  * @param  numSamples  <b>[uint32_t]</b>    number of complex samples in each vector
  * @return               None
  */
__STATIC_FORCEINLINE void stm32_hsp_cmplx_mult_cmplx_f32(float32_t *pSrcA, float32_t *pSrcB, float32_t *pDst, uint32_t numSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_CMPLX_MUL_F32);
  HSP_ACC_VectIIOS_WriteParam(stm32_cmsis_handle, (uint32_t)pSrcA, (uint32_t)pSrcB, (uint32_t)pDst, (numSamples*2));
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(stm32_cmsis_handle);
}

/**
  * @brief  Floating-point complex-by-real multiplication
  * @param  pSrcCmplx   <b>[float32_t *]</b> points to the complex input vector
  * @param  pSrcReal    <b>[float32_t *]</b> points to the real input vector
  * @param  pCmplxDst   <b>[float32_t *]</b> points to the complex output vector
  * @param  numSamples  <b>[uint32_t]</b>    number of samples in each vector
  * @return               None
  */
__STATIC_FORCEINLINE void stm32_hsp_cmplx_mult_real_f32(float32_t *pSrcCmplx, float32_t *pSrcReal, float32_t *pCmplxDst, uint32_t numSamples)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_CMPLX_RMUL_F32);
  HSP_ACC_VectIIOS_WriteParam(stm32_cmsis_handle, (uint32_t)pSrcCmplx, (uint32_t)pSrcReal, (uint32_t)pCmplxDst, (numSamples*2));
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(stm32_cmsis_handle);  
}
/**
  * @}
  */

/** @defgroup HSP_MODULES_DSP_Vector_Functions HSP Modules DSP Vector Functions
  * @{
  */
/** @defgroup HSP_CMSIS_BASIC_Macros HSP CMSIS Basic macro
  * @brief    HSP Basic CMSIS functions
  * @{
  */

/**
  * @brief Floating-point vector absolute value.
  * @param  pSrc       <b>[float32_t *]</b> points to the input buffer
  * @param  pDst       <b>[float32_t *]</b> points to the output buffer
  * @param  blockSize  <b>[uint32_t]</b>    number of samples in each vector
  * @return               None
  */
__STATIC_FORCEINLINE void stm32_hsp_abs_f32(float32_t *pSrc, float32_t *pDst, uint32_t blockSize)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_ABS_F32);
  HSP_ACC_VectIOS_WriteParam(stm32_cmsis_handle, (uint32_t)pSrc, (uint32_t)pDst, blockSize);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(stm32_cmsis_handle);  
}

/**
  * @brief Floating-point vector addition.
  * @param  pSrcA      <b>[float32_t *]</b> points to the first input vector
  * @param  pSrcB      <b>[float32_t *]</b> points to the second input vector
  * @param  pDst       <b>[float32_t *]</b> points to the output vector
  * @param  blockSize  <b>[uint32_t]</b>    number of samples in each vector
  * @return               None
  */
__STATIC_FORCEINLINE void stm32_hsp_add_f32(float32_t *pSrcA, float32_t *pSrcB, float32_t *pDst, uint32_t blockSize)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_ADD_F32);
  HSP_ACC_VectIIOS_WriteParam(stm32_cmsis_handle, (uint32_t)pSrcA, (uint32_t)pSrcB, (uint32_t)pDst, blockSize);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(stm32_cmsis_handle);   
}

/**
  * @brief  Mean value of a floating-point vector.
  * @param  pSrc       <b>[float32_t *]</b> is input pointer
  * @param  blockSize  <b>[uint32_t]</b>    is the number of samples to process
  * @param  pResult    <b>[float32_t *]</b> is output value.
  * @return               None
  */
__STATIC_FORCEINLINE void stm32_hsp_mean_f32(float32_t *pSrc, uint32_t blockSize, float32_t *pResult)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_AVG_F32);
  HSP_ACC_VectIOS_WriteParam(stm32_cmsis_handle, (uint32_t)pSrc, (uint32_t)pResult, blockSize);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(stm32_cmsis_handle);  
}

/**
  * @brief  Copies the elements of a floating-point vector.
  * @param  pSrc       <b>[float32_t *]</b> input pointer
  * @param  pDst       <b>[float32_t *]</b> output pointer
  * @param  blockSize  <b>[uint32_t]</b>    number of samples to process
  * @return               None
  */
__STATIC_FORCEINLINE void stm32_hsp_copy_f32(float32_t *pSrc, float32_t *pDst, uint32_t blockSize)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_COPY);
  HSP_ACC_VectIOS_WriteParam(stm32_cmsis_handle, (uint32_t)pSrc, (uint32_t)pDst, blockSize);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(stm32_cmsis_handle);  
}

/**
  * @brief Dot product of floating-point vectors.
  * @param  pSrcA      <b>[float32_t *]</b> points to the first input vector
  * @param  pSrcB      <b>[float32_t *]</b> points to the second input vector
  * @param  blockSize  <b>[uint32_t]</b>    number of samples in each vector
  * @param  result     <b>[float32_t *]</b> output result returned here
  * @return               None
  */
__STATIC_FORCEINLINE void stm32_hsp_dot_prod_f32(float32_t *pSrcA, float32_t *pSrcB, uint32_t blockSize, float32_t *result)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_DOTPROD_F32);
  HSP_ACC_VectIIOS_WriteParam(stm32_cmsis_handle, (uint32_t)pSrcA, (uint32_t)pSrcB, (uint32_t)result, blockSize);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(stm32_cmsis_handle);  
}

/**
  * @brief Search for value and position of the absolute biggest element of a of floating-point vector.
  * @param  pSrc       <b>[float32_t *]</b> points to the input buffer
  * @param  blockSize  <b>[uint32_t]</b>    length of the input vector
  * @param  pResult    <b>[float32_t *]</b> maximum value returned here
  * @param  pIndex     <b>[uint32_t *]</b>  index of maximum value returned here
  * @return               None
  */
__STATIC_FORCEINLINE void stm32_hsp_absmax_f32(float32_t *pSrc, uint32_t blockSize, float32_t *pResult, uint32_t *pIndex)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_ABSMAX_F32);
  HSP_ACC_VectIIOS_WriteParam(stm32_cmsis_handle, (uint32_t)pSrc, (uint32_t)pResult, (uint32_t)pIndex, blockSize);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(stm32_cmsis_handle);
}

/**
  * @brief Maximum value of a floating-point vector.
  * @param  pSrc       <b>[float32_t *]</b> points to the input buffer
  * @param  blockSize  <b>[uint32_t]</b>    length of the input vector
  * @param pResult    <b>[float32_t *]</b> maximum value returned here
  * @param pIndex     <b>[uint32_t *]</b>  index of maximum value returned here
  * @return               None
  */
__STATIC_FORCEINLINE void stm32_hsp_max_f32(float32_t *pSrc, uint32_t blockSize, float32_t *pResult, uint32_t *pIndex)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_MAX_F32);
  HSP_ACC_VectIIOS_WriteParam(stm32_cmsis_handle, (uint32_t)pSrc, (uint32_t)pResult, (uint32_t)pIndex, blockSize);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(stm32_cmsis_handle);
}

/**
  * @brief  Minimum value of a floating-point vector.
  * @param  pSrc       <b>[float32_t *]</b>  is input pointer
  * @param  blockSize  <b>[uint32_t]</b>     is the number of samples to process
  * @param  pResult    <b>[float32_t *]</b>  is output pointer
  * @param  pIndex     <b>[uint32_t *]</b>   is the array index of the minimum value in the input buffer.
  * @return               None
  */
__STATIC_FORCEINLINE void stm32_hsp_min_f32(float32_t *pSrc, uint32_t blockSize, float32_t *pResult, uint32_t *pIndex)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_MIN_F32);
  HSP_ACC_VectIIOS_WriteParam(stm32_cmsis_handle, (uint32_t)pSrc, (uint32_t)pResult, (uint32_t)pIndex, blockSize);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(stm32_cmsis_handle);
}

/**
  * @brief Floating-point vector multiplication.
  * @param  pSrcA      <b>[float32_t *]</b>  points to the first input vector
  * @param  pSrcB      <b>[float32_t *]</b>  points to the second input vector
  * @param  pDst       <b>[float32_t *]</b>  points to the output vector
  * @param  blockSize  <b>[uint32_t]</b>     number of samples in each vector
  * @return               None
  */
__STATIC_FORCEINLINE void stm32_hsp_mult_f32(float32_t *pSrcA, float32_t *pSrcB, float32_t *pDst, uint32_t blockSize)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_MUL_F32);
  HSP_ACC_VectIIOS_WriteParam(stm32_cmsis_handle, (uint32_t)pSrcA, (uint32_t)pSrcB, (uint32_t)pDst, blockSize);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(stm32_cmsis_handle);
}

/**
  * @brief  Adds a constant offset to a floating-point vector.
  * @param  pSrc        <b>[float32_t *]</b>  points to the input vector
  * @param  offset      <b>[float32_t *]</b>  is the offset to be added
  * @param  pDst        <b>[float32_t *]</b>  points to the output vector
  * @param  blockSize   <b>[uint32_t]</b>     number of samples in the vector
  * @return               None
  */
__STATIC_FORCEINLINE void stm32_hsp_offset_f32(float32_t *pSrc, float32_t offset, float32_t *pDst, uint32_t blockSize)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_OFFSET_I_F32);
  HSP_ACC_VectIFVOS_WriteParam(stm32_cmsis_handle, (uint32_t)pSrc, offset, (uint32_t)pDst, blockSize);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(stm32_cmsis_handle);  
}

/**
  * @brief  Root Mean Square of the elements of a floating-point vector.
  * @param  pSrc       <b>[float32_t *]</b>  is input pointer
  * @param  blockSize  <b>[uint32_t]</b>     is the number of samples to process
  * @param  pResult    <b>[float32_t *]</b>  is output value.
  * @return               None
  */
__STATIC_FORCEINLINE void stm32_hsp_rms_f32(float32_t *pSrc, uint32_t blockSize, float32_t *pResult)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_RMS_F32);
  HSP_ACC_VectIOS_WriteParam(stm32_cmsis_handle, (uint32_t)pSrc, (uint32_t)pResult, blockSize);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(stm32_cmsis_handle);  
}

/**
  * @brief Multiplies a floating-point vector by a scalar.
  * @param  pSrc       <b>[float32_t *]</b>  points to the input vector
  * @param  scale      <b>[float32_t]</b>    scale factor to be applied
  * @param  pDst       <b>[float32_t *]</b>  points to the output vector
  * @param  blockSize  <b>[uint32_t]</b>     number of samples in the vector
  * @return               None
  */
__STATIC_FORCEINLINE void stm32_hsp_scale_f32(float32_t *pSrc,  float32_t scale, float32_t *pDst, uint32_t blockSize)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_SCALE_I_F32);
  HSP_ACC_VectIFVOS_WriteParam(stm32_cmsis_handle, (uint32_t)pSrc, scale, (uint32_t)pDst, blockSize);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(stm32_cmsis_handle);   
}

/**
  * @brief  Fills a constant value into a floating-point vector.
  * @param  value      <b>[float32_t]</b>    input value to be filled
  * @param pDst       <b>[float32_t *]</b>  output pointer
  * @param  blockSize  <b>[uint32_t]</b>     number of samples to process
  * @return               None
  */
__STATIC_FORCEINLINE void stm32_hsp_fill_f32(float32_t value, float32_t *pDst, uint32_t blockSize)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_SET_I);
  HSP_ACC_VectIFVOS_WriteParam(stm32_cmsis_handle, (uint32_t)pDst, value, (uint32_t)pDst, blockSize);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(stm32_cmsis_handle);   
}

/**
  * @brief Floating-point vector subtraction.
  * @param  pSrcA      <b>[float32_t *]</b> points to the first input vector
  * @param  pSrcB      <b>[float32_t *]</b> points to the second input vector
  * @param  pDst       <b>[float32_t *]</b> points to the output vector
  * @param  blockSize  <b>[uint32_t]</b>    number of samples in each vector
  * @return               None
  */
__STATIC_FORCEINLINE void stm32_hsp_sub_f32(float32_t *pSrcA, float32_t *pSrcB, float32_t *pDst, uint32_t blockSize)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_SUB_F32);
  HSP_ACC_VectIIOS_WriteParam(stm32_cmsis_handle, (uint32_t)pSrcA, (uint32_t)pSrcB, (uint32_t)pDst, blockSize);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(stm32_cmsis_handle);  
}

/**
  * @brief  Converts the elements of the Q31 vector to floating-point vector.
  * @param  pSrc       <b>[q31_t *]</b>      is input pointer
  * @param  pDst       <b>[float32_t *]</b>  is output pointer
  * @param  blockSize  <b>[uint32_t]</b>     is the number of samples to process
  * @return               None
  */
__STATIC_FORCEINLINE void stm32_hsp_q31_to_float(int32_t *pSrc, float32_t *pDst, uint32_t blockSize)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_Q312F);
  HSP_ACC_VectIOS_WriteParam(stm32_cmsis_handle, (uint32_t)pSrc, (uint32_t)pDst, blockSize);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(stm32_cmsis_handle);  
}

/**
  * @brief Converts the elements of the floating-point vector to Q31 vector.
  * @param  pSrc       <b>[float32_t *]</b>  points to the floating-point input vector
  * @param pDst       <b>[q31_t *]</b>      points to the Q31 output vector
  * @param  blockSize  <b>[uint32_t]</b>     length of the input vector
  * @return               None
  */
__STATIC_FORCEINLINE void stm32_hsp_float_to_q31(float32_t *pSrc, int32_t *pDst, uint32_t blockSize)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_F2Q31);
  HSP_ACC_VectIOS_WriteParam(stm32_cmsis_handle, (uint32_t)pSrc, (uint32_t)pDst, blockSize);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(stm32_cmsis_handle);  
}

/**
  * @brief         Floating-point vector of log values.
  * @param     pSrc       <b>[float32_t *]</b>  points to the input vector
  * @param     pDst       <b>[float32_t *]</b>  points to the output vector
  * @param     blockSize  <b>[uint32_t]</b>     number of samples in each vector
  * @return        none
  */
__STATIC_FORCEINLINE void stm32_hsp_vlog_f32(float32_t *pSrc, float32_t *pDst, uint32_t blockSize)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_LN_F32);
  HSP_ACC_VectIOS_WriteParam(stm32_cmsis_handle, (uint32_t)pSrc, (uint32_t)pDst, blockSize);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(stm32_cmsis_handle);  
}

/**
  *  @brief         Floating-point vector of exp values.
  *  @param     pSrc       <b>[float32_t *]</b> points to the input vector
  *  @param     pDst       <b>[float32_t *]</b> points to the output vector
  *  @param     blockSize  <b>[uint32_t]</b>    number of samples in each vector
  *  @return        none
  */
__STATIC_FORCEINLINE void stm32_hsp_vexp_f32(float32_t *pSrc, float32_t *pDst, uint32_t blockSize)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_EXP_F32);
  HSP_ACC_VectIOS_WriteParam(stm32_cmsis_handle, (uint32_t)pSrc, (uint32_t)pDst, blockSize);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(stm32_cmsis_handle);  
}

/**
  * @brief  Sum of the squares of the elements of a floating-point vector.
  * @param  pSrc       <b>[float32_t *]</b> is input pointer
  * @param  blockSize  <b>[uint32_t]</b>    is the number of samples to process
  * @param  pResult    <b>[float32_t *]</b> is output value.
  * @return        none
  */
__STATIC_FORCEINLINE void stm32_hsp_power_f32(float32_t *pSrc, uint32_t blockSize, float32_t *pResult)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_DOTPROD_F32);
  HSP_ACC_VectIIOS_WriteParam(stm32_cmsis_handle, (uint32_t)pSrc, (uint32_t)pSrc, (uint32_t)pResult, blockSize);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(stm32_cmsis_handle);  
}

/**
  * @}
  */

/** @defgroup HSP_MODULES_DSP_Matrix_Functions HSP Modules DSP Matrix Functions
  * @{
  */
/** @defgroup HSP_CMSIS_MATRIX_Macros HSP CMSIS Matrix macro
  * @brief    HSP Matrix CMSIS functions
  * @{
  */
/**
  * @brief Floating-point matrix addition.
  * @param  pSrcA  [stm32_hsp_matrix_instance_f32 *]  points to the first input matrix structure
  * @param  pSrcB  [stm32_hsp_matrix_instance_f32 *]  points to the second input matrix structure
  * @param  pDst   [stm32_hsp_matrix_instance_f32 *]  points to output matrix structure
  * @return     None
  * <code>STM32_HSP_MATH_SIZE_MISMATCH</code> or <code>STM32_HSP_MATH_SUCCESS</code> based on the outcome of size checking.
  */
#ifdef STM32_HSP_MATH_MATRIX_CHECK
__STATIC_FORCEINLINE stm32_hsp_status stm32_hsp_mat_add_f32(stm32_hsp_matrix_instance_f32 *pSrcA, stm32_hsp_matrix_instance_f32 *pSrcB, stm32_hsp_matrix_instance_f32 *pDst)
{
  if (((pSrcA)->numRows != (pSrcB)->numRows) || ((pSrcA)->numCols != (pSrcB)->numCols) ||
      ((pSrcA)->numRows != (pDst)->numRows) || ((pSrcA)->numCols != (pDst)->numCols)) 
  {
    return STM32_HSP_MATH_SIZE_MISMATCH;
  }
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_ADD_F32);
  HSP_ACC_VectIIOS_WriteParam(stm32_cmsis_handle, (uint32_t)((pSrcA)->pData), (uint32_t)((pSrcB)->pData),
                              (uint32_t)((pDst)->pData), (pDst)->numCols * (pDst)->numRows);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(stm32_cmsis_handle);   
  return STM32_HSP_MATH_SUCCESS;
}
#else
__STATIC_FORCEINLINE stm32_hsp_status stm32_hsp_mat_add_f32(stm32_hsp_matrix_instance_f32 *pSrcA, stm32_hsp_matrix_instance_f32 *pSrcB, stm32_hsp_matrix_instance_f32 *pDst)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_ADD_F32);
  HSP_ACC_VectIIOS_WriteParam(stm32_cmsis_handle, (uint32_t)((pSrcA)->pData), (uint32_t)((pSrcB)->pData),
                              (uint32_t)((pDst)->pData), (pDst)->numCols * (pDst)->numRows);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(stm32_cmsis_handle); 
  return STM32_HSP_MATH_SUCCESS;
}
#endif

/**
  * @brief Floating-point matrix subtraction
  * @param  pSrcA  [stm32_hsp_matrix_instance_f32 *]  points to the first input matrix structure
  * @param  pSrcB  [stm32_hsp_matrix_instance_f32 *]  points to the second input matrix structure
  * @param  pDst   [stm32_hsp_matrix_instance_f32 *]  points to output matrix structure
  * @return     None
  * <code>STM32_HSP_MATH_SIZE_MISMATCH</code> or <code>STM32_HSP_MATH_SUCCESS</code> based on the outcome of size checking.
  */
#ifdef STM32_HSP_MATH_MATRIX_CHECK
__STATIC_FORCEINLINE stm32_hsp_status stm32_hsp_mat_sub_f32(stm32_hsp_matrix_instance_f32 *pSrcA, stm32_hsp_matrix_instance_f32 *pSrcB, stm32_hsp_matrix_instance_f32 *pDst)
{
  if (((pSrcA)->numRows != (pSrcB)->numRows) || ((pSrcA)->numCols != (pSrcB)->numCols) ||
      ((pSrcA)->numRows != (pDst)->numRows) || ((pSrcA)->numCols != (pDst)->numCols)) 
  {
    return STM32_HSP_MATH_SIZE_MISMATCH;
  }
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_SUB_F32);
  HSP_ACC_VectIIOS_WriteParam(stm32_cmsis_handle, (uint32_t)((pSrcA)->pData), (uint32_t)((pSrcB)->pData),
                              (uint32_t)((pDst)->pData), (pDst)->numCols * (pDst)->numRows);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(stm32_cmsis_handle);  
  return STM32_HSP_MATH_SUCCESS;
}
#else
__STATIC_FORCEINLINE stm32_hsp_status stm32_hsp_mat_sub_f32(stm32_hsp_matrix_instance_f32 *pSrcA, stm32_hsp_matrix_instance_f32 *pSrcB, stm32_hsp_matrix_instance_f32 *pDst)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_SUB_F32);
  HSP_ACC_VectIIOS_WriteParam(stm32_cmsis_handle, (uint32_t)((pSrcA)->pData), (uint32_t)((pSrcB)->pData),
                              (uint32_t)((pDst)->pData), (pDst)->numCols * (pDst)->numRows);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(stm32_cmsis_handle);  
  return STM32_HSP_MATH_SUCCESS;
}
#endif

/**
  * @brief Floating-point matrix inverse.
  * @param  pSrc   [stm32_hsp_matrix_instance_f32 *]  points to the instance of the input floating-point matrix structure.
  * @param  pDst   [stm32_hsp_matrix_instance_f32 *]  points to the instance of the output floating-point matrix structure.
  * @return The function returns STM32_HSP_MATH_SIZE_MISMATCH, if the dimensions do not match.
  * If the input matrix is singular (does not have an inverse), then the algorithm terminates and returns error status ARM_MATH_SINGULAR.
  */
#ifdef STM32_HSP_MATH_MATRIX_CHECK
__STATIC_FORCEINLINE stm32_hsp_status stm32_hsp_mat_inverse_f32(stm32_hsp_matrix_instance_f32 *pSrc, stm32_hsp_matrix_instance_f32 *pDst)
{
  if ((pSrc->numRows != pSrc->numCols) || (pDst->numRows != pDst->numCols) || (pSrc->numRows != pDst->numRows))
  {
    return STM32_HSP_MATH_SIZE_MISMATCH;
  }
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_MAT_INV_F32);
  HSP_ACC_MatInv_WriteParam(stm32_cmsis_handle, (uint32_t)((pSrc)->pData), (uint32_t)((pDst)->pData), ((pDst)->numCols), ((pDst)->numRows));
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(stm32_cmsis_handle);  
  // @TODO: lecture PARAMR0 
  // if (stm32_cmsis_handle->Instance->PARAMR0 != 0)
  // {
    // return STM32_HSP_MATH_SINGULAR;
  // }
  return STM32_HSP_MATH_SUCCESS;
}
#else
__STATIC_FORCEINLINE stm32_hsp_status stm32_hsp_mat_inverse_f32(stm32_hsp_matrix_instance_f32 *pSrc, stm32_hsp_matrix_instance_f32 *pDst)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_MAT_INV_F32);
  HSP_ACC_MatInv_WriteParam(stm32_cmsis_handle, (uint32_t)((pSrc)->pData), (uint32_t)((pDst)->pData), ((pDst)->numCols), ((pDst)->numRows));
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(stm32_cmsis_handle);  
  // @TODO: lecture PARAMR0 
  // if (stm32_cmsis_handle->Instance->PARAMR0 != 0)
  // {
    // return STM32_HSP_MATH_SINGULAR;
  // }
  return STM32_HSP_MATH_SUCCESS;
}
#endif

/**
  * @brief Floating-point matrix transpose.
  * @param  pSrc  [stm32_hsp_matrix_instance_f32 *]  points to the input matrix
  * @param  pDst  [stm32_hsp_matrix_instance_f32 *]  points to the output matrix
  * @return    The function returns either  <code>STM32_HSP_MATH_SIZE_MISMATCH</code>
  * or <code>STM32_HSP_MATH_SUCCESS</code> based on the outcome of size checking.
  */
#ifdef STM32_HSP_MATH_MATRIX_CHECK
__STATIC_FORCEINLINE stm32_hsp_status stm32_hsp_mat_trans_f32(stm32_hsp_matrix_instance_f32 *pSrc, stm32_hsp_matrix_instance_f32 *pDst)
{
  if (((pSrc)->numRows != (pDst)->numCols) || ((pSrc)->numCols != (pDst)->numRows))
  {
    return STM32_HSP_MATH_SIZE_MISMATCH;
  }
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_MAT_TRANS_F32);
  HSP_ACC_MatTrans_WriteParam(stm32_cmsis_handle, (uint32_t)((pSrc)->pData), (uint32_t)((pDst)->pData), ((pDst)->numCols), ((pDst)->numRows));
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(stm32_cmsis_handle);  
  return STM32_HSP_MATH_SUCCESS;
}
#else
__STATIC_FORCEINLINE stm32_hsp_status stm32_hsp_mat_trans_f32(stm32_hsp_matrix_instance_f32 *pSrc, stm32_hsp_matrix_instance_f32 *pDst)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_MAT_TRANS_F32);
  HSP_ACC_MatTrans_WriteParam(stm32_cmsis_handle, (uint32_t)((pSrc)->pData), (uint32_t)((pDst)->pData), ((pDst)->numCols), ((pDst)->numRows));
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(stm32_cmsis_handle);  
  return STM32_HSP_MATH_SUCCESS;
}
#endif

/**
  * @brief Floating-point matrix multiplication
  * @param  pSrcA  [stm32_hsp_matrix_instance_f32 *]  points to the first input matrix structure
  * @param  pSrcB  [stm32_hsp_matrix_instance_f32 *]  points to the second input matrix structure
  * @param  pDst   [stm32_hsp_matrix_instance_f32 *]  points to output matrix structure
  * @return     The function returns either
  * <code>STM32_HSP_MATH_SIZE_MISMATCH</code> or <code>STM32_HSP_MATH_SUCCESS</code> based on the outcome of size checking.
  */
#ifdef STM32_HSP_MATH_MATRIX_CHECK
__STATIC_FORCEINLINE stm32_hsp_status stm32_hsp_mat_mult_f32(stm32_hsp_matrix_instance_f32 *pSrcA, stm32_hsp_matrix_instance_f32 *pSrcB, stm32_hsp_matrix_instance_f32 *pDst)
{
  if (((pSrcA)->numCols != (pSrcB)->numRows) || ((pSrcA)->numRows != (pDst)->numRows) || ((pSrcB)->numCols != (pDst)->numCols))
  {
    return (STM32_HSP_MATH_SIZE_MISMATCH);
  }
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_MAT_MUL_F32);
  HSP_ACC_MatMult_WriteParam(stm32_cmsis_handle, (uint32_t)((pSrcA)->pData), (uint32_t)((pSrcB)->pData), (uint32_t)((pDst)->pData),
                             ((pSrcA)->numRows), ((pSrcA)->numCols),((pSrcB)->numRows), ((pSrcB)->numCols));
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(stm32_cmsis_handle);  
  return STM32_HSP_MATH_SUCCESS;
}
#else
__STATIC_FORCEINLINE stm32_hsp_status stm32_hsp_mat_mult_f32(stm32_hsp_matrix_instance_f32 *pSrcA, stm32_hsp_matrix_instance_f32 *pSrcB, stm32_hsp_matrix_instance_f32 *pDst)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_MAT_MUL_F32);
  HSP_ACC_MatMult_WriteParam(stm32_cmsis_handle, (uint32_t)((pSrcA)->pData), (uint32_t)((pSrcB)->pData), (uint32_t)((pDst)->pData),
                             ((pSrcA)->numRows), ((pSrcA)->numCols),((pSrcB)->numRows), ((pSrcB)->numCols));
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(stm32_cmsis_handle);  
  return STM32_HSP_MATH_SUCCESS;
}
#endif

/**
  * @brief Floating-point matrix scaling.
  * @param  pSrc   [stm32_hsp_matrix_instance_f32 *]  points to the input matrix
  * @param  scale  [stm32_hsp_matrix_instance_f32 *]  scale factor
  * @param  pDst   [stm32_hsp_matrix_instance_f32 *]  points to the output matrix
  * @return     The function returns either
  * <code>STM32_HSP_MATH_SIZE_MISMATCH</code> or <code>STM32_HSP_MATH_SUCCESS</code> based on the outcome of size checking.
  */
#ifdef STM32_HSP_MATH_MATRIX_CHECK
__STATIC_FORCEINLINE stm32_hsp_status stm32_hsp_mat_scale_f32(stm32_hsp_matrix_instance_f32 *pSrc, float32_t scale, stm32_hsp_matrix_instance_f32 *pDst)
{
  if (((pSrc)->numRows != (pDst)->numRows) || ((pSrc)->numCols != (pDst)->numCols))
  {
    return (STM32_HSP_MATH_SIZE_MISMATCH);
  }
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_SCALE_I_F32);
  HSP_ACC_VectIFVOS_WriteParam(stm32_cmsis_handle, (uint32_t)((pSrc)->pData), scale, (uint32_t)((pDst)->pData),
                              (pDst)->numCols*(pDst)->numRows);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(stm32_cmsis_handle);
  return STM32_HSP_MATH_SUCCESS;
}
#else
__STATIC_FORCEINLINE stm32_hsp_status stm32_hsp_mat_scale_f32(stm32_hsp_matrix_instance_f32 *pSrc, float32_t scale, stm32_hsp_matrix_instance_f32 *pDst)
{
  HSP_HW_IF_WRITE_DCMDIDR(HSP_DIRECT_CMD_VEC_SCALE_I_F32);
  HSP_ACC_VectIFVOS_WriteParam(stm32_cmsis_handle, (uint32_t)((pSrc)->pData), scale, (uint32_t)((pDst)->pData),
                              (pDst)->numCols*(pDst)->numRows);
  HSP_ACC_WAIT_END_OF_DIRECT_COMMAND(stm32_cmsis_handle);
  return STM32_HSP_MATH_SUCCESS;
}
#endif

/**
  * @}
  */

/** @defgroup HSP_MODULES_DSP_Init_Functions HSP Modules DSP Transform Functions
  * @{
  */
stm32_hsp_status stm32_hsp_init(hsp_core_handle_t *hmw);

/**
  *  @brief         Initialization function for the floating-point real FFT.
  *  @param     S            points to an arm_rfft_fast_instance_f32 structure
  *  @param     fftLen       length of the Real Sequence
  *  @return        execution status
  *                   - \ref STM32_HSP_MATH_SUCCESS        : Operation successful
  *                   - \ref STM32_HSP_MATH_ARGUMENT_ERROR : <code>fftLen</code> is not a supported length
  *  @par           Description
  *                   The parameter <code>fftLen</code> specifies the length of RFFT/CIFFT process.
  *                   Supported RFFT Lengths are 64, 128, 256, 512, 1024, 2048, 4096.
  */
stm32_hsp_status stm32_hsp_rfft_fast_init_f32(stm32_hsp_rfft_fast_instance_f32 *S, uint16_t fftLen);

/**
  * @brief         Initialization function for the cfft f32 function
  * @param     S              points to an instance of the floating-point CFFT structure
  * @param     fftLen         fft length (number of complex samples)
  * @return        execution status
  *                  - \ref STM32_HSP_MATH_SUCCESS        : Operation successful
  *                 - \ref STM32_HSP_MATH_ARGUMENT_ERROR : an error is detected
  */
stm32_hsp_status stm32_hsp_cfft_init_f32(stm32_hsp_cfft_instance_f32 *S, uint16_t fftLen);

/**
  * @brief  Initialization function for the floating-point FIR filter.
  * @param     S          points to an instance of the floating-point FIR filter structure.
  * @param     numTaps    Number of filter coefficients in the filter.
  * @param     pCoeffs    points to the filter coefficients.
  * @param     pState     points to the state buffer.
  * @param     blockSize  number of samples that are processed at a time.
  * @return        None
  */
void stm32_hsp_fir_init_f32(stm32_hsp_fir_instance_f32 *S, uint16_t numTaps, const float32_t *pCoeffs,
                            float32_t *pState, uint32_t blockSize);

/**
  * @brief         Initialization function for the floating-point FIR decimator.
  * @param     S          points to an instance of the floating-point FIR decimator structure
  * @param     numTaps    number of coefficients in the filter
  * @param     M          decimation factor
  * @param     pCoeffs    points to the filter coefficients
  * @param     pState     points to the state buffer
  * @param     blockSize  number of input samples to process per call
  * @return        execution status
  *                    - \ref STM32_HSP_MATH_SUCCESS      : Operation successful
  *                    - \ref STM32_HSP_MATH_LENGTH_ERROR : <code>blockSize</code> is not a multiple of <code>M</code>
  * @return        None
  */
stm32_hsp_status stm32_hsp_fir_decimate_init_f32(stm32_hsp_fir_decimate_instance_f32 *S, uint16_t numTaps, uint8_t M,
                                                 const float32_t *pCoeffs, float32_t *pState, uint32_t blockSize);

/**
  * @brief  Initialization function for the floating-point Biquad cascade filter.
  * @param S          points to an instance of the floating-point Biquad cascade structure.
  * @param     numStages  number of 2nd order stages in the filter.
  * @param     pCoeffs    points to the filter coefficients.
  * @param     pState     points to the state buffer.
  * @return        None
  */
void stm32_hsp_biquad_cascade_df1_init_f32(stm32_hsp_biquad_casd_df1_inst_f32 *S, uint8_t numStages,
                                           const float32_t *pCoeffs,
                                           float32_t *pState);
/**
  *   @brief         Initialization function for the floating-point transposed direct form II Biquad cascade filter.
  *   @param     S           points to an instance of the filter data structure.
  *   @param     numStages   number of 2nd order stages in the filter.
  *   @param     pCoeffs     points to the filter coefficients.
  *   @param     pState      points to the state buffer.
  * @return        None
  *
  *   @par           Coefficient and State Ordering
  *                  The coefficients are stored in the array <code>pCoeffs</code> in the following order
  *   <pre>
  *       {b10, b11, b12, a11, a12, b20, b21, b22, a21, a22, ...}
  *   </pre>
  *   @par
  *                    where <code>b1x</code> and <code>a1x</code> are the coefficients for the first stage,
  *                    <code>b2x</code> and <code>a2x</code> are the coefficients for the second stage,
  *                    and so on.  The <code>pCoeffs</code> array contains a total of <code>5*numStages</code> values.
  *                    and it must be initialized using the function
  *                    arm_biquad_cascade_df2T_compute_coefs_f32 which is taking the
  *                    standard array coefficient as parameters.
  *   @par
  *                    The <code>pState</code> is a pointer to state array.
  *                    Each Biquad stage has 2 state variables <code>d1,</code> and <code>d2</code>.
  *                    The 2 state variables for stage 1 are first, then the 2 state variables for stage 2, and so on.
  *                    The state array has a total length of <code>2*numStages</code> values.
  *                    The state variables are updated after each block of data is processed; the coefficients are untouched.
  */
void stm32_hsp_biquad_cascade_df2T_init_f32(stm32_hsp_biquad_cascade_df2T_instance_f32 *S, uint8_t numStages,
                                            const float32_t *pCoeffs, float32_t *pState);

/**
  * @brief Initialization function for floating-point LMS filter.
  * @param S          points to an instance of the floating-point LMS filter structure.
  * @param numTaps    number of filter coefficients.
  * @param pCoeffs    points to the coefficient buffer.
  * @param pState     points to state buffer.
  * @param mu         step size that controls filter coefficient updates.
  * @param blockSize  number of samples to process.
  * @return        None
  */
void stm32_hsp_lms_init_f32(stm32_hsp_lms_instance_f32 *S, uint16_t numTaps, float32_t *pCoeffs, float32_t *pState,
                            float32_t mu, uint32_t blockSize);

/**
  * @brief Initialization function for the floating-point IIR lattice filter.
  * @param S          points to an instance of the floating-point IIR lattice structure.
  * @param numStages  number of stages in the filter.
  * @param pkCoeffs   points to the reflection coefficient buffer.  The array is of length numStages.
  * @param pvCoeffs   points to the ladder coefficient buffer.  The array is of length numStages+1.
  * @param pState     points to the state buffer.
  * @param blockSize  number of samples to process.
  * @return        None
  */
void stm32_hsp_iir_lattice_init_f32(stm32_hsp_iir_lattice_instance_f32 *S, uint16_t numStages, float32_t *pkCoeffs,
                                    float32_t *pvCoeffs, float32_t *pState, uint32_t blockSize);

/**
  * @brief  Floating-point matrix initialization.
  * @param     S         points to an instance of the floating-point matrix structure.
  * @param     nRows     number of rows in the matrix.
  * @param     nColumns  number of columns in the matrix.
  * @param     pData     points to the matrix data array.
  * @return        None
  */
void stm32_hsp_mat_init_f32(stm32_hsp_matrix_instance_f32 *S, uint16_t nRows, uint16_t nColumns, float32_t *pData);
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

#endif /* HSP_DSP_H */
