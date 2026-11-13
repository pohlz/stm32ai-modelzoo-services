/**
  ******************************************************************************
  * @file   : motion_bm.c
  * @author : STMicroelectronics AIS application team
  * @brief  : Main program body
  * @version $Version$
  * @date    $Date$
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
#include "main.h"
#include "motion_bm.h"
#include "logging.h"
#include "preproc_dpu.h"
//#include "iks02a1_motion_sensors.h"
//#include "iks02a1_motion_sensors_ex.h"
//#include "iks02a1_motion_sensors_ex_patch.h"
#include "b_u585i_iot02a_motion_sensors.h"
#include "gs_utils.h"

#include <stdlib.h>

/* Globals--------------------------------------------------------------------*/
MotPreProc_t mot_pre_proc_ctx ;
MotProc_t    mot_proc_ctx ;
volatile int motion_acq_buff_size;

/* Private defines -----------------------------------------------------------*/
#define ISM330DHCX_TAG_ACC   ( 0x02 )

/* Private function prototypes -----------------------------------------------*/
static int InitMotionSensors(int fifo_level);
static int deInitMotionSensors(void);

/* Private variables ---------------------------------------------------------*/
static const char *sAiMotionClassLabels[CTRL_X_CUBE_AI_MODEL_CLASS_NUMBER] = \
                                                 CTRL_X_CUBE_AI_MODEL_CLASS_LIST;
/**
 * @brief initializes the motion pre-processing chain,
 * including samples acquisition
 * @param pCtx pointer to motion pre-processing context
 * @param h height of samples matrix (2D)
 * @param w width of samples matrix (2D)
 * @retval 0 if success
 */
int motion_pre_proc_init(MotPreProc_t* pCtx /*,int h , int w*/)
{
  int ret = 0 ;

  /* allocates buffers for acquisition and buffers for pre-processing */
  pCtx->acq_p = (int8_t* ) MEM_ALLOC ( pCtx->dpu.in_height * 7 * sizeof(int8_t));
  pCtx->pre_in_p = (int16_t * ) MEM_ALLOC (pCtx->dpu.in_height*pCtx->dpu.in_width * sizeof(int16_t));
 
  if ( NULL == pCtx->acq_p || NULL == pCtx->pre_in_p )
  {
    LogError("Malloc failed\n");
    ret = -1;
  }
  if ( DPU_OK != MotionPreProc_DPUInit(&pCtx->dpu))
  {
    LogError("Error while initializing motion pre-processing DPU.");
    ret = -1;
  }

  if ( 0 != InitMotionSensors(pCtx->dpu.in_height) )
  {
    LogError("Error while initializing motion sensor.");
    ret = -1;
  }

  return ret;
}

/**
 * @brief de-initializes the motion pre-processing chain,
 * including samples acquisition
 * @param pCtx pointer to motion pre-processing context
 * @retval 0 if success
 */
int motion_pre_proc_de_init(MotPreProc_t* pCtx)
{
  deInitMotionSensors();
  MEM_FREE(pCtx->acq_p);
  MEM_FREE(pCtx->pre_in_p);
  return 0;
}

/**
 * @brief initializes the motion processing chain, including
 * samples acquisition, preprocessing and AI inference
 * @param None
 * @retval 0 if success
 */
int motion_init_bm(void)
{
  /* get the AI model   */
  AiDPULoadModel(&mot_proc_ctx.ai);
  mot_proc_ctx.ai.classes = sAiMotionClassLabels;

  /* get in/out tensors */
  mot_proc_ctx.ai_in_p  = ( float* ) mot_proc_ctx.ai.p_stai_inputs[0]  ;
  mot_proc_ctx.ai_out_p = ( float* ) mot_proc_ctx.ai.p_stai_outputs[0] ;
#if (CTRL_AI_FFT            == CTRL_X_CUBE_AI_PREPROC )  
  mot_pre_proc_ctx.dpu.in_height = mot_proc_ctx.ai.in_height * 2 ;
#else
  mot_pre_proc_ctx.dpu.in_height = mot_proc_ctx.ai.in_height;
#endif

  mot_pre_proc_ctx.dpu.in_width = mot_proc_ctx.ai.in_width ;
  motion_pre_proc_init(&mot_pre_proc_ctx);
  return 0;
}

/**
 * @brief de-initializes the motion processing chain, including
 * samples acquisition, pre-processing and AI inference
 * @param None
 * @retval 0 if success
 */
int motion_de_init_bm(void)
{
  AiDPUReleaseModel(&mot_proc_ctx.ai);
  motion_pre_proc_de_init(&mot_pre_proc_ctx);
  return 0;
}

/**
 * @brief executes the motion processing chain, including
 * samples acquisition, pre-processing and AI inference
 * @param None
 * @retval None
 */
void motion_exec_bm(void)
{
  assert_param(CTRL_X_CUBE_AI_SPECTROGRAM_NFFT    == mot_pre_proc_ctx.dpu.in_height);
  assert_param(CTRL_X_CUBE_AI_SPECTROGRAM_NFFT/2  == mot_proc_ctx.ai.in_height);
  assert_param(3 == CTRL_X_CUBE_AI_SENSOR_DIM);
  
  if (motion_acq_buff_size >= mot_pre_proc_ctx.dpu.in_height)
  {
    motion_acq_buff_size -= mot_pre_proc_ctx.dpu.in_height ;
    MotionPreProc_DPU(&mot_pre_proc_ctx.dpu, mot_pre_proc_ctx.pre_in_p, mot_proc_ctx.ai_in_p);
    AiDPUProcess(&mot_proc_ctx.ai,mot_proc_ctx.ai_out_p);
    PrintAIClassesOutput(mot_proc_ctx.ai_out_p,mot_proc_ctx.ai.classes);
  }
}

/* Callback Definition -------------------------------------------------------*/
#define ISM330DHCX_TAG_ACC   ( 0x02 )
#define ISM330DHCX 0

int32_t MOTION_SENSOR_FIFO_Get_Fifo(uint32_t Instance, uint32_t Function,int8_t* p_in, int16_t * p_out, int h)
{
	return  0;
}

/**
 * @brief Acquires 3D samples from accelerometer
 * @param h height of samples 3D arrayr
 * @param p_in pointer to raw acquired samples
 * @param p_out pointer to stored samples correctly formated to int16
 * @retval None
 */
void ISM330DHCX_acquire_samples(int h, int8_t* p_in, /*float**/ int16_t * p_out)
{
  ISM330DHCX_Get_Fifo(0,h,p_in);
  for (int i = 0; i < h; i++)
  {
    if ((*p_in >> 3) == ISM330DHCX_TAG_ACC)
    {
//      p_in++; /* Skip tag */
//      *p_out++ = (float)(*(int16_t*)p_in);
//      p_in += 2;
//      *p_out++ = (float)(*(int16_t*)p_in);
//      p_in += 2;
//      *p_out++ = (float)(*(int16_t*)p_in);
//      p_in += 2;
        p_in++; /* Skip tag */
        *p_out = *(int16_t*)p_in;
        p_in += 2;
        *(p_out+h) = *(int16_t*)p_in;
        p_in += 2;
        *(p_out+2*h) = *(int16_t*)p_in;
        p_in += 2;
        p_out++;    }
    else
    {
      p_in += 7; /* Skip non-ACC data */
    }
  }
}

void  ISM330DHCX_EXTI_Callback(uint16_t GPIO_Pin)
//void HAL_GPIO_EXTI_Rising_Callback(uint16_t GPIO_Pin)
{
  if (GPIO_PIN_11==GPIO_Pin)
  {
#if 0
    IKS02A1_MOTION_SENSOR_FIFO_Get_Fifo(IKS02A1_ISM330DHCX_0 , MOTION_ACCELERO,\
        mot_pre_proc_ctx.acq_p , mot_pre_proc_ctx.pre_in_p,\
        mot_pre_proc_ctx.dpu.in_height);
    motion_acq_buff_size += mot_pre_proc_ctx.dpu.in_height ;
    MOTION_SENSOR_FIFO_Get_Fifo(ISM330DHCX , MOTION_ACCELERO,\
        mot_pre_proc_ctx.acq_p , mot_pre_proc_ctx.pre_in_p,\
        mot_pre_proc_ctx.dpu.in_height);
#endif
    ISM330DHCX_acquire_samples(mot_pre_proc_ctx.dpu.in_height,
        mot_pre_proc_ctx.acq_p, mot_pre_proc_ctx.pre_in_p);
    motion_acq_buff_size += mot_pre_proc_ctx.dpu.in_height ;

  }
}

/* Private functions Definition ----------------------------------------------*/
/**
 * @brief initializes the motion sensor (3D-accelerometer)
 * @param fifo_level,
 * @retval BSP_ERROR_NONE if success.
 */

static int InitMotionSensors(int fifo_level)
{
#if 0
	IKS02A1_MOTION_SENSOR_Init(IKS02A1_ISM330DHCX_0, MOTION_ACCELERO);

    IKS02A1_MOTION_SENSOR_SetOutputDataRate( IKS02A1_ISM330DHCX_0, \
                                  MOTION_ACCELERO , CTRL_X_CUBE_AI_SENSOR_ODR);

    IKS02A1_MOTION_SENSOR_SetFullScale( IKS02A1_ISM330DHCX_0 , MOTION_ACCELERO,\
                                              (int32_t)CTRL_X_CUBE_AI_SENSOR_FS);

    IKS02A1_MOTION_SENSOR_FIFO_Set_BDR(IKS02A1_ISM330DHCX_0, MOTION_ACCELERO, \
                                                     CTRL_X_CUBE_AI_SENSOR_ODR);

    IKS02A1_MOTION_SENSOR_FIFO_Set_Watermark_Level(IKS02A1_ISM330DHCX_0,\
                                                                    fifo_level);

    IKS02A1_MOTION_SENSOR_FIFO_Set_INT1_FIFO_Threshold(IKS02A1_ISM330DHCX_0,\
                                                                        ENABLE);

    IKS02A1_MOTION_SENSOR_FIFO_Set_Mode(IKS02A1_ISM330DHCX_0,\
                                                        ISM330DHCX_STREAM_MODE);

    IKS02A1_MOTION_SENSOR_Enable(IKS02A1_ISM330DHCX_0 , MOTION_ACCELERO);
#endif
    int32_t lBspError = BSP_ERROR_NONE;

    lBspError  = BSP_MOTION_SENSOR_Init( 0,MOTION_ACCELERO );
    lBspError |= BSP_MOTION_SENSOR_SetOutputDataRate( 0, MOTION_ACCELERO , CTRL_X_CUBE_AI_SENSOR_ODR);
    lBspError |= BSP_MOTION_SENSOR_SetFullScale( 0 , MOTION_ACCELERO , CTRL_X_CUBE_AI_SENSOR_FS);
    lBspError |= BSP_MOTION_SENSOR_Fifo( 0 , MOTION_ACCELERO , fifo_level);
    lBspError |= BSP_MOTION_SENSOR_Enable( 0 , MOTION_ACCELERO);

    return lBspError;

    
    return 0;
}

/**
 * @brief initializes the motion sensor (3D-accelerometer)
 * @param fifo_level,
 * @retval BSP_ERROR_NONE if success.
 */
static int deInitMotionSensors(void)
{
//    IKS02A1_MOTION_SENSOR_Disable(IKS02A1_ISM330DHCX_0 , MOTION_ACCELERO);
    return 0;
}
