/**
  ******************************************************************************
  * @file   : motion_bm.h
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
#ifndef __MOTION_BM_H__
#define __MOTION_BM_H__

#include "ai_dpu.h"
#include "preproc_dpu.h"

typedef struct _MotPreProc_t
{
  int8_t* acq_p;
  MotionPreProcCtx_t dpu;
  int16_t* pre_in_p;
  float*  pre_out_p;
}MotPreProc_t;

typedef struct _MotProc_t
{
  AIProcCtx_t ai;
  float*      ai_in_p;
  float*      ai_out_p;
}MotProc_t;

extern int motion_pre_proc_init(MotPreProc_t* pCtx);
extern int motion_pre_proc_de_init(MotPreProc_t* pCtx);
extern int motion_init_bm(void);
extern int motion_de_init_bm(void);
extern void motion_exec_bm(void);
extern void ISM330DHCX_acquire_samples(int h, int8_t* p_in, /*float**/ int16_t * p_out);

#endif /* MOTION_BM */
