#include <stdint.h>
#include <stdio.h>
#include "preproc_dpu.h"
#include "test_common_functions.h"
#include "LogMel_aed_test_data.h"

#define lc_print printf

static int8_t LogMel_Aed_Out[CTRL_X_CUBE_AI_SPECTROGRAM_NMEL*CTRL_X_CUBE_AI_SPECTROGRAM_COL] = {0};

int32_t Test_LogMel_Aed_hsp_bus() 
{
  static AudioPreProcCtx_t Ctx;
  enable_dwt(); 
  int16_t *pi16_pcm_in = (int16_t *) &LogMel_aed_bus1_input[0];
  int8_t *pi8_LogMel = (int8_t *) &LogMel_aed_bus1_output[0];
  int time;
  
  AudioPreProc_DPUInit(&Ctx);
  Ctx.out_Q_offset = 40;
  Ctx.out_Q_inv_scale = 1/0.054722581058740616 ;
 
  reset_timer();
  start_timer();

  AudioPreProc_DPU(&Ctx, (uint8_t *) &pi16_pcm_in[0], LogMel_Aed_Out);

  time =  get_timer();
  printf("Line Cycles %d\n\r", time);
  
  int32_t nb_errors = 0;
  nb_errors += compare_int8("MCU LogMel", (uint8_t *) LogMel_Aed_Out, (uint8_t *)  pi8_LogMel, CTRL_X_CUBE_AI_SPECTROGRAM_NMEL*CTRL_X_CUBE_AI_SPECTROGRAM_COL, P60_CHECK_DO_CMP);
  
  uint32_t *p_in = (uint32_t *) 0x200A0000;
  for(int i = 0; i < 4096; i++)
  {
      p_in[i] = 0xcafebabe;
  }

  reset_timer();
  start_timer();

  AudioPreProc_DPU(&Ctx, (uint8_t *) &pi16_pcm_in[0], LogMel_Aed_Out);

  
  time =  get_timer();
  lc_print("Line Cycles %d\n\r", time);
  
  nb_errors = 0;
  nb_errors += compare_int8("MCU LogMel", (uint8_t *) LogMel_Aed_Out, (uint8_t *)  pi8_LogMel, CTRL_X_CUBE_AI_SPECTROGRAM_NMEL*CTRL_X_CUBE_AI_SPECTROGRAM_COL, P60_CHECK_DO_CMP);

  return nb_errors;
}
