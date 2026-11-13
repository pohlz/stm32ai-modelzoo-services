#include <arm_math_types.h>
#include "test_common_functions.h"
#include <math.h>
#include <stdio.h>

#define lc_print printf

/**
 * Compare data table with ref
 * data Input vector
 * ref Reference vector to compare
 * size Number of elements to compare
 * noCompare Flag to no compare data and ref but check _errorReportCount was set by Hyperion
 * return number of check failed
 */

uint32_t compare_int8(const char *testName, uint8_t *data, uint8_t *ref, uint32_t size, uint16_t noCompare)
{
    
  uint32_t nbCheckFailed = 0;
  int8_t maxDiff = 0;
  for (uint16_t idx = 0; idx < size; idx++) {
    if (data[idx] != ref[idx]) {
      int8_t diff = 0;
      if (ref[idx] > data[idx]) {
        diff = ref[idx] - data[idx];
      } else {
        diff = data[idx] - ref[idx];
      }
      if (diff > maxDiff) {
        maxDiff = diff;
      }
     if (noCompare != P60_CHECK_DO_CMP) {
        if (nbCheckFailed < 16) {
          lc_print("ERROR at[%3d]: Res 0x%02x != Exp 0x%02x\n", idx, data[idx], ref[idx]);
        }
      } else {
        if (nbCheckFailed < 16) {
        //if (nbCheckFailed < 130) {
        lc_print("ERROR at[%3d]: Res 0x%02x != Exp 0x%02x\n", idx, data[idx], ref[idx]);
      }
      }
      nbCheckFailed++;
    }
  }
  if (nbCheckFailed == 0) {
    TEST_INFO_REPORT(testName, "\tPASSED\n");
  } else {
    TEST_INFO_REPORT(testName, "ERROR: nb check %d\n", nbCheckFailed);
    TEST_INFO_REPORT(testName, "Max difference %d (0x%02x)\n", maxDiff, maxDiff);
  }
  return nbCheckFailed;
}

void CompareFloatResults(const char* TestName, const float32_t *pf_Ref, float *pf_res, int32_t size, int frame_id)
{
  float32_t ref =  0;
  float_t diff =  0;
  float_t snr_min = 200;
  
  for(int i = 0; i < size; i++)
  {
      float32_t ref_value =  pf_Ref[i + frame_id*size];
      float32_t res_value = pf_res[i];
 
      ref += ref_value*ref_value;
      diff += (ref_value - res_value)*(ref_value - res_value);
      if(ref_value != res_value)          
      {
          float32_t snr_local = 10*log10f(ref_value*ref_value/ ((ref_value - res_value)*(ref_value - res_value)));
          if(snr_local < snr_min)
              snr_min = snr_local;
      } 
  }
  
  float snr = 10*log10f(ref/diff);
  lc_print("%s Idx = %d ,  SNR_MIN = %f, SNR : %f \n", TestName, frame_id, snr_min, snr);
}