#ifndef TEST_COMMON_FUNCTIONS_H
#define TEST_COMMON_FUNCTIONS_H

extern void lc_print(const char* fmt, ... );
#define P60_CHECK_DO_CMP 0 ///< Do comparison when call check functions
#define P60_CHECK_NO_CMP 1 ///< Do not comparison when call check functions but check _errorReportCount is set
#define TEST_INFO_REPORT(name, fmt, ...) \
{ \
  lc_print("Test %s: ", name); \
  lc_print(fmt, ##__VA_ARGS__); \
}

#define enable_dwt()    (*((volatile uint32_t *)0xE000EDFC)) |= 0x01000000;
#define start_timer()    *((volatile uint32_t*)0xE0001000) = 0x40000001  // Enable CYCCNT register
#define stop_timer()   *((volatile uint32_t*)0xE0001000) = 0x40000000  // Disable CYCCNT register
#define reset_timer()   *((volatile uint32_t*)0xE0001004) = 0
#define get_timer()   *((volatile uint32_t*)0xE0001004)       


uint32_t compare_int8(const char *testName, uint8_t *data, uint8_t *ref, uint32_t size, uint16_t noCompare);

void CompareFloatResults(const char* TestName, const float32_t *pf_Ref, float *pf_res, int32_t size, int frame_id);

#endif
