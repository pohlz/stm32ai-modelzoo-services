/**
  ******************************************************************************
  * @file    hsp_memcpy.c
  * @author  MCD Application Team
  * @brief   This file implements the optimized memcpy function
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
#include <stdint.h>
#include "hsp_memcpy.h"


extern void st_hsp_memcpy32(int8_t *pDst, int8_t *pSrc, uint32_t blockSize);

/**
* @brief Copies a int8 vector into a int8 vector.
* @param[out]   *pDst points to output vector
* @param[in]    *pSrc points to inpout vector
* @param[in]    blockSize length of the output vector
* @return none.
*
*/
void * stm32_hsp_memcpy(int8_t *pDst, int8_t *pSrc, uint32_t blockSize)
{
  /* trivial sanity check */
  if ((pSrc == pDst) || (blockSize == 0)) 
  {
    return pDst;
  }

  /* Handle buffers overlapping when address of dst is > address of src */
  const int8_t* pSrcEnd = pSrc + blockSize;
  if ((pDst > pSrc) && (pDst < pSrcEnd))
  {
    pDst += blockSize;
    while (pSrcEnd > pSrc)
    {
      *(--pDst) = *(--pSrcEnd);
    }
    return pDst;
  }

  if (blockSize < 4)
  {
    while (blockSize-- > 0)
    {
      __asm volatile ("ldrb r3, [%[in]], #1\n\t"
                      "strb r3, [%[out]], #1\n\t"
                      : [out] "+r" (pDst), [in] "+r" (pSrc) :: "r3");
    }
      return pDst;
  }

  intptr_t pDst_offset = ((intptr_t) pDst) & 0x3;
  intptr_t pSrc_offset = ((intptr_t) pSrc) & 0x3;
  int32_t n = (4 - pDst_offset) % 4;

  blockSize -= n;

  while (n-- > 0)
  {
    __asm volatile ("ldrb r3, [%[in]], #1\n\t"
                    "strb r3, [%[out]], #1\n\t"
                    : [out] "+r" (pDst), [in] "+r" (pSrc) :: "r3");
  }

  if (pDst_offset == pSrc_offset)
  {
    /* Both input and output are 32 bits aligned */
    st_hsp_memcpy32(pDst, pSrc, blockSize);
  }
  else
  {
    n = blockSize >> 4;

    /* Onlyt Output is 32 bit aligned */
    while (n-- > 0)
    {
      __asm volatile ("ldr r3, [%[in]], #4\n\t"
                      "ldr r4, [%[in]], #4\n\t"
                      "ldr r5, [%[in]], #4\n\t"
                      "ldr r6, [%[in]], #4\n\t"
                      "stmia %[out]!, {r3, r4, r5, r6}\n\t"
                      : [out] "+r" (pDst), [in] "+r" (pSrc) :: "r3", "r4", "r5", "r6" );
    }

    if ((blockSize >> 3) & 1)
    {
      __asm volatile ("ldr r3, [%[in]], #4\n\t"
                      "ldr r4, [%[in]], #4\n\t"
                      "stmia %[out]!, {r3, r4}\n\t"
                      : [out] "+r" (pDst), [in] "+r" (pSrc) :: "r3", "r4");
    }

    if ((blockSize >> 2) & 1)
    {
      __asm volatile ("ldr r3, [%[in]], #4\n\t"
                      "str r3, [%[out]], #4\n\t"
                      : [out] "+r" (pDst), [in] "+r" (pSrc) :: "r3");
    }

    if ((blockSize >> 1) & 1)
    {
      __asm volatile ("ldrh r3, [%[in]], #2\n\t"
                      "strh r3, [%[out]], #2\n\t"
                      : [out] "+r" (pDst), [in] "+r" (pSrc) :: "r3");
    }

    if (blockSize & 1)
    {
      __asm volatile ("ldrb r3, [%[in]]\n\t"
                      "strb r3, [%[out]]\n\t"
                      : [out] "+r" (pDst), [in] "+r" (pSrc) :: "r3");
    }
  } 
  return pDst;
}
