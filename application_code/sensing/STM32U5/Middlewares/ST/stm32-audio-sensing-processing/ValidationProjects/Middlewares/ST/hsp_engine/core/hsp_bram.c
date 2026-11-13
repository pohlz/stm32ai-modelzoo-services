/**
  ******************************************************************************
  * @file    hsp_bram.c
  * @author  MCD Application Team
  * @brief   This file implements the functions for the core state machine process
  *          the enumeration and the control transfer process
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
#include "hsp_bram.h"
#include "hsp_bram_if.h"
#include "hsp_hw_if.h"

/** @addtogroup HSP_ENGINE
  * @{
  */

/** @addtogroup HSP_CORE
  * @{
  */

/** @addtogroup HSP_BRAM
  * @{
  */

/** @addtogroup HSP_BRAM_Exported_Functions
  * @{
  */
/** @addtogroup HSP_BRAM_Exported_Functions_Group1
  * @{
  */

/**
  * @brief  Allocation in BRAM_AB
  * @param  hmw        HSP handle
  * @param  alloc_size Number of word to allocate
  * @param  area       HSP_BRAM_ALLOCATION_DEFAULT: allocation in all BRAM
  *                    HSP_BRAM_ALLOCATION_PERSISTENT: allocation in persistent memory
  * @retval            NULL  Allocation failure
  *                    Address value if success
  */
void *HSP_BRAM_Malloc(hsp_core_handle_t *hmw, uint32_t size_in_word, hsp_bram_allocation_t area)
{
  return (void *)HSP_BRAM_IF_Malloc(&(hmw->hbram), size_in_word, area);
}

/**
  * @brief Returns the free space in MEMAB
  * @param  hmw      HSP handle
  * @retval          Number of free word 
  */
uint32_t HSP_BRAM_GetFreeSize(hsp_core_handle_t *hmw, hsp_bram_allocation_t area)
{
  return HSP_BRAM_IF_GetFreeSize(&(hmw->hbram), area);
}

/**
  * @brief Set value address in pointer of pointer in HSP memory mapping.
  * @param hmw     HSP handle
  * @param pop     Pointer of pointer
  * @param addr    Address to set in pointer of pointer
  * @retval        HSP_CORE_OK     Deallocation succeeded.
  *                HSP_CORE_ERROR  Deallocation failed.
  */
hsp_core_status_t HSP_BRAM_SetPopAddress(hsp_core_handle_t *hmw, uint32_t *pop, uint32_t addr)
{
  if (HSP_BRAM_IF_SetPopAddress(&(hmw->hbram), pop, addr) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }
  return HSP_CORE_OK;
}

/**
  * @brief Create FIR filter state in HSP internal memory.
  * @param hhsp        HSP handle.
  * @param nbTaps      Number of filter taps
  * @param nbSamples   Number of float elements to proceed
  * @param area        HSP_BRAM_ALLOCATION_DEFAULT: allocation in all BRAM
  *                    HSP_BRAM_ALLOCATION_PERSISTENT: allocation in persistent memory
  * @retval            If allocation succeed, returns identifier of the created filter state
  *                    otherwise returns 0
  */
hsp_filter_state_identifier_t HSP_BRAM_MallocStateBuffer_Fir(hsp_core_handle_t *hmw, uint32_t nbTaps, 
                                                             uint32_t nbSamples, hsp_bram_allocation_t area)
{
  return(HSP_BRAM_IF_MallocStateBuffer_Fir(&(hmw->hbram), nbTaps, nbSamples, area));
}

/**
  * @brief Create FIR DECIMATE filter state in HSP internal memory.
  * @param hhsp         HSP handle.
  * @param nbTaps       Number of filter taps
  * @param nbSamples    Number of float elements to proceed
  * @param decimFactor  Decimation factor
  * @param area         HSP_BRAM_ALLOCATION_DEFAULT: allocation in all BRAM
  *                     HSP_BRAM_ALLOCATION_PERSISTENT: allocation in persistent memory
  * @retval             If allocation succeed, returns identifier of the created filter state
  *                     otherwise returns 0
  */
hsp_filter_state_identifier_t HSP_BRAM_MallocStateBuffer_FirDecimate(hsp_core_handle_t *hmw, uint32_t nbTaps,
                                                                     uint32_t nbSamples, uint32_t decimFactor, 
                                                                     hsp_bram_allocation_t area)
{
  return(HSP_BRAM_IF_MallocStateBuffer_FirDecimate(&(hmw->hbram), nbTaps, nbSamples, decimFactor, area));
}

/**
  * @brief Create BIQUAD Cascade Df1 filter state in HSP internal memory.
  * @param hhsp           HSP handle.
  * @param nbStages       Number of stage
  * @param area           HSP_BRAM_ALLOCATION_DEFAULT: allocation in all BRAM
  *                       HSP_BRAM_ALLOCATION_PERSISTENT: allocation in persistent memory
  * @retval               If allocation succeed, returns identifier of the created filter state
  *                       otherwise returns 0
  */
hsp_filter_state_identifier_t HSP_BRAM_MallocStateBuffer_BiquadCascadeDf1(hsp_core_handle_t *hmw, uint32_t nbStages,
                                                                          hsp_bram_allocation_t area)
{
  return(HSP_BRAM_IF_MallocStateBuffer_BiquadCascadeDf1(&(hmw->hbram), nbStages, area));
}

/**
  * @brief Create BIQUAD Cascade Df2T filter state in HSP internal memory.
  * @param hhsp           HSP handle.
  * @param nbStages       Number of stage
  * @param area           HSP_BRAM_ALLOCATION_DEFAULT: allocation in all BRAM
  *                       HSP_BRAM_ALLOCATION_PERSISTENT: allocation in persistent memory
  * @retval               If allocation succeed, returns identifier of the created filter state
  *                       otherwise returns 0
  */
hsp_filter_state_identifier_t HSP_BRAM_MallocStateBuffer_BiquadCascadeDf2t(hsp_core_handle_t *hmw, uint32_t nbStages,
                                                                           hsp_bram_allocation_t area)
{
  return(HSP_BRAM_IF_MallocStateBuffer_BiquadCascadeDf2t(&(hmw->hbram), nbStages, area));
}

/**
  * @brief Create LMS filter state in HSP internal memory.
  * @param hhsp           HSP handle.
  * @param nbTaps         Number of filter taps
  * @param area           HSP_BRAM_ALLOCATION_DEFAULT: allocation in all BRAM
  *                       HSP_BRAM_ALLOCATION_PERSISTENT: allocation in persistent memory
  * @retval               If allocation succeed, returns identifier of the created filter state
  *                       otherwise returns 0
  */
hsp_filter_state_identifier_t HSP_BRAM_MallocStateBuffer_Lms(hsp_core_handle_t *hmw, uint32_t nbTaps,
                                                             hsp_bram_allocation_t area)
{
  return(HSP_BRAM_IF_MallocStateBuffer_Lms(&(hmw->hbram), nbTaps, area));
}

/**
  * @brief Create IIR Lattice filter state in HSP internal memory.
  * @param hhsp           HSP handle.
  * @param nbStages       Number of stage
  * @param area           HSP_BRAM_ALLOCATION_DEFAULT: allocation in all BRAM
  *                       HSP_BRAM_ALLOCATION_PERSISTENT: allocation in persistent memory
  * @retval               If allocation succeed, returns identifier of the created filter state
  *                       otherwise returns 0
  */
hsp_filter_state_identifier_t HSP_BRAM_MallocStateBuffer_IirLattice(hsp_core_handle_t *hmw, uint32_t nbStages,
                                                                    hsp_bram_allocation_t area)
{
  return(HSP_BRAM_IF_MallocStateBuffer_IirLattice(&(hmw->hbram), nbStages, area));
}

/**
  * @brief Create IIR Df1 filter state in HSP internal memory.
  * @param hhsp           HSP handle.
  * @param nbStages       Number of stage
  * @param area           HSP_BRAM_ALLOCATION_DEFAULT: allocation in all BRAM
  *                       HSP_BRAM_ALLOCATION_PERSISTENT: allocation in persistent memory
  * @retval               If allocation succeed, returns identifier of the created filter state
  *                       otherwise returns 0
  */
hsp_filter_state_identifier_t HSP_BRAM_MallocStateBuffer_IirDf1(hsp_core_handle_t *hmw, uint32_t nbStages,
                                                                hsp_bram_allocation_t area)
{
  return(HSP_BRAM_IF_MallocStateBuffer_IirDf1(&(hmw->hbram), nbStages, area));
}

/**
  * @brief Create IIR 3p3z filter state in HSP internal memory.
  * @param hhsp           HSP handle.
  * @param area           HSP_BRAM_ALLOCATION_DEFAULT: allocation in all BRAM
  *                       HSP_BRAM_ALLOCATION_PERSISTENT: allocation in persistent memory
  * @retval               If allocation succeed, returns identifier of the created filter state
  *                       otherwise returns 0
  */
hsp_filter_state_identifier_t HSP_BRAM_MallocStateBuffer_Iir3p3z(hsp_core_handle_t *hmw, hsp_bram_allocation_t area)
{
  return(HSP_BRAM_IF_MallocStateBuffer_Iir3p3z(&(hmw->hbram), area));
}

/**
  * @brief Create IIR 2p2z filter state in HSP internal memory.
  * @param hhsp           HSP handle.
  * @param area           HSP_BRAM_ALLOCATION_DEFAULT: allocation in all BRAM
  *                       HSP_BRAM_ALLOCATION_PERSISTENT: allocation in persistent memory
  * @retval               If allocation succeed, returns identifier of the created filter state
  *                       otherwise returns 0
  */
hsp_filter_state_identifier_t HSP_BRAM_MallocStateBuffer_Iir2p2z(hsp_core_handle_t *hmw, hsp_bram_allocation_t area)
{
  return(HSP_BRAM_IF_MallocStateBuffer_Iir2p2z(&(hmw->hbram), area));
}
/**
  * @}
  */

/** @addtogroup HSP_BRAM_Exported_Functions_Group2
  * @{
  */
/* API to BRAM Access control ------------------------------------------------*/
hsp_core_status_t HSP_BRAM_EnableConflictAccessCounter(hsp_core_handle_t *hmw)
{
  if (HSP_HW_IF_BRAM_EnableConflictAccessCounter(hmw->hdriver) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }

  return HSP_CORE_OK;
}

hsp_core_status_t HSP_BRAM_DisableConflictAccessCounter(hsp_core_handle_t *hmw)
{
  if (HSP_HW_IF_BRAM_DisableConflictAccessCounter(hmw->hdriver) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }

  return HSP_CORE_OK;
}

uint32_t HSP_BRAM_GetConflictAccessCounter(hsp_core_handle_t *hmw)
{
  return HSP_HW_IF_BRAM_GetConflictAccessCounter(hmw->hdriver);
}


hsp_core_status_t HSP_BRAM_SetBandwidthArbitration(hsp_core_handle_t *hmw, hsp_bram_arbitration_t mode)
{
  if (HSP_HW_IF_BRAM_SetBandwidthArbitration(hmw->hdriver, (uint32_t) mode) != HSP_IF_OK)
  {
    return HSP_CORE_ERROR;
  }

  return HSP_CORE_OK;
}

hsp_bram_arbitration_t HSP_BRAM_GetBandwidthArbitration(hsp_core_handle_t *hmw)
{
  return (hsp_bram_arbitration_t)HSP_HW_IF_BRAM_GetBandwidthArbitration(hmw->hdriver);
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

/**
  * @}
  */
