/**
  ******************************************************************************
  * @file    hsp_utilities.c
  * @author  GPM Application Team
  * @brief   utilities for HSP 
  *          This file provides a set of functions to manage address for HSP 
  *          programming
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
#include "hsp_hw_if.h"
#include "hsp_bram_if.h"
#include "hsp_conf.h"
#include "hsp_if_conf.h"
#include "hsp_proclist.h"
#include "hsp_fw_def_generic.h"

/** @addtogroup HSP_ENGINE
  * @{
  */

/** @defgroup HSP_INTERFACES HSP Interfaces
  * @{
  */

/** @addtogroup HSP_UTILITIES
  * @{
  */
/* Private variables ---------------------------------------------------------*/
/* Private types -----------------------------------------------------------*/
/* Private constants ---------------------------------------------------------*/
/* Private macros -------------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/
hsp_if_status_t HSP_UTILITIES_ToBramABAddress(hsp_core_handle_t *hmw, uint32_t addr, uint32_t *addrOut);
hsp_if_status_t HSP_UTILITIES_BuildParam(hsp_core_handle_t *hmw,
                                     uint32_t inIoType, uint32_t *ouIoType,
                                     uint32_t inAddr0, uint32_t *ouAddr0,
                                     uint32_t inAddr1, uint32_t *ouAddr1,
                                     uint32_t inAddr2, uint32_t *ouAddr2,
                                     uint32_t nbParam);

/* Private functions ---------------------------------------------------------*/
/** @addtogroup HSP_UTILITIES_Private_Functions
  * @{
  */
/**
  * @brief  Translate host address in HSP address
  * @param addr    Address to be translate
  * @param ouAddr  Translated address
  * @retval        HSP_IF_OK   translation success
  * @retval        HSP_IF_ERROR  translation failure
  */  
  
hsp_if_status_t HSP_UTILITIES_ToBramABAddress(hsp_core_handle_t *hmw, uint32_t addr, uint32_t *ouAddr)
{
  hsp_bram_handle_t *hhsp_bram = &hmw->hbram;


  if ((addr >= hhsp_bram->baseSharedAddr) && (addr <= hhsp_bram->topSharedAddr))
  {
    /* Calculate HSP shared memory address */
    *ouAddr = (uint32_t)((int32_t)hhsp_bram->bramOffset + (int32_t)addr);
    return HSP_IF_OK;
  }
  if (addr == (uint32_t)&(((hal_hsp_handle_t *)(hmw->hdriver))->Instance->BUFFDR[0]))
  {
    *ouAddr = (uint32_t)HSP_REG_SPE_BUFF0DR;
    return HSP_IF_OK;
  }
  if (addr == (uint32_t)&(((hal_hsp_handle_t *)(hmw->hdriver))->Instance->BUFFDR[1]))
  {
    *ouAddr = (uint32_t)HSP_REG_SPE_BUFF1DR;
    return HSP_IF_OK;
  }
  if (addr == (uint32_t)&(((hal_hsp_handle_t *)(hmw->hdriver))->Instance->BUFFDR[2]))
  {
    *ouAddr = (uint32_t)HSP_REG_SPE_BUFF2DR;
    return HSP_IF_OK;
  }
  if (addr == (uint32_t)&(((hal_hsp_handle_t *)(hmw->hdriver))->Instance->BUFFDR[3]))
  {
    *ouAddr = (uint32_t)HSP_REG_SPE_BUFF3DR;
    return HSP_IF_OK;
  }
  return HSP_IF_ERROR;
}

/**
  * @brief  Translate host address in HSP address
  * @param hmw       Core handle
  * @param inIoType  User iotype information
  * @param ouIoType  Internal iotype
  * @param inAddr0   First address to be check and translate
  * @param ouAddr0   Translated address
  * @param inAddr1   Second address to be check and translate
  * @param ouAddr1   Translated address
  * @param inAddr2   Third address to be check and translate
  * @param ouAddr2   Translated address
  * @param nbParam   Number of address to translate (1 or 2 or 3)
  * @retval          HSP_IF_OK   translation success
  * @retval          HSP_IF_ERROR  translation failure
  */ 
hsp_if_status_t HSP_UTILITIES_BuildParam(hsp_core_handle_t *hmw,
                                     uint32_t inIoType, uint32_t *ouIoType,
                                     uint32_t inAddr0, uint32_t *ouAddr0,
                                     uint32_t inAddr1, uint32_t *ouAddr1,
                                     uint32_t inAddr2, uint32_t *ouAddr2,
                                     uint32_t nbParam)
{
  if ((nbParam == 0) || (nbParam > 3))
  {
    return HSP_IF_ERROR;
  }

  *ouIoType = 0;

  /* inAddr0 */
  if ((inIoType & HSP_SEQ_IOTYPE_IMM_0) == HSP_SEQ_IOTYPE_IMM_0)
  {
    *ouIoType |= HSP_IOTYPE_IMM;	  
  }
  else 
  {
	if ((inIoType & HSP_SEQ_IOTYPE_POP_0) == HSP_SEQ_IOTYPE_POP_0)
    {
      *ouIoType |= HSP_IOTYPE_POP0;
    }
    if (HSP_UTILITIES_ToBramABAddress(hmw, inAddr0, ouAddr0) == HSP_IF_ERROR)
    {
     return HSP_IF_ERROR;
    }
  }
  /* inAddr1 */
  if (nbParam >= 2)
  {
    if ((inIoType & HSP_SEQ_IOTYPE_IMM_1) == HSP_SEQ_IOTYPE_IMM_1)
    {
      /* No conversion required */
      *ouIoType |= HSP_IOTYPE_IMM;
    }
    else
    {
      if ((inIoType & HSP_SEQ_IOTYPE_POP_1) == HSP_SEQ_IOTYPE_POP_1)
      {
        *ouIoType |= HSP_IOTYPE_POP1;
      }
      if (HSP_UTILITIES_ToBramABAddress(hmw, inAddr1, ouAddr1) == HSP_IF_ERROR)
      {
       return HSP_IF_ERROR;
      }
    }
  }
  /* inAddr2 */
  if (nbParam >= 3)
  {
    if ((inIoType & HSP_SEQ_IOTYPE_POP_2) == HSP_SEQ_IOTYPE_POP_2)
    {
      *ouIoType |= HSP_IOTYPE_POP2;
    }
    if (HSP_UTILITIES_ToBramABAddress(hmw, inAddr2, ouAddr2) == HSP_IF_ERROR)
    {
     return HSP_IF_ERROR;
    }
  }
  return HSP_IF_OK;
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