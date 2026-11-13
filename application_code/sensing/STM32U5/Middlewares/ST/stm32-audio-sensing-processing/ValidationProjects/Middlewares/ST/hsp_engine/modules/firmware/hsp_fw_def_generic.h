/**
  ******************************************************************************
  * @file    hsp_fw_def_generic.h
  * @author  MCD Application Team
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
#ifndef HSP_FW_DEF_GENERIC_H
#define HSP_FW_DEF_GENERIC_H

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

#include "hsp_fw_version_conf.h"

#define HSP_FW_CROM_VERSION_1_1_7  (HSP_CROM_VERSION(1U, 1U, 7U))
#define HSP_FW_CRAM_VERSION_1_4    (HSP_CRAM_VERSION(1U, 4U))
#define HSP_FW_CRAM_VERSION_1_6    (HSP_CRAM_VERSION(1U, 6U))
#define HSP_FW_CRAM_VERSION_DEV    (HSP_CRAM_VERSION(0xFU, 0xFFU))


#if (HSP_DEVICE_CROM_VERSION == HSP_FW_CROM_VERSION_1_1_7)
#if (HSP_FW_CRAM_TO_LOAD_VERSION == HSP_FW_CRAM_VERSION_1_4)
#include "crom_1.1.7/cram_1.4/hsp_fw_def.h"
#elif (HSP_FW_CRAM_TO_LOAD_VERSION == HSP_FW_CRAM_VERSION_1_6)
#include "crom_1.1.7/cram_1.6/hsp_fw_def.h"
#elif (HSP_FW_CRAM_TO_LOAD_VERSION == HSP_FW_CRAM_VERSION_DEV)
#include "crom_1.1.7/cram_dev/hsp_fw_def.h"
#else
#error "CRAM FW not supported for dedicated CROM version"
#endif /* HSP_FW_CRAM_TO_LOAD_VERSION...*/
#else
#error "CROM version unknown"
#endif /* HSP_DEVICE_CROM_VERSION */


#if defined(__cplusplus)
}

#endif /* __cplusplus */

#endif /* HSP_FW_DEF_GENERIC_H */
