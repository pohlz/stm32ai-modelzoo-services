/* USER CODE BEGIN Header */
/**
******************************************************************************
* @file            : hsp_engine.c
* @version         : v1_0_Cube
* @brief           : This file implements the HSP_Engine
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
/* USER CODE END Header */

#include "main.h"
#include "hsp_engine.h"
#include "hsp_def.h"
#include "hsp_core.h"
#include "hsp_bram.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* HSP Middleware handle */
/* HSP core handle declaration */
hsp_core_handle_t hmw;

/* Private function prototypes -----------------------------------------------*/

/* Application functions prototypes *******/
extern void Error_Handler(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
 * @brief Middleware HSP Initialization Function
 * @param None
 * @retval None
 */
void MX_HSP_Engine_Init(void)
{
  /* USER CODE BEGIN HSP_Engine_Init_PreTreatment */

  /* USER CODE END HSP_Engine_Init_PreTreatment */

  if (HSP_Engine_IF_Init(&hmw) != HSP_CORE_OK)
  {
    Error_Handler();
  }
  if (HSP_CORE_Init(&hmw) != HSP_CORE_OK)
  {
    Error_Handler();
  }

  /* Configure the BRAM Access Arbitration */
  if (HSP_BRAM_SetBandwidthArbitration(&hmw, HSP_BRAM_ARBITRATION_LATENCY_8_CYCLES) != HSP_CORE_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN HSP_Engine_Init_PostTreatment */

  /* USER CODE END HSP_Engine_Init_PostTreatment */
}
