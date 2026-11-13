/**
  ******************************************************************************
  * @file    stm32u5xx_it.c
  * @brief   Interrupt Service Routines.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2023 STMicroelectronics.
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
#include "stm32u5xx_it.h"
#include "b_u585i_iot02a_audio.h"
#include "b_u585i_iot02a_motion_sensors.h"

/******************************************************************************/
/*           Cortex Processor Interruption and Exception Handlers          */
/******************************************************************************/
/**
  * @brief This function handles Non maskable interrupt.
  */
void NMI_Handler(void)
{
  while (1)
  {
  }
}

/**
  * @brief This function handles Hard fault interrupt.
  */
void HardFault_Handler(void)
{
  while (1)
  {
  }
}

/**
  * @brief This function handles Memory management fault.
  */
void MemManage_Handler(void)
{
  while (1)
  {
  }
}

/**
  * @brief This function handles Prefetch fault, memory access fault.
  */
void BusFault_Handler(void)
{
  while (1)
  {
  }
}

/**
  * @brief This function handles Undefined instruction or illegal state.
  */
void UsageFault_Handler(void)
{
  while (1)
  {
  }
}
#ifdef APP_BARE_METAL
/**
  * @brief This function handles System service call via SWI instruction.
  */
void SVC_Handler(void)
{
}
#endif
/**
  * @brief This function handles Debug monitor.
  */
void DebugMon_Handler(void)
{
}

/**
  * @brief This function handles System tick timer.
  */
void _SysTick_Handler(void)
{
  /* USER CODE BEGIN SysTick_IRQn 0 */
  /* Clear overflow flag */
  SysTick->CTRL;
  HAL_IncTick();
}

/**
  * @brief This function handles GPDMA1 Channel 6 global interrupt.
  */
void GPDMA1_Channel6_IRQHandler(void)
{
  BSP_AUDIO_IN_IRQHandler(0, AUDIO_IN_DEVICE_DIGITAL_MIC1);
  return;
}

/**
  * @brief This function handles USART1 global interrupt.
  */
void USART1_IRQHandler(void)
{
  HAL_UART_IRQHandler(&UartHandle);
}

/**
  * @brief This function handles external interrupt (11).
  */
void EXTI11_IRQHandler( void )
{
	HAL_GPIO_EXTI_IRQHandler( GPIO_PIN_11 );
	ISM330DHCX_EXTI_Callback( GPIO_PIN_11 );
}
