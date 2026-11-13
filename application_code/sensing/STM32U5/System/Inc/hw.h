/**
  ******************************************************************************
  * @file           : hw.h
  * @brief          : system hw related functions
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

/* Exported variables ---------------------------------------------------------*/
#define TIMER_PRESCALER (1600-1)

/* Private function prototypes -----------------------------------------------*/
extern void SystemClock_Config( void );
extern void SystemPower_Config( void );
extern void hw_tim5_init( void ) ;
extern void hw_gpio_init( void );
extern void hw_icache_init( void );
extern void hw_usart1_init( void );
extern uint32_t hw_timer_get_count( void );
