#!/bin/bash

stedgeai generate -m yamnet_e256_64x96_tl_int8.tflite --target stm32u5
cp st_ai_output/network.h .
cp st_ai_output/network_data.h .
cp st_ai_output/network_details.h .
cp st_ai_output/network.c .
cp st_ai_output/network_data.c .
cp st_ai_output/network_c_info.json .
cp st_ai_output/network_generate_report.txt .