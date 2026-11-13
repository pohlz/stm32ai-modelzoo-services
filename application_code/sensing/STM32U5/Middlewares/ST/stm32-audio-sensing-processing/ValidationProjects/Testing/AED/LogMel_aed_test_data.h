#ifndef LFBE_TEST_DATA_H
#define LFBE_TEST_DATA_H

#include <stdint.h>
#include "ai_model_config.h"

extern const uint32_t LogMel_aed_bus1_input[CTRL_X_CUBE_AI_SPECTROGRAM_WINDOW_LENGTH + (CTRL_X_CUBE_AI_SPECTROGRAM_COL - 1) * CTRL_X_CUBE_AI_SPECTROGRAM_HOP_LENGTH * 2/ 4];
extern const uint32_t LogMel_aed_bus1_output[CTRL_X_CUBE_AI_SPECTROGRAM_NMEL * CTRL_X_CUBE_AI_SPECTROGRAM_COL * 2 / 4];
extern const float32_t LogMel_aed_bus1_windata[CTRL_X_CUBE_AI_SPECTROGRAM_COL * CTRL_X_CUBE_AI_SPECTROGRAM_NFFT];
extern const float32_t LogMel_aed_bus1_data_fft[CTRL_X_CUBE_AI_SPECTROGRAM_COL * CTRL_X_CUBE_AI_SPECTROGRAM_NFFT] ;
extern const float32_t LogMel_aed_bus1_data_MagSqrtFFT[CTRL_X_CUBE_AI_SPECTROGRAM_COL * CTRL_X_CUBE_AI_SPECTROGRAM_NFFT/2];

#endif