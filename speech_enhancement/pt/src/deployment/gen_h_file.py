# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

from pathlib import Path

def _is_power_of_two(n):
    return (n != 0) and (n & (n - 1) == 0)

def gen_h_user_file(config):
    '''
    Generates the ai_model_config.h header file for the Getting Started C application,
    and performs a few checks for incorrect parameter values.
    Writes directly to output_dir/C_header/ai_model_config.h
    
    Inputs
    ------
    config : dict, configuration dictionary

    Outputs
    -------
    None
    '''
    # Bunch of checks to see if parameters are valid
    assert config.preprocessing.sample_rate in [16000, 48000], (
        "Target rate should match one of the available ODR rates of digital microphone, e.g. 16000 or 48000")
    assert _is_power_of_two(config.preprocessing.n_fft), (
        "n_fft must be power of 2 due to ARM CMSIS-DSP implementation constraints"
    )
    assert config.preprocessing.win_length <= config.preprocessing.n_fft, (
        "Window length must be lower or equal to n_fft"
    )

    header_path = Path(config.output_dir, "C_header/")
    header_path.mkdir(exist_ok=True)
    preproc_args = config.preprocessing
    deploy_args = config.deployment
    with open("pt/src/deployment/template_ai_model_config.h", "r") as h:
        template_lines = h.readlines()
        with open(Path(header_path, "ai_model_config.h"), "wt") as f:
            # Copy header from template to output h file
            # Header encompasses the first 29 lines
            f.writelines(template_lines[:29])
        
            # Parameters, we could just copy some over from the template

            f.write("#define CTRL_AI_HW_SELECT                        STM32N6570-DK\n")
            f.write('#define CTRL_X_CUBE_AI_MODEL_NAME                "X-CUBE-AI STFT TCN"\n')
            f.write('#define CTRL_X_CUBE_AI_MODE_NB_OUTPUT            (1U)\n')
            f.write('#define CTRL_X_CUBE_AI_MODE_OUTPUT_1             (CTRL_AI_SPECTROGRAM)\n')
            f.write('#define CTRL_X_CUBE_AI_SENSOR_TYPE               (COM_TYPE_MIC)\n')
            f.write('#define CTRL_X_CUBE_AI_AUDIO_OUT                 (COM_TYPE_HEADSET)\n')
            f.write('#define CTRL_X_CUBE_AI_SENSOR_NAME               "imp34dt05"\n')
            f.write(f'#define CTRL_X_CUBE_AI_SENSOR_ODR                ({preproc_args.sample_rate}.0F)\n')
            f.write('#define CTRL_X_CUBE_AI_PREPROC                   (CTRL_AI_STFT)\n')
            f.write('#define CTRL_X_CUBE_AI_POSTPROC                  (CTRL_AI_ISTFT)\n')
            f.write(f'#define CTRL_X_CUBE_AI_SPECTROGRAM_HOP_LENGTH    ({preproc_args.hop_length}U)\n')
            f.write(f'#define CTRL_X_CUBE_AI_SPECTROGRAM_NFFT          ({preproc_args.n_fft}U) /* power of 2 closest to CTRL_X_CUBE_AI_WINDOW_LENGTH */ \n')
            f.write(f'#define CTRL_X_CUBE_AI_SPECTROGRAM_WINDOW_LENGTH ({preproc_args.win_length}U) \n')
            f.write(f'#define CTRL_X_CUBE_AI_SPECTROGRAM_COL_NO_OVL    ({deploy_args.frames_per_patch}U) \n')
            f.write(f'#define CTRL_X_CUBE_AI_SPECTROGRAM_COL_OVL       ({deploy_args.lookahead_frames}U) \n')
            f.write(f'#define CTRL_X_CUBE_AI_AUDIO_OUT_DB_THRESHOLD    ({deploy_args.output_silence_threshold}.0F) \n')
            f.write(f'#define CTRL_X_CUBE_AI_OOD_THR (0.0F) \n') # Not used for SE. Only for compatibility with the C application
            # Write the last few lines from the template
            f.writelines(template_lines[45:])

