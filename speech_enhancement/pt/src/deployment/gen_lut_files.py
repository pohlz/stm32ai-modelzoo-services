# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import librosa
import sys
from pathlib import Path
import numpy as np


def _generate_LUTs_header_file(path,
                              window,
                              floatingPoint="flexFloatingPoint"):
    '''
    Generates the header file for the look-up tables used in the Getting Started C application
    Writes directly to path/user_mel_tables.h
    Inputs
    ------
    path : str or Posixpath, path to write the header file to.
    melFilterLut : np.ndarray, array of non-zero mel filter weights
    melFilterStartIndices : np.ndarray, array containing start indexes 
    for non-zero parts of the sparse mel filter weight array
    melFilterStopIndices : np.ndarray, array containing start indexes 
    for non-zero parts of the sparse mel filter weight array
    window : np.ndarray, array containing hanning window weights.
    floatingPoint : tells wether the floating point can be configured by app

    Outputs
    -------
    None
    '''

    # Write .h file
    with open(Path(path, "user_mel_tables.h"), "wt") as f:
        f.write('/**\n')
        f.write('******************************************************************************\n')
        f.write('* @file    user_mel_tables.h\n')
        f.write('* @author  MCD Application Team\n')
        f.write('* @brief   Header for mel_user_tables.c module\n')
        f.write('******************************************************************************\n')
        f.write('* @attention\n')
        f.write('*\n')
        f.write('* Copyright (c) 2024 STMicroelectronics.\n')
        f.write('* All rights reserved.\n')
        f.write('*\n')
        f.write('* This software is licensed under terms that can be found in the LICENSE file\n')
        f.write('* in the root directory of this software component.\n')
        f.write('* If no LICENSE file comes with this software, it is provided AS-IS.\n')
        f.write('*\n')
        f.write('******************************************************************************\n')
        f.write('*/\n')
        f.write('#ifndef _MEL_USER_TABLES_H\n')
        f.write('#define _MEL_USER_TABLES_H\n')
        f.write('#include "arm_math.h"\n')
        if floatingPoint == "default" :
            f.write('extern const float32_t user_win[{}];\n'.format(len(window)))
        elif floatingPoint == "flexFloatingPoint":
            f.write('#include "preproc_dpu.h"\n')
            f.write('extern const PREPROC_FLOAT_T user_win[{}];\n'.format(len(window)))
        f.write('#endif /* _MEL_USER_TABLES_H */\n')


def _generate_LUTs_c_file(path,
                          window,
                          floatingPoint="flexFloatingPoint"):
    '''
    Generates the C code file for the look-up tables used in the Getting Started C application
    Writes directly to path/user_mel_tables.c
    Inputs
    ------
    path : str or Posixpath, path to write the header file to.
    window : np.ndarray, array containing hanning window weights.
    floatingPoint : tells wether the floating point can be configured by app
    
    Outputs
    -------
    None
    '''
    # path = os.path.join(HydraConfig.get().runtime.output_dir, "C_header/")
    # Convert LUTs to str to be able to write to file

    window_str = np.array2string(window,
                                  separator='F ,',
                                  formatter={'float': lambda x : np.format_float_scientific(x,
                                             precision=10, unique=False)},
                                  threshold=sys.maxsize)
    

    window_str = '{' +window_str[1:-1] + 'F}'


    # Write file

    with open(Path(path, "user_mel_tables.c"), "wt") as f:
        f.write('/**\n')
        f.write('******************************************************************************\n')
        f.write('* @file    user_mel_tables.c\n')
        f.write('* @author  MCD Application Team\n')
        f.write('* @brief   This file has lookup tables for user-defined windows and mel filter banks\n')
        f.write('******************************************************************************\n')
        f.write('* @attention\n')
        f.write('*\n')
        f.write('* Copyright (c) 2024 STMicroelectronics.\n')
        f.write('* All rights reserved.\n')
        f.write('*\n')
        f.write('* This software is licensed under terms that can be found in the LICENSE file\n')
        f.write('* in the root directory of this software component.\n')
        f.write('* If no LICENSE file comes with this software, it is provided AS-IS.\n')
        f.write('*\n')
        f.write('******************************************************************************\n')
        f.write('*/\n')
        f.write('\n')
        f.write('#include "user_mel_tables.h"\n')
        f.write('\n')
        if floatingPoint =="default" :
            f.write('const float32_t user_win[{}] = {};\n'.format(
                len(window),window_str))
        elif floatingPoint == "flexFloatingPoint":
            f.write('const PREPROC_FLOAT_T user_win[{}] = {};\n'.format(
                len(window),window_str))
        f.write('\n')

def generate_LUT_files(config):
    '''
    Wrapper function to compute LUTs and write the appropriate files
    
    Inputs
    ------
    config : dict, configuration dictionary

    Outputs
    -------
    None
    '''
        
    header_path = Path(config.output_dir, "C_header/")
    header_path.mkdir(exist_ok=True)

    window = librosa.filters.get_window(window=config.preprocessing.window, Nx=config.preprocessing.win_length)
    print("[INFO] : Generating LUT header file")
    if config.deployment.hardware_setup.serie == "STM32N6":
        _generate_LUTs_header_file(header_path,
                                   window,
                                   floatingPoint="flexFloatingPoint")
    else:
        _generate_LUTs_header_file(header_path,
                                   window,
                                   floatingPoint="default")
        
    print("[INFO] : Done generating LUT header file")
    print("[INFO] : Generating LUT C file")

    if config.deployment.hardware_setup.serie == "STM32N6":
        _generate_LUTs_c_file(header_path,
                              window,
                              "flexFloatingPoint")
    else:
        _generate_LUTs_c_file(header_path,
                              window,
                              floatingPoint="default")
        
    print('[INFO] : Done generating LUT C file')
