###################################################################################
#   Copyright (c) 2021 STMicroelectronics.
#   All rights reserved.
#   This software is licensed under terms that can be found in the LICENSE file in
#   the root directory of this software component.
#   If no LICENSE file comes with this software, it is provided AS-IS.
###################################################################################
"""
STM AI runner - Common helper services to manage the STM.AI type
"""

def stellar_id_to_str(dev_id):
    """Helper function to return a Human readable device ID description"""  # noqa: DAR101,DAR201,DAR401

    switcher = {  # (see BZ119679)
        0x2511: 'SR5E1',
        0x2643: 'SR6P3',
        0x2646: 'SR6P6',
        0x2647: 'SR6P7',
        0x2633: 'SR6G3',
        0x2636: 'SR6G6',
        0x2637: 'SR6G7',
        0x2A47: 'SR6P7G7',
        0x2663: 'SR6P3E',
    }
    desc_ = f'0x{dev_id:X} - '
    desc_ += switcher.get(dev_id, 'UNKNOW')
    return desc_
