# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

"""File for the user to write their custom model"""
import torch.nn as nn


class Custom(nn.Module):
    '''Write your own model class here'''
    def __init__(self):
        pass

    def forward(self, x):
        pass