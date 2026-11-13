# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
import os
CHECKPOINT_STORAGE_URL = "https://raw.githubusercontent.com/STMicroelectronics/stm32ai-modelzoo/main/image_classification/torch_checkpoints/"
CURRENT_REPO_PATH = os.path.dirname(os.path.abspath(__file__))
SERVICES_ROOT = os.path.abspath(os.path.join(CURRENT_REPO_PATH, '../../../..'))
#CHECKPOINT_STORAGE_URL = os.path.join(os.path.dirname(SERVICES_ROOT), "stm32ai-modelzoo/image_classification/torch_checkpoints/")

MODEL_CHECKPOINTS = {
    "st_resnettiny_actrelu_pt_datasetimagenet_res224" : "st_resnettiny_actrelu_pt_224.pth.tar", 
    "st_resnetmilli_actrelu_pt_datasetimagenet_res224" : "st_resnetmilli_actrelu_pt_224.pth.tar", 
    "st_resnetmicro_actrelu_pt_datasetimagenet_res224" : "st_resnetmicro_actrelu_pt_224.pth.tar", 
    "st_resnetnano_actrelu_pt_datasetimagenet_res224" : "st_resnetnano_actrelu_pt_224.pth.tar", 
    "st_resnetpico_actrelu_pt_datasetimagenet_res224" : "st_resnetpico_actrelu_pt_224.pth.tar",
    "mobilenetv2_w035_pt_datasetimagenet_res224": "mobilenetv2_w035_pt_224.pt",      
}