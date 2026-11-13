# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def ai_runner_invoke(image_processed,ai_runner_interpreter):
    preds, _ = ai_runner_interpreter.invoke(image_processed)
    return preds
        