#  /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

def ai_runner_invoke(image_processed,ai_runner_interpreter):
    def reduce_shape(x):  # reduce shape (request by legacy API)
        old_shape = x.shape
        n_shape = [old_shape[0]]
        for v in x.shape[1:len(x.shape) - 1]:
            if v != 1:
                n_shape.append(v)
        n_shape.append(old_shape[-1])
        return x.reshape(n_shape)

    preds, _ = ai_runner_interpreter.invoke(image_processed)
    predictions = []
    for x in preds:
        x = reduce_shape(x)
        predictions.append(x.copy())
    return predictions


