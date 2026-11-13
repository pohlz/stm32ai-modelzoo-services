# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
import numpy as np
from common.utils import ai_runner_interp, ai_interp_input_quant, ai_interp_outputs_dequant
from depth_estimation.tf.src.utils import ai_runner_invoke
from depth_estimation.tf.src.preprocessing.preprocess import preprocess_input, postprocess_output_values
from depth_estimation.tf.src.prediction.utils import generate_output_image


class TFLiteQuantizedModelPredictor:
    def __init__(self, cfg, model, dataloaders):
        self.cfg = cfg
        self.model = model
        self.predict_ds = dataloaders['predict']
        self.target = cfg.prediction.target if cfg.prediction and cfg.prediction.target else "host"
        self.name_model = cfg.model.model_path
        self.model_type = cfg.model.model_type
        # Initialize the AI runner interpreter for edge devices
        self.ai_runner_interpreter = ai_runner_interp(self.target, self.name_model)

    def predict(self):
        input_details = self.model.get_input_details()[0]
        input_index_quant = input_details["index"]
        output_details = self.model.get_output_details()[0]
        output_index_quant = output_details["index"]
        if self.predict_ds is None:
            raise ValueError(
                f"Dataloader returned None. Check that the prediction path is valid and contains images. "
                f"Config prediction_path: {getattr(self.cfg.dataset, 'prediction_path', None)}"
            )
        for img, image_path in self.predict_ds:
            if self.target == "host":
                img = preprocess_input(img, input_details=input_details)
                self.model.set_tensor(input_index_quant, img)
                self.model.invoke()
                raw_prediction = self.model.get_tensor(output_index_quant)
                output = postprocess_output_values(raw_prediction, output_details)
            elif self.target in ['stedgeai_host', 'stedgeai_n6', 'stedgeai_h7p']:
                data = ai_interp_input_quant(self.ai_runner_interpreter, img.numpy()[None], '.tflite')
                output = ai_runner_invoke(data, self.ai_runner_interpreter)
                output = ai_interp_outputs_dequant(self.ai_runner_interpreter, output)[0]

            else:
                raise TypeError("Unknown or unsupported target for TFLite prediction.")
            output = np.squeeze(output)
            if self.model_type == "fast_depth":
                max_val = np.max(output)
                output = max_val - output
            generate_output_image(
                image_path=image_path,
                output=output,
                cfg=self.cfg,
                input_size=[self.model.get_input_details()[0]['shape'][1], self.model.get_input_details()[0]['shape'][2]],
                output_details=output_details
            )
        print("[INFO] : prediction complete")
