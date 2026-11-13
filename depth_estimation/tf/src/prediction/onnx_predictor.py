# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
import numpy as np
from common.utils import ai_runner_interp, ai_interp_input_quant, ai_interp_outputs_dequant
from common.evaluation import predict_onnx
from depth_estimation.tf.src.utils import ai_runner_invoke
from depth_estimation.tf.src.preprocessing.preprocess import preprocess_input, postprocess_output_values
from depth_estimation.tf.src.prediction.utils import generate_output_image
import onnxruntime


class ONNXModelPredictor:
    def __init__(self, cfg, model, dataloaders):
        self.cfg = cfg
        self.model = model
        self.predict_ds = dataloaders['predict']
        self.target = cfg.prediction.target if cfg.prediction and cfg.prediction.target else "host"
        self.name_model = cfg.model.model_path
        self.model_type = cfg.model.model_type
        # Initialize ONNX runtime session and AI runner interpreter
        self.sess = onnxruntime.InferenceSession(model.model_path)
        self.ai_runner_interpreter = ai_runner_interp(self.target, self.name_model)

    def predict(self):
        for img, image_path in self.predict_ds:
            if self.target == "host":
                img = preprocess_input(img, input_details=None)
                img = img.numpy()
#                raw_prediction = self.model.run(None, {self.model.get_inputs()[0].name: img})[0]
                raw_prediction = predict_onnx(self.sess, img)   # Use ONNX runtime for inference
                output = postprocess_output_values(raw_prediction)
                output = output.transpose(0, 2, 3, 1)
            elif self.target in ['stedgeai_host', 'stedgeai_n6', 'stedgeai_h7p']:
                if self.cfg.prediction.input_chpos == "chfirst":
                    img = np.transpose(img, [2, 0, 1])
                data = ai_interp_input_quant(self.ai_runner_interpreter, img[None], '.onnx')
                output = ai_runner_invoke(data, self.ai_runner_interpreter)
                output = ai_interp_outputs_dequant(self.ai_runner_interpreter, output)[0]
            else:
                raise TypeError("Unknown or unsupported target for ONNX prediction.")
            output = output[0, :, :, 0]
            if self.model_type == "fast_depth":
                max_val = np.max(output)
                output = max_val - output
            generate_output_image(
                image_path=image_path,
                output=output,
                cfg=self.cfg,
                input_size=[self.model.get_inputs()[0].shape[2], self.model.get_inputs()[0].shape[3]],
                output_details=None
            )
        print("[INFO] : prediction complete")
