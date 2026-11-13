# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
import warnings
import tensorflow as tf
import numpy as np
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from .base import BaseAEDPredictor
from common.utils import ai_runner_interp
from audio_event_detection.tf.src.preprocessing import preprocess_input

class AEDTFLitePredictor(BaseAEDPredictor):
    '''Predictor class for AED TFLite quantized models'''

    def __init__(self, cfg, model, dataloaders):
        """
        Initialize TFLite predictor.

        Parameters
        ----------
        cfg : DictConfig
            User configuration.
        model : tf.lite.Interpreter or tf.keras.Model
            TFLite interpreter/model used for prediction.
        dataloaders : dict
            Contains `pred_ds`, `pred_clip_labels`, `pred_filenames`.
        """
        super().__init__(cfg=cfg,
                         model=model,
                         dataloaders=dataloaders)

        self.target = getattr(cfg.prediction, 'target', 'host') if hasattr(cfg, 'prediction') else 'host'
        self.model_name = os.path.basename(model.model_path)

        # For now pred_ds is assumed to be a np.ndarray but
        # handle the case where it is a tf dataset anyways
        try:
            self.input_shape = self.pred_ds.shape
        except:
            self.input_shape = self.pred_ds.element_spec.shape
        
        if self.target in ['stedgeai_host', 'stedgeai_n6', 'stedgeai_h7p']:
            self.ai_runner_interpreter = ai_runner_interp(self.target, self.model_name)


    def _get_preds(self):
        """
        Dispatch prediction to host or target backend.

        Returns
        -------
        np.ndarray
            Patch-level prediction scores.
        """
        if self.target in ['stedgeai_host', 'stedgeai_n6', 'stedgeai_h7p']:
            return self._get_preds_on_target()
        elif self.target == "host":
            return self._get_preds_on_host()
        else:
            raise ValueError(f"Target must be one of 'stedgeai_host', 'stedgeai_n6', 'host', but was {self.target}")
    
    def _get_preds_on_host(self):
        """
        Run inference on host using the TFLite interpreter.

        Returns
        -------
        np.ndarray
            Patch-level prediction scores.
        """

        # Load the Tflite model and allocate tensors
        interpreter_quant = tf.lite.Interpreter(model_path=self.model.model_path,
                                                num_threads=getattr(self.cfg.general, 'num_threads_tflite', 1))
        input_details = interpreter_quant.get_input_details()[0]
        input_index_quant = input_details["index"]
        output_index_quant = interpreter_quant.get_output_details()[0]["index"]

        interpreter_quant.resize_tensor_input(input_index_quant, self.input_shape)
        interpreter_quant.allocate_tensors()

        patches_processed = preprocess_input(self.pred_ds, input_details)
        interpreter_quant.set_tensor(input_index_quant, patches_processed)
        interpreter_quant.invoke()
        preds = interpreter_quant.get_tensor(output_index_quant)
        return preds

    def _get_preds_on_target(self):
        """
        Run inference on target via AI Runner.

        Returns
        -------
        np.ndarray
            Patch-level prediction scores.
        """

        ai_runner_input_details = self.ai_runner_interpreter.get_inputs()
        input_details = {}
        input_details["dtype"] = ai_runner_input_details[0].dtype
        input_details["quantization"] = [ai_runner_input_details[0].scale, ai_runner_input_details[0].zero_point]
        
        # Again, we assume here pred_ds is a np.ndarray, change things here
        # If we want to support pred_ds being an actual tf dataset
        X = preprocess_input(self.pred_ds, input_details)

        preds, _ = self.ai_runner_interpreter.invoke(X)
        preds = preds[0]
        # Yes it HAS to be a tuple
        dims_to_squeeze = tuple(np.arange(1, preds.ndim - 1))

        preds = np.squeeze(preds, axis=dims_to_squeeze)

        return preds
