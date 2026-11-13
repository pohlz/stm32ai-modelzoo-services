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
import numpy as np
import onnxruntime
from omegaconf import DictConfig
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from .base import BaseAEDPredictor
from common.utils import ai_runner_interp
from audio_event_detection.tf.src.preprocessing import preprocess_input
from common.evaluation import predict_onnx

class AEDONNXPredictor(BaseAEDPredictor):
    """
    Predictor for ONNX models (float or quantized).
    """
    def __init__(self, cfg: DictConfig = None, model: onnxruntime.InferenceSession = None, dataloaders: dict = None):
        """
        Initialize ONNX predictor.

        Parameters
        ----------
        cfg : DictConfig
            User configuration.
        model : onnxruntime.InferenceSession
            ONNX runtime session used for prediction.
        dataloaders : dict
            Contains `pred_ds`, `pred_clip_labels`, `pred_filenames`.
        """

        super().__init__(cfg=cfg,
                         model=model,
                         dataloaders=dataloaders)

        self.target = getattr(cfg.prediction, 'target', 'host') if hasattr(cfg, 'prediction') else 'host'
        self.model_name = os.path.basename(model.model_path)
        
        if self.target in ['stedgeai_host', 'stedgeai_n6', 'stedgeai_h7p']:
            self.ai_runner_interpreter = ai_runner_interp(self.target, self.model_name)

    def _get_preds_on_host(self):
        """
        Run ONNX inference on host.

        Returns
        -------
        np.ndarray
            Patch-level prediction scores.
        """

        # Same as with the other predictors, we assume
        # pred_ds is a np array already
        # and that model is already an onnxruntime InferenceSession
        preds =  predict_onnx(self.model, self.pred_ds)
        return preds
    
    def _get_preds_on_target(self):
        """
        Run ONNX inference on target via AI Runner.

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
