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
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from omegaconf import DictConfig
import tensorflow as tf
from .base import BaseAEDPredictor

class AEDKerasPredictor(BaseAEDPredictor):
    '''Predictor class for AED Keras float models'''
    def __init__(self, cfg: DictConfig = None, model: tf.keras.Model = None, dataloaders: dict = None):
        """
        Initialize Keras AED predictor.

        Parameters
        ----------
        cfg : DictConfig
            User configuration.
        model : tf.keras.Model
            Trained Keras model used for prediction.
        dataloaders : dict
            Contains `pred_ds`, `pred_clip_labels`, `pred_filenames`.
        """
        super().__init__(cfg=cfg,
                         model=model,
                         dataloaders=dataloaders)
        
    def _get_preds(self):
        """
        Generate patch-level predictions using the Keras model.

        Returns
        -------
        np.ndarray
            Patch-level prediction scores.
        """
        preds = self.model.predict(self.pred_ds)
        return preds