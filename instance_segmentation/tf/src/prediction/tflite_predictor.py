# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
import os
import cv2
import numpy as np
import tensorflow as tf
from instance_segmentation.tf.src.postprocessing.postprocess import postprocess
from instance_segmentation.tf.src.preprocessing import preprocess_input, read_image

class TFLiteQuantizedModelPredictor:
    """
    Instance segmentation TFLite predictor class, matching image classification structure.
    Handles image loading, preprocessing, inference, and output creation.
    """
    def __init__(self, cfg, model, dataloaders):
        self.cfg = cfg
        self.model = model
        self.predict_ds = dataloaders['predict']
        self.prediction_result_dir = os.path.join(cfg.output_dir, 'predictions')
        self.channels = 1 if self.cfg.preprocessing.color_mode == "grayscale" else 3
        self.display_figures = getattr(cfg.general, 'display_figures', False)
        self.input_details = self.model.get_input_details()[0]
        self.input_index = self.input_details["index"]
        self.output_details = self.model.get_output_details()[0]
        self.output_index = self.output_details["index"]
        self.n_masks = 32
        self.iou_threshold = getattr(cfg.postprocessing, 'IoU_eval_thresh', 0.45)
        self.conf_threshold = getattr(cfg.postprocessing, 'confidence_thresh', 0.25)
        self.class_names = getattr(self.cfg.postprocessing, 'class_names', None)
        self.output_index_quant = self.model.get_output_details()
        self.input_details = self.model.get_input_details()[0]
        self.height, self.width = self.input_details['shape'][1], self.input_details['shape'][2]


    def predict(self):
        for img, img_path in self.predict_ds:
            img = preprocess_input(img, self.input_details)
            self.model.set_tensor(self.input_index, img)
            self.model.invoke()
            detections = self.model.get_tensor(self.output_index_quant[0]["index"])
            masks = self.model.get_tensor(self.output_index_quant[1]["index"])
            img = read_image(img_path, channels=self.channels)
            postprocess(img, masks, detections, self.output_index_quant, self.conf_threshold, self.iou_threshold, self.n_masks,
                        self.width, self.height, self.prediction_result_dir, img_path, self.class_names)
        print('[INFO] : Prediction complete.')
