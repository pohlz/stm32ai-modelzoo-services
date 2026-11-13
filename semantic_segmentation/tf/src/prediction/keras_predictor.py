import os
import tensorflow as tf
import numpy as np
from pathlib import Path
from omegaconf import DictConfig
from semantic_segmentation.tf.src.utils import vis_segmentation

def _generate_output_image(image_path, raw_output, input_size, nchw=False, cfg=None):
    if nchw and raw_output.ndim == 4:
        raw_output = raw_output.transpose(0, 2, 3, 1)
    seg_map = tf.argmax(tf.image.resize(raw_output, size=input_size), axis=3)
    seg_map = tf.squeeze(seg_map).numpy().astype(np.int8)
    vis_segmentation(image_path=image_path, seg_map=seg_map, cfg=cfg, input_size=input_size)

class KerasModelPredictor:
    def __init__(self, cfg: DictConfig, model: tf.keras.Model, dataloaders: list):
        self.cfg = cfg
        self.model = model
        self.predict_ds = dataloaders['predict']  # (image, path) batched
        if self.predict_ds is None:
            raise ValueError("Prediction dataset is None")
        in_shape = tuple(model.inputs[0].shape)
        if in_shape[0] is None:
            in_shape = in_shape[1:]
        self.height, self.width = int(in_shape[0]), int(in_shape[1])

    def predict(self):
        print("[INFO] : Keras segmentation prediction started")
        for sample in self.predict_ds:
            # sample: (image_batch, path_batch) with batch_size=1
            if isinstance(sample, (tuple, list)) and len(sample) == 2:
                img = sample[0]           # (1,H,W,C) or (H,W,C)
                path_t = sample[1]        # (1,)
                raw = path_t.numpy()[0]
                image_path = raw.decode() if isinstance(raw, bytes) else str(raw)
            else:
                img = sample[0] if isinstance(sample, (tuple, list)) else sample
                image_path = None

            if len(img.shape) == 3:
                img = tf.expand_dims(img, 0)

            preds = self.model.predict(img, verbose=0)  # (1,h,w,classes)
            if preds.shape[1] != self.height or preds.shape[2] != self.width:
                preds = tf.image.resize(preds, (self.height, self.width)).numpy()

            _generate_output_image(image_path=image_path if image_path and os.path.isfile(image_path) else None,
                                   raw_output=preds,
                                   input_size=[self.height, self.width],
                                   nchw=False,
                                   cfg=self.cfg)
        print("[INFO] : prediction complete")