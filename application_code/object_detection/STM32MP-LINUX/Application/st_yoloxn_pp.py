#!/usr/bin/python3
#
# Copyright (c) 2026 STMicroelectronics.
# All rights reserved.
#
# This software is licensed under terms that can be found in the LICENSE file
# in the root directory of this software component.
# If no LICENSE file comes with this software, it is provided AS-IS.

from stai_mpu import stai_mpu_network
import numpy as np
from timeit import default_timer as timer
import math


class NeuralNetwork:
    """
    Neural Network inference wrapper for st_yoloxn.

    Public API:
      - __init__(model_file, label_file, input_mean, input_std,
                 confidence_thresh, iou_threshold)
      - get_img_size()
      - launch_inference(img)
      - get_results()
      - get_label(idx, classes)

    Outputs:
      locations: shape (1, N, 4)  with [x_min, y_min, x_max, y_max] in [0,1]
      classes:   shape (1, N)
      scores:    shape (1, N)
    """

    def __init__(self, model_file, label_file, input_mean, input_std,
                 confidence_thresh, iou_threshold):

        def load_labels(filename):
            labels = []
            with open(filename, "r") as f:
                for l in f:
                    labels.append(l.strip())
            return labels

        self._model_file = model_file
        print("NN model used : ", self._model_file)

        # For compatibility with object_detection.py
        self.model_type = "st_yoloxn"

        self._label_file = label_file
        self._input_mean = input_mean
        self._input_std = input_std
        self.confidence_threshold = confidence_thresh
        self.iou_threshold = iou_threshold
        self.number_of_boxes = 0

        # Initialize NN model
        if ".nb" in self._model_file:
            self.stai_mpu_model = stai_mpu_network(
                model_path=self._model_file,
                use_hw_acceleration=True
            )
        else:
            self.stai_mpu_model = stai_mpu_network(
                model_path=self._model_file,
                use_hw_acceleration=False
            )

        # Input info
        self.num_inputs = self.stai_mpu_model.get_num_inputs()
        self.input_tensor_infos = self.stai_mpu_model.get_input_infos()

        # Output info
        self.num_outputs = self.stai_mpu_model.get_num_outputs()
        self.output_tensor_infos = self.stai_mpu_model.get_output_infos()

        # Labels
        self._labels = load_labels(self._label_file)

        # Image size
        in_shape = self.input_tensor_infos[0].get_shape()  # (N,H,W,C)
        self._in_height = in_shape[1]
        self._in_width = in_shape[2]

        # Infer number of classes
        first_out_shape = self.output_tensor_infos[0].get_shape()
        # For you: (1, H, W, 6) where 6 = 4 bbox + 1 obj + 1 class
        C_out = first_out_shape[-1]
        self._n_attrs = C_out
        self._num_classes = self._n_attrs - 5
        print("st_yoloxn: inferred num_classes =", self._num_classes)

        # Training anchors and strides
        base_anchors = np.array([0.5, 0.5, 0.07, 0.25, 0.23, 0.7], dtype=np.float32)
        self.network_stride = [8, 16, 32]
        self.image_size = (self._in_height, self._in_width)

        # Map anchors to levels: one anchor (w,h) per level
        # Level 0 (largest H,W) -> first pair, Level 1 -> second, Level 2 -> third
        base_anchor_pairs = base_anchors.reshape(-1, 2)  # (3,2)

        self.level_anchor = []
        for i, ns in enumerate(self.network_stride):
            # anchors * (image_size[0]/ns) as in TF code, then take one pair per level
            pair = base_anchor_pairs[i] * (self.image_size[0] / ns)
            self.level_anchor.append(pair.astype(np.float32))   # (2,)

    # ----------------------------------------------------------------------
    # Generic helpers
    # ----------------------------------------------------------------------
    def get_labels(self):
        return self._labels

    def get_img_size(self):
        """
        :return: (width, height, channels)
        """
        input_tensor_shape = self.input_tensor_infos[0].get_shape()
        print("input_tensor_shape", input_tensor_shape)
        input_width = input_tensor_shape[2]
        input_height = input_tensor_shape[1]
        input_channel = input_tensor_shape[3]
        return (input_width, input_height, input_channel)

    def launch_inference(self, img):
        """
        :param img: RGB image, shape (H,W,3), uint8
        :return: inference time (seconds)
        """
        input_data = np.expand_dims(img, axis=0)

        if self.input_tensor_infos[0].get_dtype() == np.float32:
            input_data = (np.float32(input_data) - self._input_mean) / self._input_std

        self.stai_mpu_model.set_input(0, input_data)
        start = timer()
        self.stai_mpu_model.run()
        end = timer()
        return end - start

    # ----------------------------------------------------------------------
    # YOLOX decoding helpers
    # ----------------------------------------------------------------------
    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def _generate_grids_and_strides_hw(self, feat_h, feat_w, stride):
        """
        Generate grid offsets for a feature map of size (H, W).
        Returns:
          grid:   (H, W, 2) with (x, y) cell indices
          stride: float
        """
        yv, xv = np.meshgrid(np.arange(feat_h), np.arange(feat_w), indexing="ij")
        grid = np.stack((xv, yv), axis=-1).astype(np.float32)  # (H, W, 2)
        return grid, float(stride)
    
    def _yolo_head(self, feats, anchor, num_classes):
        """
        Single-anchor variant of yolo_head for st_yoloxn export.

        feats:  np.array, shape (1, H, W, 5+num_classes) = (1,H,W,6)
                layout: [tx, ty, tw, th, conf_logit, cls_logit...]
        anchor: np.array, shape (2,) in pixels (w,h)
        returns:
          box_xy:         (1, H, W, 1, 2) normalized [0,1]
          box_wh:         (1, H, W, 1, 2) normalized [0,1]
          box_confidence: (1, H, W, 1, 1)
          box_class_probs:(1, H, W, 1, num_classes)
        """

        feats_arr = feats
        if feats_arr.ndim != 4 or feats_arr.shape[0] != 1:
            raise ValueError(f"Expected feats shape (1,H,W,C), got {feats_arr.shape}")

        batch, H, W, C = feats_arr.shape
        n_attrs = 5 + num_classes
        if C != n_attrs:
            raise ValueError(
                f"Unexpected channel size C={C}, expected 5+num_classes={n_attrs}"
            )

        feats_arr = feats_arr.reshape((batch, H, W, 1, n_attrs))
        box_xy = 1.0 / (1.0 + np.exp(-feats_arr[..., 0:2]))
        box_wh = np.exp(feats_arr[..., 2:4])
        box_confidence = 1.0 / (1.0 + np.exp(-feats_arr[..., 4:5]))
        logits = feats_arr[..., 5:]
        e = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        box_class_probs = e / np.sum(e, axis=-1, keepdims=True)

        conv_dims = np.array([H, W], dtype=np.float32).reshape((1, 1, 1, 1, 2))
        ones = np.ones((W, H), dtype=bool)
        i = np.array(np.where(ones)).T                 # (W*H, 2) -> [x_idx, y_idx]
        conv_index = np.stack([i[:, 1], i[:, 0]], axis=-1)  # [y_idx, x_idx]
        conv_index = conv_index.reshape((H, W, 2)).astype(np.float32)
        conv_index = conv_index.reshape((1, H, W, 1, 2))     # (1,H,W,1,2)

        box_xy = (box_xy + conv_index) / conv_dims
        anchor_arr = np.array(anchor, dtype=np.float32).reshape((1, 1, 1, 1, 2))
        box_wh = box_wh * anchor_arr / conv_dims

        return box_xy, box_wh, box_confidence, box_class_probs

    def _decode_yolo_predictions(self, feats, num_classes, anchor, image_size):
        """
        Single-anchor decode_yolo_predictions equivalent.

        feats:    (1, H, W, 5+num_classes) = (1,H,W,6)
        anchor:   (2,) in pixels for this level
        image_size: (img_h, img_w) (not used here; normalization already done)
        """
        box_xy, box_wh, box_confidence, box_class_probs = \
            self._yolo_head(feats, anchor, num_classes)

        x = box_xy[..., 0]
        y = box_xy[..., 1]
        w = box_wh[..., 0]
        h = box_wh[..., 1]

        boxes = np.stack([x - w / 2.0, y - h / 2.0,
                          x + w / 2.0, y + h / 2.0],
                         axis=-1)
        boxes = np.clip(boxes, 0.0, 1.0)

        # box_confidence: (1, H, W, 1, 1)
        B, H, W, A, _ = box_confidence.shape
        num_boxes = H * W * A  # A=1

        boxes = boxes.reshape((B, num_boxes, 4))                 # (1, N, 4)
        box_confidence = box_confidence.reshape((B, num_boxes, 1))
        box_class_probs = box_class_probs.reshape((B, num_boxes, num_classes))

        scores = box_confidence * box_class_probs               # (1, N, num_classes)

        return boxes, scores

    def _nms(self, boxes, scores, iou_threshold, score_threshold):
        """
        Class-agnostic NMS.

        boxes:  (N, 4)  [x_min,y_min,x_max,y_max]
        scores: (N,)    best-class score
        """
        if boxes.shape[0] == 0:
            return []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            if scores[i] < score_threshold:
                break
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            union = areas[i] + areas[order[1:]] - inter
            iou = inter / (union + 1e-6)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    # ----------------------------------------------------------------------
    # Public results API
    # ----------------------------------------------------------------------
    def get_results(self):
        predictions = []
        for i in range(self.num_outputs):
            out = self.stai_mpu_model.get_output(index=i)
            predictions.append(out)

        # Sort like TF: key=lambda x: x.shape[1], reverse=True
        predictions_sorted = sorted(predictions, key=lambda x: x.shape[1], reverse=True)

        levels_boxes = []
        levels_scores = []

        for i, pred_level in enumerate(predictions_sorted):
            anchor_i = self.level_anchor[i]      # shape (2,)
            boxes_i, scores_i = self._decode_yolo_predictions(
                pred_level, self._num_classes, anchor_i, self.image_size
            )
            levels_boxes.append(boxes_i)    # (1, Ni, 4)
            levels_scores.append(scores_i)  # (1, Ni, num_classes)

        boxes = np.concatenate(levels_boxes, axis=1)[0]        # (N_total, 4)
        scores_all = np.concatenate(levels_scores, axis=1)[0]  # (N_total, num_classes)

        class_ids = np.argmax(scores_all, axis=1)
        scores = scores_all[np.arange(scores_all.shape[0]), class_ids]
        
        # boxes: (N,4) normalized [x_min,y_min,x_max,y_max]
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]

        min_size = 0.05   # at least 5% of image in both dimensions
        max_aspect = 3.0  # reject boxes with h/w > 3.0 (too tall & thin)

        aspect = h / (w + 1e-6)
        mask = (w > min_size) & (h > min_size) & (aspect < max_aspect)

        boxes = boxes[mask]
        scores_all = scores_all[mask]

        if boxes.shape[0] == 0:
            return np.array([]), np.array([]), np.array([])

        class_ids = np.argmax(scores_all, axis=1)
        scores = scores_all[np.arange(scores_all.shape[0]), class_ids]

        keep = self._nms(boxes, scores,
                         iou_threshold=self.iou_threshold,
                         score_threshold=self.confidence_threshold)

        if len(keep) == 0:
            return np.array([]), np.array([]), np.array([])

        boxes_kept = boxes[keep]
        scores_kept = scores[keep]
        classes_kept = class_ids[keep]

        boxes_kept = np.expand_dims(boxes_kept, axis=0)
        scores_kept = np.expand_dims(scores_kept, axis=0)
        classes_kept = np.expand_dims(classes_kept, axis=0)

        self.stai_backend = self.stai_mpu_model.get_backend_engine()
        return boxes_kept, classes_kept, scores_kept

    def get_label(self, idx, classes):
        """
        get label from index
        """
        labels = self.get_labels()
        return labels[int(classes[0][idx])]