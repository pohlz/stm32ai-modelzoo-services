# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
import os
import tqdm
import numpy as np
import tensorflow as tf
import mlflow
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from common.utils import count_h5_parameters, log_to_file
from pose_estimation.tf.src.postprocessing import spe_postprocess, heatmaps_spe_postprocess, yolo_mpe_postprocess
from pose_estimation.tf.src.evaluation.metrics import single_pose_oks, multi_pose_oks_mAP, compute_ap


# Define a class for evaluating Keras models
class KerasModelEvaluator:
    """
    A class to evaluate TensorFlow Keras models.

    Args:
        cfg (DictConfig): Configuration object for evaluation.
        model (object): The Keras model to evaluate.
        dataloaders (dict): Dictionary containing datasets for testing and validation.
    """
    def __init__(self, cfg: DictConfig = None, model: object = None, dataloaders: dict = None):
        self.cfg = cfg
        self.model = model
        self.model_type = cfg.model.model_type
        self.test_ds = dataloaders['test']
        self.valid_ds = dataloaders['valid']
        self.output_dir = HydraConfig.get().runtime.output_dir
        self.display_figures = cfg.general.display_figures

    def _run_evaluate(self):
        metric = 0
        nb_images = 0
        predictions_all = []
        images_full = []
        # Use the test dataset if available; otherwise, use the validation dataset
        if self.test_ds:
            self.eval_ds = self.test_ds
        else:
            self.eval_ds = self.valid_ds
        tp = None
        count_h5_parameters(output_dir=self.output_dir, model=self.model)  # model_path not needed

        for images, labels in tqdm.tqdm(self.eval_ds):
            predictions = self.model.predict_on_batch(images)
            predictions = tf.cast(predictions, tf.float32)

            if self.model_type=='heatmaps_spe':
                poses = heatmaps_spe_postprocess(predictions,pred_size=predictions.shape[1:3])
                oks   = single_pose_oks(labels,poses)
                metric += tf.reduce_sum(oks)
                nb_images += tf.reduce_prod(tf.shape(oks)).numpy()
            elif self.model_type=='spe':
                poses = spe_postprocess(predictions)
                oks   = single_pose_oks(labels,poses)
                metric += tf.reduce_sum(oks)
                nb_images += tf.reduce_prod(tf.shape(oks)).numpy()
            elif self.model_type=='yolo_mpe':
                poses = yolo_mpe_postprocess(predictions,
                                             max_output_size=self.cfg.postprocessing.max_detection_boxes,
                                             iou_threshold=self.cfg.postprocessing.NMS_thresh,
                                             score_threshold=self.cfg.postprocessing.confidence_thresh)

                oks   = multi_pose_oks_mAP(labels,poses) # (batch*M,thresh), (batch*M,), (1,), (batch*M,)

                tdet_ind = tf.where(oks[1]>0)[:,0]    # (true_detections,)
                ttp      = tf.gather(oks[0],tdet_ind) # (true_detections,thresh)
                tconf    = tf.gather(oks[1],tdet_ind) # (true_detections,)
                tnb_gt   = oks[2]                     # (1,)
                tmaskpad = tf.gather(oks[3],tdet_ind) # (true_detections,)

                if tp==None:
                    tp      = ttp
                    conf    = tconf
                    nb_gt   = tnb_gt
                    maskpad = tmaskpad

                else:
                    tp      = tf.concat([tp,ttp],0)
                    conf    = tf.concat([conf,tconf],0)
                    nb_gt  += tnb_gt
                    maskpad = tf.concat([maskpad,tmaskpad],0)
            else:
                print('No post-processing found for the model type : '+self.model_type)

        # Saves evaluation dataset in a .npy
        if "evaluation" in self.cfg and self.cfg.evaluation:
            if "gen_npy_input" in self.cfg.evaluation and self.cfg.evaluation.gen_npy_input==True: 
                if "npy_in_name" in self.cfg.evaluation and self.cfg.evaluation.npy_in_name:
                    npy_in_name = self.cfg.evaluation.npy_in_name
                else:
                    npy_in_name = "unknown_npy_in_name"
                images_full = np.concatenate(images_full, axis=0)
                print("[INFO] : Shape of npy input dataset = {}".format(images_full.shape))
                np.save(os.path.join(self.output_dir, f"{npy_in_name}.npy"), images_full)

        # Saves model output in a .npy
        if "evaluation" in self.cfg and self.cfg.evaluation:
            if "gen_npy_output" in self.cfg.evaluation and self.cfg.evaluation.gen_npy_output==True: 
                if "npy_out_name" in self.cfg.evaluation and self.cfg.evaluation.npy_out_name:
                    npy_out_name = self.cfg.evaluation.npy_out_name
                else:
                    npy_out_name = "unknown_npy_out_name"
                predictions_all = np.concatenate(predictions_all, axis=0)
                np.save(os.path.join(self.output_dir, f"{npy_out_name}.npy"), predictions_all)

        if self.model_type in ['heatmaps_spe','spe']:
            metric /= nb_images
            print("The mean OKS is : {:.2f}%".format(metric.numpy()*100))
            # logging the OKS in the stm32ai_main.log file
            mlflow.log_metric("float_OKS", round(metric.numpy()*100, 2))
            mlflow.log_metric("float_mAP_0.5", 0)   # not used for those models, but helps for pe validation
            log_to_file(self.output_dir, "float_model_OKS : {:.2f}%".format(metric.numpy()*100)) 
        elif self.model_type=='yolo_mpe':
            metric = compute_ap(tp, conf, nb_gt, maskpad, self.display_figures)
            print('mAP@0.5        -> {:.2f}%'.format(metric[0]*100))
            print('mAP@[0.5:0.95] -> {:.2f}%'.format(np.mean(metric)*100))
            # logging the mAP@0.5 and mAP@[0.5:0.95] in the stm32ai_main.log file
            mlflow.log_metric("float_mAP_0.5", round(metric[0]*100, 2))
            mlflow.log_metric("float_mAP_0.5_0.95", round(np.mean(metric)*100, 2))
            mlflow.log_metric("float_OKS", 0)   # not used for those models, but helps for pe validation
            log_to_file(self.output_dir, "float_model_mAP@0.5        -> {:.2f}%".format(metric[0]*100))
            log_to_file(self.output_dir, "float_model_mAP@[0.5:0.95] -> {:.2f}%".format(np.mean(metric)*100))
        else:
            print('No metric found for the model type : '+self.model_type)

    def evaluate(self):
        """
        Executes the full evaluation process.

        Returns:
            float: Accuracy of the model on the evaluation dataset.
        """
        self._run_evaluate()  # Run the evaluation
        print('[INFO] : Evaluation complete.')