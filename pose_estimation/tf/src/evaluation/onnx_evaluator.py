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

from common.utils import log_to_file, ai_runner_interp, ai_interp_input_quant, ai_interp_outputs_dequant
from common.evaluation import model_is_quantized
from pose_estimation.tf.src.utils import ai_runner_invoke
from pose_estimation.tf.src.postprocessing import spe_postprocess, heatmaps_spe_postprocess, yolo_mpe_postprocess
from pose_estimation.tf.src.evaluation.metrics import single_pose_oks, multi_pose_oks_mAP, compute_ap


# Define a class for evaluating ONNX models
class ONNXModelEvaluator:
    """
    A class to evaluate ONNX models.

    Args:
        cfg (DictConfig): Configuration object for evaluation.
        model (object): The ONNX model to evaluate.
        dataloaders (dict): Dictionary containing datasets for testing and validation.
    """
    def __init__(self, cfg: DictConfig, model: object, dataloaders: dict = None):
        self.cfg = cfg
        self.model = model
        self.model_type = cfg.model.model_type
        self.test_ds = dataloaders['test']
        self.valid_ds = dataloaders['valid']
        self.output_dir = HydraConfig.get().runtime.output_dir
        self.display_figures = cfg.general.display_figures

    def _ensure_output_dir(self):
        """
        Ensures that the output directory exists.
        """
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

    def _get_target(self):
        """
        Retrieves the evaluation target from the configuration.
        """
        if self.cfg.evaluation and self.cfg.evaluation.target:
            return self.cfg.evaluation.target
        return "host"

    def _get_ai_runner_interpreter(self, target):
        """
        Retrieves the AI runner interpreter for the specified target.
        """
        name_model = os.path.basename(self.model.model_path)
        return ai_runner_interp(target, name_model)

    def _run_evaluate(self):

        self._ensure_output_dir()       # Ensure the output directory exists
        target = self._get_target()     # Get the evaluation target
        model_path = self.model.model_path
        ai_runner_interpreter = self._get_ai_runner_interpreter(target=target)   # Get the AI runner interpreter

        # Use the test dataset if available; otherwise, use the validation dataset
        if self.test_ds:
            self.eval_ds = self.test_ds
        else:
            self.eval_ds = self.valid_ds
        inputs = self.model.get_inputs()
        input_shape = self.model.get_inputs()[0].shape  # e.g. [batch, channels, height, width] or similar
        model_batch_size = input_shape[0]
        outputs = self.model.get_outputs()
        metric = 0
        nb_images = 0
        predictions_all = []
        images_full = []
        tp = None


        for images, labels in tqdm.tqdm(self.eval_ds):
            # Here we consider that the dataloader provided chlast uint8 data by default.
            # If the user declares that input will be chfirst, a transpose of the input is needed else
            # the compiler will have added a transpose in the onnx model
            if self.cfg.evaluation.input_chpos=="chfirst" or target == 'host':
                # The transpose is doing chlast -> chfirst as the host model is onnx channel first
                images = np.transpose(images, [0,3,1,2])
            else:
                images = tf.cast(images, dtype=tf.float32).numpy()
            
            if model_batch_size!=1:
                if target == 'host':
                    predictions = self.model.run([o.name for o in outputs], {inputs[0].name: images.astype('float32')})[0]
                elif target in ['stedgeai_host', 'stedgeai_n6', 'stedgeai_h7p']:
                    data = ai_interp_input_quant(ai_runner_interpreter, images, '.onnx')
                    predictions = ai_runner_invoke(data, ai_runner_interpreter)
                    predictions = ai_interp_outputs_dequant(ai_runner_interpreter, predictions)[0]

                # Here we consider that the post processing is always expecting chlast data
                # If the user declares that output will be chfirst, a transpose of the output is needed else
                # the compiler will have added a transpose in the onnx model
            else:
                if target == 'host':
                    predictions = []
                    for i in range(images.shape[0]):
                        pred = self.model.run([o.name for o in outputs], {inputs[0].name: images[i:i+1].astype('float32')})[0]
                        predictions.append(pred)
                    predictions = np.concatenate(predictions, axis=0)
                elif target in ['stedgeai_host', 'stedgeai_n6', 'stedgeai_h7p']:
                    predictions = []
                    for i in range(images.shape[0]):
                        data = ai_interp_input_quant(ai_runner_interpreter, images[i:i+1], '.onnx')
                        pred = ai_runner_invoke(data, ai_runner_interpreter)
                        pred = ai_interp_outputs_dequant(ai_runner_interpreter, pred)[0]
                        predictions.append(pred)
                    predictions = np.concatenate(predictions, axis=0)

            if (target == 'host') and self.cfg.model.model_type not in ["yolo_mpe"]:
                # Make the transpose chfirst -> chlast
                if len(predictions.shape)==3:
                    predictions = tf.transpose(predictions,[0,2,1])
                elif len(predictions.shape)==4:
                    predictions = tf.transpose(predictions,[0,2,3,1])

            predictions = tf.cast(predictions,tf.float32)
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

                mask_s = tf.constant([images.shape[2]]*4 + [1.] + [images.shape[2],images.shape[2],1.]*17)
                poses /= mask_s[None,None]

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
            if model_is_quantized(model_path) == "False":
                mlflow.log_metric("float_OKS", round(metric.numpy()*100, 2))
                mlflow.log_metric("float_mAP_0.5", 0)   # not used for those models, but helps for pe validation
                log_to_file(self.output_dir, "float_model_OKS : {:.2f}%".format(metric.numpy()*100)) 
            else:
              mlflow.log_metric("quantized_OKS", round(metric.numpy()*100, 2))
              mlflow.log_metric("quantized_mAP_0.5", 0)   # not used for those models, but helps for pe validation
              log_to_file(self.output_dir, "quantized_model_OKS : {:.2f}%".format(metric.numpy()*100))
        elif self.model_type=='yolo_mpe':
            metric = compute_ap(tp, conf, nb_gt, maskpad, self.display_figures)
            print('mAP@0.5        -> {:.2f}%'.format(metric[0]*100))
            print('mAP@[0.5:0.95] -> {:.2f}%'.format(np.mean(metric)*100))
            # logging the mAP@0.5 and mAP@[0.5:0.95] in the stm32ai_main.log file
            if model_is_quantized(model_path) == "False":
                mlflow.log_metric("float_mAP_0.5", round(metric[0]*100, 2))
                mlflow.log_metric("float_mAP_0.5_0.95", round(np.mean(metric)*100, 2))
                mlflow.log_metric("float_OKS", 0)   # not used for those models, but helps for pe validation
                log_to_file(self.output_dir, "float_model_mAP@0.5        -> {:.2f}%".format(metric[0]*100))
                log_to_file(self.output_dir, "float_model_mAP@[0.5:0.95] -> {:.2f}%".format(np.mean(metric)*100))
            else:
                mlflow.log_metric("quantized_mAP_0.5", round(metric[0]*100, 2))
                mlflow.log_metric("quantized_mAP_0.5_0.95", round(np.mean(metric)*100, 2))
                mlflow.log_metric("quantized_OKS", 0)   # not used for those models, but helps for pe validation
                log_to_file(self.output_dir, "quantized_model_mAP@0.5        -> {:.2f}%".format(metric[0]*100))
                log_to_file(self.output_dir, "quantized_model_mAP@[0.5:0.95] -> {:.2f}%".format(np.mean(metric)*100))
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