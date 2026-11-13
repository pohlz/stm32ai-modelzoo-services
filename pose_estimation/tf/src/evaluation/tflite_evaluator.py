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
from pose_estimation.tf.src.postprocessing import spe_postprocess, heatmaps_spe_postprocess, yolo_mpe_postprocess
from pose_estimation.tf.src.evaluation.metrics import single_pose_oks, multi_pose_oks_mAP, compute_ap
from pose_estimation.tf.src.utils import ai_runner_invoke


# Define a class for evaluating TFLite quantized models
class TFLiteQuantizedModelEvaluator:
    """
    A class to evaluate TensorFlow Lite (TFLite) quantized models.

    Args:
        cfg (DictConfig): Configuration object for evaluation.
        model (object): The quantized TFLite model to evaluate.
        dataloaders (dict): Dictionary containing datasets for testing and validation.
    """
    def __init__(self, cfg: DictConfig, model: object, dataloaders: dict = None):
        self.cfg = cfg
        self.quantized_model = model
        self.model_type = cfg.model.model_type
        self.test_ds = dataloaders['test']
        self.valid_ds = dataloaders['valid']
        self.output_dir = HydraConfig.get().runtime.output_dir
        self.display_figures = cfg.general.display_figures

    def _get_target(self):
        """
        Retrieves the evaluation target from the configuration.
        """
        if self.cfg.evaluation and self.cfg.evaluation.target:
            return self.cfg.evaluation.target
        return "host"

    def _get_interpreter(self, target):
        """
        Retrieves the AI runner interpreter for the specified target.
        """
        name_model = os.path.basename(self.quantized_model.model_path)
        return ai_runner_interp(target, name_model)

    def _run_evaluate(self):

        target = self._get_target()     # Get the evaluation target
        ai_runner_interpreter = self._get_interpreter(target=target)    # Get the AI runner interpreter
        # Use the test dataset if available; otherwise, use the validation dataset
        if self.test_ds:
            self.eval_ds = self.test_ds
        else:
            self.eval_ds = self.valid_ds
        input_details = self.quantized_model.get_input_details()
        outputs_details = self.quantized_model.get_output_details()
        self.quantized_model.allocate_tensors()
        metric = 0
        nb_images = 0
        predictions_all = []
        images_full = []
        tp = None
        for images, labels in tqdm.tqdm(self.eval_ds):
            image_processed = images.numpy()
            if input_details[0]['dtype'] == np.uint8:
                image_processed = (image_processed - self.cfg.preprocessing.rescaling.offset) / self.cfg.preprocessing.rescaling.scale
                image_processed = np.clip(np.round(image_processed), np.iinfo(input_details[0]['dtype']).min, np.iinfo(input_details[0]['dtype']).max)
            elif input_details[0]['dtype'] == np.int8:
                image_processed = (image_processed - self.cfg.preprocessing.rescaling.offset) / self.cfg.preprocessing.rescaling.scale
                image_processed -= 128
                image_processed = np.clip(np.round(image_processed), np.iinfo(input_details[0]['dtype']).min, np.iinfo(input_details[0]['dtype']).max)
            elif input_details[0]['dtype'] == np.float32:
                image_processed = image_processed
            else:
                print('[ERROR] : input dtype not recognized -> ', input_details[0]['dtype'])
            if "evaluation" in self.cfg and self.cfg.evaluation:
                if "gen_npy_input" in self.cfg.evaluation and self.cfg.evaluation.gen_npy_input == True:
                    images_full.append(image_processed)
            imags = image_processed.astype(input_details[0]['dtype'])
            predictions = []
            for indx, imge in enumerate(imags):
                imgee = imge[None]
                if target == 'host':
                    self.quantized_model.set_tensor(input_details[0]['index'], imgee)
                    self.quantized_model.invoke()
                    prediction = [self.quantized_model.get_tensor(outputs_details[j]["index"]) for j in range(len(outputs_details))][0]
                elif target in ['stedgeai_host', 'stedgeai_n6', 'stedgeai_h7p']:
                    data = ai_interp_input_quant(ai_runner_interpreter, images.numpy()[indx][None], '.tflite')
                    prediction = ai_runner_invoke(data, ai_runner_interpreter)
                    prediction = ai_interp_outputs_dequant(ai_runner_interpreter, prediction)[0]
                predictions.append(prediction[0])
            predictions = np.stack(predictions)
            if "evaluation" in self.cfg and self.cfg.evaluation:
                if "gen_npy_output" in self.cfg.evaluation and self.cfg.evaluation.gen_npy_output == True:
                    predictions_all.append(predictions)
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
            mlflow.log_metric("quantized_OKS", round(metric.numpy()*100, 2))
            mlflow.log_metric("quantized_mAP_0.5", 0)   # not used for those models, but helps for pe validation
            log_to_file(self.output_dir, "quantized_model_OKS : {:.2f}%".format(metric.numpy()*100))
        elif self.model_type=='yolo_mpe':
            metric = compute_ap(tp, conf, nb_gt, maskpad, self.display_figures)
            print('mAP@0.5        -> {:.2f}%'.format(metric[0]*100))
            print('mAP@[0.5:0.95] -> {:.2f}%'.format(np.mean(metric)*100))
            # logging the mAP@0.5 and mAP@[0.5:0.95] in the stm32ai_main.log file
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