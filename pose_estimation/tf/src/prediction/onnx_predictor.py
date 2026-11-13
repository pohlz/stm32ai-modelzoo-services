# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

# Import necessary libraries
import os
import cv2
import numpy as np
import tensorflow as tf
from hydra.core.hydra_config import HydraConfig
from common.utils import ai_runner_interp, ai_interp_input_quant, ai_interp_outputs_dequant
from pose_estimation.tf.src.utils import ai_runner_invoke, skeleton_connections_dict, stm32_to_opencv_colors_dict
from pose_estimation.tf.src.postprocessing  import spe_postprocess, heatmaps_spe_postprocess, yolo_mpe_postprocess, hand_landmarks_postprocess, head_landmarks_postprocess

# Suppress warnings and TensorFlow logs for cleaner output
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ONNXModelPredictor:
    """
    A class to handle predictions using a ONNX model. This class includes methods for:
    - Loading and preprocessing images
    - Making predictions
    - Annotating and saving prediction results
    - Displaying results in a tabular format
    """
    def __init__(self, cfg, model, dataloaders):
        """
        Initialize the predictor with configuration, model, and dataloaders.

        Args:
            cfg: Configuration object containing settings for the predictor.
            model: The trained ONNX model to use for predictions.
            dataloaders: A dictionary containing the prediction dataset.
        """
        self.cfg = cfg
        self.model = model
        self.model_type = cfg.model.model_type
        self.class_name = cfg.dataset.class_names[0] if cfg.dataset.class_names is not None else 'None'
        self.skeleton_connections = []
        self.predict_ds = dataloaders['predict']
        self.prediction_result_dir = os.path.join(cfg.output_dir, 'predictions')
        os.makedirs(self.prediction_result_dir, exist_ok=True)

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
        name_model = os.path.basename(self.model.model_path)
        return ai_runner_interp(target, name_model)

    def _load_image(self, img_path):
        """
        Load an image from the given path and convert it to RGB format.

        Args:
            img_path: Path to the image file.

        Returns:
            The loaded image in RGB format, or None if the image could not be loaded.
        """
        image = cv2.imread(img_path)
        if image is None:
            print(f"[ERROR] : Could not load image {img_path}")
            return None
        if len(image.shape) != 3: # If the image is grayscale, convert it to BGR
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _get_prediction(self, img, target, ai_runner_interpreter):
        """
        Perform prediction on the given image.

        Args:
            img: The preprocessed image to predict.

        Returns:
            A tuple containing the predicted label and the prediction score.
        """

        inputs = self.model.get_inputs()
        outputs = self.model.get_outputs()

        if self.cfg.prediction.input_chpos=="chfirst" or target == 'host':
            # The transpose is doing chlast -> chfirst as the host model is onnx channel first
            img = np.transpose(img, [0,3,1,2])

        if target == 'host':
            predictions = self.model.run([o.name for o in outputs], {inputs[0].name: img.astype('float32')})[0]
        elif target in ['stedgeai_host', 'stedgeai_n6', 'stedgeai_h7p']:
            data = ai_interp_input_quant(ai_runner_interpreter, img, '.onnx')
            predictions = ai_runner_invoke(data, ai_runner_interpreter)
            predictions = ai_interp_outputs_dequant(ai_runner_interpreter, predictions)[0]

        if self.cfg.prediction:
            if (target == 'host') and self.cfg.model.model_type not in ["yolo_mpe"]:
                # Make the transpose chfirst -> chlast
                if len(predictions.shape)==3:
                    predictions = tf.transpose(predictions,[0,2,1])
                elif len(predictions.shape)==4:
                    predictions = tf.transpose(predictions,[0,2,3,1])

        predictions = tf.cast(predictions,tf.float32)

        htype,hprob = None,None

        if self.model_type=='heatmaps_spe':
            poses = heatmaps_spe_postprocess(predictions,pred_size=predictions.shape[1:3])[0]
        elif self.model_type=='spe':
            poses = spe_postprocess(predictions)[0]    
        elif self.model_type=='yolo_mpe':
            poses = yolo_mpe_postprocess(predictions,
                                         max_output_size = self.cfg.postprocessing.max_detection_boxes,
                                         iou_threshold   = self.cfg.postprocessing.NMS_thresh,
                                         score_threshold = self.cfg.postprocessing.confidence_thresh)[0]
        elif self.model_type=='hand_spe':
            poses,norm_poses,htype,hprob = hand_landmarks_postprocess(predictions)
        elif self.model_type=='head_spe':
            poses = head_landmarks_postprocess(predictions)
        else:
            print('No post-processing found for the ONNX model type : '+self.model_type)

        return poses,htype,hprob

    def _annotate_and_save(self, image_rgb, poses, htype, hprob, img_path, img_id, input_shape):
        """
        Annotate the image with the prediction and save it.

        Args:
            image_rgb: The RGB image to annotate.
            poses: list of poses to be display on the image
            img_path: The original image path (used for naming the saved file).
        """

        height, width, _ = image_rgb.shape
        img_name = os.path.splitext(img_path)[0]
        threshSkeleton = self.cfg.postprocessing.kpts_conf_thresh
        bbox_thick = int(0.6 * (height + width) / 600)
        radius = 3

        for ids,p in enumerate(poses):
            if self.model_type in ['heatmaps_spe','spe']:
                xx, yy, pp = p[0::3],p[1::3],p[2::3]
            elif self.model_type == 'hand_spe':
                xx = p[0::3]/input_shape[0]
                yy = p[1::3]/input_shape[1]
                pp = tf.ones_like(xx) * hprob[ids]
                x1 = int(np.min(xx)*width)
                x2 = int(np.max(xx)*width)
                y1 = int(np.min(yy)*height)
                y2 = int(np.max(yy)*height)
            elif self.model_type == 'head_spe':
                radius = 1
                xx = p[0::2]/input_shape[0]
                yy = p[1::2]/input_shape[1]
                pp = tf.ones_like(xx)
                x1 = int(np.min(xx)*width)
                x2 = int(np.max(xx)*width)
                y1 = int(np.min(yy)*height)
                y2 = int(np.max(yy)*height)
            elif self.model_type == 'yolo_mpe':
                x,y,w,h,conf = p[:5]
                xx, yy, pp = p[5::3],p[5+1::3],p[5+2::3]
                x  /= input_shape[0]
                y  /= input_shape[1]
                w  /= input_shape[0]
                h  /= input_shape[1]
                xx /= input_shape[0]
                yy /= input_shape[1]
                x1 = int((x - w/2)*width)
                x2 = int((x + w/2)*width)
                y1 = int((y - h/2)*height)
                y2 = int((y + h/2)*height)
            if img_id==0:
                kpts_nbr = self.cfg.dataset.keypoints if (self.cfg.dataset.keypoints is not None and self.cfg.dataset.keypoints==len(xx)) else len(xx)
                try:
                    self.skeleton_connections = skeleton_connections_dict[self.class_name][kpts_nbr]
                except:
                    list_of_connections=''
                    for s in skeleton_connections_dict : list_of_connections += s+": "+str(list(skeleton_connections_dict[s].keys()))+" | "
                    print(f'Skeleton for the class [{self.class_name}] & number of keypoints [{kpts_nbr}] is unknown -> use {list_of_connections[:-2]}')
                    print('You can add your own in the utils/connections.py file')
            if not tf.reduce_all(tf.constant(pp)==0):
                for i in range(0,len(xx)):
                    if float(pp[i])>threshSkeleton:
                        cv2.circle(image_rgb,(int(xx[i]*width),int(yy[i]*height)),radius=radius,color=(255, 255, 255), thickness=-1)
                    else:
                        cv2.circle(image_rgb,(int(xx[i]*width),int(yy[i]*height)),radius=radius,color=(255, 0, 0), thickness=-1)
                for k,l,color in self.skeleton_connections:
                    if float(pp[k])>threshSkeleton and float(pp[l])>threshSkeleton: 
                        cv2.line(image_rgb,(int(xx[k]*width),int(yy[k]*height)),(int(xx[l]*width),int(yy[l]*height)),stm32_to_opencv_colors_dict[color])            
            if self.model_type=='yolo_mpe':
                btext = '{}-{:.2f}'.format(self.class_name,conf)
            elif self.model_type=='hand_spe':
                btext = '{}'.format(['left_hand','right_hand'][htype[ids]>0.5])
            if self.model_type in ['yolo_mpe','hand_spe']:
                cv2.rectangle(image_rgb,(x1,y1), (x2, y2),(255, 0, 255),1)
                cv2.putText(image_rgb, btext, (x1,y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), bbox_thick//2, lineType=cv2.LINE_AA)
        pred_res_filename = os.path.join(self.prediction_result_dir,os.path.basename(img_name)+'.png')
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(pred_res_filename,image_bgr)
        if self.cfg.general.display_figures:
            cv2.imshow('image',image_bgr)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def predict(self):
        """
        Perform predictions on the dataset and save the results.
        """

        target = self._get_target()     # Get the evaluation target
        ai_runner_interpreter = self._get_interpreter(target=target)    # Get the AI runner interpreter

        img_id = 0
        for img, img_path in self.predict_ds:   # Iterate over the prediction dataset
            image_path = img_path.numpy()[0].decode()   # Decode the image path
            image_rgb = self._load_image(image_path)
            if image_rgb is None:
                continue
            poses,htype,hprob = self._get_prediction(img,target,ai_runner_interpreter)
            input_shape = img.shape[1:3]
            self._annotate_and_save(image_rgb, poses, htype, hprob, image_path, img_id, input_shape)
            img_id+=1
        print('[INFO] : Prediction complete.')
