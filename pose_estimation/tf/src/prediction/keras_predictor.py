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
from pose_estimation.tf.src.utils import skeleton_connections_dict, stm32_to_opencv_colors_dict
from pose_estimation.tf.src.postprocessing  import spe_postprocess, heatmaps_spe_postprocess

# Suppress warnings and TensorFlow logs for cleaner output
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class KerasModelPredictor:
    """
    A class to handle predictions using a Keras model. This class includes methods for:
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
            model: The trained Keras model to use for predictions.
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

    def _get_prediction(self, img):
        """
        Perform prediction on the given image.

        Args:
            img: The preprocessed image to predict.

        Returns:
            A tuple containing the predicted label and the prediction score.
        """
        predictions = self.model.predict_on_batch(img)

        if self.model_type=='heatmaps_spe':
            poses = heatmaps_spe_postprocess(predictions,pred_size=predictions.shape[1:3])[0]
        elif self.model_type=='spe':
            poses = spe_postprocess(predictions)[0]
        else:
            print('No post-processing found for the Keras model type : '+self.model_type)

        return poses

    def _annotate_and_save(self, image_rgb, poses, img_path, img_id):
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
        img_id = 0
        for img, img_path in self.predict_ds:   # Iterate over the prediction dataset
            image_path = img_path.numpy()[0].decode()   # Decode the image path
            image_rgb = self._load_image(image_path)
            if image_rgb is None:
                continue
            poses = self._get_prediction(img)
            self._annotate_and_save(image_rgb, poses, image_path, img_id)
            img_id+=1
        print('[INFO] : Prediction complete.')
