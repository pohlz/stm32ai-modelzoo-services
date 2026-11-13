# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
import sys
from pathlib import Path
from omegaconf import DictConfig
from tabulate import tabulate
import numpy as np
import tensorflow as tf
import onnxruntime

import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from common.utils import get_model_name_and_its_input_shape, ai_runner_interp, ai_interp_input_quant, ai_interp_outputs_dequant
from common.evaluation import predict_onnx
from re_identification.tf.src.utils import ai_runner_invoke, pairwise_distance
from re_identification.tf.src.preprocessing import preprocess_input



def _load_test_data(directory: str):
    """
    Parse the training data and return a list of paths to annotation files.
    
    Args:
    - directory: A string representing the path to test set directory.
    
    Returns:
    - A list of strings representing the paths to test images.
    """
    annotation_lines = []
    path = directory+'/'
    for file in os.listdir(path):
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
            new_path = path+file
            annotation_lines.append(new_path)
    return annotation_lines

def _prepare_data(img_path: str, model_path: str, channels: int, model_input_shape: tuple, cpp: DictConfig):
    """
    Prepares the image for the model inference.

    Args:
        img_path (str): The path to the image to be prepared.
        channels (int): The number of channels of the image (1 for grayscale, 3 for RGB).
        model_input_shape (tuple): The input shape of the model.
        cpp (DictConfig): A dictionary containing the preprocessing parameters.

    Returns:
        A tensor representing the preprocessed image.
    Errors:
        The image file can't be loaded.
    """
    # Load the image with Tensorflow for the model inference
    try:
        data = tf.io.read_file(img_path)
        img = tf.image.decode_image(data, channels=channels)
    except:
        raise ValueError(f"\nUnable to load image file {img_path}\n"
                         "Supported image file formats are BMP, GIF, JPEG and PNG.")
    # Resize the image            
    height, width = model_input_shape[1:] if Path(model_path).suffix == '.onnx' else model_input_shape[0:2]
    if cpp.resizing.aspect_ratio == "fit":
        img = tf.image.resize(img, [height, width], method=cpp.resizing.interpolation, preserve_aspect_ratio=False)
    else:
        img = tf.image.resize_with_crop_or_pad(img, height, width)
    # Rescale the image
    img = cpp.rescaling.scale * tf.cast(img, tf.float32) + cpp.rescaling.offset
    return img

def _features_from_keras(model, img):
    if len(img.shape) == 3:
        img = tf.expand_dims(img, 0)
    features = model.predict(img)
    return features

def _features_from_tflite(interpreter, input_details, input_index, output_index, img, output_details, target, ai_runner_interpreter, cfg):
    image_processed = preprocess_input(img, input_details)
    if target == 'host':
        interpreter.set_tensor(input_index, image_processed)
        interpreter.invoke()
        features = interpreter.get_tensor(output_index)
        if output_details['dtype'] in [np.uint8, np.int8]:
            features = (np.float32(features) - output_details['quantization'][1]) * output_details['quantization'][0]
    elif target in ['stedgeai_host', 'stedgeai_n6', 'stedgeai_h7p']:
        imagee = ai_interp_input_quant(ai_runner_interpreter,img[None].numpy(),'.tflite')
        features = ai_runner_invoke(imagee,ai_runner_interpreter)
        features = ai_interp_outputs_dequant(ai_runner_interpreter,[features])[0]
    return features

def _features_from_onnx(sess, img, target, ai_runner_interpreter, cfg):
    if len(img.shape) == 3:
        image_processed = np.expand_dims(img, 0)
    else:
        image_processed = img.numpy()
    image_processed = np.transpose(image_processed,[0,3,1,2])
    if target == 'host':
        features = predict_onnx(sess, image_processed)
    elif target in ['stedgeai_host', 'stedgeai_n6', 'stedgeai_h7p']:
        imagee = ai_interp_input_quant(ai_runner_interpreter,image_processed,'.onnx')
        features = ai_runner_invoke(imagee,ai_runner_interpreter)
        features = ai_interp_outputs_dequant(ai_runner_interpreter,[features])[0]
    return features

def predict(cfg: DictConfig = None, model: tf.keras.Model = None) -> None:
    """
    Predicts class for all the images that are inside a given query directory based on the distances with gallery images.
    The model used for the predictions can be either a .h5, .keras or .tflite file.

    Args:
        cfg (dict): A dictionary containing the entire configuration file.

    Returns:
        None
    
    Errors:
        The directory containing the images cannot be found.
        The directory does not contain any file.
        An image file can't be loaded.
    """
    import cv2
    model_path = model.model_path
    test_images_dir = cfg.dataset.prediction_query_path
    gallery_images_dir = cfg.dataset.prediction_gallery_path
    cpp = cfg.preprocessing
    if cfg.prediction and cfg.prediction.target:
        target = cfg.prediction.target
    else:
        target = "host"
    name_model = os.path.basename(model_path)

    _, model_input_shape = get_model_name_and_its_input_shape(model_path)
    
    
    print("[INFO] : Making predictions using:")
    print("  model:", model_path)
    print("  images query directory:", test_images_dir)
    print("  images gallery directory:", gallery_images_dir)

    channels = 1 if cpp.color_mode == "grayscale" else 3
    results_table = []
    file_extension = Path(model_path).suffix

    if test_images_dir and gallery_images_dir:
        image_filenames =  _load_test_data(test_images_dir)
        image_gallery_filenames =  _load_test_data(gallery_images_dir)
    else:
        print("no test set found")

    if file_extension in [".h5",".keras"]:
        # Load the .h5 or .keras model
        model = tf.keras.models.load_model(model_path)
    elif file_extension == ".tflite":
        # Load the Tflite model and allocate tensors
        interpreter_quant = tf.lite.Interpreter(model_path=model_path)
        interpreter_quant.allocate_tensors()
        input_details = interpreter_quant.get_input_details()[0]
        input_index_quant = input_details["index"]
        output_details = interpreter_quant.get_output_details()[0]
        output_index_quant = interpreter_quant.get_output_details()[0]["index"]
        ai_runner_interpreter = ai_runner_interp(target,name_model)
    elif file_extension == ".onnx":
        sess = onnxruntime.InferenceSession(model_path)
        ai_runner_interpreter = ai_runner_interp(target,name_model)

    prediction_result_dir = f'{cfg.output_dir}/predictions/'
    os.makedirs(prediction_result_dir, exist_ok=True)

    # prepare the imgs of gallery set
    img_gallery = []
    for i in range(len(image_gallery_filenames)):
        if image_gallery_filenames[i].endswith(".jpg") or image_gallery_filenames[i].endswith(".png") or image_gallery_filenames[i].endswith(".jpeg"):
            im_path = image_gallery_filenames[i]
            # # Load the image with Tensorflow for the model inference
            img = _prepare_data(im_path, model_path, channels, model_input_shape, cpp)
            img_gallery.append(img)
    print("number of gallery images:",len(img_gallery))

    # calculate the features of gallery images
    gallery_features = []
    if file_extension in [".h5",".keras"]:
        gallery_features = _features_from_keras(model, tf.stack(img_gallery))
    elif file_extension == ".onnx":
        gallery_features = _features_from_onnx(sess, tf.stack(img_gallery), target, ai_runner_interpreter, cfg)
    else:
        for i in range(len(image_gallery_filenames)):
            if image_gallery_filenames[i].endswith(".jpg") or image_gallery_filenames[i].endswith(".png") or image_gallery_filenames[i].endswith(".jpeg"):
                im_path = image_gallery_filenames[i]
                # # Load the image with Tensorflow for the model inference
                img = _prepare_data(im_path, model_path, channels, model_input_shape, cpp)
                if file_extension == ".tflite":
                    features = _features_from_tflite(interpreter_quant, input_details, input_index_quant, output_index_quant, img, output_details, target, ai_runner_interpreter, cfg)
                else:
                    raise TypeError(f"Unknown or unsupported model type. Received path {model_path}")
                gallery_features.append(features)
    gallery_features = np.vstack(gallery_features)
    print("gallery_features shape:", gallery_features.shape)

    distance_metric = cfg.prediction.reid_distance_metric if 'reid_distance_metric' in cfg.prediction else 'cosine'
    print (f"[INFO] : Using `{distance_metric}` distance metric for re-identification prediction")

    # loop over all the images to predict
    for i in range(len(image_filenames)):
        if image_filenames[i].endswith(".jpg") or image_filenames[i].endswith(".png") or image_filenames[i].endswith(".jpeg"):

            print('Inference on image : ',image_filenames[i])
            im_path = image_filenames[i]

            # # Load the image for the model inference
            img = _prepare_data(im_path, model_path, channels, model_input_shape, cpp)

            # Load the image with OpenCV to print it on screen
            image = cv2.imread(im_path)
            if len(image.shape) != 3:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, _ = image.shape
            thick = int(0.6 * (height + width) / 600)
            img_name = os.path.splitext(image_filenames[i])[0]

            if file_extension in [".h5",".keras"]:
                features = _features_from_keras(model, img)
            elif file_extension == ".tflite":
                features = _features_from_tflite(interpreter_quant, input_details, input_index_quant, output_index_quant, img, output_details, target, ai_runner_interpreter, cfg)
            elif file_extension == ".onnx":
                features = _features_from_onnx(sess, img, target, ai_runner_interpreter, cfg)
            else:
                raise TypeError(f"Unknown or unsupported model type. Received path {model_path}")

            # calculate the distance between query image and gallery images
            distances = pairwise_distance(features, gallery_features, distance_metric=distance_metric)
            # Find the label of the closest gallery image
            min_index = np.argmin(distances)
            predicted_label = os.path.basename(image_gallery_filenames[min_index]).split('_')[0]
            print("Predicted label:",predicted_label)

            # Add result to the table
            results_table.append([predicted_label, "{:.1f}".format(min_index), image_filenames[i]])
            pred_text = str(predicted_label) + ": " +"{:.1f}".format(min_index)+ "%"

            # disply the test image and its 5 closest gallery images 
            # the test image is on the left and the gallery images are on the right
            # the test image is separated from the gallery images by a larger white space
            # the gallery images are separated from each other by a white line
            # add the image name on the top of the test image (outside image) + "test: "
            # add the label of each gallery image on the top of the image (outside image)
            gallery_images = []
            im_test = cv2.imread(im_path)
            im_test = cv2.cvtColor(im_test, cv2.COLOR_BGR2RGB)
            im_test = cv2.resize(im_test, (200, 400))
            im_test = cv2.copyMakeBorder(im_test, 50, 10, 30, 70, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            cv2.putText(im_test, "test: "+os.path.basename(img_name), (35, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, lineType=cv2.LINE_AA)
            gallery_images.append(im_test)
            for j in range(5):
                im_path_gallery = image_gallery_filenames[np.argsort(distances)[0][j]]
                im_gallery = cv2.imread(im_path_gallery)
                im_gallery = cv2.cvtColor(im_gallery, cv2.COLOR_BGR2RGB)
                im_gallery = cv2.resize(im_gallery, (200, 400))
                im_gallery = cv2.copyMakeBorder(im_gallery, 50, 10, 5, 5, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                label = os.path.splitext(os.path.basename(im_path_gallery))[0]
                color = (0, 0, 255)
                cv2.putText(im_gallery, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, lineType=cv2.LINE_AA)
                gallery_images.append(im_gallery)
                if j < 4:
                    gallery_images.append(255 * np.ones((460, 10, 3), dtype=np.uint8))
            im_vis = np.hstack(gallery_images)
            # add a global title on the top of the image
            im_vis = cv2.copyMakeBorder(im_vis, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            cv2.putText(im_vis, "Top 5 closest gallery images", (im_vis.shape[1]//2 - 100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, lineType=cv2.LINE_AA)
            pred_res_filename = f'{prediction_result_dir}/{os.path.basename(img_name)}.png'
            cv2.imwrite(pred_res_filename, cv2.cvtColor(im_vis, cv2.COLOR_RGB2BGR))
            if cfg.general.display_figures:
                cv2.imshow('image', cv2.cvtColor(im_vis, cv2.COLOR_RGB2BGR))
                cv2.waitKey(0)
                cv2.destroyAllWindows()

