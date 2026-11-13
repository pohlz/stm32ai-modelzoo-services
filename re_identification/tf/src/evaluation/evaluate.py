# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
from pathlib import Path
import warnings
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from typing import Tuple, Optional, List
import numpy as np
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import onnxruntime
import tensorflow as tf
import mlflow
import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix

from re_identification.tf.src.preprocessing import apply_rescaling, postprocess_output, preprocess_input
from re_identification.tf.src.utils import ai_runner_invoke, get_loss, pairwise_distance, calculate_rank_accuracy, calculate_map
from common.evaluation import model_is_quantized, predict_onnx_batch
from common.utils import tf_dataset_to_np_array, compute_confusion_matrix, count_h5_parameters, \
                         ai_runner_interp, ai_interp_input_quant, ai_interp_outputs_dequant, \
                         plot_confusion_matrix, log_to_file

def _plot_cumulative_matching_characteristics_curve(accuracy_rank_n: list, map: float,
                           output_dir: str, model_name: str):
    plt.figure(figsize=(14, 8))
    n_rank = len(accuracy_rank_n)
    plt.plot(range(1, n_rank + 1), accuracy_rank_n, marker='o')
    plt.xticks(range(1, n_rank + 1, 5))
    plt.xlabel('Rank')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Re-identification Cumulative Matching Characteristics (CMC) Curve\n{model_name}')
    plt.text(0.8, 0.2, f'mAP: {map}%', transform=plt.gca().transAxes, fontsize=20,
             bbox=dict(facecolor='white', alpha=0.5))
    plt.grid()
    plt.savefig(os.path.join(output_dir, f'{model_name}_reid_rank_curve.png'))
    plt.close()

def _plot_histogram_of_distances_for_same_and_different_identities(distances: np.ndarray,
                                                                   query_labels: np.ndarray,
                                                                   gallery_labels: np.ndarray,
                                                                   output_dir: str,
                                                                   model_name: str) -> float:
    same_id_distances = []
    different_id_distances = []
    for i in range(distances.shape[0]):
        for j in range(distances.shape[1]):
            if query_labels[i] == gallery_labels[j]:
                same_id_distances.append(distances[i][j])
            else:
                different_id_distances.append(distances[i][j])
    # calculate the best threshold to separate same and different identities
    # Vectorized threshold search for best balanced accuracy
    same_id_distances = np.array(same_id_distances)
    different_id_distances = np.array(different_id_distances)
    all_distances = np.concatenate([same_id_distances, different_id_distances])
    y_true = np.concatenate([np.ones_like(same_id_distances), np.zeros_like(different_id_distances)])
    thresholds = np.linspace(min(all_distances), max(all_distances), num=100)
    # shape: (num_thresholds, num_samples)
    y_pred_matrix = (all_distances[None, :] <= thresholds[:, None]).astype(int)
    y_true_matrix = np.broadcast_to(y_true, y_pred_matrix.shape)
    # Compute TP, TN, FP, FN for each threshold
    tp = np.sum((y_pred_matrix == 1) & (y_true_matrix == 1), axis=1)
    tn = np.sum((y_pred_matrix == 0) & (y_true_matrix == 0), axis=1)
    fp = np.sum((y_pred_matrix == 1) & (y_true_matrix == 0), axis=1)
    fn = np.sum((y_pred_matrix == 0) & (y_true_matrix == 1), axis=1)
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        tpr = np.where((tp + fn) > 0, tp / (tp + fn), 0)
        tnr = np.where((tn + fp) > 0, tn / (tn + fp), 0)
        balanced_accuracy = (tpr + tnr) / 2
    # Mask out thresholds where either class is missing
    valid = ((tp + fn) > 0) & ((tn + fp) > 0)
    if np.any(valid):
        best_idx = np.argmax(balanced_accuracy * valid)
        best_threshold = thresholds[best_idx]
    else:
        best_threshold = 0.0

    plt.figure(figsize=(14, 8))
    # display histogram of distances based on density
    plt.hist(np.array(same_id_distances), bins=50, alpha=0.5, label='Same Identity', density=True)
    plt.hist(np.array(different_id_distances), bins=50, alpha=0.5, label='Different Identity', density=True)
    plt.axvline(x=best_threshold, color='r', linestyle='--', label=f'Best Threshold: {best_threshold:.3f}')
    plt.xlabel('Distance')
    plt.ylabel('Density')
    plt.title(f'Histogram of Distances for Same and Different Identities\n{model_name}')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, f'{model_name}_reid_distance_histogram.png'))
    plt.close()

    return best_threshold

def evaluate_h5_model(model_path: str = None, eval_ds: tf.data.Dataset = None, class_names: list = None,
                      output_dir: str = None, name_ds: Optional[str] = 'test_set',
                      display_figures: bool = None) -> float:
    """
    Evaluates a trained Keras model saved in .h5 or .keras format on the provided test data.

    Args:
        model_path (str): The file path to the .h5 or .keras model.
        eval_ds (tf.data.Dataset): The test data to evaluate the model on.
        class_names (list): A list of class names for the confusion matrix.
        output_dir (str): The directory where to save the image.
        name_ds (str): The name of the chosen eval_ds to be mentioned in the prints and figures.
    Returns:
        float: The accuracy of the model on the test data.
    """

    # Load the .h5 or .keras model
    model = tf.keras.models.load_model(model_path)
    loss = get_loss(num_classes=len(class_names))
    model.compile(loss=loss, metrics=['accuracy'])

    # Evaluate the model on the test data
    tf.print(f'[INFO] : Evaluating the float model using {name_ds}...')
    loss, accuracy = model.evaluate(eval_ds)
    # compute confusion matrix
    cm, accuracy = compute_confusion_matrix(test_set=eval_ds, model=model)

    if display_figures:
        print(f"[INFO] : Plotting confusion matrix for float model on {name_ds}...")
        model_name = f"float_model_confusion_matrix_{name_ds}"
        plot_confusion_matrix(cm=cm,
                              class_names=class_names,
                              model_name=model_name,
                              title=f'{model_name}\naccuracy: {accuracy}',
                              output_dir=output_dir)
    
    print(f"[INFO] : Accuracy of float model on {name_ds} = {accuracy}%")
    print(f"[INFO] : Loss of float model on {name_ds} = {loss}")
    mlflow.log_metric(f"float_acc_{name_ds}", accuracy)
    mlflow.log_metric(f"float_loss_{name_ds}", loss)
    log_to_file(output_dir, f"Float model {name_ds}:")
    log_to_file(output_dir, f"Accuracy of float model : {accuracy} %")
    log_to_file(output_dir, f"Loss of float model : {round(loss,2)} ")
    return accuracy

def evaluate_h5_model_reid(model_path: str = None, eval_query_ds: tf.data.Dataset = None, eval_gallery_ds: tf.data.Dataset = None,
                           output_dir: str = None, distance_metric: str = 'euclidean',
                           name_ds: Optional[str] = 'test_set',
                           display_figures: bool = None) -> float:
    """
    Evaluates a trained Keras model saved in .h5 or .keras format on the provided test data.

    Args:
        model_path (str): The file path to the .h5 or .keras model.
        eval_query_ds (tf.data.Dataset): The test query data to evaluate the model on.
        eval_gallery_ds (tf.data.Dataset): The test gallery data to evaluate the model on.
        class_names (list): A list of class names for the confusion matrix.
        output_dir (str): The directory where to save the image.
        name_ds (str): The name of the chosen eval_ds to be mentioned in the prints and figures.
    Returns:
        float: The rank-1, rank-5, rank-10 accuracy and mAP of the model on the test data.
    """

    # Load the .h5 or .keras model
    model = tf.keras.models.load_model(model_path)
    model_name = os.path.basename(model_path)
    # loss = get_loss(num_classes=len(class_names))
    # model.compile(loss=loss, metrics=['accuracy'])

    # # remove the last layer (softmax) for re-identification
    # model = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)

    # Prepare dataset with only images and batch it
    query_images = eval_query_ds.map(lambda x, y: x)
    gallery_images = eval_gallery_ds.map(lambda x, y: x)

    # Evaluate the model on the test data
    tf.print(f'[INFO] : Evaluating the float model using {name_ds}...')
    # extract features
    query_ds, query_labels = tf_dataset_to_np_array(eval_query_ds, model)
    query_features = model.predict(query_images, verbose=1)
    gallery_ds, gallery_labels = tf_dataset_to_np_array(eval_gallery_ds, model)
    gallery_features = model.predict(gallery_images, verbose=1)
    
    # compute pairwise distances
    distances = pairwise_distance(query_features, gallery_features, distance_metric=distance_metric)
    print(f'[INFO] : Computed pairwise distances shape: {distances.shape}')
    accuracy_rank_n = []
    for i in range(min(100, len(gallery_labels)//2)):
        accuracy_rank_n.append(round(calculate_rank_accuracy(distances, query_labels, gallery_labels, top_k=i+1)*100, 2))
    accuracy_rank1 = accuracy_rank_n[0]
    accuracy_rank5 = accuracy_rank_n[4]
    accuracy_rank10 = accuracy_rank_n[9]
    print(f'[INFO] : Evaluation Rank-1 accuracy: {accuracy_rank1} %')
    print(f'[INFO] : Evaluation Rank-5 accuracy: {accuracy_rank5} %')
    print(f'[INFO] : Evaluation Rank-10 accuracy: {accuracy_rank10} %')

    # compute mAP       
    mAP = round(calculate_map(distances, query_labels, gallery_labels)*100, 2)
    print(f'[INFO] : Evaluation mAP: {mAP} %')

    log_to_file(output_dir,  f"Float model {name_ds}:")
    log_to_file(output_dir, f"Rank-1 accuracy of float model : {accuracy_rank1} %")
    log_to_file(output_dir, f"Rank-5 accuracy of float model : {accuracy_rank5} %")
    log_to_file(output_dir, f"Rank-10 accuracy of float model : {accuracy_rank10} %")
    log_to_file(output_dir, f"mAP of float model : {mAP} %")
    mlflow.log_metric(f"float_acc_rank1_{name_ds}", accuracy_rank1)
    mlflow.log_metric(f"float_acc_rank5_{name_ds}", accuracy_rank5)
    mlflow.log_metric(f"float_acc_rank10_{name_ds}", accuracy_rank10)
    mlflow.log_metric(f"float_mAP_{name_ds}", mAP)
    if display_figures:
        _plot_cumulative_matching_characteristics_curve(accuracy_rank_n, mAP, output_dir, model_name)
        _plot_histogram_of_distances_for_same_and_different_identities(distances, query_labels, gallery_labels, output_dir, model_name)
    return accuracy_rank1, accuracy_rank5, accuracy_rank10, mAP


def _inference_tflite_dataset(cfg: DictConfig = None, eval_ds: tf.data.Dataset = None,
                              input_details: dict = None, output_details: dict = None,
                              interpreter_quant: tf.lite.Interpreter = None,
                              ai_runner_interpreter = None,
                              input_index_quant: int = None, output_index_quant: int = None,
                              target: str = 'host') -> Tuple[np.ndarray, np.ndarray]:
    test_pred = []
    test_labels = []
    for images, labels in tqdm.tqdm(eval_ds, total=len(eval_ds)):
        for image, label in zip(images, labels):
            image_processed = preprocess_input(image, input_details)
            if target == 'host':
                interpreter_quant.set_tensor(input_index_quant, image_processed)
                interpreter_quant.invoke()
                test_pred_features = interpreter_quant.get_tensor(output_index_quant)
            elif target in ['stedgeai_host', 'stedgeai_n6', 'stedgeai_h7p']:
                image_preproc = ai_interp_input_quant(ai_runner_interpreter, image[None].numpy(),
                                                      '.tflite')
                test_pred_features = ai_runner_invoke(image_preproc, ai_runner_interpreter)
                test_pred_features = ai_interp_outputs_dequant(ai_runner_interpreter, [test_pred_features])[0]
                test_pred_features = np.reshape(test_pred_features, [1, -1])
            test_pred.append(test_pred_features)
            test_labels.append(label.numpy())
    test_pred = np.concatenate(test_pred, axis=0)
    test_labels = np.array(test_labels)
    return test_pred, test_labels

def _evaluate_tflite_quantized_model_reid(cfg: DictConfig = None,
                                          quantized_model_path: str = None, 
                                          eval_query_ds: tf.data.Dataset = None,
                                          eval_gallery_ds: tf.data.Dataset = None,
                                          output_dir: str = None,
                                          distance_metric: str = 'euclidean',
                                          name_ds: Optional[str] = 'test_set',
                                          num_threads: Optional[int] = 1,
                                          display_figures: bool = None) -> float:
    """
    Evaluates the accuracy of a quantized TensorFlow Lite model using tflite.interpreter and plots the confusion matrix.
    Args:
        cfg (config): The configuration file.
        quantized_model_path (str): The file path to the quantized TensorFlow Lite model.
        eval_query_ds (tf.data.Dataset): The test query dataset to evaluate the model on.
        eval_gallery_ds (tf.data.Dataset): The test gallery dataset to evaluate the model on.
        class_names (list): A list of class names for the confusion matrix.
        output_dir (str): The directory where to save the image.
        distance_metric (str): Distance metric to be used for pairwise distance computation. Choices are 'euclidean' or 'cosine'.
        name_ds (str): The name of the chosen eval_ds to be mentioned in the prints and figures.
        num_threads (int): number of threads for the tflite interpreter
    Returns:
        float: The accuracy of the quantized model.
    """
    tf.print(f'[INFO] : Evaluating the quantized model using {name_ds}...')
    if cfg.evaluation and cfg.evaluation.target:
        target = cfg.evaluation.target
    else:
        target = "host"
    model_name = os.path.basename(quantized_model_path)
    ai_runner_interpreter = ai_runner_interp(target,model_name)
    interpreter_quant = tf.lite.Interpreter(model_path=quantized_model_path, num_threads=num_threads)
    interpreter_quant.allocate_tensors()
    input_details = interpreter_quant.get_input_details()[0]
    input_index_quant = input_details["index"]
    output_index_quant = interpreter_quant.get_output_details()[0]["index"]
    output_details = interpreter_quant.get_output_details()[0]
    query_pred, query_labels = _inference_tflite_dataset(cfg=cfg, eval_ds=eval_query_ds,
                                                        input_details=input_details, output_details=output_details,
                                                        interpreter_quant=interpreter_quant,
                                                        ai_runner_interpreter=ai_runner_interpreter,
                                                        input_index_quant=input_index_quant, output_index_quant=output_index_quant,
                                                        target=target)
    gallery_pred, gallery_labels = _inference_tflite_dataset(cfg=cfg, eval_ds=eval_gallery_ds,
                                                            input_details=input_details, output_details=output_details,
                                                            interpreter_quant=interpreter_quant,
                                                            ai_runner_interpreter=ai_runner_interpreter,
                                                            input_index_quant=input_index_quant, output_index_quant=output_index_quant,
                                                            target=target)
    # compute pairwise distances
    distances = pairwise_distance(query_pred, gallery_pred, distance_metric=distance_metric)
    print(f'[INFO] : Computed pairwise distances shape: {distances.shape}')
    accuracy_rank_n = []
    for i in range(min(100, len(gallery_labels)//2)):
        accuracy_rank_n.append(round(calculate_rank_accuracy(distances, query_labels, gallery_labels, top_k=i+1)*100, 2))
    accuracy_rank1 = accuracy_rank_n[0]
    accuracy_rank5 = accuracy_rank_n[4]
    accuracy_rank10 = accuracy_rank_n[9]
    print(f'[INFO] : Evaluation Rank-1 accuracy: {accuracy_rank1} %')
    print(f'[INFO] : Evaluation Rank-5 accuracy: {accuracy_rank5} %')
    print(f'[INFO] : Evaluation Rank-10 accuracy: {accuracy_rank10} %')
    # compute mAP
    mAP = round(calculate_map(distances, query_labels, gallery_labels)*100, 2)
    print(f'[INFO] : Evaluation mAP: {mAP} %')
    log_to_file(output_dir,  f"Quantized model {name_ds}:")
    log_to_file(output_dir, f"Rank-1 accuracy of quantized model : {accuracy_rank1} %")
    log_to_file(output_dir, f"Rank-5 accuracy of quantized model : {accuracy_rank5} %")
    log_to_file(output_dir, f"Rank-10 accuracy of quantized model : {accuracy_rank10} %")
    log_to_file(output_dir, f"mAP of quantized model : {mAP} %")
    mlflow.log_metric(f"int_acc_rank1_{name_ds}", accuracy_rank1)
    mlflow.log_metric(f"int_acc_rank5_{name_ds}", accuracy_rank5)
    mlflow.log_metric(f"int_acc_rank10_{name_ds}", accuracy_rank10)
    mlflow.log_metric(f"int_mAP_{name_ds}", mAP)
    if display_figures:
        _plot_cumulative_matching_characteristics_curve(accuracy_rank_n, mAP, output_dir, model_name)
        _plot_histogram_of_distances_for_same_and_different_identities(distances, query_labels, gallery_labels, output_dir, model_name)
    return accuracy_rank1, accuracy_rank5, accuracy_rank10, mAP

def _evaluate_onnx_model_reid(cfg: DictConfig,
                              input_query_ds: tf.data.Dataset,
                              input_gallery_ds: tf.data.Dataset,
                              input_model_path: str,
                              output_dir: str,
                              distance_metric: str = 'euclidean',
                              name_ds: Optional[str] = 'test_set',
                              display_figures: bool = None) -> Tuple[float, np.ndarray]:
    """Evaluate an ONNX model for re-identification.
    Args:
        cfg (DictConfig): dict containing all yaml parameters.
        input_query_ds (tf.data.Dataset): input query dataset
        input_gallery_ds (tf.data.Dataset): input gallery dataset
        input_model_path (str): The path to the onnx model to be evaluated.
        class_labels (List[str]): The list of the class labels.
        output_dir(str): The name of the output directory where confusion matrix and the logs are to be saved.
        distance_metric (str): Distance metric to be used for pairwise distance computation. Choices are 'euclidean' or 'cosine'.
        name_ds (str): name of the set on which we evaluate
    Returns:
        Tuple[float, np.ndarray]: The rank-1, rank-5, rank-10 accuracy and mAP of the model on the test data.
    """
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if cfg.evaluation and cfg.evaluation.target:
        target = cfg.evaluation.target
    else:
        target = "host"
    model_name = os.path.basename(input_model_path)
    ai_runner_interpreter = ai_runner_interp(target, model_name)
    sess = onnxruntime.InferenceSession(input_model_path)
    model_type = 'quantized' if model_is_quantized(input_model_path) else 'float'
    if model_type == 'float' or target == 'host':
        query_pred, query_labels = predict_onnx_batch(sess, input_query_ds)
        gallery_pred, gallery_labels = predict_onnx_batch(sess, input_gallery_ds)
    elif (target in ['stedgeai_host', 'stedgeai_n6', 'stedgeai_h7p']) and model_type == 'quantized':
        query_pred = []
        query_ds, query_labels = tf_dataset_to_np_array(input_query_ds)
        for i in tqdm.tqdm(range(query_ds.shape[0])):
            data = ai_interp_input_quant(ai_runner_interpreter, query_ds[i][None],
                                         '.onnx')
            prd_label = ai_runner_invoke(data, ai_runner_interpreter)
            prd_label = ai_interp_outputs_dequant(ai_runner_interpreter, [prd_label])[0]
            query_pred.append(prd_label)
        query_pred = np.array(query_pred, dtype=np.float32)
        gallery_pred = []
        gallery_ds, gallery_labels = tf_dataset_to_np_array(input_gallery_ds)
        for i in tqdm.tqdm(range(gallery_ds.shape[0])):
            data = ai_interp_input_quant(ai_runner_interpreter, gallery_ds[i][None],
                                         '.onnx')
            prd_label = ai_runner_invoke(data, ai_runner_interpreter)
            prd_label = ai_interp_outputs_dequant(ai_runner_interpreter, [prd_label])[0]
            gallery_pred.append(prd_label)
        gallery_pred = np.array(gallery_pred, dtype=np.float32)
    else:
        raise TypeError("Only supported targets are \"host\", \"stedgeai_host\" or \"stedgeai_n6\". "
                        "Check the \"evaluation\" section of your configuration file.")
    # compute pairwise distances
    distances = pairwise_distance(query_pred, gallery_pred, distance_metric=distance_metric)
    print(f'[INFO] : Computed pairwise distances shape: {distances.shape}')
    accuracy_rank_n = []
    for i in range(min(100, len(gallery_labels)//2)):
        accuracy_rank_n.append(round(calculate_rank_accuracy(distances, query_labels, gallery_labels, top_k=i+1)*100, 2))
    accuracy_rank1 = accuracy_rank_n[0]
    accuracy_rank5 = accuracy_rank_n[4]
    accuracy_rank10 = accuracy_rank_n[9]
    print(f'[INFO] : Evaluation Rank-1 accuracy: {accuracy_rank1} %')
    print(f'[INFO] : Evaluation Rank-5 accuracy: {accuracy_rank5} %')
    print(f'[INFO] : Evaluation Rank-10 accuracy: {accuracy_rank10} %')
    # compute mAP
    mAP = round(calculate_map(distances, query_labels, gallery_labels)*100, 2)
    print(f'[INFO] : Evaluation mAP: {mAP} %')
    log_to_file(output_dir,  f"{model_type} onnx model {name_ds}:")
    log_to_file(output_dir, f"Rank-1 accuracy of {model_type} onnx model : {accuracy_rank1} %")
    log_to_file(output_dir, f"Rank-5 accuracy of {model_type} onnx model : {accuracy_rank5} %")
    log_to_file(output_dir, f"Rank-10 accuracy of {model_type} onnx model : {accuracy_rank10} %")
    log_to_file(output_dir, f"mAP of {model_type} onnx model : {mAP} %")
    acc_metric_name = f"int_acc_rank1_{name_ds}" if model_is_quantized(input_model_path) else f"float_acc_rank1_{name_ds}"
    mlflow.log_metric(acc_metric_name, accuracy_rank1)
    acc_metric_name = f"int_acc_rank5_{name_ds}" if model_is_quantized(input_model_path) else f"float_acc_rank5_{name_ds}"
    mlflow.log_metric(acc_metric_name, accuracy_rank5)
    acc_metric_name = f"int_acc_rank10_{name_ds}" if model_is_quantized(input_model_path) else f"float_acc_rank10_{name_ds}"
    mlflow.log_metric(acc_metric_name, accuracy_rank10)
    acc_metric_name = f"int_mAP_{name_ds}" if model_is_quantized(input_model_path) else f"float_mAP_{name_ds}"
    mlflow.log_metric(acc_metric_name, mAP)
    if display_figures:
        _plot_cumulative_matching_characteristics_curve(accuracy_rank_n, mAP, output_dir, model_name)
        _plot_histogram_of_distances_for_same_and_different_identities(distances, query_labels, gallery_labels, output_dir, model_name)
    return accuracy_rank1, accuracy_rank5, accuracy_rank10, mAP

def evaluate(cfg: DictConfig = None, eval_query_ds: tf.data.Dataset = None,
             eval_gallery_ds: tf.data.Dataset = None,
             model: tf.keras.Model = None, name_ds: Optional[str] = 'test_set') -> float:
    """
    Evaluates and benchmarks a TensorFlow Lite or Keras model, and generates a Config header file if specified.

    Args:
        cfg (config): The configuration file.
        eval_query_ds (tf.data.Dataset): The test query dataset to evaluate the model on.
        eval_gallery_ds (tf.data.Dataset): The test gallery dataset to evaluate the model on.
        model (tf.keras.Model): The model to evaluate.
        name_ds (str): The name of the chosen test_data to be mentioned in the prints and figures.

    Returns:
        acc (float): evaluation accuracy
    """
    output_dir = HydraConfig.get().runtime.output_dir
    if cfg.evaluation and 'reid_distance_metric' in cfg.evaluation:
        distance_metric = cfg.evaluation.reid_distance_metric
    else:
        distance_metric = 'cosine'
    print (f"[INFO] : Using `{distance_metric}` distance metric for re-identification evaluation")
    model_path = model.model_path
    file_extension = Path(model_path).suffix

    # Pre-process test dataset
    if eval_query_ds is None or eval_gallery_ds is None:
        raise ValueError("Both eval_query_ds and eval_gallery_ds should be provided for evaluation.")
    eval_query_ds = apply_rescaling(dataset=eval_query_ds, scale=cfg.preprocessing.rescaling.scale,
                                  offset=cfg.preprocessing.rescaling.offset)
    eval_gallery_ds = apply_rescaling(dataset=eval_gallery_ds, scale=cfg.preprocessing.rescaling.scale,
                                  offset=cfg.preprocessing.rescaling.offset)

    try:
        # Check if the model is a TensorFlow Lite model
        if file_extension in ['.h5','.keras']:
            # count_h5_parameters(output_dir=output_dir, 
            #                     model_path=model_path)
            # Evaluate Keras model
            acc_rank1, acc_rank5, acc_rank10, map = evaluate_h5_model_reid(model_path=model_path, 
                                                                           eval_query_ds=eval_query_ds,
                                                                           eval_gallery_ds=eval_gallery_ds,
                                                                           output_dir=output_dir, 
                                                                           distance_metric=distance_metric,
                                                                           name_ds=name_ds, 
                                                                           display_figures=cfg.general.display_figures)
        elif file_extension == '.tflite':
            # Evaluate quantized TensorFlow Lite model
            acc_rank1, acc_rank5, acc_rank10, map = _evaluate_tflite_quantized_model_reid(cfg=cfg, 
                                                                                          quantized_model_path=model_path, 
                                                                                          eval_query_ds=eval_query_ds,
                                                                                          eval_gallery_ds=eval_gallery_ds,
                                                                                          output_dir=output_dir, 
                                                                                          distance_metric=distance_metric,
                                                                                          name_ds=name_ds,
                                                                                          num_threads=cfg.general.num_threads_tflite,
                                                                                          display_figures=cfg.general.display_figures)
        elif file_extension == '.onnx':
            # Evaluate quantized or float ONNX model
            #data, labels = tf_dataset_to_np_array(eval_ds)
            acc_rank1, acc_rank5, acc_rank10, map = _evaluate_onnx_model_reid(cfg=cfg, 
                                                                              input_query_ds=eval_query_ds, 
                                                                              input_gallery_ds=eval_gallery_ds,
                                                                              input_model_path=model_path,
                                                                              output_dir=output_dir, 
                                                                              distance_metric=distance_metric,
                                                                              name_ds=name_ds, 
                                                                              display_figures=cfg.general.display_figures)
    except Exception:
        raise ValueError(f"Model accuracy evaluation failed\nReceived model path: {model_path}")

    return acc_rank1, acc_rank5, acc_rank10, map