# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

from omegaconf import DictConfig
from typing import Tuple, Dict, Union
from arc_fault_detection.tf.src.preprocessing import downsample_data
from sklearn.model_selection import train_test_split
import pandas as pd
import zipfile
import os
import numpy as np
import tensorflow as tf


def load_afd_test_bench(cfg: DictConfig = None) -> Dict:
    """
    load afd_test_bench data based on the provided configuration. It builds the
    tf.data.Dataset objects according to the operation_mode, model.input_shape
    and preprocessing parameters.

    Args:
        cfg (DictConfig): Configuration object containing the settings.

    Returns:
        Dict: A dictionary containing the following:
            - train_ds: tf.data.Dataset, Training dataset.
            - valid_ds: tf.data.Dataset, Validation dataset.
            - test_ds: tf.data.Dataset, Test dataset.
            - quantization_ds: tf.data.Dataset, Quantization dataset.
            - predict_ds: np.ndarray,  Prediction dataset.
    """

    mode = cfg.operation_mode
    training_path=cfg.dataset.training_path
    validation_path=cfg.dataset.validation_path
    test_path=cfg.dataset.test_path
    quantization_path=cfg.dataset.quantization_path
    prediction_path=cfg.dataset.prediction_path
    validation_split=cfg.dataset.validation_split
    test_split=cfg.dataset.test_split
    quantization_split = getattr(cfg.dataset, "quantization_split", None)
    quantization_split = 1.0 if quantization_split is None else quantization_split
    print(f"[INFO] : quantization_split {quantization_split}, {mode} mode with the following paths: \n")
    # Adjust paths based on operation mode to avoid loading unnecessary datasets
    if mode == 'benchmarking':
        training_path = None
        validation_path = None
        test_path = None
        quantization_path = None
        prediction_path = None
    elif mode == 'prediction':
         if not cfg.preprocessing.normalization:
                training_path = None
         validation_path = None
         validation_split = 0.0
         test_path = None
         test_split = 0.0
         quantization_path = None
    elif mode == 'evaluation':
         if not cfg.preprocessing.normalization:
                training_path = None
         validation_path = None
         validation_split = 0.0
         quantization_path = None
         prediction_path = None
    elif mode in ['chain_eqe', 'chain_eqeb']:
         if not cfg.preprocessing.normalization and quantization_path:
                training_path = None
         validation_path = None
         validation_split = 0.0
         prediction_path = None
    elif mode in ['quantization', 'chain_qb']:
        if not cfg.preprocessing.normalization and quantization_path:
                training_path = None
        validation_path = None
        validation_split = 0.0
        test_path = None
        test_split = 0.0
        prediction_path = None
    elif mode in ['training', 'chain_tb']:
        quantization_path = None
        prediction_path = None
    else: 
        prediction_path = None

    # Set default batch size        
    batch_size = 32
    if hasattr(cfg, 'training') and cfg.training is not None:
        if hasattr(cfg.training, 'batch_size') and cfg.training.batch_size is not None:
            batch_size = cfg.training.batch_size
            
    # check if the batch dimension is included in the input shape and remove it if present
    input_shape = cfg.model.input_shape
    if len (input_shape) == 4 : input_shape = input_shape[1:]

    train_ds, valid_ds, test_ds, quantization_ds, predict_ds = load_dataset(
                                                mode=mode,
                                                training_path=training_path,
                                                validation_path=validation_path,
                                                test_path=test_path,
                                                test_split=test_split,
                                                validation_split=validation_split,
                                                quantization_path=quantization_path,
                                                quantization_split=quantization_split,
                                                prediction_path=prediction_path,
                                                input_shape=input_shape,
                                                downsampling=cfg.preprocessing.downsampling,
                                                normalization=cfg.preprocessing.normalization,
                                                time_domain=cfg.preprocessing.time_domain,
                                                seed=cfg.dataset.seed,
                                                batch_size=batch_size,
                                                to_cache=cfg.dataset.to_cache
                                                )

    return {'train': train_ds, 'valid': valid_ds, 'quantization': quantization_ds, 'test': test_ds, 'predict': predict_ds}

def preprocess_data(X: np.ndarray, y: np.ndarray, input_shape: Tuple[int], downsampling: bool, 
                     time_domain: bool, normalization) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
    """ Preprocess input data by downsampling and normalizing.

    Args:
        X (np.ndarray): Input data.
        y (np.ndarray): Labels.
        input_shape (Tuple[int]): Desired input shape.
            (n_channels, seq_len, 1) for Conv2D or dense architectures.
        downsampling (bool): Flag to enable downsampling.
        time_domain (bool): Flag to indicate if data should be processed in time domain.
        normalization (Tuple[float] or bool): Parameters for normalization (mean, std) or a boolean flag.

    Returns:
        If normalization is True:
            X_processed, y, (mean, std)
        else:
            X_processed, y
    """

    seq_len = input_shape[1]
    n_channels = input_shape[0]
    if not time_domain:
         target_length = seq_len * 2 
    else:
         target_length = seq_len
         
    if downsampling:
        X = downsample_data(X, target_length = target_length)
    else:
        if  X.shape[1] < target_length:
            raise ValueError(f"Data width {X.shape[1]} is less than or equal to target length {target_length}. \
                             loading data is not possible.")
        X = X[:, :target_length]

    if not time_domain:
        # rfft output length is target_length/2 + 1; we only keep seq_len bins.
        X = abs(np.fft.rfft(X))[:, :seq_len] 
    
    if type(normalization) == bool and normalization:
        if not time_domain:
            # compute normalization parameters on frequency domain data
            normalization_params = (np.mean(X, axis=0), np.std(X, axis=0) + 1e-8)
        else:
            # compute normalization parameters on time domain data, mean and std over all samples
            normalization_params = (np.mean(X), np.std(X) + 1e-8)    
        X = (X - normalization_params[0]) / normalization_params[1]
    elif type(normalization) == tuple:  
        X = (X - normalization[0]) / normalization[1]

    # reshape X to match multichannel dataset format (n_samples/n_channels, n_channels, seq_len)
    # X.shape[0] must be divisible by n_channels
    reminder = X.shape[0] % n_channels
    if reminder != 0:
        print("[WARNING] : Reshaping the data to match the multichannel format may lead to data loss. "
                f"n_samples={X.shape[0]}, n_channels={n_channels}" )
        X = X[:-reminder, :]
        y = y[:-reminder] if y is not None else y

   # reshape and expand dims
    X = X.reshape((-1, n_channels, seq_len))
    X = np.expand_dims(X, axis=-1)
    
    #reshape y to match multichannel dataset format (n_samples/n_channels, )
    y= y.reshape((-1, n_channels)) if y is not None else y

    if type(normalization) == bool and normalization:
        return X, y, normalization_params
    else:
        return X, y

def load_file_from_zip(path, prediction=False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads a CSV file, extracting from a zip if necessary.
    For prediction, returns all columns as X.
    For training/validation/test, returns X (features) and Y (labels).

    Args:
        path (str): Path to the CSV file, e.g., 'afd_test_bench/train.csv'
        prediction (bool): If True, return all columns as X. If False, return X and Y.

    Returns:
        np.ndarray: X (features, float32), None if prediction=True
        np.ndarray: Y (labels, int32) [only if prediction=False]
    """

    # Check if the CSV file exists
    if not os.path.exists(path):
        # Try to extract from zip
        folder = os.path.dirname(path)
        zip_path = folder + '.zip'
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extract(os.path.relpath(path, folder), folder)
            print(f"[INFO] : Extracted {os.path.relpath(path, folder)} to {folder}")

    # Now load the CSV
    df = pd.read_csv(path, header=None)
    if prediction:
        X = df.values.astype(np.float32)
        return X, None
    else:
        X = df.iloc[:, :-1].values.astype(np.float32)
        Y = df.iloc[:, -1].values.astype(np.int32)
        return X, Y

def split_per_channel_with_group_reshape(X: np.ndarray, Y: np.ndarray,split_size: float, 
                                         seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split multichannel data by treating each channel as an independent sample
    for stratification, then regroup channels into blocks so that the final
    tensors have shape (n_groups, n_channels, seq_len, 1) for X and
    (n_groups, n_channels) for Y.

    Args:
        X (np.ndarray): Input data of shape (n_samples, n_channels, seq_len, 1).
        Y (np.ndarray): Labels of shape (n_samples, n_channels) or flattened.
        split_size (float): Fraction of the data used for test split.
        seed (int): Random seed for reproducibility.

    Returns:
        X_train (np.ndarray): Training data of shape (n_groups_train, n_channels, seq_len, 1).
        X_test (np.ndarray): Test data of shape (n_groups_test, n_channels, seq_len, 1).
        Y_train (np.ndarray): Training labels of shape (n_groups_train, n_channels).
        Y_test (np.ndarray): Test labels of shape (n_groups_test, n_channels).
    """
    # Reference shapes
    n_samples, n_channels, seq_len, _ = X.shape  # (N, C, L, 1)

    # Each channel becomes an independent sample
    X_train_flat = X.reshape(n_samples * n_channels, seq_len, 1)   # (N*C, L, 1)
    Y_train_flat = Y.reshape(n_samples * n_channels)               # (N*C,)

    # Split with stratification on channels
    X_train_f, X_test_f, Y_train_f, Y_test_f = train_test_split(
        X_train_flat,
        Y_train_flat,
        test_size=split_size,
        random_state=seed,
        stratify=Y_train_flat,
    )

    # Make first split a multiple of n_channels by borrowing from the second
    r = len(X_train_f) % n_channels
    if r != 0:
        needed = n_channels - r
        X_train_f = np.concatenate([X_train_f, X_test_f[:needed]], axis=0)
        Y_train_f = np.concatenate([Y_train_f, Y_test_f[:needed]], axis=0)
        X_test_f = X_test_f[needed:]
        Y_test_f = Y_test_f[needed:]

    # Reshape back into channel groups
    X_train = X_train_f.reshape(-1, n_channels, seq_len, 1)
    Y_train = Y_train_f.reshape(-1, n_channels)
    X_test = X_test_f.reshape(-1, n_channels, seq_len, 1)
    Y_test = Y_test_f.reshape(-1, n_channels)

    return X_train, X_test, Y_train, Y_test
    
def load_dataset(
    mode: str,
    training_path: str,
    validation_path: str = None,
    test_path: str = None,
    test_split: float = 0.2,
    validation_split: float = 0.2,
    quantization_path: str = None,
    quantization_split: float = None,
    prediction_path: str = None,
    input_shape: Tuple = None,
    downsampling: bool = True,
    normalization: bool = False,
    time_domain: bool = False,
    seed: int = None,
    batch_size: int = 32,
    to_cache: bool = False
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, np.ndarray]:
    """ Loads a dataset. Returns a training dataset, validation dataset and optionally a quantization
    and test dataset. 

    Inputs
    ------
    mode (str): operation mode.
    training_path : str, path to the training set .csv file.
    validation_path : str, path to the validation set .csv file.
        If None, the training set is split into training and validation sets.
    validation_split : float. If validation_csv_path is None, split the training set into 
        training/validation set according to the value of this arg.
    quantization_path : str, path to the quantization set .csv file.
        If None, no quantization dataset is returned.
    quantization_split : float. the fraction of quantization/training dataset to use for quantization.
    test_path :Path to the test set .csv file. If None, no test dataset is returned.
    test_split (float): Fraction of the data to use for test.
    downsampling (bool): a flag to downsample data
    input_shape (tuple[int]): shape of the input 
    normalization (bool): a flag to implement standard normalization data
    time_domain (bool): a flag to indicate if data should be processed in time domain
    seed (int): Seed to use for shuffling the data.
    batch_size (int): Batch size to use for the datasets.
    to_cache (boolean): flag to cache the tensorflow_datasets
    
    Outputs
    -------
    train_dataset : tf.data.Dataset, training dataset
    val_dataset : tf.data.Dataset, validation dataset
    test_dataset : tf.data.Dataset, Test dataset.
    quantization_dataset : tf.data.Dataset, quantization dataset,
    predict_dataset : np.ndarray, prediction dataset,
    """

    train_dataset, val_dataset, test_dataset, quantization_dataset, predict_dataset = None, None, None, None, None
    X_train_full, Y_train_full = None, None
    normalization_params = None

    if normalization:
        X_train_full, Y_train_full = load_file_from_zip(training_path)
        X_train_full, Y_train_full, normalization_params = preprocess_data(X_train_full, Y_train_full, input_shape, downsampling, time_domain, normalization)

        
    if prediction_path:
        predict_dataset, _ = load_file_from_zip(prediction_path, prediction=True)
        predict_dataset, _ = preprocess_data(predict_dataset, None, input_shape, downsampling, time_domain, normalization_params)    

    if training_path:
        if X_train_full is None:
            X_train_full, Y_train_full = load_file_from_zip(training_path)
            # Preprocess features
            X_train_full, Y_train_full = preprocess_data(X_train_full, Y_train_full, input_shape, downsampling, time_domain, normalization_params)

        X_train, Y_train = X_train_full, Y_train_full
        # Validation split logic
        if validation_path:
            X_val, Y_val = load_file_from_zip(validation_path)
            X_val, Y_val = preprocess_data(X_val, Y_val, input_shape, downsampling, time_domain, normalization_params)   
        elif validation_split > 0.0:
            X_train, X_val, Y_train, Y_val = split_per_channel_with_group_reshape(X_train, Y_train, validation_split, seed)

        # Test split logic
        if test_path:
            X_test, Y_test = load_file_from_zip(test_path)
            X_test, Y_test = preprocess_data(X_test, Y_test, input_shape, downsampling, time_domain, normalization_params)
        elif test_split > 0.0:
            X_train, X_test, Y_train, Y_test = split_per_channel_with_group_reshape(X_train, Y_train, test_split, seed)
            
        # Create TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        print("Train class distribution:", np.bincount(Y_train.flatten()))
        train_dataset = train_dataset.cache() if to_cache else train_dataset
        train_dataset = train_dataset.shuffle(buffer_size=len(X_train), seed=seed).batch(batch_size)

        if validation_path or validation_split > 0.0:
            val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
            print("Validation class distribution:", np.bincount(Y_val.flatten()))
            val_dataset = val_dataset.cache() if to_cache else val_dataset
            val_dataset = val_dataset.batch(batch_size)

        if test_path or test_split > 0.0:
            test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
            print("Test class distribution:", np.bincount(Y_test.flatten()))
            test_dataset = test_dataset.cache() if to_cache else test_dataset
            test_dataset = test_dataset.batch(batch_size)

    if quantization_path and quantization_split > 0.0:
            X_quant, Y_quant = load_file_from_zip(quantization_path)
            X_quant, Y_quant = preprocess_data(X_quant, Y_quant, input_shape, downsampling, time_domain, normalization_params)
            quantization_dataset = tf.data.Dataset.from_tensor_slices((X_quant, Y_quant))
            print("Quantization class distribution:", np.bincount(Y_quant.flatten()))
            quantization_dataset = quantization_dataset.cache()
            quantization_dataset = quantization_dataset.batch(batch_size)
            print(f'[INFO] : Quantization will be performed by using {quantization_split * 100} % of the provided dataset...')
            quantization_dataset = quantization_dataset.take(int(len(quantization_dataset) * float(quantization_split)))
    elif  training_path and mode in ['chain_eqe','chain_tbqeb', 'chain_tqe', 'chain_eqeb', 'quantization', 'chain_qb'] and quantization_split > 0.0:
            print(f'[INFO] : Quantization will be performed by using {quantization_split * 100} % of the training dataset...')
            quantization_dataset = train_dataset
            quantization_dataset = quantization_dataset.take(int(len(quantization_dataset) * float(quantization_split)))
    elif mode in ['chain_eqe','chain_tbqeb', 'chain_tqe', 'chain_eqeb', 'quantization', 'chain_qb']:
            print(f'[INFO] : No quantization dataset provided or quantization_split is set to 0. Quantization will be performed by using fake dataset...')


    if validation_path and not training_path:
                    X_val, Y_val = load_file_from_zip(validation_path)
                    X_val, Y_val = preprocess_data(X_val, Y_val, input_shape, downsampling, time_domain, normalization_params)
                    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
                    print("Validation class distribution:", np.bincount(Y_val.flatten()))
                    val_dataset = val_dataset.cache() if to_cache else val_dataset
                    val_dataset = val_dataset.batch(batch_size)
                    
    if test_path and not training_path:
                    X_test, Y_test = load_file_from_zip(test_path)
                    X_test, Y_test = preprocess_data(X_test, Y_test, input_shape, downsampling, time_domain, normalization_params)
                    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
                    print("Test class distribution:", np.bincount(Y_test.flatten()))
                    test_dataset = test_dataset.cache() if to_cache else test_dataset
                    test_dataset = test_dataset.batch(batch_size)

    return train_dataset, val_dataset, test_dataset, quantization_dataset, predict_dataset
