# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
import tensorflow as tf
import pandas as pd
import numpy as np

from typing import Tuple, List
import pandas as pd
from tqdm import tqdm
from .utils import get_data_segments, preprocess_data, read_pkl_dataset
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_mobility_v1(train_path: str,
                     test_path: str,
                     validation_split: float,
                     class_names: List[str],
                     input_shape: Tuple[int],
                     gravity_rot_sup: bool,
                     normalization: bool,
                     batch_size: int,
                     seed: int,
                     to_cache: bool = False):
    """
    Loads the mobility dataset and return two TensorFlow datasets for training, validation and test.
    args:
        train_path: path to the training dataset pickle file
        test_path: path to the test dataset pickle file
        validation_split: ratio for validation dataset split from training dataset
        class_names: list of class names for the dataset
        input_shape: input shape of the model
        gravity_rot_sup: whether to apply gravity and rotation suppression
        normalization: whether to apply normalization
        batch_size: batch size for the datasets
        seed: random seed for shuffling and splitting
        to_cache: whether to cache the datasets
    returns:
        train_ds: TensorFlow dataset for training
        valid_ds: TensorFlow dataset for validation
        test_ds: TensorFlow dataset for testing
    """
    # for deployment operation mode
    if train_path is None and test_path is None:
        return None, None, None

    train_dataset = read_pkl_dataset(train_path, class_names)
    test_dataset = read_pkl_dataset(test_path, class_names)

    train_dataset[train_dataset.columns[:3]] = train_dataset[train_dataset.columns[:3]] * 9.8
    test_dataset[test_dataset.columns[:3]] = test_dataset[test_dataset.columns[:3]] * 9.8

    print('[INFO] : Building train segments!')
    train_segments, train_labels = get_data_segments(dataset=train_dataset,
                                                     seq_len=input_shape[0])
    print('[INFO] : Building test segments!')
    train_segments = preprocess_data(train_segments, gravity_rot_sup, normalization)
    train_labels = to_categorical([class_names.index(label)
                                  for label in train_labels], num_classes=len(class_names))
    test_segments, test_labels = get_data_segments(dataset=test_dataset,
                                                   seq_len=input_shape[0])
    test_segments = preprocess_data(test_segments, gravity_rot_sup, normalization)
    test_labels = to_categorical([class_names.index(label)
                                  for label in test_labels], num_classes=len(class_names))

    train_x, valid_x, train_y, valid_y = train_test_split(train_segments, train_labels,
                                                          test_size=validation_split,
                                                          shuffle=True,
                                                          random_state=seed)
    train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    train_ds = train_ds.shuffle(train_x.shape[0],
                                reshuffle_each_iteration=True,
                                seed=seed).batch(batch_size)
    valid_ds = tf.data.Dataset.from_tensor_slices((valid_x, valid_y))
    valid_ds = valid_ds.shuffle(valid_x.shape[0],
                                reshuffle_each_iteration=True,
                                seed=seed).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((test_segments, test_labels))
    test_ds = test_ds.shuffle(test_segments.shape[0],
                              reshuffle_each_iteration=True,
                              seed=seed).batch(batch_size)

    if to_cache:
        train_ds = train_ds.cache()
        test_ds = test_ds.cache()
        valid_ds = valid_ds.cache()
    return train_ds, valid_ds, test_ds