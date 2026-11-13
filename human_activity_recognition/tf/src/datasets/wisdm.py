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
from .utils import get_data_segments, preprocess_data, clean_csv
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_wisdm(dataset_path: str,
               class_names: List[str],
               input_shape: Tuple,
               gravity_rot_sup: bool,
               normalization: bool,
               val_split: float,
               test_split: float,
               seed: int,
               batch_size: int,
               to_cache: bool = False):
    """
    Loads the wisdm dataset and return two TensorFlow datasets for training and validation.
    args:
        dataset_path: path to the dataset file
        class_names: list of class names for the dataset
        input_shape: input shape of the model
        gravity_rot_sup: whether to apply gravity and rotation suppression
        normalization: whether to apply normalization
        val_split: ratio for validation dataset split from training dataset
        test_split: ratio for test dataset split from training dataset
        seed: random seed for shuffling and splitting
        batch_size: batch size for the datasets
        to_cache: whether to cache the datasets
    returns:
        train_ds: TensorFlow dataset for training
        valid_ds: TensorFlow dataset for validation
        test_ds: TensorFlow dataset for testing
    """
    # for deployment case where no dataset paths are provided
    if dataset_path is None or dataset_path == "":
        return None, None, None

    clean_csv(dataset_path)

    # read all the data in csv 'WISDM_ar_v1.1_raw.txt' into a dataframe
    #  called dataset
    columns = ['User', 'Activity_Label', 'Arrival_Time',
               'x', 'y', 'z']  # headers for the columns

    dataset = pd.read_csv(dataset_path, header=None,
                          names=columns, delimiter=',')

    # removing the ; at the end of each line and casting the last variable
    #  to datatype float from string
    dataset['z'] = [float(str(char).replace(";", "")) for char in dataset['z']]

    # remove the user column as we do not need it
    dataset = dataset.drop('User', axis=1)

    # as we are workign with numbers, let us replace all the empty columns
    # entries with nan (not a number)
    dataset.replace(to_replace='null', value=np.nan)

    # remove any data entry which contains nan as a member
    dataset = dataset.dropna(axis=0, how='any')
    if len(class_names) == 4:
        dataset['Activity_Label'] = ['Stationary' if activity == 'Standing' or activity == 'Sitting' else activity
                                     for activity in dataset['Activity_Label']]
        dataset['Activity_Label'] = ['Stairs' if activity == 'Upstairs' or activity == 'Downstairs' else activity
                                     for activity in dataset['Activity_Label']]

    # removing the columns for time stamp and rearranging remaining columns
    dataset = dataset[['x', 'y', 'z', 'Activity_Label']]
    segments, labels = get_data_segments(dataset=dataset,
                                         seq_len=input_shape[0])

    segments = preprocess_data(segments, gravity_rot_sup, normalization)
    labels = to_categorical([class_names.index(label)
                            for label in labels], num_classes=len(class_names))

    # split data into train and test
    train_x, test_x, train_y, test_y = train_test_split(segments, labels,
                                                        test_size=test_split,
                                                        shuffle=True,
                                                        random_state=seed)
    # split data into train and valid
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y,
                                                          test_size=val_split,
                                                          shuffle=True,
                                                          random_state=seed)
    train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    if batch_size is None:
        batch_size=32
    train_ds = train_ds.shuffle(train_x.shape[0],
                                reshuffle_each_iteration=True,
                                seed=seed).batch(batch_size)
    valid_ds = tf.data.Dataset.from_tensor_slices((valid_x, valid_y))
    valid_ds = valid_ds.shuffle(valid_x.shape[0],
                                reshuffle_each_iteration=True,
                                seed=seed).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    test_ds = test_ds.shuffle(test_x.shape[0],
                              reshuffle_each_iteration=True,
                              seed=seed).batch(batch_size)
    if to_cache:
        train_ds = train_ds.cache()
        valid_ds = valid_ds.cache()
        test_ds = test_ds.cache()
    return train_ds, valid_ds, test_ds