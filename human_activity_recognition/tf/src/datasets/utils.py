
# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
from omegaconf import DictConfig
import tensorflow as tf
from typing import Tuple, List
import numpy as np
import statistics
import pandas as pd
from tqdm import tqdm

# preprocessing imports
from scipy.signal import butter
from human_activity_recognition.tf.src.preprocessing import gravity_rotation

def prepare_kwargs_for_dataloader(cfg: DictConfig):
    '''
    prepare_kwargs_for_dataloader prepares the kwargs for the dataloader functions
    args:
        cfg: omegaconf DictConfig object containing the configuration parameters
    returns:
        dataloader_kwargs: dictionary containing the kwargs for the dataloader functions
    '''
    # Extract input shape from the model
    model_path = cfg.model.model_path
    file_extension = str(model_path).split('.')[-1].lower()
    input_shape = cfg.model.input_shape
    if cfg.model.model_path and file_extension in ['h5', 'keras']:
        model_tmp = tf.keras.models.load_model(model_path, compile=False)
        input_shape = model_tmp.inputs[0].shape[1:]
    else:
        raise NotImplementedError(f"Model format '{file_extension}' not supported for dataloader preparation.")
    
    print("input_shape=", input_shape)
    
    # Prepare kwargs
    batch_size = getattr(cfg.training, 'batch_size', 32) if cfg.training else 32
    dataloader_kwargs = {
        'training_path': getattr(cfg.dataset, 'training_path', None),
        'validation_path': getattr(cfg.dataset, 'validation_path', None),
        'test_path': getattr(cfg.dataset, 'test_path', None),
        'validation_split': getattr(cfg.dataset, 'validation_split', 0.25),
        'test_split': getattr(cfg.dataset, 'test_split', 0.25),
        'class_names': getattr(cfg.dataset, 'class_names', None),
        'batch_size': batch_size, 
        'seed': getattr(cfg.dataset, 'seed', 127),
        'gravity_rot_sup': getattr(cfg.preprocessing, 'gravity_rot_sup', False),
        'normalization': getattr(cfg.preprocessing, 'normalization', False),
        'data_dir':  getattr(cfg.dataset, 'data_dir', './datasets/'),
        'data_download': getattr(cfg.dataset, 'data_download', True),
        'input_shape': input_shape,
        'to_cache': getattr(cfg.dataset, 'to_cache', False)
    }

    return dataloader_kwargs

def read_pkl_dataset(pkl_file_path: str,
                     class_names: List[str]):
    '''
    _read_pkl_dataset reads a pkl dataset and returns a pandas DataFrame
    arg:
        pkl_file_path: path to the pkl file to be read
        class_names: a list of strings containing the names of the activities
    returns:
        dataset: a pandas dataframe containing all the data combined in a single object
                 containing four columns 'x', 'y', 'z', 'Activity_Labels'.
    '''
    # initialize the script
    class_id = 0
    file_nr = 0
    my_array = []

    # read pkl dataset
    dataset = pd.read_pickle(pkl_file_path)

    # list with nr files for every activity
    nr_files_per_class = []
    # we know there are only five activities in the dataset
    #  with labels from 0->4 so let us count nr of files for every activity
    for lbl in range(len(class_names)):
        nr_files_per_class.append(dataset['act'].count(lbl))

    # acceleration data in the dataset
    array_data = dataset['acc']

    # now let us get data for every activity one by one
    for nr_files in nr_files_per_class:
        # for every occurance of the label
        for i in range(file_nr, file_nr + nr_files):
            my_array.append([])
            for j in range(array_data[i].shape[0]):
                # for every sample in the file
                my_array[i].append([])
                my_array[i][j].extend(array_data[i][j])
                my_array[i][j].append(class_id)
        file_nr += nr_files
        class_id += 1

    # preparing a vertical stack for the dataset
    my_array = np.vstack(my_array[:])

    # creating a data frame withonly four columns 
    # 'x', 'y', 'z', and 'Activity_Label' to be 
    # consistent with WISDM data
    columns = ['x', 'y', 'z', 'Activity_Label']
    my_dataset = pd.DataFrame(my_array, columns=columns)

    # replace activity code with activity labels to be consistent with WISDM dataset
    my_dataset['Activity_Label'] = [str(num).replace(str(num),
                                                     class_names[int(num)])
                                    for num in my_dataset['Activity_Label']]
    return my_dataset


def clean_csv(file_path):
    '''
    This function is specifically written for WISDM dataset.
    This function takes as an input the path to the csv file,
    cleans it and rewrites the cleaned data in the same file.
    args:
        file_path: path of the csv file to be cleaned.
    '''
    # read data file
    with open(file_path, "rt", encoding='utf-8') as fin:
        # read file contents to string
        data = fin.read()

    # fix all problems by replacing ';\n' with '\n's etc
    data = data.replace(';\n', '\n').replace(
        '\n;\n', '\n').replace(';', '\n').replace(',\n', '\n')

    # open the data file in write mode
    with open(file_path, "wt", encoding='utf-8', newline='') as f_out:
        # overrite the data file with the correct data
        f_out.write(data)

    # close the file
    fin.close()


def get_segment_indices(data_column: List, win_len: int):
    '''
    this function gets the start and end indices for the segments
    args:
        data_column: indices
        win_len: segment length
    yields:
        init: int
        end: int
    '''
    # get segment indices to window the data into overlapping frames
    init = 0
    while init < len(data_column):
        yield int(init), int(init + win_len)
        init = init + win_len


def get_data_segments(dataset: pd.DataFrame,
                      seq_len: int) -> Tuple[np.ndarray, np.ndarray]:

    '''
    This function segments the data into (non)overlaping frames
    args:
        dataset: a dataframe containing 'x', 'y', 'z', and 'Activity_Label' columns
        seq_len: length of each segment
    returns:
        A Tuple of np.ndarray containing segments and labels
    '''
    data_indices = dataset.index.tolist()
    n_samples = len(data_indices)
    segments = []
    labels = []

    # need the following variable for tqdm to show the progress bar
    num_segments = int(np.floor((n_samples - seq_len) / seq_len)) + 1

    # creating segments until the _get_segment_indices keep on yielding the start and end of the segments
    for (init, end) in tqdm(get_segment_indices(data_indices, seq_len),
                            unit=' segments', desc='[INFO] : Segments built ',
                            total=num_segments):

        # check if the nr of remaing samples are enough to create a frame
        if end < n_samples:
            segments.append(np.transpose([dataset['x'].values[init: end],
                            dataset['y'].values[init: end],
                            dataset['z'].values[init: end]]))

            # use the label which occured the most in the frame
            # print(labels, statistics.mode(dataset['Activity_Label'][init: end]))
            labels.append(statistics.mode(dataset['Activity_Label'][init: end]))

    # converting the segments from list to numpy array
    segments = np.asarray(segments, dtype=np.float64)
    segments = segments.reshape(segments.shape[0], segments.shape[1],
                                segments.shape[2], 1)
    labels = np.asarray(labels)
    return segments, labels

def preprocess_data(data: np.ndarray,
                    gravity_rot_sup: bool,
                    normalization: bool) -> np.ndarray:

    '''
    Preprocess the data by applying gravity rotation and supression and/or normalization
    args:
        data: numpy array containing the data to be preprocessed
        gravity_rot_sup: boolean flag to apply gravity rotation and supression
        normalization: boolean flag to apply normalization
    returns:
        data: numpy array containing the preprocessed data
    '''

    if gravity_rot_sup:
        # choose a sampling frequency
        # hardcoding to have code equivalency with C-implementation
        f_sample = 26
        # create a copy of data to avoid overwriting the passed dataframe
        data_copy = data.copy()
        # create highpass filter to remove dc components
        f_cut = 0.4
        filter_type = 'highpass'
        f_norm = f_cut / f_sample
        num, den = butter(4, f_norm, btype=filter_type)

        # preprocess the dataset by finding and rotating the gravity axis
        for i in range(data_copy.shape[0]):
            data_copy[i, :, :, 0] = gravity_rotation(np.array(data_copy[i, :, :, 0],
                                                              dtype=float),
                                                     A_COEFF=den, B_COEFF=num)
        return data_copy
    if normalization:
        data_copy = data.copy()

        # Reshape the array to treat each seq_len x 3 window as a seq_len x 1 window
        data_copy = np.reshape(data_copy, (data_copy.shape[0], data_copy.shape[1], -1))

        # Calculate the mean and standard deviation of each seq_len x 1 window
        mean = np.mean(data_copy, axis=(1, 2), keepdims=True)
        std = np.std(data_copy, axis=(1, 2), keepdims=True)

        # Normalize each seq_len x 1 window
        data_norm = (data_copy - mean) / std

        # Reshape the normalized array back to its original shape
        data_norm = np.reshape(data_norm, (data_norm.shape[0], data_norm.shape[1], 3, 1))
        return data_norm

    return data