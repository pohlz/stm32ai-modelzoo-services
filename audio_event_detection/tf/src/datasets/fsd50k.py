# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
import pandas as pd
import json
from typing import List
import numpy as np
from typing import List
from .base import BaseAEDTFDataset

class FSD50KTFDataset(BaseAEDTFDataset):
    """
    FSD50K dataset integration for TensorFlow AED pipelines.

    Prepares FSD50K CSVs by unsmearing labels and converting to ESC-compatible
    format, then constructs training (dev), validation (eval), quantization,
    testing, and prediction datasets. Supports optional restriction to
    monolabel samples and class filtering via `class_names`.

    Notes
    -----
    - The dev set is used for training; the eval set is used for validation.
    - Input CSVs and ontology files are processed to generate model-zoo-ready
      CSVs before dataset creation.
    """
    def __init__(self,
                 time_pipeline,
                 freq_pipeline,
                 dev_audio_folder: str ,
                 eval_audio_folder: str ,
                 audioset_ontology_path: str, 
                 csv_folder: str,
                 class_names: List[str],
                 only_keep_monolabel: bool,
                 quantization_csv_path: str = None,
                 quantization_audio_path: str = None,
                 quantization_split: float = 1.0,
                 test_csv_path: str = None,
                 test_audio_path: str = None,
                 pred_audio_path: str = None,
                 use_garbage_class: bool = False,
                 file_extension: str = ".wav",
                 n_samples_per_garbage_class: int = None,
                 expand_last_dim: bool = True,
                 seed: int = 133):
        """
        Initialize the FSD50K dataset wrapper.

        Parameters
        ----------
        time_pipeline : callable
            Time-domain preprocessing applied to raw waveforms.
        freq_pipeline : callable
            Frequency-domain conversion producing spectrogram patches.
        dev_audio_folder : str
            Path to the dev set audio directory (used for training).
        eval_audio_folder : str
            Path to the eval set audio directory (used for validation).
        audioset_ontology_path : str
            Path to AudioSet ontology JSON file.
        csv_folder : str
            Directory containing FSD50K CSVs; also used to write processed CSVs.
        class_names : List[str]
            Classes to include when constructing datasets.
        only_keep_monolabel : bool
            If True, only retain monolabel samples when generating model-zoo CSVs.
        quantization_csv_path : str, optional
            Path to quantization CSV; if None, training data may be reused.
        quantization_audio_path : str, optional
            Directory for the quantization dataset audio.
        quantization_split : float, default 1.0
            Fraction of the quantization dataset to keep.
        test_csv_path : str, optional
            Path to testing CSV.
        test_audio_path : str, optional
            Directory containing test audio files.
        pred_audio_path : str, optional
            Directory of unlabeled audio for building a prediction dataset.
        use_garbage_class : bool, default False
            If True, aggregate out-of-scope classes into an `other` class.
        file_extension : str, default ".wav"
            Audio file extension to append where needed.
        n_samples_per_garbage_class : int, optional
            Number of samples per out-of-scope class kept for `other`; balanced
            heuristic used if None.
        expand_last_dim : bool, default True
            If True, expand spectrogram patches with a trailing singleton channel.
        seed : int, default 133
            Random seed for reproducible operations.
        """
    
        self.dev_audio_folder = dev_audio_folder
        self.eval_audio_folder = eval_audio_folder
        self.audioset_ontology_path = audioset_ontology_path
        self.csv_folder = csv_folder
        self.only_keep_monolabel = only_keep_monolabel
        self.class_names = class_names

        self.quantization_csv_path = quantization_csv_path
        self.quantization_audio_path = quantization_audio_path
        self.quantization_split = quantization_split
        self.test_csv_path = test_csv_path
        self.test_audio_path = test_audio_path


        self._prepare_fsd50k_csvs()
    
        self.dev_csv_path = os.path.join(csv_folder, "model_zoo_unsmeared_dev.csv")
        self.eval_csv_path = os.path.join(csv_folder, "model_zoo_unsmeared_eval.csv")

        # We're providing the training_* and validation_* args to the base class' __init__ here
        # but it is optional and doesn't really matter

        super().__init__(
                time_pipeline=time_pipeline,
                freq_pipeline=freq_pipeline,
                training_csv_path=self.dev_csv_path,
                training_audio_path=dev_audio_folder,
                validation_csv_path=self.eval_csv_path,
                validation_audio_path=eval_audio_folder,
                validation_split=None, # We use the eval set as valid set for FSD50K and not a split of the training set
                quantization_csv_path=self.quantization_csv_path,
                quantization_audio_path=self.quantization_audio_path,
                test_csv_path=self.test_csv_path,
                test_audio_path=self.test_audio_path,
                pred_audio_path=pred_audio_path,
                use_garbage_class=use_garbage_class,
                file_extension=file_extension,
                n_samples_per_garbage_class=n_samples_per_garbage_class,
                expand_last_dim=expand_last_dim,
                seed=seed)
        

    def get_tf_datasets(self,
                        batch_size: int,
                        to_cache: bool = True,
                        shuffle: bool = True):
        """
        Construct TensorFlow datasets for training, validation, quantization,
        testing, and prediction based on provided CSVs and audio folders.

        Parameters
        ----------
        batch_size : int
            Batch size for output datasets.
        to_cache : bool, default True
            If True, cache the constructed datasets.
        shuffle : bool, default True
            If True, shuffle the training dataset.

        Returns
        -------
        dict
            Dictionary containing the following keys:
            - `train_ds`: tf.data.Dataset or None
            - `val_ds`: tf.data.Dataset or None
            - `quantization_ds`: tf.data.Dataset or None
            - `test_ds`: tf.data.Dataset or None
            - `val_clip_labels`: np.ndarray or None
            - `test_clip_labels`: np.ndarray or None
            - `pred_ds`: tf.data.Dataset, or tuple of arrays (X, clip_labels) when `return_arrays=True`, or None
            - `pred_clip_labels`: np.ndarray or None
            - `pred_filenames`: list of Path objects or None

        Raises
        ------
        ValueError
            If `class_names` is missing while `training_csv_path` is provided.
        """

        print("[INFO] : Loading FSD50K. The dev set will be used as the training set")
        print("and the eval set will be used as the validation set.")
        print("[INFO] : The dataset.training_*, and dataset.validation_* args in your config file will be ignored.")
        if not self.class_names:
            raise ValueError("Argument class_names was not provided ! \
                            Please provide at least one class in class_names")
        else:
            used_classes = self.class_names

        train_df = pd.read_csv(self.dev_csv_path)
        val_df = pd.read_csv(self.eval_csv_path) 
        # If a validation path is provided, use it
        # If a validation split is provided, use it
        # If none of these are provided, use fold 5 as validation set.
    
        print("[INFO] : Loading training dataset")
        train_ds = self.get_ds(
            df=train_df,
            audio_path=self.dev_audio_folder,
            used_classes=used_classes,
            batch_size=batch_size,
            to_cache=to_cache,
            shuffle=shuffle,
            return_clip_labels=False,
            return_arrays=False
            )
        
        # Load validation data 
        # Use the eval set
        print("[INFO] : Loading validation dataset, using FSD50K's eval set as validation dataset.")
        val_ds, val_clip_labels = self.get_ds(
            df=val_df,
            audio_path=self.eval_audio_folder,
            used_classes=used_classes,
            batch_size=batch_size,
            to_cache=to_cache,
            shuffle=shuffle,
            return_clip_labels=True,
            return_arrays=False
            )
        
        if self.quantization_csv_path:
            quant_df = pd.read_csv(self.quantization_csv_path)
            if self.class_names is None:
                quant_class_names = quant_df["category"].unique().tolist()
            else:
                quant_class_names = self.class_names

            quantization_ds = self.get_ds(
                df=quant_df,
                used_classes=quant_class_names,
                audio_path=self.quantization_audio_path,
                batch_size=batch_size,
                to_cache=to_cache,
                shuffle=False,
                return_clip_labels=False,
                return_arrays=False
                )
        
        elif train_ds is not None:
            quantization_ds = train_ds
            
        else:
            quantization_ds = None

        if quantization_ds:
            quantization_ds = quantization_ds.take(int(len(quantization_ds) * float(self.quantization_split)))
        
        if self.test_csv_path:
            test_df = pd.read_csv(self.test_csv_path)
            if self.class_names is None:
                test_class_names = test_df["category"].unique().tolist()
            else:
                test_class_names = self.class_names

            test_ds, test_clip_labels = self.get_ds(
                df=test_df,
                audio_path=self.test_audio_path,
                used_classes=test_class_names,
                batch_size=batch_size,
                to_cache=to_cache,
                shuffle=False,
                return_clip_labels=True,
                return_arrays=False
                )
        elif val_ds is not None:
            test_ds = val_ds # Use FSD50K eval set if not custom test set given
            test_clip_labels = val_clip_labels
        else:
            test_ds = None
            test_clip_labels = None

        if self.pred_audio_path:
            pred_ds, pred_clip_labels, pred_filenames = self.get_ds_no_df(
                audio_path=self.pred_audio_path,
                batch_size=batch_size,
                to_cache=to_cache,
                return_clip_labels=True,
                return_arrays=True,
                return_filenames=True)
        else:
            pred_ds = None
            pred_clip_labels = None
            pred_filenames = None

        out_dict = {"train_ds":train_ds,
                    "val_ds":val_ds,
                    "quantization_ds":quantization_ds,
                    "test_ds":test_ds,
                    "val_clip_labels":val_clip_labels,
                    "test_clip_labels":test_clip_labels,
                    "pred_ds":pred_ds,
                    "pred_clip_labels":pred_clip_labels,
                    "pred_filenames":pred_filenames}
        
        return out_dict


    def _prepare_fsd50k_csvs(self):
        ''' Converts FSD50K dataset to an ESC-like format expected by the model zoo.
            Unsmears labels, and discards samples belonging to classes not specified in class_names.
            Can additionally discard multilabel samples.
            Writes csvs compatible with the model zoo format to 
            csv_folder/unsmeared_dev.csv and csv_folder/unsmeared_eval.csv
        '''
        preferred_collapse_classes = self.class_names
        vocabulary_path = os.path.join(self.csv_folder, 'vocabulary.csv')
        dev_csv_path = os.path.join(self.csv_folder, 'dev.csv')
        eval_csv_path = os.path.join(self.csv_folder, 'eval.csv')

        unsmeared_dev_csv_path = os.path.join(self.csv_folder, "unsmeared_dev.csv")
        unsmeared_eval_csv_path = os.path.join(self.csv_folder, "unsmeared_eval.csv")
        # Unsmear labels
        print("Unsmearing labels for training set.")
        unsmear_labels(dev_csv_path,
                       vocabulary_path,
                       self.audioset_ontology_path,
                       output_file=unsmeared_dev_csv_path,
                       save=True)
        print("Unsmearing labels for test set")
        unsmear_labels(eval_csv_path,
                       vocabulary_path,
                       self.audioset_ontology_path,
                       output_file=unsmeared_eval_csv_path,
                       save=True)
        
        print("Generating model zoo-compatible CSV for training set.")
        output_file = os.path.join(self.csv_folder, "model_zoo_unsmeared_dev.csv")
        make_model_zoo_compatible(unsmeared_dev_csv_path,
                                  classes_to_keep=None,
                                  only_keep_monolabel=self.only_keep_monolabel,
                                  collapse_to_monolabel=False,
                                  preferred_collapse_classes=preferred_collapse_classes,
                                  output_file=output_file,
                                  save=True,
                                  quick_hack=False)
        print("Successfully generated model zoo-compatible CSV for training set at {}".format(output_file))
        print("Generating model zoo-compatible CSV for test set.")
        output_file = os.path.join(self.csv_folder, "model_zoo_unsmeared_eval.csv")
        make_model_zoo_compatible(unsmeared_eval_csv_path,
                                classes_to_keep=None,
                                only_keep_monolabel=self.only_keep_monolabel,
                                collapse_to_monolabel=False,
                                preferred_collapse_classes=preferred_collapse_classes,
                                output_file=output_file,
                                save=True,
                                quick_hack=False)
        print("Successfully generated model zoo-compatible CSV for test set at {}".format(output_file))

        print("Done preparing FSD50K csv files !")


def generate_ranked_vocabulary(vocabulary,
                               audioset_ontology,
                               output_file=None):
    '''Takes as input the vocabulary.csv file of FSD50K and adds a "ranking" column
       indicating whether each class is a leaf or intermediate node according to the ontology

       Inputs
       ------
       vocabulary : str or PosixPath : Path to the vocabulary.csv file contained in the FSD50K dataset
       audioset_ontology : str or Posixpath : Path to the audioset ontology json file. 
            Download here : https://github.com/audioset/ontology/blob/master/ontology.json
       output_file : str or PosixPath : Path to write the output csv file.
       
       Outputs
       -------
       vocabulary_ranked.csv : The vocabulary.csv file with an extra ranking column added 
            indicating whether each class is a leaf or intermediate node.'''

            
    fsd_vocab = pd.read_csv(vocabulary, header=None,
                            names= ['id', 'name', 'mids'])
                            
    with open(audioset_ontology, 'r') as f:
        audioset_ontology = json.load(f)

    ranks = []
    set_of_mids = set(fsd_vocab['mids'].tolist())
    for i in range(len(fsd_vocab)):
        for cl in audioset_ontology:
            # If this is the corresponding audioset class
            if cl['id'] == fsd_vocab['mids'].iloc[i]:
                # Then determine if it has any children in the audioset ontology
                if len(cl['child_ids']) == 0:
                    # If it doesn't it's a leaf
                    ranks.append('Leaf')
                # If it does then see if these children are also in the FSD50k ontology
                else:
                    child_set = set(cl['child_ids'])
                    if len(child_set.intersection(set_of_mids)) == 0:
                        # if the children aren't in FSD50k ontology then it's a leaf
                        ranks.append('Leaf')
                    else:
                        # if they are then it's an intermediate node
                        ranks.append('Intermediate')
                break

    fsd_vocab['ranks'] = ranks
    if output_file is None:
        output_file = 'vocabulary_ranked.csv'
    fsd_vocab.to_csv(output_file, header=False, index=False)


def _remove_labels(labels, to_keep):
    '''Internal utility function'''
    labels = labels.split(',')
    kept = [l for l in labels if l in to_keep]
    labels = ','.join(kept)
    return labels

def filter_labels(ground_truth, ranked_vocabulary, node_rank='Leaf',
                  output_file=None, save=False, drop_no_label=True):
    '''Filters labels based on label node rank in audioset ontology.
    
    Inputs
    ------
    ground_truth : csv file with FSD50K ground truth labels
    ranked_vocabulary : ranked FSD50K vocabulary csv file. 
    node_rank : str, if "Leaf" keep leaf labels, if "Intermediate" keep all other labels
    output_file : path to the file to where the filtered dataframe should be saved
    save : bool, set to True to save the filtered dataframe
    drop_no_label : bool, if True drop rows with no remaining labels after filtering
    
    Outputs
    -------
    ground_truth_df : pandas.Dataframe, dataframe containing the filtered labels'''
    ground_truth_df = pd.read_csv(ground_truth)
    names = ['id', 'name', 'mids', 'rank']
    ranked_vocabulary = pd.read_csv(ranked_vocabulary, header=None, names=names)
    leaves = ranked_vocabulary[ranked_vocabulary['rank'] == 'Leaf']['name'].to_list()
    intermediates = ranked_vocabulary[ranked_vocabulary['rank'] == 'Intermediate']['name'].to_list()

    if node_rank == 'Leaf':
        to_keep = leaves
    elif node_rank == 'Intermediate':
        to_keep = intermediates
    else:
        raise ValueError('node_rank arg must be either "Leaf" or "Intermediate"')

    filtered_labels = []
    for i in range(len(ground_truth_df)):
        labels = ground_truth_df['labels'].iloc[i]
        filtered_labels.append(_remove_labels(labels, to_keep=to_keep))
    ground_truth_df.drop('labels', axis=1, inplace=True)
    ground_truth_df['labels'] = filtered_labels

    if drop_no_label:
        nan_value = float('NaN')
        ground_truth_df.replace("", nan_value, inplace=True)
        ground_truth_df.dropna(subset=['labels'], inplace=True)
    
    if output_file == None:
        output_file = 'filtered_{}_{}'.format(node_rank, ground_truth.split('/')[-1])

    if save:
        ground_truth_df.to_csv(output_file, index=False)

    return ground_truth_df

def _has_children_in_list(mids, ontology):
    '''Returns list of mids who have children in the input list.'''
    output = []
    for mid in mids:
        for cl in ontology:
            if cl['id'] == mid:
                children = set(cl['child_ids'])
        num_children = len(children.intersection(set(mids)))
        if num_children > 0:
            output.append(mid)
    return output

def unsmear_labels(ground_truth, vocabulary, audioset_ontology, output_file=None, save=False):
    '''
    Unsmears labels from an FSD50K ground truth csv.
    Certain labels imply other labels (e.g. all clips labeled "Electric guitar"
    will automatically be labeled "Music". Unsmearing removes these automatically applied labels
    , only keeping the original label.)

    Inputs
    ------
    ground_truth : path to csv file with FSD50K ground truth labels
    vocabulary : path to FSD50K vocabulary csv file. 
    audioset_ontology : path to audioset ontology json file
    output_file : path to the file where the unsmeared dataframe should be saved
    save : bool, set to True to save the unsmeared dataframe

    Outputs
    -------
    ground_truth_df : FSD50K ground truth dataframe with unsmeared labels
    '''

    fsd_vocab = pd.read_csv(vocabulary, header=None,
                            names= ['id', 'name', 'mids'])

    ground_truth_df = pd.read_csv(ground_truth)

    with open(audioset_ontology, 'r') as f:
        audioset_ontology = json.load(f)

    filtered_labels = []
    for i in range(len(ground_truth_df)):
        mids = ground_truth_df['mids'].iloc[i]
        # Convert to list
        mids = mids.split(',')
        # For each mid, check if they have children already in the list.
        #  If they have at least one, remove them.
        parents = _has_children_in_list(mids, audioset_ontology)
        remainder = list(set(mids).difference(set(parents)))

        # Convert remainder back to labels
        labels = fsd_vocab[fsd_vocab['mids'].isin(remainder)]['name'].to_list()
        labels = ','.join(labels)
        filtered_labels.append(labels)

    ground_truth_df.drop('labels', axis=1, inplace=True)
    ground_truth_df['labels'] = filtered_labels

    if output_file == None:
        output_file = 'unsmeared_{}'.format(ground_truth.split('/')[-1])

    if save:
        ground_truth_df.to_csv(output_file, index=False)

    return ground_truth_df
            
def _filter_samples_by_class(df, classes_to_keep):
    '''Filter a dataframe to only keep samples which have the desired classes
       Inputs
       ------
       df : Pandas DataFrame, dataframe to perform filtering on
       classes_to_keep : list of str, list of the classes to keep after filtering
       
       Outputs
       -------
       df : pandas.Dataframe, filtered dataframe'''

    splitter = splitter = lambda s : s.split(',')
    # Have to use sets instead of doing substring match with regex because
    # if some class is a substring of another class, the latter would fail
    to_keep = df['labels'].apply(splitter).apply(set(classes_to_keep).isdisjoint)
    # Invert the boolean series
    to_keep = ~to_keep

    df = df[to_keep]

    return df


def make_model_zoo_compatible(path_to_csv,
                              classes_to_keep=None,
                              only_keep_monolabel=True, 
                              collapse_to_monolabel=False,
                              preferred_collapse_classes=None,
                              output_file=None,
                              save=False,
                              quick_hack=False):
    '''Makes the input csv compatible with the ST model zoo. Renames two columns, and has
       the option to only keep monolabel files.
       
       Inputs
       ------
       path_to_csv : PosixPath, path to the csv you want to load
       classes_to_keep : List of str, list of the classes you want to keep.
                         Set to None to keep everything.
       only_keep_monolabel : bool, if set to True we discard all samples which have multiple labels.
                             Make sure you only apply this to an unsmeared csv otherwise it will make
                             no sense
       collapse_to_monolabel : bool. If set to True, we collapse all multi_label sample to monolabel
                                     by picking a sample at random.
                                     Make sure to only apply this to an unsmeared csv
       output_file : PosixPath, path to the output file to save the resulting dataframe in
       save : bool, set to True to save the resulting dataframe.
       
       Outputs
       -------
       df : Dataframe in ESC format. '''
    
    df = pd.read_csv(path_to_csv)

    if classes_to_keep:
        df = _filter_samples_by_class(df, classes_to_keep=classes_to_keep)
    splitter = lambda s : s.split(',')

    if only_keep_monolabel:
        df = df[~df['labels'].str.contains(',')]
        
    elif collapse_to_monolabel:
        if preferred_collapse_classes is None:
            if classes_to_keep is None:
                raise ValueError("If collapse_to_monolabel is true, \
                                  at least one of classes_to_keep or preferred_collapse_classes must be specified ")
            preferred_collapse_classes = classes_to_keep
        seed = 42 # For reproducibility
        rng = np.random.default_rng(seed=seed)
        intersecter = lambda li : list(set(preferred_collapse_classes).intersection(li))
        # Frankenstein monster of a code line
        df['labels'] = df['labels'].apply(splitter).apply(intersecter).apply(rng.choice)

    # Convert labels to list if we didn't collapse to monolabel or discard multilabel
    if not only_keep_monolabel and not collapse_to_monolabel:
        df['labels'] = df['labels'].apply(splitter)
    
    # Rename columns
    mapper = {'fname':'filename', 'labels':'category'}
    df.rename(columns=mapper, inplace=True)

    # Convert dtype of filename column to str because apparently it's np.int64 by default 
    df['filename'] = df['filename'].astype('str')
    appender = lambda s : s + '.wav'
    if quick_hack:
        df['filename'] = df['filename'].apply(appender)

    # Save resulting df
    if output_file == None:
        output_file = 'model_zoo_ready_{}'.format(path_to_csv.split('/')[-1])

    if save:
        df.to_csv(output_file, index=False)

    return df