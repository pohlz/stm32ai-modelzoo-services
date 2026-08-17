# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import pandas as pd
from typing import List
from re import A
import numpy as np
import tensorflow as tf
from math import ceil
from pathlib import Path
import librosa


class BaseAEDTFDataset:
    """
    Base class for TensorFlow audio event detection datasets.

    Notes
    -----
    - The sampling rate (`sr`) is inferred from either the `time_pipeline` or
        `freq_pipeline` via an `sr` attribute. One of the pipelines must expose it.
    - CSV files are expected in ESC-like format with columns `filename` and
        `category`.
    """
    def __init__(self,
                 time_pipeline,
                 freq_pipeline,
                 training_csv_path: str = None,
                 training_audio_path: str = None,
                 validation_csv_path: str = None,
                 validation_audio_path: str = None,
                 validation_split: float = None,
                 quantization_csv_path: str = None,
                 quantization_audio_path: str = None,
                 test_csv_path: str = None,
                 test_audio_path: str = None,
                 pred_audio_path: str = None,
                 use_garbage_class: bool = False,
                 file_extension: str = ".wav",
                 n_samples_per_garbage_class: int = None,
                 expand_last_dim: bool = True,
                 seed: int = 133
                 ):
        """
        Initialize the base AED dataset with preprocessing pipelines and paths.

        Parameters
        ----------
        time_pipeline : callable
            Function or callable object that takes a 1D waveform and returns a
            processed waveform (e.g., resampling, trimming, normalization).
        freq_pipeline : callable
            Function or callable object that takes a processed waveform and
            returns a list/array of spectrogram patches.
        training_csv_path : str, optional
            Path to training CSV file (ESC-like format with `filename`, `category`).
        training_audio_path : str, optional
            Directory containing training audio files referenced by the CSV.
        validation_csv_path : str, optional
            Path to validation CSV file. If not provided, a split may be used by
            inheriting classes.
        validation_audio_path : str, optional
            Directory containing validation audio files. If not provided, falls
            back to `training_audio_path` in inheriting classes.
        validation_split : float, optional
            Proportion (0-1) of training CSV to allocate to validation when a
            dedicated validation CSV is not provided.
        quantization_csv_path : str, optional
            Path to a CSV used to create a representative dataset for post-training
            quantization; if not provided, training data may be reused.
        quantization_audio_path : str, optional
            Directory containing audio for the quantization dataset.
        test_csv_path : str, optional
            Path to testing CSV file.
        test_audio_path : str, optional
            Directory containing test audio files.
        pred_audio_path : str, optional
            Directory of unlabeled audio clips used to build a prediction dataset.
        use_garbage_class : bool, default False
            If True, pool samples of classes not in `used_classes` into an
            additional `other` class.
        file_extension : str, default ".wav"
            File extension to append when CSV `filename` lacks an extension.
        n_samples_per_garbage_class : int, optional
            Number of samples to keep per out-of-scope class when forming the
            `other` class. If None, a balanced heuristic is used.
        expand_last_dim : bool, default True
            If True, expand spectrogram patches with a trailing singleton channel
            dimension.
        seed : int, default 133
            Random seed for shuffling and sampling operations.

        Raises
        ------
        ValueError
            If neither `time_pipeline` nor `freq_pipeline` exposes an `sr`
            attribute to infer the sampling rate.
        """
        
        self.time_pipeline = time_pipeline
        self.freq_pipeline = freq_pipeline
        self.training_csv_path = training_csv_path
        self.training_audio_path = training_audio_path
        self.validation_csv_path = validation_csv_path
        self.validation_audio_path = validation_audio_path
        self.validation_split = validation_split
        self.quantization_csv_path = quantization_csv_path
        self.quantization_audio_path = quantization_audio_path
        self.test_csv_path = test_csv_path
        self.test_audio_path = test_audio_path
        self.pred_audio_path = pred_audio_path
        self.use_garbage_class = use_garbage_class
        self.file_extension = file_extension
        self.n_samples_per_garbage_class = n_samples_per_garbage_class
        self.expand_last_dim = expand_last_dim
        self.seed = seed

        # Infer sampling rate from pipelines
        try:
            self.sr = self.time_pipeline.sr
        except:
            try:
                self.sr = self.freq_pipeline.sr
            except:
                raise ValueError("Could not infer sampling rate from pipelines." 
                                 "Make sure either your time or frequency pipeline has a 'sr' attribute.")

    # Allow get_ds to use audio_path and df that aren't from self
    # This is e.g. in case user requests a custom quantization set while in a chain_tqe for example.
    def get_ds(self,
               audio_path: str,
               used_classes: List[str],
               batch_size: int,
               to_cache: bool,
               shuffle: bool, 
               return_clip_labels: bool,
               return_arrays: bool,
               df=None):
        '''
        Takes as input a csv, a folder with audio files,
        a string lookup Tensorflow layers (to convert classes to one-hot vectors),
        and all the parameters needed to compute spectrogram patches,
        and returns a tf.data.Dataset of spectrogram patches & one-hot vectors

        Inputs
        -------
        sd : pandas DataFrame of the dataset.
        audio_path : str, Posixpath or pathlib.Path object, path to folder containing audio files
        patch_length : int, number of spectrogram frames per patch.
                        Note that the length is specified in frames, e.g. if each frame in 
                        the spectrogram represents 50 ms, a patch of 1s would need 20 frames. 
        n_mels : int, number of mel filters. passed to librosa.feature.melspectrogram
        target_rate : int, Sample rate of output waveform
        overlap : float between 0 and 1.0, proportion of overlap between consecutive spectrograms.
            For example, with an overlap of 0.25 and a patch length of 20, 
            patch number n would share its 5 first frames with patch (n-1),
            and its 5 last frames with patch (n+1). 
        n_fft : int, length of FFT window, passed to librosa.feature.melspectrogram
        spec_hop_length : int, hop length between FFTs, passed to librosa.feature.melspectrogram
        min_length : int,  Minimum length of output waveform in seconds. 
            If input waveform is below this duration, it will be repeated to reach min_length.
        max_length : int, Maximum length of output waveform in seconds.
            If input waveform is longer, it will be cut from the end.

        top_db : int, Passed to librosa.split. Frames below -top_db will be considered as silence and cut.
            Higher values induce more tolerance to silence.
        frame_length : int, frame length used for silence removal
        hop_length : int, hop length used for silence removal
        trim_last_second : bool. Set to True to cut the output waveform to an integer number of seconds.
            For example, if the output waveform is 7s and 350 ms long, this option will cut the last 350 ms.
        include_last_patch : bool. If set to False, the last spectrogram frames 
            will be discarded if they are not enough to build a new patch with.
            If set to True, they will be kept in a new patch which will more heavily overlap with the previous one.
            For example, with a patch length of 20 frames, and overlap of 0.25 (thus 5 frames),
            and an overall clip length of 127 frames, the last 7 frames would not be enough to build a new patch,
            since we'd need 20 - 5 = 15 new frames. If include_last_patch is set to False,
            these last 7 frames will be discarded. If it is set to True, 
            they will be included in a new patch along with the 13 previous frames.

        win_length : int, passed to librosa.feature.melspectrogram. If None default to n_fft
        window : str, passed to librosa.feature.melspectrogram
        center : bool, passed to librosa.feature.melspectrogram 
        pad_mode : str, passed to librosa.feature.melspectrogram
        power : float, 1.0 or 2.0 only are allowed. Set to 2.O for power melspectrogram, 
            and 1.0 for amplitude melspectrogram

        fmin : int, min freq of spectrogram, passed to librosa.feature.melspectrogram
        fmax : int, max freq of spectrogram, passed to librosa.feature.melspectrogram
        power_to_db_ref, : func, reference used to convert linear scale mel spectrogram to db scale. 
            Passed to passed to librosa.power_to_db or librosa.amplitude_to_db
        norm : str, normalization used for triangular mel weights. Passer to librosa.feature.melspectrogram
        Defaults to "slaney", in which case the triangular mel weights are divided by the width of the mel band.
        htk : bool, if True use the HTK formula to compute mel weights, else use Slaney's.
            Passed to librosa.feature.melspectrogram
        to_db : bool, if True convert the output spectrogram to decibel scale.
            if False we just take log(spec + 1e-4)
        use_garbage_class : bool, set to True to pool samples from all classes not included in class_names
            into an additional "garbage" (i.e. "other") class.
        n_samples_per_garbage_class : int, number of samples per unused class 
            to pool into the "garbage" class. If None, gets automatically determined 
            trying to keep a balanced dataset.
        expand_last_dim : bool, if True expand the last dim of each spectrogram patch
            (i.e. shape of patches becomes (n_mels, n_patches, 1))
        file_extension : str, file extension of audio files, e.g. ".wav". 
        batch_size : int, batch size of the output dataset
        to_cache : bool, if True cache output dataset
        shuffle : bool, if True shuffle output dataset
        return_clip_labels : bool, if True returns an additional numpy array 
                                containing labels linking each patch to its origin audio clip
        return_arrays : bool, if True returns a tuple (X, y) of numpy arrays
                        instead of a tf.data.Dataset
        seed : seed used for shuffling the dataset, and sampling data for the "garbage" class if asked.
        Outputs
        -------
        ds : tf.data.Dataset or tuple of np.ndarrays containing the spectrogram patches and labels.
            is a tf.data.Dataset is return_arrays is set to False, and a tuple of np.ndarrays
            if return_arrays is set to True.
        clip_labels : optional, np.ndarray containing labels linking each patch to its origin audio clip.
                    only returned if return_clip_labels is set to True.
        '''
        
        df['filename'] = df['filename'].astype('str')
        # Determine if we need to add file extension to file names
        add_file_extension = str(self.file_extension) not in df['filename'].iloc[0]

        # Raise error if some requested classes are not actually present in the DF
        available_classes = df["category"].unique()
        if not set(used_classes).issubset(set(available_classes)):
            raise ValueError(f"""Some classes in class_names were not found
            in the csv file associated with your dataset.
            Please check the class_names argument of the config file. 
            Classes that were not found in csv : {set(used_classes).difference(set(available_classes))}""")

        # If used_classes is not provided, warn and use all available classes
        if not used_classes:
            raise ValueError("Did not receive any classes to use. Please check the class_names arg in your config file")
            
            
        # Handle all the "garbage class" stuff
        if self.use_garbage_class:
            if self.n_samples_per_garbage_class:
                pass
            else:
                # If user doesn't provide this arg, determine automatically.
                num_ind_samples = df['category'].isin(used_classes).sum()
                num_other_class = len(set(df['category'].unique()).difference(set(used_classes)))
                self.n_samples_per_garbage_class = ceil(num_ind_samples / num_other_class)
            
            df = self._add_garbage_class_to_df(
                dataframe=df,
                classes_to_keep=used_classes,
                n_samples_per_other_class=self.n_samples_per_garbage_class,
                seed=self.seed,
                shuffle=True)
            
            num_other_samples = len(df[df['category'] == "other"])
        else:
            # Keep only user-provided classes
            df = df[df['category'].isin(used_classes)]

        num_samples = len(df)
        print("[INFO] : Loading dataset.")
        print(f"[INFO] : Using {len(used_classes)} classes.")
        print(f"[INFO] : Classes used : {used_classes}")
        print(f"[INFO] : Loading {num_samples} audio clips.")
        if self.use_garbage_class:
            print('[INFO] : Using the additional "garbage" class.')
            print(f'[INFO] : Added {num_other_samples} samples to the "garbage" class.')

        # Load data
        X = []
        y= []
        if return_clip_labels:
            clip_labels = []

        n_patches_generated = 0
        for i in range(len(df)):
            if add_file_extension:
                fname = df['filename'].iloc[i] + str(self.file_extension)
            else:
                fname = df['filename'].iloc[i]

            label = df['category'].iloc[i]
            filepath = Path(audio_path, fname)
            wave, _ = librosa.load(filepath, sr=self.sr)
            wave = self.time_pipeline(wave)
            patches = self.freq_pipeline(wave)
            n_patches_generated += len(patches)
            X.extend(patches)
            y.extend([label] * len(patches))
            if return_clip_labels :
                clip_labels.extend([i] * len(patches))

        # Concatenate X into a single array
        X = np.stack(X, axis=0)
        if self.expand_last_dim:
            X = np.expand_dims(X, axis=-1)
        print(f"[INFO] : Generated {n_patches_generated} patches")
        
        # One-hot encode label vectors
        vocab = df['category'].unique()
        string_lookup_layer = tf.keras.layers.StringLookup(
            vocabulary=sorted(list(vocab)), # Note we're sorting classes alphabetically here
            num_oov_indices=0,
            output_mode='one_hot')
        
        y = np.array(string_lookup_layer(y))

        ds = tf.data.Dataset.from_tensor_slices((X, y))
        ds = ds.batch(batch_size)
        if shuffle:
            if return_clip_labels:
                # Can't use actual warnings because they are silenced in the main script
                # print("[WARNING] Tried to shuffle a dataset which has associated clip labels. \n \
                #     This would break the clip labels. Skipping shuffle.")
                pass
            else:
                ds = ds.shuffle(len(ds), reshuffle_each_iteration=True, seed=self.seed)
        if to_cache:
            ds = ds.cache()
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        if return_arrays:
            ds = (X, y)
        if return_clip_labels:
            return ds, np.array(clip_labels)
        else:
            return ds

    def get_gsc_ds(self,
                   audio_path: str,
                   used_classes: List[str],
                   batch_size: int,
                   to_cache: bool,
                   shuffle: bool,
                   return_clip_labels: bool,
                   split_name: str,
                   df=None):
        """Build a lazy GSC dataset with per-iteration waveform augmentation.

        Unlike :meth:`get_ds`, this method keeps file paths in the TensorFlow
        dataset and performs waveform preprocessing and feature extraction from
        ``Dataset.map``. Consequently, training placement, shift, and noise are
        sampled again on every epoch. Deterministic splits may be cached.
        """
        if df is None or df.empty:
            raise ValueError(f"GSC split {split_name!r} is empty")
        if self.use_garbage_class:
            raise ValueError(
                "GSC uses explicit 'unknown' and 'silence' classes; "
                "dataset.use_garbage_class must be False"
            )

        dataframe = df.copy()
        dataframe["filename"] = dataframe["filename"].astype(str)
        available_classes = set(dataframe["category"].unique())
        missing_classes = set(used_classes) - available_classes
        if missing_classes:
            raise ValueError(
                f"Classes missing from GSC {split_name} CSV: {sorted(missing_classes)}"
            )

        dataframe = dataframe[dataframe["category"].isin(used_classes)].reset_index(drop=True)
        self.time_pipeline.validate_silence_sources(dataframe, split_name)

        class_names = sorted(used_classes)
        class_to_index = {name: index for index, name in enumerate(class_names)}
        paths = []
        label_indices = []
        label_names = []
        extension = str(self.file_extension)

        for row in dataframe.itertuples(index=False):
            filename = str(row.filename)
            if extension and not filename.lower().endswith(extension.lower()):
                filename += extension
            filepath = Path(audio_path, filename)
            if not filepath.is_file():
                raise FileNotFoundError(f"GSC audio file not found: {filepath}")
            paths.append(str(filepath))
            label_indices.append(class_to_index[row.category])
            label_names.append(str(row.category))

        training = split_name == "training"
        ds = tf.data.Dataset.from_tensor_slices(
            (paths, np.asarray(label_indices, dtype=np.int32), label_names)
        )
        if training and shuffle:
            ds = ds.shuffle(
                buffer_size=len(paths),
                seed=self.seed,
                reshuffle_each_iteration=True,
            )

        feature_shape = (
            self.freq_pipeline.n_mels,
            self.freq_pipeline.patch_length,
        )
        if self.expand_last_dim:
            feature_shape += (1,)

        def _decode_string(value):
            value = np.asarray(value).item()
            return value.decode("utf-8") if isinstance(value, bytes) else str(value)

        def _load_and_preprocess(path, label_index, label_name):
            decoded_path = _decode_string(path)
            decoded_label = _decode_string(label_name)
            wave, _ = librosa.load(decoded_path, sr=self.sr, mono=True)
            wave = self.time_pipeline(
                wave,
                label=decoded_label,
                training=training,
            )
            patches = self.freq_pipeline(wave)
            if len(patches) != 1:
                raise ValueError(
                    "GSC preprocessing must produce exactly one feature patch; "
                    f"received {len(patches)} from {decoded_path}"
                )

            feature = np.asarray(patches[0], dtype=np.float32)
            if self.expand_last_dim:
                feature = np.expand_dims(feature, axis=-1)

            one_hot = np.zeros(len(class_names), dtype=np.float32)
            one_hot[int(np.asarray(label_index).item())] = 1.0
            return feature, one_hot

        def _tf_map(path, label_index, label_name):
            feature, label = tf.numpy_function(
                _load_and_preprocess,
                [path, label_index, label_name],
                [tf.float32, tf.float32],
            )
            feature.set_shape(feature_shape)
            label.set_shape((len(class_names),))
            return feature, label

        # One worker keeps the seeded NumPy generator deterministic and avoids
        # concurrent access to Librosa from Python callbacks.
        ds = ds.map(_tf_map, num_parallel_calls=1)
        if to_cache and not training:
            ds = ds.cache()
        elif to_cache and training:
            print("[INFO] : Ignoring dataset.to_cache for randomized GSC training data")
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        if return_clip_labels:
            # Fixed one-second inputs produce one patch per source clip.
            return ds, np.arange(len(dataframe), dtype=np.int32)
        return ds
        
    def get_ds_no_df(self,
                     audio_path: str,
                     batch_size: int,
                     to_cache: bool,
                     return_clip_labels: bool,
                     return_arrays: bool,
                     return_filenames: bool = True):
        '''
        Analogue to get_ds, but does not require a dataframe as input.
        Outputs a TF dataset with only patches, and no labels.
        Optionally outputs filenames of the files in the dataset
        '''


        filenames = Path(audio_path).glob('*'+self.file_extension)
        filenames = [f for f in filenames if f.is_file()]
        X = []
        if return_clip_labels:
            clip_labels = []

        print("[INFO] : Loading dataset.")
        print(f"[INFO] : Loading {len(filenames)} audio clips.")
        n_patches_generated = 0
        for i, fname in enumerate(filenames):
            wave, _ = librosa.load(fname, sr=self.sr)
            wave = self.time_pipeline(wave)
            patches = self.freq_pipeline(wave)
            n_patches_generated += len(patches)
            X.extend(patches)
            if return_clip_labels :
                clip_labels.extend([i] * len(patches))

        X = np.stack(X, axis=0)
        if self.expand_last_dim:
            X = np.expand_dims(X, axis=-1)
        print(f"[INFO] : Generated {n_patches_generated} patches")
        
        if not return_arrays:
            ds = tf.data.Dataset.from_tensor_slices(X)
            ds = ds.batch(batch_size)
            if to_cache:
                ds = ds.cache()
            ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        else:
            ds = X
        
        out = [ds]
        if return_clip_labels:
            out.append(np.array(clip_labels))
        if return_filenames:
            out.append(filenames)
        return out
        
    def get_tf_datasets(self):
        """
        Build and return the set of TensorFlow datasets.

        Notes
        -----
        This base method is abstract and must be implemented by inheriting
        classes, which will define how training, validation, quantization,
        testing, and prediction datasets are constructed from provided CSVs and
        audio directories.

        Raises
        ------
        NotImplementedError
            Always raised in the base class. Implement in subclasses.
        """
        raise NotImplementedError("This method should be implemented in the inheriting class.")

    def _add_garbage_class_to_df(dataframe: pd.DataFrame,
                             classes_to_keep: List[str] = None,
                             n_samples_per_other_class: int = 2,
                             seed: int = 42,
                             shuffle: bool = True):
        '''
        Takes a dataframe, a list of classes, subsamples classes not in said list, 
        and changes their label to "other".
        NOTE : We expect the labels to be in a column named "category" (ESC format)
        
        Inputs
        ------
        dataframe : Pandas Dataframe. Initial dataframe.
        classes_to_keep : list of str. List of classes which should not
            be subsampled and have their label changed
        n_samples_per_other_class : int. How many samples of each class 
            NOT in classes_to_keep to keep after filtering.
        seed : int or numpy RandomGenerator : Seed for the random subsampling
        shuffle : bool. Set to true to shuffle the output dataframe. 
            If set to False, all the "other" samples will be at the bottom of the output dataframe.

        Outputs
        -------
        kept_samples : Pandas Dataframe. Dataframe which contains all samples of the classes
            in classes_to_keep, and the specified number of samples for the other classes.
            These samples have their labels changed to "other"
        '''
        # Determine which classes should be treated as "other" (i.e. ood classes)
        other_classes = set(dataframe['category'].unique()).difference(set(classes_to_keep))
        # Start by keeping samples of ind classes
        kept_samples = dataframe[dataframe['category'].isin(classes_to_keep)]
        # Keep only a certain amount of samples per ood class, 
        # and concatenate with the previous dataframe
        for ood_class in other_classes:
            sample = dataframe[dataframe['category'] == ood_class].sample(n=n_samples_per_other_class,
                                                                        random_state=seed,
                                                                        replace=True)
            # We don't check for duplicates because it slows stuff down 
            # and this should not introduce duplicates anyways
            kept_samples = pd.concat([kept_samples, sample], axis=0)

        # Finally, update the category column with the "other" label for anything not in
        # classes_to_keep.
        replacer = lambda s : "other" if (s in other_classes) else s
        kept_samples['category'] = kept_samples['category'].apply(replacer)

        if shuffle:
            kept_samples = kept_samples.sample(frac=1)

        return kept_samples

