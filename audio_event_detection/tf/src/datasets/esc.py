# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

from .base import BaseAEDTFDataset
import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split

class ESC10TFDataset(BaseAEDTFDataset):
    """
    ESC-10 dataset integration for TensorFlow AED pipelines.

    Notes
    -----
    - Requires CSVs with `filename` and `category` columns; ESC fold information
        is expected when fold-based validation is used.
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
                 quantization_split: float = 1.0,
                 test_csv_path: str = None,
                 test_audio_path: str = None,
                 pred_audio_path: str = None,
                 class_names: List[str] = None,
                 use_garbage_class: bool = False,
                 file_extension: str = ".wav",
                 n_samples_per_garbage_class: int = None,
                 expand_last_dim: bool = True,
                 seed: int = 133):
        """
        Initialize the ESC-10 dataset wrapper.

        Parameters
        ----------
        time_pipeline : callable
            Time-domain preprocessing callable applied to raw waveforms.
        freq_pipeline : callable
            Frequency-domain conversion callable producing spectrogram patches.
        training_csv_path : str, optional
            Path to training CSV (ESC-like format with optional `fold`).
        training_audio_path : str, optional
            Directory containing training audio files.
        validation_csv_path : str, optional
            Path to validation CSV; if None, `validation_split` or fold 5 is used.
        validation_audio_path : str, optional
            Directory containing validation audio files; defaults to training
            directory if not provided.
        validation_split : float, optional
            Proportion (0-1) used for stratified train/validation split when no
            validation CSV is provided.
        quantization_csv_path : str, optional
            Path to CSV for building a quantization representative dataset.
        quantization_audio_path : str, optional
            Directory containing audio for the quantization dataset.
        quantization_split : float, default 1.0
            Fraction of the quantization dataset to keep.
        test_csv_path : str, optional
            Path to testing CSV.
        test_audio_path : str, optional
            Directory containing test audio files.
        pred_audio_path : str, optional
            Directory of unlabeled audio to build a prediction dataset.
        class_names : List[str], optional
            List of classes to include; required when training CSV is provided.
        use_garbage_class : bool, default False
            If True, aggregate out-of-scope classes into an `other` class.
        file_extension : str, default ".wav"
            Audio file extension to append when needed.
        n_samples_per_garbage_class : int, optional
            Number of samples per out-of-scope class retained in `other`; if None,
            a balanced heuristic is used.
        expand_last_dim : bool, default True
            If True, expand spectrogram patches with a trailing singleton channel.
        seed : int, default 133
            Random seed for reproducible operations.
        """
    
        super().__init__(
            time_pipeline=time_pipeline,
            freq_pipeline=freq_pipeline,
            training_csv_path=training_csv_path,
            training_audio_path=training_audio_path,
            validation_csv_path=validation_csv_path,
            validation_audio_path=validation_audio_path,
            validation_split=validation_split,
            quantization_csv_path=quantization_csv_path,
            quantization_audio_path=quantization_audio_path,
            test_csv_path=test_csv_path,
            test_audio_path=test_audio_path,
            pred_audio_path=pred_audio_path,
            use_garbage_class=use_garbage_class,
            file_extension=file_extension,
            n_samples_per_garbage_class=n_samples_per_garbage_class,
            expand_last_dim=expand_last_dim,
            seed=seed)
        
        self.class_names = class_names
        self.quantization_split = quantization_split

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

        if self.training_csv_path:
            if not self.class_names:
                raise ValueError("Argument class_names was not provided ! \
                            Please provide at least two classes in class_names")
            else:
                used_classes = self.class_names
        
            esc_df = pd.read_csv(self.training_csv_path)

            # If a validation path is provided, use it
            # If a validation split is provided, use it
            # If none of these are provided, use fold 5 as validation set.
            if  self.validation_csv_path is not None:
                train_esc_df = esc_df
                val_esc_df = pd.read_csv(self.validation_csv_path)

            elif self.validation_split is not None:
                train_esc_df, val_esc_df = train_test_split(esc_df, test_size=self.validation_split,
                                                            random_state=self.seed, stratify=esc_df['category'])

            else:
                # Use fold 5
                train_esc_df, val_esc_df = esc_df[esc_df["fold"] != 5], esc_df[esc_df["fold"] == 5]
            print("[INFO] : Loading training dataset")

            train_ds = self.get_ds(
                df=train_esc_df,
                audio_path=self.training_audio_path,
                used_classes=used_classes,
                batch_size=batch_size,
                to_cache=to_cache,
                shuffle=shuffle,
                return_clip_labels=False,
                return_arrays=False
                )
            
            # Use validation audio path if provided
            audio_path = self.validation_audio_path if self.validation_audio_path else self.training_audio_path
            print("[INFO] : Loading validation dataset")
            val_ds, val_clip_labels = self.get_ds(
                df=val_esc_df,
                audio_path=audio_path,
                used_classes=used_classes,
                batch_size=batch_size,
                to_cache=to_cache,
                shuffle=False,
                return_clip_labels=True,
                return_arrays=False
                )
        else:
            train_ds = None
            val_ds = None
            val_clip_labels = None

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
        

