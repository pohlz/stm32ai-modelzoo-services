# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
import torch
import librosa
import warnings
from pathlib import Path
import torchaudio
import numpy as np
import random
from math import floor
from sklearn.model_selection import train_test_split

"""Torch dataset class for the Valentini (Voicebank+DEMAND dataset)
   NOTE : May move the frequency data augmentation to a seperate dataloader
   or training loop, as doing it per sample instead of per-batch is expensive.
   You can use both torchaudio and librosa preprocessing pipelines. Note that using
   torchaudio will return torch tensors and using librosa will return numpy arrays.
"""

class CustomValentiniLike(torch.utils.data.Dataset):
    def __init__(self,
                 set: str, 
                 noisy_files_path: Path,
                 sample_rate: int,
                 input_pipeline,
                 target_pipeline = None,
                 clean_files_path: Path = None,
                 file_extension: str = ".wav",
                 return_transcript: bool = False,
                 txt_folder: Path = None,
                 time_data_aug = None,
                 freq_data_aug = None,
                 preproc_lib: str = "librosa",
                 device: str = "cpu",
                 n_clips=None,
                 val_split=None,
                 random_seed=42
                 ):
        '''
        Parameters
        ----------
        clean_files_path, str or Path : path to the folder containing clean audio files. 
            If None, only the noisy input is returned.
        noisy_files_path, str or Path : path to the folder containing noisy audio files,
        set, str : One of "train", "test", or "valid". If "train" or "valid", splits the dataset into two
                    and returns the appropriate one.
                    "test" is ignored (because the test set is assumed to come from different folders)
                    and is just accepted for implementation convenience
        sample_rate, int : Audio sampling rate. Dataset default is 48kHz, setting a different sampling rate
                            will resample the audio data.
        input_pipeline : Preprocessing pipeline used to process input to the model.
                            Pipelines need to be callable on a single waveform at a time.
                            See the preprocessing folder for examples of pipelines
        target_pipeline : Preprocessing pipeline used to process the target.
                            Pipelines need to be callable on a single waveform at a time.
                            See the preprocessing folder for examples of pipelines
        file_extension, str : File extension of audio files
        return_transcript, bool : if True, returns a transcript with each (noisy, clean) pair
                                    __getitem__ output thus becomes a (noisy, clean, transcript) tuple
        time_data_aug : Data augmentation pipeline for time domain data
        freq_data_aug : Data augmentation pipeline for frequency domain data
        preproc_lib, str : One of "librosa" or "torchaudio". 
                            Will use the corresponding library to load the data.
                            Use whatever is compatible with your pipeline.
        device, str : If using preproc_lib = "torchaudio", sends the loaded audio data to this device.
                        Useful when using GPU-accelerated preprocessing pipelines with torchaudio.
        n_clips, int or float : Number of audio clips to include in this dataset object. 
                        If None, includes all audio clips found in the dataset folder.
                        If an int, randomly samples n_clips from those found in the dataset folder.
                        If float, samples a fraction of clips.
        val_split, int or float : If set is 'train' or 'valid', number or fraction of clips to include in the validation set.
        random_seed, int : Random seed used for the random sampling of clips in case n_clips is not None.s
        '''
        super().__init__()
        self.clean_files_path = clean_files_path
        self.set = set
        self.noisy_files_path = noisy_files_path
        self.file_extension = file_extension
        self.sample_rate = sample_rate
        self.input_pipeline = input_pipeline
        self.target_pipeline = target_pipeline
        self.return_transcript = return_transcript
        self.time_data_aug = time_data_aug
        self.freq_data_aug = freq_data_aug
        self.device = device
        self.preproc_lib = preproc_lib
        self.n_clips = n_clips
        self.random_seed = random_seed
        self.val_split = val_split
        self.txt_folder = txt_folder
        random.seed(self.random_seed)
        # Some warnings
        # Warn if resampling because it's slow
        if self.sample_rate != 48000:
            warnings.warn("Sample rate is different from the original 48 kHz. \n"
                            f"Audio will be resampled to {self.sample_rate} Hz, which will incur additional overhead")
        # Warn if using GPU device and no torchaudio preprocessing 
        # To send already pre-processed tensors to the GPU, do this in the dataloader or training loop.
        if self.device != "cpu" and self.preproc_lib == "librosa":
            warnings.warn("Sending waveform tensor to GPU, but not using any torch processing pipeline."
                            "This will probably break a few things, set device='cpu' when instanciating this class")
            
        assert self.preproc_lib in ["librosa", "torchaudio"], "preproc_lib must be one of ['librosa', 'torchaudio']"
        # Load file list
        if self.clean_files_path:
            self.clean_file_list = list((self.clean_files_path).glob("*" + file_extension))

        self.noisy_file_list = list((self.noisy_files_path).glob("*" + file_extension))

        # Perform train/val split if needed
        if self.set in ["train", "valid"] and self.val_split > 0:
            if self.val_split is None:
                print("[INFO] No number of validation samples given" 
                      "using default value of 0.1")
                self.val_split = 0.1
            if self.clean_files_path:
                pairs = list(zip(self.clean_file_list, self.noisy_file_list))
                train_pairs, valid_pairs = train_test_split(pairs,
                                                            test_size=self.val_split,
                                                            random_state=self.random_seed)
                train_clean_file_list, train_noisy_file_list = zip(*train_pairs)
                valid_clean_file_list, valid_noisy_file_list = zip(*valid_pairs)
            else:
                train_noisy_file_list, valid_noisy_file_list = train_test_split(self.noisy_file_list,
                                                                                test_size=self.val_split,
                                                                                random_state=self.random_seed)
            if self.set == "train":
                if self.clean_files_path:
                    self.clean_file_list = train_clean_file_list
                    self.noisy_file_list = train_noisy_file_list
                else:
                    self.noisy_file_list = train_noisy_file_list
            elif self.set == "valid":
                if self.clean_files_path:
                    self.clean_file_list = valid_clean_file_list
                    self.noisy_file_list = valid_noisy_file_list
                else:
                    self.noisy_file_list = valid_noisy_file_list

        # Select random corresponding samples from both noisy and clean files
        if self.n_clips and self.set != "valid":
            print("[INFO] Loading a portion of the dataset")
            if self.clean_files_path:
                pairs = list(zip(self.clean_file_list, self.noisy_file_list))
                if isinstance(self.n_clips, int):
                    print(f"[INFO] Loading {self.n_clips} clips")
                    pairs = random.sample(pairs, self.n_clips)
                elif isinstance(self.n_clips, float):
                    print(f"[INFO] Loading {floor(self.n_clips * len(pairs))} clips")
                    pairs = random.sample(pairs, floor(self.n_clips * len(pairs)))
                self.clean_file_list, self.noisy_file_list = zip(*pairs)
            else:
                if isinstance(self.n_clips, int):
                    print(f"[INFO] Loading {self.n_clips} clips")
                    self.noisy_file_list = random.sample(self.noisy_file_list, n_clips)
                elif isinstance(self.n_clips, float):
                    print(f"[INFO] Loading {floor(self.n_clips * len(self.noisy_file_list))} clips")
                    self.noisy_file_list = random.sample(self.noisy_file_list, floor(n_clips * len(self.noisy_file_list)))

        if self.clean_files_path:
            assert len(self.clean_file_list) == len(self.noisy_file_list), \
            ("Different number of noisy and clean speech files. \n" 
            f"Found {len(self.clean_file_list)} clean files and {len(self.noisy_file_list)} noisy files")

    def __len__(self):
        return len(self.noisy_file_list)

    def __getitem__(self, idx):
        if self.preproc_lib == "librosa":
            # Load
            noisy_wave, sr = librosa.load(path = self.noisy_file_list[idx], sr=self.sample_rate)
            if self.clean_files_path:
                clean_wave, sr = librosa.load(path = self.clean_file_list[idx], sr=self.sample_rate)

        elif self.preproc_lib == "torchaudio":
            noisy_wave, sr = torchaudio.load(uri=self.noisy_file_list[idx], sr=self.sample_rate)
            if self.clean_files_path:
                clean_wave, sr = torchaudio.load(uri=self.clean_file_list[idx], sr=self.sample_rate)
            if self.device != "cpu": # It's already on CPU to begin with
                noisy_wave = noisy_wave.to(self.device)
                if self.clean_files_path:
                    clean_wave = clean_wave.to(self.device)

        # Apply time-domain data augmentation if there is any
        # I might move this to a dataloader or training loop later
        if self.time_data_aug:
            noisy_wave = self.time_data_aug(noisy_wave)

        # Apply preproc pipelines
        if isinstance(self.input_pipeline, list):
            preproc_input = [pipe(noisy_wave) for pipe in self.input_pipeline]
        else:
            preproc_input = self.input_pipeline(noisy_wave)

        if self.clean_files_path:
            preproc_target = self.target_pipeline(clean_wave)
            # Sometimes preproc_target becomes unwriteable and I have absolutely no idea why
            # So, make a copy
            try:
                preproc_target = np.copy(preproc_target)
            except:
                pass
        
        # Apply frequency-domain data augmentation if there is any
        if self.freq_data_aug:
            if isinstance(preproc_input, list):
                preproc_input = [self.freq_data_aug(inp) for inp in preproc_input]
            else:
                preproc_input = self.freq_data_aug(preproc_input)
        
        if self.return_transcript:
            txt_file = (self.txt_folder / self.clean_file_list[idx].parts[-1]).with_suffix(".txt")
            with open(txt_file, 'r') as f:
                transcript = f.readlines()
            if self.clean_files_path:
                return preproc_input, preproc_target, transcript
            else:
                return preproc_input, transcript

        else:
            if self.clean_files_path:
                return preproc_input, preproc_target
            else:
                return preproc_input


class Valentini(CustomValentiniLike):
    '''Torch dataset class for the Valentini (Voicebank + DEMAND) dataset.'''
    def __init__(self,
                 dataset_folder: Path,
                 set: str ,
                 n_speakers: int,
                 sample_rate: int,
                 input_pipeline,
                 target_pipeline = None,
                 return_transcript: bool = False,
                 time_data_aug = None,
                 freq_data_aug = None,
                 preproc_lib: str = "librosa",
                 device: str = "cpu",
                 n_clips=None,
                 random_seed=42,
                 val_split=None,
                 return_only_noisy=False
                 ):
        '''
        Parameters
        ----------
        dataset_folder, str or Path : path to the root folder of the dataset
        set, str : One of "train", "test", or "valid". If "train" or "valid", splits the dataset into two
                    and returns the appropriate one.
                    "test" is ignored (because the test set is assumed to come from different folders)
                    and is just accepted for implementation convenience
        n_speakers, int : Set to 28 to use the 28-speaker dataset, 56 to use the 56-speaker dataset
        sample_rate, int : Audio sampling rate. Dataset default is 48kHz, setting a different sampling rate
                            will resample the audio data.
        input_pipeline : Preprocessing pipeline used to process input to the model.
                         Pipelines need to be callable on a single waveform at a time.
                         See the preprocessing folder for examples of pipelines
        target_pipeline : Preprocessing pipeline used to process the target.
                         Pipelines need to be callable on a single waveform at a time.
                         See the preprocessing folder for examples of pipelines
        return_transcript, bool : if True, returns a transcript with each (noisy, clean) pair
                                  __getitem__ output thus becomes a (noisy, clean, transcript) tuple
        time_data_aug : Data augmentation pipeline for time domain data
        freq_data_aug : Data augmentation pipeline for frequency domain data
        preproc_lib, str : One of "librosa" or "torchaudio". 
                           Will use the corresponding library to load the data.
                           Use whatever is compatible with your pipeline.
        device, str : If using preproc_lib = "torchaudio", sends the loaded audio data to this device.
                      Useful when using GPU-accelerated preprocessing pipelines with torchaudio.
        n_clips, int : Number of audio clips to include in this dataset object. 
                       If None, includes all audio clips found in the dataset folder.
                       If an int, randomly samples n_clips from those found in the dataset folder.
        random_seed, int : Random seed used for the random sampling of clips in case n_clips is not None.
        return_only_noisy : If True, only returns the noisy input, and not a (noisy, clean) pair.
        val_split, int or float : If set is 'train' or 'valid', number or fraction of clips to include in the validation set.
        '''
        self.dataset_folder = Path(dataset_folder)
        self.set = set
        
        self.n_speakers = int(n_speakers)
        assert self.n_speakers in [28, 56], "n_speakers must be either 28 or 56"
        
        self.file_extension = ".wav"
        if self.set in ["train", "valid"]:
            clean_files_path = (self.dataset_folder / f"clean_trainset_{self.n_speakers}spk_wav")
            noisy_files_path = (self.dataset_folder / f"noisy_trainset_{self.n_speakers}spk_wav")
        elif self.set == "test":
            clean_files_path =  (self.dataset_folder / f"clean_testset_wav")
            noisy_files_path = (self.dataset_folder / f"noisy_testset_wav")
        else:
            raise ValueError(f"set must be one of 'train', 'valid', 'test', was {self.set}")
        
        if return_only_noisy:
            clean_files_path = None
            target_pipeline = None

        if self.set in ["train", "valid"]:
            self.txt_folder = self.dataset_folder / f"trainset_{self.n_speakers}spk_txt"
        elif self.set == "test":
            self.txt_folder = self.dataset_folder / f"testset_txt"
        else:
            raise ValueError(f"set must be one of 'train', 'valid', 'test', was {self.set}")

        super().__init__(clean_files_path=clean_files_path,
                         set=set,
                         noisy_files_path=noisy_files_path,
                         sample_rate=sample_rate,
                         input_pipeline=input_pipeline,
                         target_pipeline=target_pipeline,
                         file_extension=self.file_extension,
                         return_transcript=return_transcript,
                         txt_folder=self.txt_folder,
                         time_data_aug=time_data_aug,
                         freq_data_aug=freq_data_aug,
                         preproc_lib=preproc_lib,
                         device=device,
                         n_clips=n_clips,
                         val_split=val_split,
                         random_seed=random_seed)