# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

""" Frequency domain preprocessing pipelines"""
import librosa
import numpy as np
import warnings
import torch
import torch.nn as nn

class LibrosaSpecPipeline:
    """Wrapper around the Librosa STFT function.
       Returns the whole spectrogram corresponding to the input waveform, and not patches."""
    def __init__(self,
                 sample_rate: int, 
                 n_fft: int, 
                 hop_length: int,
                 win_length: int = None,
                 window: str = 'hann',
                 center: bool = True,
                 pad_mode : str = 'constant',
                 magnitude : bool = True,
                 power : int = 1.0,
                 peak_normalize: bool = False
                 ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode
        self.magnitude = magnitude
        self.power = power
        self.peak_normalize = peak_normalize

    def __call__(self, wave):
        if self.peak_normalize:
            wave /= np.max(np.abs(wave))
        spec = librosa.stft(wave,
                            n_fft=self.n_fft,
                            hop_length=self.hop_length,
                            win_length=self.win_length,
                            window=self.window,
                            center=self.center,
                            pad_mode=self.pad_mode)
        if self.magnitude:
            spec = np.abs(spec)**self.power
        return spec
    
class TorchSpecPipeline(nn.Module):
    def __init__(self, 
                 sample_rate: int, 
                 n_fft: int, 
                 hop_length: int,
                 win_length: int = None,
                 window: str = 'hann',
                 center: bool = True,
                 pad_mode : str = 'constant',
                 magnitude : bool = True,
                 normalized : bool = False,
                 power : int = 1.0,
                 return_complex : bool = True,
                 peak_normalize: bool = False
                 ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode
        self.magnitude = magnitude
        self.normalized = normalized
        self.power = power
        self.return_complex = return_complex
        self.peak_normalize = peak_normalize

    def forward(self, wave):
        
        if type(self.window) not in [np.ndarray, torch.Tensor]:
            # If window is a string or a tuple, pass to librosa.filters.get_window
            # instead of trying to match it to one of the window functions
            # implemented in torch
            self.window = torch.Tensor(librosa.filters.get_window(self.window, Nx=self.win_length))
        # Convert wave to tensor if needed
        if not isinstance(wave, torch.Tensor):
            try:
                wave = torch.Tensor(wave)
            except:
                pass
        if self.peak_normalize:
            wave /= torch.max(torch.abs(wave))
        spec = torch.stft(wave,
                            n_fft=self.n_fft,
                            hop_length=self.hop_length,
                            win_length=self.win_length,
                            window=self.window,
                            center=self.center,
                            pad_mode=self.pad_mode,
                            normalized=self.normalized,
                            return_complex=self.return_complex
                            )
        if self.magnitude:
            spec = torch.abs(spec)**self.power
        return spec


class LibrosaMelSpecPipeline:
    """Wrapper around the Librosa melspectrogram function.
       Returns the whole spectrogram corresponding to the input waveform, and not patches."""
    def __init__(self,
                 sample_rate: int, 
                 n_fft: int, 
                 hop_length: int,
                 win_length: int = None,
                 window: str = 'hann',
                 center: bool = True,
                 pad_mode : str = 'constant',
                 power: float = 2.0,
                 n_mels: int = 64,
                 fmin:int = 20,
                 fmax:int = 20000,
                 power_to_db_ref = np.max,
                 norm: str = None,
                 htk: bool = True,
                 db_scale: bool = False,
                 log_scale: bool = True,
                 peak_normalize: bool = False):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode
        self.power = power
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.power_to_db_ref = power_to_db_ref
        self.norm = norm
        self.htk = htk
        self.db_scale = db_scale
        self.log_scale = log_scale
        self.peak_normalize = peak_normalize

        if self.db_scale and self.log_scale:
            raise ValueError("Both db_scale and log_scale were set to True, but are mutually exclusive.")
        if self.db_scale and self.power_to_db_ref != np.max:
            warnings.warn("Decibel scale was chosen, but power_to_db_ref was not set to np.max \n"
                          "Resulting spectrogram will not be in DbFS. Ignore this warning if this is deliberate \n"
                          "Or if you used some other maximum function")
        
    def __call__(self, wave, **kwargs):
        if self.peak_normalize:
            wave /= np.max(np.abs(wave))
        # Additional kwargs go to librosa.feature.melspectrogram()
        melspec = librosa.feature.melspectrogram(y=wave, 
                                                 sample_rate=self.sample_rate, 
                                                 n_fft=self.n_fft,
                                                 hop_length=self.hop_length,
                                                 win_length=self.win_length,
                                                 window=self.window,
                                                 center=self.center,
                                                 pad_mode=self.pad_mode,
                                                 power=self.power,
                                                 n_mels=self.n_mels,
                                                 fmin=self.fmin,
                                                 fmax=self.fmax,
                                                 norm=self.norm,
                                                 htk=self.htk,
                                                 **kwargs)
    
        if self.db_scale:
            if self.power == 2.0:
                db_melspec = librosa.power_to_db(melspec, ref=self.power_to_db_ref)
            elif self.power == 1.0:
                db_melspec = librosa.amplitude_to_db(melspec, ref=self.power_to_db_ref)
            else:
                raise ValueError('Power must be either 2.0 or 1.0')
            return db_melspec
        
        elif self.log_scale:
            return np.log(melspec + 1e-6) 
        
        
        
class LibrosaMFCCPipeline:
    """Wrapper around librosa.feature.mfcc
       Allows for a bit more customization than a direct call to librosa.feature.mfcc 
       with the waveform as input"""

    def __init__(self,
                 sample_rate: int, 
                 n_fft: int, 
                 hop_length: int,
                 win_length: int = None,
                 window: str = 'hann',
                 center: bool = True,
                 pad_mode : str = 'constant',
                 n_mels: int = 64,
                 fmin:int = 20,
                 fmax:int = 20000,
                 power_to_db_ref = np.max,
                 mel_norm: str = None,
                 htk: bool = True,
                 n_mfcc : int = 16,
                 dct_type : int = 2,
                 dct_norm = 'ortho',
                 lifter : float = 0,
                 peak_normalize: bool = False):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.power_to_db_ref = power_to_db_ref
        self.mel_norm = mel_norm
        self.htk = htk
        self.n_mfcc = n_mfcc
        self.dct_type = dct_type
        self.dct_norm = dct_norm
        self.lifter = lifter
        self.peak_normalize = peak_normalize

    def __call__(self, wave):
        if self.peak_normalize:
            wave /= np.max(np.abs(wave))
        melspec = librosa.feature.melspectrogram(y=wave, 
                                                 sample_rate=self.sample_rate, 
                                                 n_fft=self.n_fft,
                                                 hop_length=self.hop_length,
                                                 win_length=self.win_length,
                                                 window=self.window,
                                                 center=self.center,
                                                 pad_mode=self.pad_mode,
                                                 power=2.0,
                                                 n_mels=self.n_mels,
                                                 fmin=self.fmin,
                                                 fmax=self.fmax,
                                                 norm=self.mel_norm,
                                                 htk=self.htk)
        
        melspec = librosa.power_to_db(melspec, ref=self.power_to_db_ref)
        mfccs = librosa.feature.mfcc(S=melspec,
                             sample_rate=self.sample_rate,
                             n_mfcc=self.n_mfcc,
                             dct_type=self.dct_type,
                             norm=self.dct_norm,
                             lifter=self.lifter
                             )
        
        return mfccs