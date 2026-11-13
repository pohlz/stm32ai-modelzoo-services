###############################################################################
#
# Copyright(c) 2025 STMicroelectronics.
# All rights reserved.
#
# This software is licensed under terms that can be found in the LICENSE file
# in the root directory of this software component.
# If no LICENSE file comes with this software, it is provided AS-IS.
#
###############################################################################
""" Frequency domain preprocessing pipelines"""
import librosa
import numpy as np
import warnings

class LibrosaSpecPipeline:
    """Wrapper around the Librosa STFT function.
       Returns the whole spectrogram corresponding to the input waveform, and not patches."""
    def __init__(self,
                 sr: int, 
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
        self.sr = sr
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


class LibrosaMelSpecPipeline:
    """Wrapper around the Librosa melspectrogram function.
       Returns the whole spectrogram corresponding to the input waveform, and not patches."""
    def __init__(self,
                 sr: int, 
                 n_fft: int, 
                 hop_length: int,
                 win_length: int = None,
                 window: str = 'hann',
                 center: bool = True,
                 pad_mode : str = 'constant',
                 power: float = 2.0,
                 n_mels: int = 64,
                 fmin: int = 0.0,
                 fmax:int = None,
                 power_to_db_ref = np.max,
                 norm: str = None,
                 htk: bool = True,
                 db_scale: bool = False,
                 log_scale: bool = True,
                 peak_normalize: bool = False):
        self.sr = sr
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
                                                 sr=self.sr, 
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
        

class LibrosaMelSpecPatchesPipeline(LibrosaMelSpecPipeline):
    """Wrapper around the Librosa melspectrogram function.
       Returns spectrogram patches."""  
    def __init__(self,
                 patch_length: int,
                 overlap_frames: int,
                 sr: int, 
                 n_fft: int, 
                 hop_length: int,
                 win_length: int = None,
                 window: str = 'hann',
                 center: bool = True,
                 pad_mode : str = 'constant',
                 power: float = 2.0,
                 n_mels: int = 64,
                 fmin: int = 0,
                 fmax:int = None,
                 power_to_db_ref = np.max,
                 norm: str = None,
                 htk: bool = True,
                 db_scale: bool = False,
                 log_scale: bool = True,
                 peak_normalize: bool = False):
        
        super().__init__(sr=sr, 
                 n_fft=n_fft,
                 hop_length=hop_length,
                 win_length=win_length,
                 window=window,
                 center=center,
                 pad_mode=pad_mode,
                 power=power,
                 n_mels=n_mels,
                 fmin=fmin,
                 fmax=fmax,
                 power_to_db_ref=power_to_db_ref,
                 norm=norm,
                 htk=htk,
                 db_scale=db_scale,
                 log_scale=log_scale,
                 peak_normalize=peak_normalize)
        
        self.patch_length = patch_length
        self.overlap_frames = overlap_frames

    def __call__(self, wave, **kwargs):
        if self.peak_normalize:
            wave /= np.max(np.abs(wave))
        # Additional kwargs go to librosa.feature.melspectrogram()
        melspec = librosa.feature.melspectrogram(y=wave, 
                                                    sr=self.sr, 
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
                melspec = librosa.power_to_db(melspec, ref=self.power_to_db_ref)
            elif self.power == 1.0:
                melspec = librosa.amplitude_to_db(melspec, ref=self.power_to_db_ref)
            else:
                raise ValueError('Power must be either 2.0 or 1.0')
        
        elif self.log_scale:
            melspec = np.log(melspec + 1e-6) 

        
        
        # Cut melspec into patches
        patches = librosa.util.frame(melspec,
                                      frame_length=self.patch_length,
                                      hop_length=self.patch_length - self.overlap_frames)
        
        patches = np.transpose(patches, axes=(2, 0, 1))
        return patches
    
class LibrosaMFCCPatchesPipeline:
    def __init__(self,
                patch_length: int,
                overlap_frames: int,
                sr: int, 
                n_fft: int, 
                hop_length: int,
                n_mfcc: int = 20,
                dct_type: int = 2,
                win_length: int = None,
                window: str = 'hann',
                center: bool = True,
                pad_mode : str = 'constant',
                power: float = 2.0,
                n_mels: int = 64,
                fmin: int = 0,
                fmax:int = None,
                power_to_db_ref = np.max,
                mfcc_norm: str = 'ortho',
                norm:str = 'slaney',
                htk: bool = True,
                peak_normalize: bool = False,
                **kwargs):
        """Wrapper around the Librosa MFCC function.
        Returns MFCC patches.""" 
         
        self.patch_length = patch_length
        self.overlap_frames = overlap_frames
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc
        self.dct_type = dct_type
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
        self.peak_normalize = peak_normalize
        self.mfcc_norm = mfcc_norm

    def __call__(self, wave):
        if self.peak_normalize:
            wave /= np.max(np.abs(wave))

        mfccs = librosa.feature.mfcc(
            y=wave,
            n_mfcc=self.n_mfcc,
            dct_type=self.dct_type,
            norm=self.mfcc_norm,
            mel_norm=self.norm,
            sr=self.sr,
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
            htk=self.htk)
        
        patches = librosa.util.frame(mfccs,
                                      frame_length=self.patch_length,
                                      hop_length=self.patch_length - self.overlap_frames)
        
        patches = np.transpose(patches, axes=(2, 0, 1))

        return patches