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
import librosa
import numpy as np

class LibrosaSilenceRemovalPipeline:
    '''Removes silence from a waveform, and if necessary, trims/repeat the waveform
       in order to fit a minimum and maximum duration.'''
    
    def __init__(self,
                 min_length: int = 1, 
                 max_length: int = 10,
                 sr: int = 16000 ,
                 top_db: int = 30,
                 frame_length: int = 2048,
                 hop_length: int = 512,
                 verbose: bool = False
                 ):
        
        self.min_length = min_length
        self.max_length = max_length
        self.sr = sr
        self.top_db = top_db
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.verbose = verbose
    def __call__(self, wave):
        intervals = librosa.effects.split(wave,
                                          top_db=self.top_db,
                                          frame_length=self.frame_length,
                                          hop_length=self.hop_length)

        wave = np.concatenate([wave[interval[0]:interval[1]] for interval in intervals])

        if len(wave) > self.sr * self.max_length:
            wave = wave[:self.max_length*self.sr]
        
        if len(wave) < self.sr * self.min_length:
            n_repeats = self.sr * self.min_length // len(wave)
            wave = np.tile(wave, n_repeats + 1)
            wave = wave[:self.min_length*self.sr]
            if self.verbose:
                print(f"Waveform was shorter than minimum length, repeated {n_repeats} times")


        return wave