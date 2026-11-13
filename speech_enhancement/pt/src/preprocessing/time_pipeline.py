# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

""" Time domain preprocessing pipelines"""
import librosa
import numpy as np
import copy
import torch

class IdentityPipeline:
    "Dummy pipeline in case you don't need to preprocess either the input or the target."
    def __init__(self, 
                 peak_normalize=False):
        self.peak_normalize = peak_normalize
    def __call__(self, wave):
        if isinstance(wave, torch.Tensor):
            if self.peak_normalize:
                wave /= torch.max(torch.abs(wave))
        elif isinstance(wave, np.ndarray):
            if self.peak_normalize:
                wave /= np.max(np.abs(wave))
        else:
            raise TypeError("input must be np.ndarray or torch.Tensor")
        return wave

class LibrosaFramingPipeline:
    """ Time domain pipeline that cuts the input signal into frames.
        Beware that the shape of the array will depend on the length of the input signal.
        To constrain the number of output frames, use the max_num_frames argument. 
        Inputs
        ------
        frame_length : int, number of samples per frame
        hop_length : int, number of steps to advance between frames
        max_num_frames : int, number of frames desired in the output. Default is None
            If None, output will have all frames. 
            If max_num_frames < n_frames, the last frames will be discarded
            If max_num_frames > n_frames, the output frames array will be zero-padded to have max_num_frames frames.
        axis : int, Frame axis in the output array. Passed to librosa.utils.frame.
            Note that if your input signal is multidimensional
            (e.g. (batch_size, wave_len)) different axes (e.g. axis=1 vs axis=-1) 
            can give different output shape
        copy : bool, whether a deep copy of the frames array should be returned.

    """
    def __init__(self, frame_length:int ,
                 hop_length:int ,
                 max_num_frames: int = None,
                 axis: int = 0,
                 copy: bool = False,
                 peak_normalize: bool = False):
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.max_num_frames = max_num_frames
        self.axis = axis
        self.copy_array = copy
        self.peak_normalize = peak_normalize

    def __call__(self, wave):
        if self.peak_normalize:
            wave /= np.max(np.abs(wave))
        frames = librosa.util.frame(wave,
                                    frame_length=self.frame_length,
                                    hop_length=self.hop_length,
                                    axis=self.axis)
        if self.copy_array:
        # Make a deep copy because librosa.util.frame only creates a view of its input array
        # And so changing frames would change the input signal array.
        # Probably useless in the vast majority of cases, but you never know.
            frames = copy.deepcopy(frames)
        
        # Zero-pad or clip the resulting frame array from the right.
        if self.max_num_frames is not None:
            n_frames = frames.shape[self.axis]
            frame_diff = self.max_num_frames - n_frames

            if frame_diff > 0:
                pad_lengths = [(0, 0)] * frames.ndim
                pad_lengths[self.axis] = (0, frame_diff)
                frames = np.pad(frames, pad_width=pad_lengths, mode="constant")
            else:
                frames = np.delete(arr=frames, obj=np.arange(self.max_num_frames, n_frames), axis=self.axis)
        return frames