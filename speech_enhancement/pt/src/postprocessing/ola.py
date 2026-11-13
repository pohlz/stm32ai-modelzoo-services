# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import numpy as np
from librosa.filters import get_window
"""Time-domain overlap-add and reconstruction. For frequency domain please use librosa.istft instead"""

def overlap_add(frames, hop_length, window=None, output_length=None):
    """
    Overlap-add in the time domain.
    Accepts windowed signals. Functions as an inverse to librosa.utils.frames
	NOTE : The reconstructed signal will likely not be the same length
	as the input signal to librosa.util.frame, because some samples at the 
	end of the signal are discarded during the framing operation.
	The exact number of samples discarded is (signal_length - frame_length) % hop_length.
	To avoid this you can clip the input to librosa.util.frame so that 
	(signal_length - frame_length) % hop_length = 0

	Inputs
	------
	frames : np.ndarray of shape (..., n_frames, frame_length)
		Input signal frames. Frames are assumed to be the same length.
	hop_length : hop length between frames
    window : string, tuple, number, callable, or list-like
        - If string, it's the name of the window function (e.g., `'hann'`)
        - If tuple, it's the name of the window function and any parameters
          (e.g., `('kaiser', 4.0)`)
        - If numeric, it is treated as the beta parameter of the `'kaiser'`
          window, as in `scipy.signal.get_window`.
        - If callable, it's a function that accepts one integer argument
          (the window length, here frame_length)
        - If list-like, it's a pre-computed window of the correct length `frame_length`
        NOTE : This is passed to librosa.filter.get_window, so anything you'd use in a "window" arg
        in librosa works here.
    output_length : Desired signal length of reconstructed output. 
		If specified, the reconstructed output will be clipped or zero-padded
        to the right to fit output_length.
        If None, the reconstructed output will be of 
        shape (..., n_samples) where n_samples = frame_length + (hop_length - 1) * n_frames
	Outputs
	-------
	out : np.ndarray of shape (..., n_samples) where n_samples = frame_length + (hop_length - 1) * n_frames
		 or n_samples=output_length if output_length was specified.

    """
    n_frames = frames.shape[-2]
    frame_length = frames.shape[-1]
    out_len = frame_length + (n_frames - 1) * hop_length
    out_shape = frames.shape[:-2] + (out_len,)
    out = np.zeros(out_shape)

    current_sample = 0
    normalization_factor = np.zeros_like(out)
    if window is None:
        # Constant pseudo-window equal to 1
        window = np.ones_like(frames[..., 0, :])
    else:
        window = get_window(window, Nx=frame_length)

    for i in range(n_frames):
        # Overlap-add
        out[..., current_sample:current_sample+frame_length] += frames[..., i, :]
        # Normalize by sum of window functions.
        # Doesn't need to be the square of the window unlike in the frequency domain
        # Because we don't have an ill-posed inverse problem here.
        normalization_factor[..., current_sample:current_sample+frame_length] += window
        current_sample += hop_length
    # Normalize and avoid division by zero if there is any
    nonzero_pos = normalization_factor != 0
    out = out[nonzero_pos] / normalization_factor[nonzero_pos]

    # Clip or pad to fit user_provided output_length
    if output_length is not None and output_length != out_len:
        if out_len > output_length:
            out = out[..., :out_len]
        elif out_len < output_length:
            pad_lengths = [(0, 0)] * out.ndim
            pad_lengths[-1] = (0, output_length-out_len)
            out = np.pad(out, pad_width=pad_lengths, mode="constant")

    return out

def reconstruct(frames, hop_length, output_length=None):
    """
    Signal reconstruction in the time domain.
	Assumes the time-domain signal has not been windowed.
	Intended as an inverse of librosa.util.frame.
	NOTE : The reconstructed signal will likely not be the same length
	as the input signal to librosa.util.frame, because some samples at the 
	end of the signal are discarded during the framing operation.
	The exact number of samples discarded is (signal_length - frame_length) % hop_length.
	To avoid this you can clip the input to librosa.util.frame so that 
	(signal_length - frame_length) % hop_length = 0

	Inputs
	------
	frames : np.ndarray of shape (..., n_frames, frame_length)
		Input signal frames. Frames are assumed to be the same length.
	hop_length : hop length between frames
    output_length : Desired signal length of reconstructed output. 
		If specified, the reconstructed output will be clipped or zero-padded
        to the right to fit output_length.
        If None, the reconstructed output will be of 
        shape (..., n_samples) where n_samples = frame_length + (hop_length - 1) * n_frames
	Outputs
	-------
	out : np.ndarray of shape (..., n_samples) where n_samples = frame_length + (hop_length - 1) * n_frames
		 or n_samples=output_length if output_length was specified.

    """

    n_frames = frames.shape[-2]
    frame_length = frames.shape[-1]
    out_len = frame_length + (n_frames - 1) * hop_length
    out_shape = frames.shape[:-2] + (out_len,)
    out = np.zeros(out_shape)
    for i in range(n_frames):
        if i == 0:
            out[..., :frame_length] = frames[..., 0, :]
        else:
            out[..., frame_length + (i-1) * hop_length:frame_length + i * hop_length] = frames[..., i, -hop_length:]
    if output_length is not None and output_length != out_len:
        if out_len > output_length:
            out = out[..., :out_len]
        elif out_len < output_length:
            pad_lengths = [(0, 0)] * out.ndim
            pad_lengths[-1] = (0, output_length-out_len)
            out = np.pad(out, pad_width=pad_lengths, mode="constant")
    return out


    




    
        