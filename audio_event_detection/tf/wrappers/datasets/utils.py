# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

from omegaconf import DictConfig
from audio_event_detection.tf.src.preprocessing import (
    GSCWaveformPipeline,
    LibrosaMelSpecPatchesPipeline,
    LibrosaSilenceRemovalPipeline,
)
from math import floor

def get_pipelines(cfg: DictConfig):
    '''Grabs preprocessing args from config and instantiates preprocessing pipelines'''
    time_pipeline = _get_time_pipeline(cfg)
    freq_pipeline = _get_freq_pipeline(cfg)

    return time_pipeline, freq_pipeline

def _get_time_pipeline(cfg: DictConfig):
    ''' Grabs time domain preproc args from config and instantiates LibrosaSilenceRemovalPipeline'''
    preproc_section = cfg.preprocessing

    if preproc_section.get("gsc_default", False):
        return GSCWaveformPipeline(
            sr=preproc_section.get("target_rate", 16000),
            sample_length=preproc_section.get("sample_length", 1.0),
            random_position=preproc_section.get("random_position", True),
            time_shift_ms=preproc_section.get("time_shift_ms", 100),
            background_noise_path=preproc_section.get("background_noise_path", None),
            background_frequency=preproc_section.get("background_frequency", 0.8),
            background_volume=preproc_section.get("background_volume", 0.1),
            reserved_background_files=preproc_section.get("reserved_background_files", []),
            seed=cfg.dataset.get("seed", 123),
        )

    min_length = preproc_section.get("min_length", 1)
    max_length = preproc_section.get("max_length", 10)
    sr = preproc_section.get("target_rate", 16000)
    top_db = preproc_section.get("top_db", 30)
    frame_length = preproc_section.get("frame_length", 2048)
    hop_length = preproc_section.get("hop_length", 512)

    pipeline = LibrosaSilenceRemovalPipeline(
        min_length=min_length,
        max_length=max_length,
        sr=sr,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length,
        verbose=False
    )
    
    return pipeline
    
def _get_freq_pipeline(cfg: DictConfig):
    '''Grabs frequency domain preproc args from config and instantiates LibrosaMelSpecPatchesPipeline'''
    preproc_section = cfg.feature_extraction
    
    patch_length = preproc_section.get("patch_length", 96)
    overlap = preproc_section.get("overlap", 0.25)
    sr = cfg.preprocessing.get("target_rate", 16000)
    n_mels = preproc_section.get("n_mels", 64)
    n_fft = preproc_section.get("n_fft", 512)
    hop_length = preproc_section.get("hop_length", 160)
    win_length = preproc_section.get("window_length", 400)
    window = preproc_section.get("window", "hann")
    center = preproc_section.get("center", True)
    pad_mode = preproc_section.get("pad_mode", "constant")
    power = preproc_section.get("power", 1.0)
    fmin = preproc_section.get("fmin", 125)
    fmax = preproc_section.get("fmax", 7500)
    norm = preproc_section.get("norm", None)
    htk = preproc_section.get("htk", True) 
    to_db = preproc_section.get("to_db", False)

    overlap_frames = floor(patch_length * overlap)

    pipeline = LibrosaMelSpecPatchesPipeline(
        patch_length=patch_length,
        overlap_frames=overlap_frames,
        sr=sr,
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
        norm=norm,
        htk=htk,
        db_scale=to_db,
        log_scale=not to_db,
        peak_normalize=False
        )
    
    return pipeline
