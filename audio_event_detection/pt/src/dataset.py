"""GSC CSV adapter using the existing waveform policy and log-mel features.

The waveform module itself has no TensorFlow dependency. Load it directly to
avoid the TensorFlow imports in its parent package's __init__.
"""

import importlib.util
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

_source = Path(__file__).resolve().parents[2] / "tf/src/preprocessing/gsc_pipeline.py"
_spec = importlib.util.spec_from_file_location("aed_gsc_waveform", _source)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
GSCWaveformPipeline = _module.GSCWaveformPipeline


class GSCDataset(Dataset):
    def __init__(self, cfg, split):
        self.training = split == "training"
        self.cfg = cfg
        ds, pre = cfg.dataset, cfg.preprocessing
        self.root = Path(ds[f"{split}_audio_path"])
        self.rows = pd.read_csv(ds[f"{split}_csv_path"])
        if not {"filename", "category"}.issubset(self.rows.columns) or self.rows.empty:
            raise ValueError(f"{split} CSV must have filename/category columns and samples")
        self.labels = {name: i for i, name in enumerate(ds.class_names)}
        unknown = set(self.rows.category) - self.labels.keys()
        if unknown:
            raise ValueError(f"Unmapped CSV categories: {sorted(unknown)}")
        self.waveform = GSCWaveformPipeline(
            sr=pre.target_rate, sample_length=pre.sample_length,
            random_position=pre.random_position, time_shift_ms=pre.time_shift_ms,
            background_noise_path=pre.background_noise_path,
            background_frequency=pre.background_frequency if self.training else 0,
            background_volume=pre.background_volume,
            reserved_background_files=pre.reserved_background_files,
            seed=ds.seed,
        )
        self.waveform.validate_silence_sources(self.rows, split)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        path = self.root / row.filename
        if not path.suffix:
            path = path.with_suffix(self.cfg.dataset.file_extension)
        wave, _ = librosa.load(path, sr=self.cfg.preprocessing.target_rate, mono=True)
        wave = self.waveform(wave, label=row.category, training=self.training)
        f = self.cfg.feature_extraction
        mel = librosa.feature.melspectrogram(
            y=wave, sr=self.cfg.preprocessing.target_rate, n_fft=f.n_fft,
            hop_length=f.hop_length, win_length=f.window_length, window=f.window,
            center=f.center, pad_mode=f.pad_mode, power=f.power, n_mels=f.n_mels,
            fmin=f.fmin, fmax=f.fmax, norm=f.norm, htk=f.htk,
        )
        if mel.shape[1] < f.patch_length:
            raise ValueError(f"Only {mel.shape[1]} frames available for patch_length={f.patch_length}")
        # Same first complete patch and natural log as the GSC TF configuration.
        mel = np.log(mel[:, :f.patch_length] + 1e-6).astype(np.float32)
        return torch.from_numpy(mel[None]), self.labels[row.category]
