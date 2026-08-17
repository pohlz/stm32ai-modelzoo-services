# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2026.
#  *
#  * This file adds Google Speech Commands waveform preprocessing to the
#  * STM32 Model Zoo Services audio-event-detection pipeline.
#  *--------------------------------------------------------------------------------------------*/

"""Google Speech Commands waveform preprocessing and augmentation."""

from pathlib import Path
import re
from typing import Iterable, Optional

import librosa
import numpy as np
import pandas as pd


class GSCWaveformPipeline:
    """Prepare and augment one-second Google Speech Commands waveforms.

    Training examples are zero padded/cropped to a fixed duration, shifted in
    time, and optionally mixed with a random segment from a training-only
    background-noise pool. Evaluation and quantization calls are deterministic.
    """

    gsc_default = True
    _SILENCE_SOURCE_RE = re.compile(r"^silence_\d+_(.+)_\d+\.wav$")

    def __init__(
        self,
        sr: int = 16000,
        sample_length: float = 1.0,
        random_position: bool = True,
        time_shift_ms: int = 100,
        background_noise_path: Optional[str] = None,
        background_frequency: float = 0.8,
        background_volume: float = 0.1,
        reserved_background_files: Optional[Iterable[str]] = None,
        seed: int = 123,
    ):
        self.sr = int(sr)
        self.num_samples = int(round(self.sr * float(sample_length)))
        self.random_position = bool(random_position)
        self.max_shift_samples = int(round(self.sr * float(time_shift_ms) / 1000.0))
        self.background_frequency = float(background_frequency)
        self.background_volume = float(background_volume)
        self.rng = np.random.default_rng(seed)

        if self.num_samples <= 0:
            raise ValueError("sample_length must produce at least one sample")
        if self.max_shift_samples < 0:
            raise ValueError("time_shift_ms must not be negative")
        if not 0.0 <= self.background_frequency <= 1.0:
            raise ValueError("background_frequency must be between 0 and 1")
        if not 0.0 <= self.background_volume <= 1.0:
            raise ValueError("background_volume must be between 0 and 1")

        self.background_noise_path = (
            Path(background_noise_path).resolve() if background_noise_path else None
        )
        self.reserved_background_files = {
            Path(name).name for name in (reserved_background_files or [])
        }
        self.reserved_background_stems = {
            Path(name).stem for name in self.reserved_background_files
        }
        self.training_background_files = []
        self.background_waves = []
        self._load_training_backgrounds()

    def _load_training_backgrounds(self) -> None:
        """Load only noise recordings that are not reserved for validation/test."""
        if self.background_frequency == 0.0:
            return
        if self.background_noise_path is None:
            raise ValueError(
                "background_noise_path is required when background_frequency is non-zero"
            )
        if not self.background_noise_path.is_dir():
            raise FileNotFoundError(
                f"Background-noise directory not found: {self.background_noise_path}"
            )

        all_sources = sorted(self.background_noise_path.glob("*.wav"))
        known_names = {path.name for path in all_sources}
        unknown_reserved = self.reserved_background_files - known_names
        if unknown_reserved:
            raise FileNotFoundError(
                "Reserved background WAV files were not found: "
                f"{sorted(unknown_reserved)}"
            )

        training_sources = [
            path for path in all_sources if path.name not in self.reserved_background_files
        ]
        if not self.reserved_background_files:
            raise ValueError(
                "At least one background WAV must be reserved for validation/test"
            )
        if not training_sources:
            raise ValueError(
                "All background WAV files are reserved; at least one is required for training"
            )

        for source_path in training_sources:
            wave, _ = librosa.load(source_path, sr=self.sr, mono=True)
            wave = np.asarray(wave, dtype=np.float32)
            if len(wave) >= self.num_samples:
                self.training_background_files.append(source_path.name)
                self.background_waves.append(wave)

        if not self.background_waves:
            raise ValueError(
                "No non-reserved background WAV is long enough for one input window"
            )

        print(
            "[INFO] : GSC training background WAVs: "
            f"{self.training_background_files}"
        )
        print(
            "[INFO] : GSC validation/test background WAVs: "
            f"{sorted(self.reserved_background_files)}"
        )

    def fix_length(self, wave: np.ndarray, training: bool = False) -> np.ndarray:
        """Crop or zero-pad a waveform to exactly ``num_samples`` samples."""
        wave = np.asarray(wave, dtype=np.float32).reshape(-1)
        target = self.num_samples

        if len(wave) > target:
            max_start = len(wave) - target
            start = int(self.rng.integers(0, max_start + 1)) if training else max_start // 2
            return wave[start : start + target]

        if len(wave) < target:
            missing = target - len(wave)
            if training and self.random_position:
                left = int(self.rng.integers(0, missing + 1))
            else:
                left = 0
            return np.pad(wave, (left, missing - left), mode="constant")

        return wave.copy()

    def time_shift(self, wave: np.ndarray, training: bool = False) -> np.ndarray:
        """Shift training audio without wraparound, filling uncovered samples with zero."""
        if not training or self.max_shift_samples == 0:
            return wave

        shift = int(
            self.rng.integers(-self.max_shift_samples, self.max_shift_samples + 1)
        )
        shifted = np.zeros_like(wave)
        if shift > 0:
            shifted[shift:] = wave[:-shift]
        elif shift < 0:
            shifted[:shift] = wave[-shift:]
        else:
            shifted[:] = wave
        return shifted

    def sample_background(self) -> np.ndarray:
        """Select a random fixed-length segment from a training-only noise source."""
        source = self.background_waves[int(self.rng.integers(0, len(self.background_waves)))]
        max_start = len(source) - self.num_samples
        start = int(self.rng.integers(0, max_start + 1)) if max_start else 0
        return source[start : start + self.num_samples]

    def mix_background(
        self,
        wave: np.ndarray,
        label: Optional[str],
        training: bool = False,
    ) -> np.ndarray:
        """Apply the GSC reference random background mixture to non-silence classes."""
        if (
            not training
            or label == "silence"
            or self.background_frequency == 0.0
            or self.rng.random() >= self.background_frequency
        ):
            return wave

        gain = float(self.rng.uniform(0.0, self.background_volume))
        mixed = wave + gain * self.sample_background()
        return np.clip(mixed, -1.0, 1.0).astype(np.float32)

    def __call__(
        self,
        wave: np.ndarray,
        label: Optional[str] = None,
        training: bool = False,
    ) -> np.ndarray:
        wave = self.fix_length(wave, training=training)
        wave = self.time_shift(wave, training=training)
        wave = self.mix_background(wave, label=label, training=training)
        return np.asarray(wave, dtype=np.float32)

    @classmethod
    def silence_sources(cls, dataframe: pd.DataFrame) -> set[str]:
        """Extract source-noise stems encoded in generated silence filenames."""
        sources = set()
        silence_rows = dataframe[dataframe["category"] == "silence"]
        for filename in silence_rows["filename"].astype(str):
            match = cls._SILENCE_SOURCE_RE.match(Path(filename).name)
            if not match:
                raise ValueError(
                    "Cannot verify GSC silence provenance from filename: "
                    f"{filename}. Regenerate the CSVs with prepare_st_csv.py."
                )
            sources.add(match.group(1))
        return sources

    def validate_silence_sources(self, dataframe: pd.DataFrame, split_name: str) -> None:
        """Reject source leakage between training and validation/test silence rows."""
        sources = self.silence_sources(dataframe)
        if not sources:
            raise ValueError(f"GSC split {split_name!r} has no silence examples")

        if split_name in {"training", "quantization"}:
            leaked = sources & self.reserved_background_stems
            if leaked:
                raise ValueError(
                    f"Reserved validation/test background sources leak into {split_name}: "
                    f"{sorted(leaked)}. Regenerate the CSVs with prepare_st_csv.py."
                )
        elif split_name in {"validation", "test"}:
            unreserved = sources - self.reserved_background_stems
            if unreserved:
                raise ValueError(
                    f"Non-reserved training background sources occur in {split_name}: "
                    f"{sorted(unreserved)}. Regenerate the CSVs with prepare_st_csv.py."
                )

