"""Focused tests for Google Speech Commands preprocessing."""

import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from audio_event_detection.tf.src.preprocessing import (  # noqa: E402
    GSCWaveformPipeline,
    LibrosaMelSpecPatchesPipeline,
)
from audio_event_detection.tf.src.datasets.base import BaseAEDTFDataset  # noqa: E402


def write_pcm16(path: Path, samples: np.ndarray, sample_rate: int = 16000) -> None:
    samples = np.clip(samples, -1.0, 1.0)
    pcm = np.round(samples * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


class GSCWaveformPipelineTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.noise_dir = self.root / "_background_noise_"
        self.noise_dir.mkdir()

        samples = np.linspace(-0.5, 0.5, 32000, dtype=np.float32)
        write_pcm16(self.noise_dir / "training_noise.wav", samples)
        write_pcm16(self.noise_dir / "second_training_noise.wav", samples[::-1])
        write_pcm16(self.noise_dir / "reserved_noise.wav", np.full(32000, 0.25))

        self.pipeline = GSCWaveformPipeline(
            sr=16000,
            sample_length=1.0,
            random_position=True,
            time_shift_ms=100,
            background_noise_path=str(self.noise_dir),
            background_frequency=1.0,
            background_volume=0.1,
            reserved_background_files=["reserved_noise.wav"],
            seed=120,
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def test_fixed_length_uses_zero_padding(self):
        short = np.ones(8000, dtype=np.float32)
        output = self.pipeline.fix_length(short, training=False)
        self.assertEqual(output.shape, (16000,))
        np.testing.assert_array_equal(output[:8000], short)
        np.testing.assert_array_equal(output[8000:], np.zeros(8000))

    def test_evaluation_is_deterministic(self):
        wave_in = np.linspace(-0.25, 0.25, 12000, dtype=np.float32)
        first = self.pipeline(wave_in, label="yes", training=False)
        second = self.pipeline(wave_in, label="yes", training=False)
        np.testing.assert_array_equal(first, second)

    def test_training_noise_excludes_reserved_recording(self):
        self.assertEqual(
            set(self.pipeline.training_background_files),
            {"training_noise.wav", "second_training_noise.wav"},
        )
        self.assertNotIn("reserved_noise.wav", self.pipeline.training_background_files)

    def test_silence_does_not_receive_a_second_noise_mix(self):
        silence = np.zeros(16000, dtype=np.float32)
        output = self.pipeline(silence, label="silence", training=True)
        np.testing.assert_array_equal(output, silence)

    def test_strict_silence_provenance(self):
        training = pd.DataFrame(
            {
                "filename": [
                    "_st_generated/silence/training/"
                    "silence_00000_training_noise_00000000.wav"
                ],
                "category": ["silence"],
            }
        )
        validation = pd.DataFrame(
            {
                "filename": [
                    "_st_generated/silence/validation/"
                    "silence_00000_reserved_noise_00000000.wav"
                ],
                "category": ["silence"],
            }
        )
        self.pipeline.validate_silence_sources(training, "training")
        self.pipeline.validate_silence_sources(validation, "validation")

        with self.assertRaises(ValueError):
            self.pipeline.validate_silence_sources(validation, "training")
        with self.assertRaises(ValueError):
            self.pipeline.validate_silence_sources(training, "validation")

    def test_lazy_training_recomputes_augmentation(self):
        write_pcm16(self.root / "yes.wav", np.full(16000, 0.2, dtype=np.float32))
        silence_path = (
            self.root
            / "silence_00000_training_noise_00000000.wav"
        )
        write_pcm16(silence_path, np.zeros(16000, dtype=np.float32))
        dataframe = pd.DataFrame(
            {
                "filename": ["yes.wav", silence_path.name],
                "category": ["yes", "silence"],
            }
        )
        freq_pipeline = LibrosaMelSpecPatchesPipeline(
            patch_length=96,
            overlap_frames=24,
            sr=16000,
            n_fft=512,
            hop_length=160,
            win_length=400,
            window="hann",
            center=False,
            pad_mode="constant",
            power=1.0,
            n_mels=64,
            fmin=125,
            fmax=7500,
            norm=None,
            htk=True,
            db_scale=False,
            log_scale=True,
            peak_normalize=False,
        )
        dataset_builder = BaseAEDTFDataset(
            time_pipeline=self.pipeline,
            freq_pipeline=freq_pipeline,
            file_extension=".wav",
            expand_last_dim=True,
            seed=120,
        )
        dataset = dataset_builder.get_gsc_ds(
            df=dataframe,
            audio_path=str(self.root),
            used_classes=["yes", "silence"],
            batch_size=1,
            to_cache=True,
            shuffle=False,
            return_clip_labels=False,
            split_name="training",
        )

        first_epoch = next(iter(dataset))[0].numpy()
        second_epoch = next(iter(dataset))[0].numpy()
        self.assertEqual(first_epoch.shape, (1, 64, 96, 1))
        self.assertFalse(np.array_equal(first_epoch, second_epoch))


if __name__ == "__main__":
    unittest.main()
