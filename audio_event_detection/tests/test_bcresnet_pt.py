"""Run from audio_event_detection: python -m unittest discover -s tests -p test_bcresnet_pt.py."""

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from omegaconf import OmegaConf
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pt.src.dataset import GSCDataset
from pt.src.models import get_custom_model
from pt.src.runner import evaluate, learning_rate, load_weights, run, validate_config
from tools.plot_bcresnet_pt_run import generate_reports

CONFIG = Path(__file__).resolve().parents[1] / "user_config_gsc12_gsc_preproc_bcresnet_py_v03.yaml"


class BCResNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_shapes_gradients_and_scales(self):
        for tau in (1, 1.5, 2, 3, 6, 8):
            model = get_custom_model(tau=tau)
            x = torch.randn(2, 1, 40, 96)
            logits = model(x)
            self.assertEqual(tuple(logits.shape), (2, 12))
            torch.nn.functional.cross_entropy(logits, torch.tensor([0, 11])).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            self.assertEqual(model.c, [int(tau * 8) * 2, int(tau * 8), int(tau * 12),
                                       int(tau * 16), int(tau * 20), int(tau * 32)])
        with self.assertRaises(ValueError):
            get_custom_model((40, 96, 1))

    def test_checkpoint_roundtrip_and_class_order(self):
        cfg = OmegaConf.load(CONFIG)
        model = get_custom_model().eval()
        x = torch.randn(1, 1, 40, 96)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.pth"
            torch.save(model.state_dict(), path)
            restored = get_custom_model().eval()
            load_weights(restored, path, cfg)
            torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)
            torch.save({"state_dict": model.state_dict(), "tau": 1,
                        "class_names": list(reversed(cfg.dataset.class_names))}, path)
            with self.assertRaisesRegex(ValueError, "ordering"):
                load_weights(restored, path, cfg)

    def test_config_and_schedule(self):
        cfg = OmegaConf.load(CONFIG)
        validate_config(cfg)
        cfg.operation_mode = "chain_tqe"
        with self.assertRaises(ValueError):
            validate_config(cfg)
        self.assertAlmostEqual(learning_rate(1, 100, 5, 0.1), 0.02)
        self.assertAlmostEqual(learning_rate(5, 100, 5, 0.1), 0.1)
        self.assertAlmostEqual(learning_rate(100, 100, 5, 0.1), 0)

    def test_confusion_counts_preserve_label_order_and_missing_classes(self):
        import pandas as pd
        from PIL import Image
        loader = [(torch.tensor([[0., 3., 0.], [4., 0., 0.], [0., 2., 0.]]),
                   torch.tensor([0, 1, 1]))]
        with tempfile.TemporaryDirectory() as tmp:
            metrics = evaluate(torch.nn.Identity(), loader, "cpu", output_dir=tmp,
                               class_names=["yes", "no", "silence"])
            self.assertAlmostEqual(metrics["accuracy"], 1 / 3)
            for level in ("patch", "clip"):
                base = Path(tmp) / f"float_model_{level}_confusion_matrix_validation_set"
                counts = pd.read_csv(base.with_suffix(".csv"), index_col=0)
                self.assertEqual(list(counts.columns), ["yes", "no", "silence"])
                np.testing.assert_array_equal(counts.values, [[0, 1, 0], [1, 1, 0], [0, 0, 0]])
                with Image.open(base.with_suffix(".png")) as image:
                    image.verify()

    def test_end_to_end_and_silence_leakage(self):
        cfg = OmegaConf.load(CONFIG)
        cfg.general.device = "cpu"
        cfg.training.epochs = 2
        cfg.training.warmup_epochs = 0
        cfg.training.batch_size = 2
        cfg.preprocessing.background_frequency = 0
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wave = (0.1 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)).astype(np.float32)
            for split, source in (("training", "white_noise"), ("validation", "running_tap")):
                names = [f"{split}_yes.wav", f"silence_00000_{source}_00000000.wav"]
                for name in names:
                    sf.write(root / name, wave, 16000)
                csv_path = root / f"{split}.csv"
                with csv_path.open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["filename", "category"])
                    writer.writerows(zip(names, ["yes", "silence"]))
                cfg.dataset[f"{split}_audio_path"] = str(root)
                cfg.dataset[f"{split}_csv_path"] = str(csv_path)
            data = GSCDataset(cfg, "validation")
            torch.testing.assert_close(data[0][0], data[0][0], rtol=0, atol=0)
            self.assertEqual(tuple(data[0][0].shape), (1, 40, 96))
            output = root / "train"
            result = run(cfg, output)
            self.assertEqual(result["validation"]["samples"], 2)
            self.assertEqual(len(json.loads((output / "history.json").read_text())), 2)
            self.assertTrue((output / "Training_curves.png").is_file())
            preserved = {name: (output / name).read_bytes() for name in ("config.yaml", "metrics.json", "history.json")}
            reports = generate_reports(output, device="cpu", threads=1)
            self.assertEqual(result, reports)
            for name, content in preserved.items():
                self.assertEqual((output / name).read_bytes(), content)
            cfg.operation_mode = "evaluation"
            cfg.model.model_path = str(output / "saved_models/best_model.pth")
            again = run(cfg, root / "eval")
            self.assertEqual(result, again)
            self.assertTrue((root / "eval/float_model_patch_confusion_matrix_validation_set.png").is_file())
            self.assertNotIn("tensorflow", sys.modules)
            cfg.preprocessing.reserved_background_files = ["white_noise.wav"]
            with self.assertRaisesRegex(ValueError, "Non-reserved"):
                GSCDataset(cfg, "validation")


if __name__ == "__main__":
    unittest.main()
