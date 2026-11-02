"""Run directly; NumPy metric tests do not require the training runtime."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "tf" / "src"


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helpers = load_source("rejection_labels", SOURCE / "utils/rejection_labels.py")
try:
    import tensorflow as tf
except ImportError:
    tf = None


class RejectionAccuracyTest(unittest.TestCase):
    def test_only_rejection_swaps_are_forgiven(self):
        names = ["yes", "silence", "no", "unknown"]
        truth = np.eye(4)[[1, 3, 0, 2, 1, 0]]
        predicted = np.eye(4)[[3, 1, 3, 0, 2, 0]]
        self.assertEqual(helpers.merged_rejection_accuracy(truth, predicted, names), 0.5)

    def test_probabilities_are_not_summed(self):
        # Command wins argmax even though rejection probabilities sum to more.
        truth = [[0, 1, 0]]
        predicted = [[0.4, 0.3, 0.3]]
        self.assertEqual(helpers.merged_rejection_accuracy(
            truth, predicted, ["yes", "unknown", "silence"]
        ), 0.0)

    def test_missing_silence_keeps_strict_accuracy(self):
        self.assertEqual(helpers.merged_rejection_accuracy(
            np.eye(2), np.eye(2)[::-1], ["yes", "unknown"]
        ), 0.0)

    def test_class_order_does_not_matter(self):
        names = ["silence", "yes", "unknown", "no"]
        self.assertEqual(helpers.merged_rejection_accuracy(
            np.eye(4)[[0, 2]], np.eye(4)[[2, 0]], names
        ), 1.0)

    @unittest.skipIf(tf is None, "TensorFlow is not installed")
    def test_training_metric_matches_numpy_and_serializes(self):
        module = load_source("merged_accuracy", SOURCE / "training/merged_accuracy.py")
        metric = module.MergedRejectionAccuracy(rejection_indices=[1, 3])
        truth = np.eye(4)[[1, 3, 0, 2, 1, 0]]
        predicted = np.eye(4)[[3, 1, 3, 0, 2, 0]]
        metric.update_state(truth[:2], predicted[:2])
        metric.update_state(truth[2:], predicted[2:])
        self.assertAlmostEqual(float(metric.result()), 0.5)
        restored = tf.keras.metrics.deserialize(tf.keras.metrics.serialize(metric))
        self.assertEqual(restored.rejection_indices, (1, 3))

    @unittest.skipIf(tf is None, "TensorFlow is not installed")
    def test_shortlist_uses_merged_metric(self):
        module = load_source("candidate_checkpoint", SOURCE / "training/candidate_checkpoint.py")

        class Saver:
            def save(self, path):
                Path(path).write_text("checkpoint", encoding="utf-8")

        with tempfile.TemporaryDirectory() as directory:
            callback = module.CandidateCheckpoints(
                directory, top_k=1, monitor="val_merged_rejection_accuracy"
            )
            callback.set_model(Saver())
            for epoch, strict, merged, loss in [(0, .9, .91, .2), (1, .85, .95, .3)]:
                callback.on_epoch_end(epoch, {
                    "val_accuracy": strict, "val_merged_rejection_accuracy": merged,
                    "val_loss": loss,
                })
            manifest = json.loads((Path(directory) / "manifest.json").read_text())
            top = [row for row in manifest["candidates"] if row["top_accuracy"]]
            self.assertEqual(top[0]["epoch"], 2)
            self.assertEqual(len(manifest["candidates"]), 2)


if __name__ == "__main__":
    unittest.main()
