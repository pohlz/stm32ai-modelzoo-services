"""Adapt PyTorch metrics to the shared STM plotting functions without TensorFlow."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")  # Save figures on training servers without a display.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import the standalone plotting module without executing common.utils.__init__,
# which imports TensorFlow. This is the same renderer used by the STM AED services.
_source = Path(__file__).resolve().parents[3] / "common/utils/visualize_utils.py"
_spec = importlib.util.spec_from_file_location("aed_stm_visualize", _source)
_plots = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_plots)


def plot_training_history(history, output_dir):
    if not history:
        raise ValueError("Cannot plot empty training history")
    adapted = SimpleNamespace(history={
        "accuracy": [row["train_accuracy"] for row in history],
        "val_accuracy": [row["validation"]["accuracy"] for row in history],
        "loss": [row["train_loss"] for row in history],
        "val_loss": [row["validation"]["loss"] for row in history],
    })
    try:
        _plots.vis_training_curves(adapted, str(output_dir))
    finally:
        plt.close("all")


def plot_confusion_counts(counts, class_names, output_dir, split):
    """Rows=true, columns=predicted; PNGs normalized by row, CSVs raw counts.

    Absent true classes remain blank in the shared renderer (undefined recall).
    GSC produces one patch per clip, hence both STM report levels are identical.
    """
    counts = np.asarray(counts, dtype=np.int64)
    if counts.shape != (len(class_names), len(class_names)) or counts.sum() == 0:
        raise ValueError("Confusion counts must be a nonempty square matrix matching class_names")
    accuracy = np.trace(counts) / counts.sum() * 100
    for level in ("patch", "clip"):
        name = f"float_model_{level}_confusion_matrix_{split}_set"
        frame = pd.DataFrame(counts, index=class_names, columns=class_names)
        frame.index.name = "True Label / Predicted Label"
        frame.to_csv(Path(output_dir) / f"{name}.csv")
        try:
            with np.errstate(invalid="ignore", divide="ignore"):
                _plots.plot_confusion_matrix(
                    cm=counts, class_names=list(class_names),
                    title=f"Float model {level}-level accuracy on {split} set: {accuracy:.2f}%",
                    model_name=name, output_dir=str(output_dir),
                )
        finally:
            plt.close("all")
