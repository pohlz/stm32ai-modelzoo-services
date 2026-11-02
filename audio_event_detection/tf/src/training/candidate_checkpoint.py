import json
import math
from pathlib import Path

import tensorflow as tf


class CandidateCheckpoints(tf.keras.callbacks.Callback):
    """Retain top K accuracy checkpoints plus the minimum-loss checkpoint."""

    def __init__(self, directory, top_k=5, monitor="val_accuracy"):
        super().__init__()
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self.monitor = monitor
        self.top_accuracy = []
        self.best_loss = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        accuracy = logs.get("val_accuracy")
        loss = logs.get("val_loss")
        selection_accuracy = logs.get(self.monitor)

        if accuracy is None or loss is None or selection_accuracy is None:
            raise ValueError(
                f"Candidate selection requires val_accuracy, val_loss and {self.monitor}."
            )

        accuracy, loss = float(accuracy), float(loss)
        selection_accuracy = float(selection_accuracy)
        if not all(math.isfinite(value) for value in (accuracy, loss, selection_accuracy)):
            raise ValueError("Non-finite validation metrics.")

        current = {
            "epoch": epoch + 1,
            "val_accuracy": accuracy,
            "selection_accuracy": selection_accuracy,
            "val_loss": loss,
            "checkpoint": f"epoch_{epoch + 1:04d}.keras",
        }

        previous = {row["epoch"]: row for row in self.top_accuracy}
        if self.best_loss is not None:
            previous[self.best_loss["epoch"]] = self.best_loss

        # Higher accuracy wins; ties keep the earlier epoch.
        self.top_accuracy = sorted(
            [*self.top_accuracy, current],
            key=lambda row: (-row["selection_accuracy"], row["epoch"]),
        )[:self.top_k]

        if (
            self.best_loss is None
            or loss < self.best_loss["val_loss"]
        ):
            self.best_loss = current

        retained = {row["epoch"]: row for row in self.top_accuracy}
        retained[self.best_loss["epoch"]] = self.best_loss

        if current["epoch"] in retained:
            # Saves the same augmented wrapper used by the existing trainer.
            self.model.save(self.directory / current["checkpoint"])

        # Delete only checkpoint files previously managed by this callback.
        for old_epoch in previous.keys() - retained.keys():
            old_path = self.directory / previous[old_epoch]["checkpoint"]
            old_path.unlink(missing_ok=True)

        accuracy_epochs = {row["epoch"] for row in self.top_accuracy}
        candidates = []
        for candidate_epoch, row in sorted(retained.items()):
            candidates.append({
                **row,
                "top_accuracy": candidate_epoch in accuracy_epochs,
                "best_loss": candidate_epoch == self.best_loss["epoch"],
            })

        manifest = {
            "top_k": self.top_k,
            "monitor": self.monitor,
            "candidates": candidates,
        }
        temporary = self.directory / "manifest.tmp"
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary.replace(self.directory / "manifest.json")
