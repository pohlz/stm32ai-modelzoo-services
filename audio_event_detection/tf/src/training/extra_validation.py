"""Reporting-only validation that leaves training metric state untouched."""

import tensorflow as tf


class ExtraValidation(tf.keras.callbacks.Callback):
    """Evaluate fixed datasets after an epoch without optimizer or metric updates.

    Forward passes share the current model weights, but use inference mode and
    independent metric objects. No nested model.evaluate() resets the metrics
    used by fit, and no model cloning consumes initializer random numbers.
    """

    def __init__(self, datasets):
        super().__init__()
        self.datasets = datasets
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(
            from_logits=False, reduction="none"
        )
        self.loss_metric = tf.keras.metrics.Mean()
        self.accuracy_metric = tf.keras.metrics.CategoricalAccuracy()

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            raise ValueError("Extra validation requires the epoch logs dictionary")
        for name, dataset in self.datasets.items():
            self.loss_metric.reset_state()
            self.accuracy_metric.reset_state()
            for features, labels in dataset:
                predictions = self.model(features, training=False)
                losses = self.loss_fn(labels, predictions)
                # Match the compiled validation loss, including regularization.
                if self.model.losses:
                    losses += tf.add_n([
                        tf.cast(value, losses.dtype) for value in self.model.losses
                    ])
                self.loss_metric.update_state(losses)
                self.accuracy_metric.update_state(labels, predictions)
            if float(self.loss_metric.count.numpy()) == 0:
                raise ValueError(f"Extra validation dataset {name!r} is empty")
            prefix = f"val_{name}"
            logs[f"{prefix}_loss"] = float(self.loss_metric.result().numpy())
            logs[f"{prefix}_accuracy"] = float(self.accuracy_metric.result().numpy())
            print(
                f"[INFO] : Epoch {epoch + 1} {name}: "
                f"loss={logs[f'{prefix}_loss']:.5f}, "
                f"accuracy={logs[f'{prefix}_accuracy']:.5f}",
                flush=True,
            )
