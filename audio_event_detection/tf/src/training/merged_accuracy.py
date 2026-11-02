"""Serializable training metric for a shared silence/unknown rejection label."""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="AED")
class MergedRejectionAccuracy(tf.keras.metrics.CategoricalAccuracy):
    def __init__(self, rejection_indices=(), name="merged_rejection_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.rejection_indices = tuple(rejection_indices)

    def update_state(self, y_true, y_pred, sample_weight=None):
        if len(self.rejection_indices) == 2:
            # Remap hard labels, preserving command-versus-rejection mistakes.
            first, second = self.rejection_indices
            depth = tf.shape(y_pred)[-1]
            truth = tf.argmax(y_true, axis=-1, output_type=tf.int32)
            predicted = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
            truth = tf.where(truth == second, first, truth)
            predicted = tf.where(predicted == second, first, predicted)
            y_true = tf.one_hot(truth, depth)
            y_pred = tf.one_hot(predicted, depth)
        return super().update_state(y_true, y_pred, sample_weight)

    def get_config(self):
        return {**super().get_config(), "rejection_indices": list(self.rejection_indices)}
