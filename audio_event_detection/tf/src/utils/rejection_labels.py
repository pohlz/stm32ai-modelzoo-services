"""Accuracy helpers that forgive only silence/unknown label swaps."""

import numpy as np


def rejection_indices(class_names):
    """Resolve indices by name rather than assuming a particular class order."""
    names = list(class_names)
    return [names.index(name) for name in ("unknown", "silence") if name in names]


def merged_rejection_accuracy(y_true, y_pred, class_names):
    """Map hard decisions after argmax; do not sum rejection probabilities."""
    truth = np.argmax(y_true, axis=-1)
    predicted = np.argmax(y_pred, axis=-1)
    indices = rejection_indices(class_names)
    if len(indices) == 2:
        truth = np.where(np.isin(truth, indices), indices[0], truth)
        predicted = np.where(np.isin(predicted, indices), indices[0], predicted)
    return float(np.mean(truth == predicted))
