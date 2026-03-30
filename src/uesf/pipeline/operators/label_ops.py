"""Built-in label stream preprocessing operators."""

from __future__ import annotations

import numpy as np


def smooth(labels: np.ndarray, params: dict) -> np.ndarray:
    """Apply label smoothing with a moving window.

    For discrete labels, uses mode within window.

    Args:
        labels: Label array.
        params: 'window_size' (int).

    Returns:
        Smoothed labels.
    """
    window_size = params.get("window_size", 5)
    if window_size <= 1:
        return labels

    # Flatten, smooth, reshape
    original_shape = labels.shape
    flat = labels.ravel()
    smoothed = np.copy(flat)

    half = window_size // 2
    for i in range(len(flat)):
        start = max(0, i - half)
        end = min(len(flat), i + half + 1)
        window = flat[start:end]
        # Mode: most frequent value
        values, counts = np.unique(window, return_counts=True)
        smoothed[i] = values[np.argmax(counts)]

    return smoothed.reshape(original_shape)
