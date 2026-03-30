"""Built-in joint stream preprocessing operators.

Joint operators modify both data and labels simultaneously,
ensuring dimensional alignment.
"""

from __future__ import annotations

import numpy as np


def sliding_window(
    data: np.ndarray, labels: np.ndarray, sampling_rate: float, params: dict
) -> tuple[np.ndarray, np.ndarray, float]:
    """Apply sliding window epoching.

    Segments continuous recordings into fixed-length windows.
    New windows expand along the 'recording' dimension.

    Args:
        data: Shape (sessions, recordings, channels, samples).
        labels: Shape (sessions, recordings) or compatible.
        sampling_rate: Sampling rate in Hz.
        params:
            window_size_sec: Window duration in seconds.
            stride_sec: Stride duration in seconds.
            window_type: "rect" or "hanning".
            label_strategy: "mode" or "last".

    Returns:
        (windowed_data, windowed_labels, sampling_rate)
    """
    window_size_sec = params["window_size_sec"]
    stride_sec = params.get("stride_sec", window_size_sec)
    window_type = params.get("window_type", "rect")
    label_strategy = params.get("label_strategy", "mode")

    window_samples = int(window_size_sec * sampling_rate)
    stride_samples = int(stride_sec * sampling_rate)

    n_sessions = data.shape[0]
    n_recordings = data.shape[1]
    n_samples = data.shape[3]

    # Build window function
    if window_type == "hanning":
        window_func = np.hanning(window_samples).astype(data.dtype)
    else:
        window_func = np.ones(window_samples, dtype=data.dtype)

    # Compute number of windows per recording
    n_windows = max(0, (n_samples - window_samples) // stride_samples + 1)

    if n_windows == 0:
        return data, labels, sampling_rate

    # Output arrays
    out_data_list = []
    out_label_list = []

    for s in range(n_sessions):
        session_windows = []
        session_labels = []

        for r in range(n_recordings):
            for w in range(n_windows):
                start = w * stride_samples
                end = start + window_samples
                segment = data[s, r, :, start:end] * window_func
                session_windows.append(segment)

                # Label strategy
                if label_strategy == "last":
                    lbl = labels[s, r] if labels.ndim > 1 else labels[s]
                else:  # mode
                    lbl = labels[s, r] if labels.ndim > 1 else labels[s]
                session_labels.append(lbl)

        out_data_list.append(np.stack(session_windows, axis=0))
        out_label_list.append(np.array(session_labels))

    # Shape: (sessions, n_windows * n_recordings, channels, window_samples)
    windowed_data = np.stack(out_data_list, axis=0)
    windowed_labels = np.stack(out_label_list, axis=0)

    return windowed_data, windowed_labels, sampling_rate


def epoch_normalize(
    data: np.ndarray, labels: np.ndarray, sampling_rate: float, params: dict
) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-epoch normalization (safe, no cross-subject leakage).

    Args:
        data: Shape (..., channels, samples) or (..., samples).
        labels: Labels array (unchanged).
        sampling_rate: Sampling rate.
        params:
            method: "zscore" or "minmax".
            axis: Axis to normalize along (default: -1, samples axis).

    Returns:
        (normalized_data, labels, sampling_rate)
    """
    method = params.get("method", "zscore")
    axis = params.get("axis", -1)

    if method == "zscore":
        mean = data.mean(axis=axis, keepdims=True)
        std = data.std(axis=axis, keepdims=True)
        std = np.where(std == 0, 1.0, std)
        data = (data - mean) / std
    elif method == "minmax":
        dmin = data.min(axis=axis, keepdims=True)
        dmax = data.max(axis=axis, keepdims=True)
        drange = dmax - dmin
        drange = np.where(drange == 0, 1.0, drange)
        data = (data - dmin) / drange

    return data.astype(np.float32), labels, sampling_rate
