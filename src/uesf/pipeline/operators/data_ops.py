"""Built-in data stream preprocessing operators.

Each operator processes EEG signal data without changing the number of dimensions.
Operates on a single subject's data at a time.
"""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample_poly


def resample(data: np.ndarray, sampling_rate: float, params: dict) -> tuple[np.ndarray, float]:
    """Resample data to a target sampling rate.

    Args:
        data: EEG data array (sessions, recordings, channels, samples).
        sampling_rate: Current sampling rate in Hz.
        params: Must contain 'target_rate' (Hz).

    Returns:
        (resampled_data, new_sampling_rate)
    """
    target_rate = params["target_rate"]
    if target_rate == sampling_rate:
        return data, sampling_rate

    # Use rational resampling
    sr_int = int(sampling_rate)
    tr_int = int(target_rate)
    g = gcd(sr_int, tr_int)
    up = tr_int // g
    down = sr_int // g

    resampled = resample_poly(data, up, down, axis=-1)
    return resampled.astype(data.dtype), float(target_rate)


def bandpass_filter(data: np.ndarray, sampling_rate: float, params: dict) -> tuple[np.ndarray, float]:
    """Apply band-pass/high-pass/low-pass filter.

    Args:
        data: EEG data (sessions, recordings, channels, samples).
        sampling_rate: Sampling rate in Hz.
        params: 'l_freq' and/or 'h_freq' in Hz.

    Returns:
        (filtered_data, sampling_rate)
    """
    l_freq = params.get("l_freq")
    h_freq = params.get("h_freq")
    order = params.get("order", 5)
    nyq = sampling_rate / 2.0

    if l_freq is not None and h_freq is not None:
        b, a = butter(order, [l_freq / nyq, h_freq / nyq], btype="band")
    elif l_freq is not None:
        b, a = butter(order, l_freq / nyq, btype="high")
    elif h_freq is not None:
        b, a = butter(order, h_freq / nyq, btype="low")
    else:
        return data, sampling_rate

    filtered = filtfilt(b, a, data, axis=-1)
    return filtered.astype(data.dtype), sampling_rate


def notch_filter(data: np.ndarray, sampling_rate: float, params: dict) -> tuple[np.ndarray, float]:
    """Apply notch filter to remove powerline interference.

    Args:
        data: EEG data (sessions, recordings, channels, samples).
        sampling_rate: Sampling rate in Hz.
        params: 'notch_freq' (e.g., 50.0 or 60.0 Hz).

    Returns:
        (filtered_data, sampling_rate)
    """
    notch_freq = params["notch_freq"]
    quality = params.get("quality", 30.0)
    b, a = iirnotch(notch_freq, quality, sampling_rate)
    filtered = filtfilt(b, a, data, axis=-1)
    return filtered.astype(data.dtype), sampling_rate


def reference(data: np.ndarray, sampling_rate: float, params: dict) -> tuple[np.ndarray, float]:
    """Re-reference EEG data.

    Args:
        data: EEG data (sessions, recordings, channels, samples).
        sampling_rate: Sampling rate in Hz.
        params: 'type' - "CAR" (common average reference).

    Returns:
        (re-referenced_data, sampling_rate)
    """
    ref_type = params.get("type", "CAR")
    if ref_type == "CAR":
        # Common Average Reference: subtract mean across channels
        mean = data.mean(axis=-2, keepdims=True)
        data = data - mean
    return data.astype(data.dtype), sampling_rate
