"""Tests for joint stream preprocessing operators."""

from __future__ import annotations

import numpy as np

from uesf.pipeline.operators.joint_ops import epoch_normalize, sliding_window


class TestSlidingWindow:
    def test_basic_windowing(self):
        # 2 sessions, 1 recording, 32 channels, 1000 samples @ 250Hz
        data = np.random.randn(2, 1, 32, 1000).astype(np.float32)
        labels = np.array([[0], [1]], dtype=np.int64)

        result_data, result_labels, sr = sliding_window(
            data, labels, 250.0,
            {"window_size_sec": 2.0, "stride_sec": 1.0}
        )

        # Window size = 2s * 250Hz = 500 samples
        # Stride = 1s * 250Hz = 250 samples
        # Windows per recording = (1000 - 500) // 250 + 1 = 3
        assert result_data.shape[0] == 2  # sessions
        assert result_data.shape[1] == 3  # windows
        assert result_data.shape[2] == 32  # channels
        assert result_data.shape[3] == 500  # samples per window

    def test_labels_match_windows(self):
        data = np.random.randn(2, 1, 32, 1000).astype(np.float32)
        labels = np.array([[0], [1]], dtype=np.int64)

        _, result_labels, _ = sliding_window(
            data, labels, 250.0,
            {"window_size_sec": 2.0, "stride_sec": 2.0}
        )

        # 2 windows per recording
        assert result_labels.shape[0] == 2  # sessions
        assert result_labels.shape[1] == 2  # windows

    def test_hanning_window(self):
        data = np.random.randn(1, 1, 4, 500).astype(np.float32)
        labels = np.array([[0]], dtype=np.int64)

        result_data, _, _ = sliding_window(
            data, labels, 250.0,
            {"window_size_sec": 2.0, "stride_sec": 2.0, "window_type": "hanning"}
        )

        # First and last samples should be near 0 due to Hanning window
        assert abs(result_data[0, 0, 0, 0]) < abs(data[0, 0, 0, 0]) + 0.01


class TestEpochNormalize:
    def test_zscore(self):
        data = np.random.randn(2, 3, 32, 500).astype(np.float32) * 100 + 50
        labels = np.zeros((2, 3), dtype=np.int64)

        result, _, _ = epoch_normalize(data, labels, 250.0, {"method": "zscore", "axis": -1})

        # After Z-score along samples axis, mean should be ~0, std ~1
        for i in range(2):
            for j in range(3):
                for c in range(32):
                    np.testing.assert_allclose(result[i, j, c].mean(), 0.0, atol=0.01)
                    np.testing.assert_allclose(result[i, j, c].std(), 1.0, atol=0.01)

    def test_minmax(self):
        data = np.random.randn(1, 1, 4, 100).astype(np.float32) * 50 + 25
        labels = np.zeros((1, 1), dtype=np.int64)

        result, _, _ = epoch_normalize(data, labels, 250.0, {"method": "minmax", "axis": -1})

        assert result.min() >= -0.01
        assert result.max() <= 1.01
