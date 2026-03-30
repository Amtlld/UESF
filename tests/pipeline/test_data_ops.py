"""Tests for data stream preprocessing operators."""

from __future__ import annotations

import numpy as np

from uesf.pipeline.operators.data_ops import bandpass_filter, notch_filter, reference, resample


class TestResample:
    def test_downsample(self):
        data = np.random.randn(2, 1, 32, 1000).astype(np.float32)
        result, new_sr = resample(data, 500.0, {"target_rate": 250})
        assert new_sr == 250.0
        assert result.shape[-1] == 500

    def test_same_rate_noop(self):
        data = np.random.randn(2, 1, 32, 500).astype(np.float32)
        result, new_sr = resample(data, 250.0, {"target_rate": 250})
        assert new_sr == 250.0
        np.testing.assert_array_equal(result, data)


class TestBandpassFilter:
    def test_bandpass(self):
        data = np.random.randn(2, 1, 32, 500).astype(np.float32)
        result, sr = bandpass_filter(data, 250.0, {"l_freq": 1.0, "h_freq": 45.0})
        assert result.shape == data.shape
        assert sr == 250.0

    def test_highpass_only(self):
        data = np.random.randn(2, 1, 32, 500).astype(np.float32)
        result, sr = bandpass_filter(data, 250.0, {"l_freq": 1.0})
        assert result.shape == data.shape

    def test_lowpass_only(self):
        data = np.random.randn(2, 1, 32, 500).astype(np.float32)
        result, sr = bandpass_filter(data, 250.0, {"h_freq": 45.0})
        assert result.shape == data.shape

    def test_no_freqs_noop(self):
        data = np.random.randn(2, 1, 32, 500).astype(np.float32)
        result, sr = bandpass_filter(data, 250.0, {})
        np.testing.assert_array_equal(result, data)


class TestNotchFilter:
    def test_notch_50hz(self):
        data = np.random.randn(2, 1, 32, 500).astype(np.float32)
        result, sr = notch_filter(data, 250.0, {"notch_freq": 50.0})
        assert result.shape == data.shape
        assert sr == 250.0


class TestReference:
    def test_car(self):
        data = np.random.randn(2, 1, 32, 500).astype(np.float32)
        result, sr = reference(data, 250.0, {"type": "CAR"})
        # After CAR, mean across channels should be ~0
        mean_across_channels = result.mean(axis=-2)
        np.testing.assert_allclose(mean_across_channels, 0.0, atol=1e-5)
