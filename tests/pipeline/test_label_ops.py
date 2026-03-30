"""Tests for label stream preprocessing operators."""

from __future__ import annotations

import numpy as np

from uesf.pipeline.operators.label_ops import smooth


class TestSmooth:
    def test_identity_window_1(self):
        labels = np.array([0, 1, 0, 1, 0])
        result = smooth(labels, {"window_size": 1})
        np.testing.assert_array_equal(result, labels)

    def test_smoothing_removes_isolated(self):
        # Single 1 surrounded by 0s should become 0
        labels = np.array([0, 0, 1, 0, 0])
        result = smooth(labels, {"window_size": 5})
        assert result[2] == 0

    def test_preserves_shape(self):
        labels = np.random.randint(0, 3, size=(2, 5))
        result = smooth(labels, {"window_size": 3})
        assert result.shape == labels.shape
