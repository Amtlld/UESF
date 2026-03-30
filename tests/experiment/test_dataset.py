"""Tests for EEGDataset."""

from __future__ import annotations

import numpy as np
import torch

from uesf.experiment.dataset import EEGDataset


class TestEEGDataset:
    def test_length(self):
        data = np.random.randn(100, 32, 500)
        labels = np.zeros(100, dtype=np.int64)
        ds = EEGDataset(data, labels)
        assert len(ds) == 100

    def test_getitem(self):
        data = np.random.randn(10, 32, 500)
        labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0], dtype=np.int64)
        ds = EEGDataset(data, labels)

        x, y = ds[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert x.shape == (32, 500)
        assert y.item() == 0
        assert x.dtype == torch.float32
        assert y.dtype == torch.long

    def test_data_label_mismatch_raises(self):
        data = np.random.randn(10, 32, 500)
        labels = np.zeros(5)
        with __import__("pytest").raises(AssertionError, match="mismatch"):
            EEGDataset(data, labels)
