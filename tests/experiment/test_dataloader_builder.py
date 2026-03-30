"""Tests for DataLoader builder and CombinedIterator."""

from __future__ import annotations

import numpy as np

from uesf.experiment.dataloader_builder import build_dataloaders
from uesf.experiment.dataset import EEGDataset


class TestCombinedIterator:
    def test_basic_iteration(self):
        ds1 = EEGDataset(np.random.randn(20, 10), np.zeros(20, dtype=np.int64))
        ds2 = EEGDataset(np.random.randn(20, 10), np.ones(20, dtype=np.int64))

        loader = build_dataloaders({"src": ds1, "tgt": ds2}, batch_size=5, phase="train")

        batches = list(loader)
        assert len(batches) == 4  # 20 / 5

        for batch in batches:
            assert "src" in batch
            assert "tgt" in batch
            src_data, src_labels = batch["src"]
            assert src_data.shape[0] == 5

    def test_stops_at_shortest(self):
        ds1 = EEGDataset(np.random.randn(10, 5), np.zeros(10, dtype=np.int64))
        ds2 = EEGDataset(np.random.randn(20, 5), np.ones(20, dtype=np.int64))

        loader = build_dataloaders({"a": ds1, "b": ds2}, batch_size=5, phase="train")
        batches = list(loader)
        assert len(batches) == 2  # Limited by ds1 (10 / 5)

    def test_single_channel(self):
        ds = EEGDataset(np.random.randn(16, 8), np.zeros(16, dtype=np.int64))
        loader = build_dataloaders({"main": ds}, batch_size=4, phase="val")
        batches = list(loader)
        assert len(batches) == 4

    def test_no_shuffle_for_val(self):
        data = np.arange(10).reshape(10, 1).astype(np.float32)
        labels = np.zeros(10, dtype=np.int64)
        ds = EEGDataset(data, labels)

        loader = build_dataloaders({"main": ds}, batch_size=10, shuffle_train=True, phase="val")
        batch = list(loader)[0]
        x, _ = batch["main"]
        # In val mode, no shuffle, so order should be preserved
        np.testing.assert_array_equal(x.numpy().flatten(), data.flatten())

    def test_len(self):
        ds = EEGDataset(np.random.randn(20, 5), np.zeros(20, dtype=np.int64))
        loader = build_dataloaders({"main": ds}, batch_size=5, phase="train")
        assert len(loader) == 4
