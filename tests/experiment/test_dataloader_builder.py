"""Tests for dataloader_builder (prepare_*, DataloaderBuilder, CombinedIterator)."""

from __future__ import annotations

import numpy as np

from uesf.experiment.dataloader_builder import (
    CombinedIterator,
    DataloaderBuilder,
    flatten_3d,
    get_sample_input,
    prepare_channel_data,
    prepare_uda_channel_data,
)
from uesf.experiment.splitter import (
    MultiDatasetSplitResult,
    SplitResult,
    UDASplitResult,
)


def _labels_5d(shape):
    return np.arange(np.prod(shape[:3]), dtype=np.int64).reshape(shape[:3])


def _cache_pair(spec):
    """Return (dataset_cache, labels_cache) with dense indexing labels."""
    cache: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    for alias, shape in spec.items():
        cache[alias] = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        labels[alias] = np.arange(shape[0] * shape[1] * shape[2], dtype=np.int64)
    return cache, labels


class TestPrepareChannelDataSingle:
    def test_returns_main_channel(self):
        cache, labels = _cache_pair({"a": (3, 1, 2, 2, 4)})
        sr = SplitResult(
            train_indices=np.array([0, 1, 2, 3]),
            val_indices=np.array([4]),
            test_indices=np.array([5]),
        )
        data, lbl = prepare_channel_data(sr, cache, labels, "train")
        assert list(data.keys()) == ["main"]
        assert data["main"].shape == (4, 2, 4)
        np.testing.assert_array_equal(lbl["main"], np.array([0, 1, 2, 3]))

    def test_empty_phase_returns_empty(self):
        cache, labels = _cache_pair({"a": (2, 1, 2, 2, 2)})
        sr = SplitResult(train_indices=np.arange(4))
        data, lbl = prepare_channel_data(sr, cache, labels, "val")
        assert data == {} and lbl == {}


class TestPrepareChannelDataMulti:
    def test_concatenates_across_aliases(self):
        cache, labels = _cache_pair({"a": (1, 1, 2, 2, 2), "b": (1, 1, 3, 2, 2)})
        msr = MultiDatasetSplitResult(
            phase_indices={
                "train": {
                    "a": np.array([0, 1]),
                    "b": np.array([0, 1, 2]),
                },
                "val": {},
                "test": {},
            },
        )
        data, lbl = prepare_channel_data(msr, cache, labels, "train")
        assert data["main"].shape == (5, 2, 2)
        assert lbl["main"].shape == (5,)


class TestPrepareUDAChannelData:
    def test_cross_dataset_transductive(self):
        cache, labels = _cache_pair({"a": (2, 1, 2, 2, 2), "b": (2, 1, 2, 2, 2)})
        uda = UDASplitResult(
            source_train={"a": np.arange(4)},
            source_val={"a": np.array([], int)},
            target_train={"b": np.arange(4)},
            target_val={"b": np.array([], int)},
            target_test={"b": np.arange(4)},
        )
        phases = prepare_uda_channel_data(uda, cache, labels)
        assert "source" in phases["train"][0]
        assert "target" in phases["train"][0]
        assert phases["val"] == ({}, {})
        assert "main" in phases["test"][0]
        assert phases["test"][0]["main"].shape == (4, 2, 2)

    def test_inductive_val_target_val(self):
        cache, labels = _cache_pair({"m": (4, 1, 2, 2, 2)})
        uda = UDASplitResult(
            source_train={"m": np.arange(4)},
            source_val={"m": np.array([4])},
            target_train={"m": np.array([5, 6])},
            target_val={"m": np.array([7])},
            target_test={"m": np.array([])},
        )
        phases = prepare_uda_channel_data(uda, cache, labels)
        assert "source_val" in phases["val"][0]
        assert "target_val" in phases["val"][0]
        assert phases["val"][0]["source_val"].shape[0] == 1


class TestDataloaderBuilder:
    def test_build_with_two_channels(self):
        b = DataloaderBuilder()
        data = {
            "source": np.random.randn(10, 2, 4).astype(np.float32),
            "target": np.random.randn(10, 2, 4).astype(np.float32),
        }
        labels = {
            "source": np.zeros(10, dtype=np.int64),
            "target": np.ones(10, dtype=np.int64),
        }
        loader = b.build(data, labels, batch_size=5, shuffle=False)
        assert isinstance(loader, CombinedIterator)
        batches = list(loader)
        assert len(batches) == 2
        assert set(batches[0]) == {"source", "target"}

    def test_empty_channel_skipped(self):
        b = DataloaderBuilder()
        data = {"main": np.random.randn(4, 2, 2).astype(np.float32), "extra": np.array([])}
        labels = {"main": np.zeros(4, dtype=np.int64), "extra": np.array([])}
        loader = b.build(data, labels, batch_size=2, shuffle=False)
        assert list(next(iter(loader))) == ["main"]


class TestCombinedIterator:
    def test_len_uses_shortest(self):
        b = DataloaderBuilder()
        d = {
            "a": np.random.randn(10, 2, 2).astype(np.float32),
            "b": np.random.randn(20, 2, 2).astype(np.float32),
        }
        lb = {"a": np.zeros(10, dtype=np.int64), "b": np.zeros(20, dtype=np.int64)}
        loader = b.build(d, lb, batch_size=5, shuffle=False)
        assert len(loader) == 2  # 10 / 5


class TestShapeHelpers:
    def test_flatten_3d(self):
        data = np.zeros((2, 3, 4, 5, 6))
        flat = flatten_3d(data)
        assert flat.shape == (2 * 3 * 4, 5, 6)


class TestGetSampleInput:
    def test_returns_first_channel_batch(self):
        b = DataloaderBuilder()
        d = {"main": np.random.randn(4, 2, 2).astype(np.float32)}
        lb = {"main": np.zeros(4, dtype=np.int64)}
        loader = b.build(d, lb, batch_size=2, shuffle=False)
        sample = get_sample_input(loader)
        assert sample.shape[0] == 2
