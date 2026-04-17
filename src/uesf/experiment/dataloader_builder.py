"""UESF DataLoader builder + channel-data preparation.

The preparation functions turn ``SplitResult``/``MultiDatasetSplitResult``/
``UDASplitResult`` index containers into per-channel ``dict[str, np.ndarray]``
blocks; :class:`DataloaderBuilder` wraps those blocks in ``EEGDataset`` and
stitches them into a :class:`CombinedIterator`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from uesf.core.exceptions import ConfigError
from uesf.core.logging import get_logger
from uesf.experiment.dataset import EEGDataset
from uesf.experiment.splitter import (
    MultiDatasetSplitResult,
    SplitResult,
    UDASplitResult,
)

logger = get_logger("experiment.dataloader")


# ---------------------------------------------------------------------------
# Combined iterator
# ---------------------------------------------------------------------------


class CombinedIterator:
    """Iterate multiple DataLoaders in parallel, yielding a dict batch.

    Stops at the shortest component loader.
    """

    def __init__(self, loaders: dict[str, DataLoader]) -> None:
        self.loaders = loaders

    def __iter__(self):
        iterators = {name: iter(loader) for name, loader in self.loaders.items()}
        while True:
            batch: dict[str, Any] = {}
            try:
                for name, it in iterators.items():
                    batch[name] = next(it)
            except StopIteration:
                break
            yield batch

    def __len__(self) -> int:
        if not self.loaders:
            return 0
        return min(len(loader) for loader in self.loaders.values())


# ---------------------------------------------------------------------------
# Shape helper
# ---------------------------------------------------------------------------


def flatten_3d(data: np.ndarray) -> np.ndarray:
    """Reshape 5-D ``[sub, sess, rec, ch, samp]`` → 3-D ``[N, ch, samp]``."""
    if data.ndim != 5:
        raise ConfigError(
            f"flatten_3d expects 5-D data, got {data.ndim}-D with shape {data.shape}",
        )
    return data.reshape(-1, data.shape[3], data.shape[4])


# ---------------------------------------------------------------------------
# Regular prepare_channel_data
# ---------------------------------------------------------------------------


def prepare_channel_data(
    split_result: SplitResult | MultiDatasetSplitResult,
    dataset_cache: dict[str, np.ndarray],
    labels_cache: dict[str, np.ndarray],
    phase: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Turn a Regular split result into ``{channel: array}`` data + labels.

    Returns:
        ``(channel_data, channel_labels)``. Each dict has the single key
        ``"main"``. Empty arrays are returned when the requested phase has
        no samples.
    """
    if isinstance(split_result, MultiDatasetSplitResult):
        alias_indices = split_result.phase_indices.get(phase, {})
        parts_d: list[np.ndarray] = []
        parts_l: list[np.ndarray] = []
        for alias, idx in alias_indices.items():
            if len(idx) == 0:
                continue
            flat = flatten_3d(dataset_cache[alias])
            parts_d.append(flat[idx])
            parts_l.append(labels_cache[alias][idx])
        if not parts_d:
            return {}, {}
        return (
            {"main": np.concatenate(parts_d, axis=0)},
            {"main": np.concatenate(parts_l, axis=0)},
        )

    # Single-dataset SplitResult.
    if len(dataset_cache) != 1:
        raise ConfigError(
            "prepare_channel_data: SplitResult requires a single-dataset cache, "
            f"got aliases {list(dataset_cache.keys())}.",
        )
    alias = next(iter(dataset_cache.keys()))
    indices_attr = f"{phase}_indices"
    idx = getattr(split_result, indices_attr, None)
    if idx is None or len(idx) == 0:
        return {}, {}
    flat = flatten_3d(dataset_cache[alias])
    return {"main": flat[idx]}, {"main": labels_cache[alias][idx]}


# ---------------------------------------------------------------------------
# UDA prepare_uda_channel_data
# ---------------------------------------------------------------------------


def prepare_uda_channel_data(
    uda_split: UDASplitResult,
    dataset_cache: dict[str, np.ndarray],
    labels_cache: dict[str, np.ndarray],
) -> dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]]:
    """Produce per-phase ``(channel_data, channel_labels)`` tuples for UDA.

    Phase channels (per 04 doc §4.2):
      - train: ``{"source": ..., "target": ...}`` (target omitted if empty)
      - val:   ``{"source_val": ..., "target_val": ...}`` (keys omitted if empty)
      - test:  ``{"main": target_test}``
    """

    def _gather(index_map: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        chunks_d: list[np.ndarray] = []
        chunks_l: list[np.ndarray] = []
        for alias, idx in index_map.items():
            if len(idx) == 0:
                continue
            flat = flatten_3d(dataset_cache[alias])
            chunks_d.append(flat[idx])
            chunks_l.append(labels_cache[alias][idx])
        if not chunks_d:
            return np.array([]), np.array([])
        return np.concatenate(chunks_d, axis=0), np.concatenate(chunks_l, axis=0)

    train_data: dict[str, np.ndarray] = {}
    train_labels: dict[str, np.ndarray] = {}
    src_d, src_l = _gather(uda_split.source_train)
    if len(src_d):
        train_data["source"] = src_d
        train_labels["source"] = src_l
    tgt_d, tgt_l = _gather(uda_split.target_train)
    if len(tgt_d):
        train_data["target"] = tgt_d
        train_labels["target"] = tgt_l

    val_data: dict[str, np.ndarray] = {}
    val_labels: dict[str, np.ndarray] = {}
    sv_d, sv_l = _gather(uda_split.source_val)
    if len(sv_d):
        val_data["source_val"] = sv_d
        val_labels["source_val"] = sv_l
    tv_d, tv_l = _gather(uda_split.target_val)
    if len(tv_d):
        val_data["target_val"] = tv_d
        val_labels["target_val"] = tv_l

    test_data: dict[str, np.ndarray] = {}
    test_labels: dict[str, np.ndarray] = {}
    te_d, te_l = _gather(uda_split.target_test)
    if len(te_d):
        test_data["main"] = te_d
        test_labels["main"] = te_l

    return {
        "train": (train_data, train_labels),
        "val": (val_data, val_labels),
        "test": (test_data, test_labels),
    }


# ---------------------------------------------------------------------------
# DataloaderBuilder
# ---------------------------------------------------------------------------


class DataloaderBuilder:
    """Wrap per-channel arrays as a :class:`CombinedIterator`."""

    def __init__(self, num_workers: int = 0) -> None:
        self.num_workers = num_workers

    def build(
        self,
        channel_data: dict[str, np.ndarray],
        channel_labels: dict[str, np.ndarray],
        batch_size: int,
        shuffle: bool = True,
    ) -> CombinedIterator:
        if set(channel_data.keys()) != set(channel_labels.keys()):
            raise ConfigError(
                "DataloaderBuilder.build: channel_data and channel_labels "
                f"must share keys. Got {sorted(channel_data)} vs {sorted(channel_labels)}.",
            )
        loaders: dict[str, DataLoader] = {}
        for name, data in channel_data.items():
            labels = channel_labels[name]
            if len(data) == 0:
                continue
            ds: Dataset = EEGDataset(data, labels)
            loaders[name] = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                drop_last=False,
            )
        return CombinedIterator(loaders)


# ---------------------------------------------------------------------------
# Sample extraction for log_graph
# ---------------------------------------------------------------------------


def get_sample_input(train_loader: CombinedIterator) -> torch.Tensor:
    """Extract a single-sample input tensor from a train loader.

    Uses the first channel in batch-dict order (``"source"`` for UDA,
    ``"main"`` for Regular). Returns a tensor on CPU.
    """
    iterator = iter(train_loader)
    batch = next(iterator)
    if not batch:
        raise ConfigError("get_sample_input: train_loader produced an empty batch.")
    first_key = next(iter(batch))
    data, _ = batch[first_key]
    return data


# ---------------------------------------------------------------------------
# Legacy shim — will be removed once ExperimentManager is rewritten (Stage 6).
# ---------------------------------------------------------------------------


def build_dataloaders(
    datasets: dict[str, Dataset],
    batch_size: int = 32,
    shuffle_train: bool = True,
    num_workers: int = 0,
    phase: str = "train",
) -> CombinedIterator:
    """Legacy helper — accepts pre-wrapped ``dict[str, Dataset]``.

    Retained temporarily so the unmodified ExperimentManager keeps
    importing. Stage 6 deletes this along with the caller.
    """
    should_shuffle = shuffle_train and phase == "train"
    loaders: dict[str, DataLoader] = {}
    for name, dataset in datasets.items():
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=should_shuffle,
            num_workers=num_workers,
            drop_last=False,
        )
    return CombinedIterator(loaders)
