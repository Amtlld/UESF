"""Splitter result dataclasses.

All index arrays live in the flatten_3d sample-index space (unless stated).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _empty_int() -> np.ndarray:
    return np.array([], dtype=int)


@dataclass
class SplitResult:
    """Indices for one Regular split/fold (train/val/test).

    ``val_indices`` and ``test_indices`` may be empty arrays when the stage
    is absent (e.g. ValSplit has no test, train-only exotic cases).
    """

    train_indices: np.ndarray
    val_indices: np.ndarray = field(default_factory=_empty_int)
    test_indices: np.ndarray = field(default_factory=_empty_int)
    fold_info: dict = field(default_factory=dict)


@dataclass
class DatasetLevelSplitResult:
    """Alias-level assignment for dimension=dataset Regular splits.

    No sample indices are stored here — downstream Strategy fills them in via
    :class:`MultiDatasetSplitResult` after invoking a ValSplitter per alias.
    """

    phase_aliases: dict[str, list[str]]
    fold_info: dict = field(default_factory=dict)


@dataclass
class MultiDatasetSplitResult:
    """Cross-dataset Regular split result (phase -> alias -> indices).

    Produced by :class:`RegularExecutionStrategy` after combining a
    :class:`DatasetLevelSplitResult` with per-alias ``ValSplitter`` output.
    """

    phase_indices: dict[str, dict[str, np.ndarray]]
    fold_info: dict = field(default_factory=dict)


@dataclass
class DomainSplitResult:
    """One UDA domain fold."""

    source_indices: dict[str, np.ndarray]
    target_indices: dict[str, np.ndarray]
    fold_info: dict = field(default_factory=dict)


@dataclass
class UDASplitResult:
    """Final UDA split — domain + inner-split combined, flattened."""

    source_train: dict[str, np.ndarray]
    source_val: dict[str, np.ndarray] = field(default_factory=dict)
    target_train: dict[str, np.ndarray] = field(default_factory=dict)
    target_val: dict[str, np.ndarray] = field(default_factory=dict)
    target_test: dict[str, np.ndarray] = field(default_factory=dict)
    fold_info: dict = field(default_factory=dict)
