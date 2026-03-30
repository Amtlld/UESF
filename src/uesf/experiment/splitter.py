"""UESF Splitter - dataset splitting strategies with dimension isolation.

Supports Holdout, K-Fold, and Leave-One-Out (LOOCV) strategies.
Dimension-based isolation prevents data leakage (e.g., same subject
never appears in both train and test).
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from uesf.core.exceptions import ConfigError
from uesf.core.logging import get_logger

logger = get_logger("experiment.splitter")


class SplitResult:
    """Holds index arrays for a single split/fold."""

    def __init__(
        self,
        train_indices: np.ndarray,
        val_indices: np.ndarray | None = None,
        test_indices: np.ndarray | None = None,
    ) -> None:
        self.train_indices = train_indices
        self.val_indices = val_indices if val_indices is not None else np.array([], dtype=int)
        self.test_indices = test_indices if test_indices is not None else np.array([], dtype=int)


def create_splitter(split_config: dict[str, Any]) -> HoldoutSplitter | KFoldSplitter:
    """Factory function to create the appropriate splitter."""
    strategy = split_config.get("strategy", "holdout")
    if strategy == "holdout":
        return HoldoutSplitter(split_config)
    if strategy == "k-fold":
        return KFoldSplitter(split_config)
    raise ConfigError(
        f"Unknown split strategy: '{strategy}'",
        hint="Use 'holdout' or 'k-fold'.",
    )


class HoldoutSplitter:
    """Simple train/val/test holdout split."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.train_ratio = config.get("train_ratio", 0.7)
        self.val_ratio = config.get("val_ratio", 0.15)
        self.test_ratio = config.get("test_ratio", 0.15)
        self.dimension = config.get("dimension", "none")
        self.shuffle = config.get("shuffle", True)
        self.seed = config.get("seed")

    def split(self, data: np.ndarray) -> list[SplitResult]:
        """Split data into train/val/test sets.

        Args:
            data: Full dataset array with shape [subject, session, recording, channel, sample]
                  or similar multi-dimensional array.

        Returns:
            List with a single SplitResult.
        """
        groups = _get_groups(data, self.dimension)
        n = len(groups)

        indices = list(range(n))
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(indices)

        n_train = max(1, int(n * self.train_ratio))
        n_val = max(0, int(n * self.val_ratio))

        train_groups = [groups[i] for i in indices[:n_train]]
        val_groups = [groups[i] for i in indices[n_train:n_train + n_val]]
        test_groups = [groups[i] for i in indices[n_train + n_val:]]

        train_idx = np.concatenate(train_groups) if train_groups else np.array([], dtype=int)
        val_idx = np.concatenate(val_groups) if val_groups else np.array([], dtype=int)
        test_idx = np.concatenate(test_groups) if test_groups else np.array([], dtype=int)

        return [SplitResult(train_idx, val_idx, test_idx)]


class KFoldSplitter:
    """K-Fold cross-validation splitter. k=-1 or k='total' for LOOCV."""

    def __init__(self, config: dict[str, Any]) -> None:
        k = config.get("k-folds", config.get("k_folds", 5))
        self.k = k
        self.dimension = config.get("dimension", "none")
        self.shuffle = config.get("shuffle", True)
        self.seed = config.get("seed")
        self.val_ratio_in_train = config.get("val_ratio_in_train", 0.0)

    def split(self, data: np.ndarray) -> list[SplitResult]:
        """Split data into K folds.

        Args:
            data: Full dataset array.

        Returns:
            List of SplitResult, one per fold.
        """
        groups = _get_groups(data, self.dimension)
        n = len(groups)

        # LOOCV
        k = n if (self.k == -1 or self.k == "total") else self.k
        if k > n:
            logger.warning("k=%d > number of groups=%d, using k=%d (LOOCV)", k, n, n)
            k = n

        indices = list(range(n))
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(indices)

        fold_size = n // k
        remainder = n % k

        results = []
        start = 0
        for fold_idx in range(k):
            end = start + fold_size + (1 if fold_idx < remainder else 0)
            test_group_indices = indices[start:end]
            train_group_indices = indices[:start] + indices[end:]

            # Split off validation from train if requested
            val_group_indices = []
            if self.val_ratio_in_train > 0 and len(train_group_indices) > 1:
                n_val = max(1, int(len(train_group_indices) * self.val_ratio_in_train))
                val_group_indices = train_group_indices[-n_val:]
                train_group_indices = train_group_indices[:-n_val]

            train_idx = (
                np.concatenate([groups[i] for i in train_group_indices])
                if train_group_indices else np.array([], dtype=int)
            )
            val_idx = (
                np.concatenate([groups[i] for i in val_group_indices])
                if val_group_indices else np.array([], dtype=int)
            )
            test_idx = (
                np.concatenate([groups[i] for i in test_group_indices])
                if test_group_indices else np.array([], dtype=int)
            )

            results.append(SplitResult(train_idx, val_idx, test_idx))
            start = end

        return results


def _get_groups(data: np.ndarray, dimension: str) -> list[np.ndarray]:
    """Group sample indices by the specified dimension.

    For 'none' dimension, each sample is its own group.
    For 'subject', samples are grouped by the first dimension.
    For 'session', samples are grouped by (subject, session) pairs.
    For 'recording', samples are grouped by (subject, session, recording) triples.

    The data is expected to have shape [subject, session, recording, channel, sample]
    or a flattened version thereof. We flatten to total samples and group accordingly.
    """
    if dimension == "none":
        total = data.shape[0] if data.ndim <= 2 else int(np.prod(data.shape[:-2]))
        return [np.array([i]) for i in range(total)]

    if data.ndim < 3:
        # Flat data: treat as single group per sample
        return [np.array([i]) for i in range(len(data))]

    # Multi-dimensional: shape is [subject, session, recording, channel, sample] or subset
    if dimension == "subject":
        n_subjects = data.shape[0]
        samples_per_subject = int(np.prod(data.shape[1:-2])) if data.ndim > 3 else 1
        groups = []
        for s in range(n_subjects):
            start = s * samples_per_subject
            groups.append(np.arange(start, start + samples_per_subject))
        return groups

    if dimension == "session":
        n_subjects = data.shape[0]
        n_sessions = data.shape[1]
        samples_per_session = int(np.prod(data.shape[2:-2])) if data.ndim > 4 else 1
        groups = []
        for s in range(n_subjects):
            for sess in range(n_sessions):
                start = (s * n_sessions + sess) * samples_per_session
                groups.append(np.arange(start, start + samples_per_session))
        return groups

    if dimension == "recording":
        n_subjects = data.shape[0]
        n_sessions = data.shape[1]
        n_recordings = data.shape[2]
        samples_per_recording = 1  # Each recording is one sample unit
        groups = []
        for s in range(n_subjects):
            for sess in range(n_sessions):
                for rec in range(n_recordings):
                    idx = (s * n_sessions * n_recordings + sess * n_recordings + rec) * samples_per_recording
                    groups.append(np.arange(idx, idx + samples_per_recording))
        return groups

    raise ConfigError(
        f"Unknown split dimension: '{dimension}'",
        hint="Use 'subject', 'session', 'recording', or 'none'.",
    )
