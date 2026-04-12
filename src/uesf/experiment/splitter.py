"""UESF Splitter - dataset splitting strategies with dimension isolation.

Supports Holdout, K-Fold, and Leave-One-Out (LOOCV) strategies for
regular deep learning, as well as UDA (Unsupervised Domain Adaptation)
splitting for cross-dataset and intra-dataset domain adaptation.

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


def create_splitter(
    split_config: dict[str, Any] | None = None,
    *,
    mode: str = "regular",
    uda_config: dict[str, Any] | None = None,
) -> HoldoutSplitter | KFoldSplitter | DatasetLevelSplitter | IntraDatasetUDASplitter | CrossDatasetUDASplitter:
    """Factory function to create the appropriate splitter.

    Args:
        split_config: Split configuration dict (for regular mode).
        mode: ``"regular"`` or ``"uda"``.
        uda_config: UDA configuration dict (required when mode is ``"uda"``).

    Returns:
        A splitter instance.
    """
    if mode == "uda":
        if uda_config is None:
            raise ConfigError("uda_config is required when mode='uda'.")
        uda_type = uda_config.get("type")
        if uda_type == "cross-dataset":
            return CrossDatasetUDASplitter(uda_config)
        if uda_type == "intra-dataset":
            return IntraDatasetUDASplitter(uda_config)
        raise ConfigError(
            f"Unknown UDA type: '{uda_type}'",
            hint="Use 'cross-dataset' or 'intra-dataset'.",
        )

    # Regular mode
    if split_config is None:
        split_config = {"strategy": "holdout"}

    dimension = split_config.get("dimension", "none")
    if dimension == "dataset":
        return DatasetLevelSplitter(split_config)

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

        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ConfigError(
                f"Split ratios must sum to 1.0, got {total_ratio:.4f}",
                hint=f"train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio}",
            )

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

        n_train = max(1, round(n * self.train_ratio))
        n_val = max(0, round(n * self.val_ratio))
        # Clamp so train + val never exceeds n
        if n_train + n_val > n:
            n_val = n - n_train

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
        if k != -1 and k != "total" and (not isinstance(k, int) or k < 2):
            raise ConfigError(
                f"k-folds must be an integer >= 2, -1, or 'total', got {k!r}",
                hint="Use k >= 2 for k-fold CV, or -1 / 'total' for LOOCV.",
            )
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

    Data must have shape [subject, session, recording, channel, sample] (5D).
    Each recording is one sample unit; total units = sub * sess * rec.

    For 'none', each recording is its own group.
    For 'subject', recordings are grouped by subject.
    For 'session', recordings are grouped by (subject, session).
    For 'recording', each recording is its own group (same as 'none').
    """
    if data.ndim != 5:
        raise ConfigError(
            f"Data must be 5-D [subject, session, recording, channel, sample], "
            f"got {data.ndim}-D with shape {data.shape}",
        )

    n_subjects, n_sessions, n_recordings = data.shape[:3]
    total = n_subjects * n_sessions * n_recordings

    if dimension == "none" or dimension == "recording":
        return [np.array([i]) for i in range(total)]

    if dimension == "subject":
        samples_per_subject = n_sessions * n_recordings
        groups = []
        for s in range(n_subjects):
            start = s * samples_per_subject
            groups.append(np.arange(start, start + samples_per_subject))
        return groups

    if dimension == "session":
        groups = []
        for s in range(n_subjects):
            for sess in range(n_sessions):
                start = (s * n_sessions + sess) * n_recordings
                groups.append(np.arange(start, start + n_recordings))
        return groups

    raise ConfigError(
        f"Unknown split dimension: '{dimension}'",
        hint="Use 'subject', 'session', 'recording', or 'none'.",
    )


# ---------------------------------------------------------------------------
# UDA split result
# ---------------------------------------------------------------------------


class UDASplitResult:
    """Holds index arrays for a single UDA split/fold.

    Each field is ``dict[alias, np.ndarray]`` mapping a dataset alias to
    flat sample indices within that dataset.  For intra-dataset UDA all
    dicts have a single key (the sole dataset alias).
    """

    def __init__(
        self,
        source_train_indices: dict[str, np.ndarray],
        source_val_indices: dict[str, np.ndarray] | None = None,
        target_train_indices: dict[str, np.ndarray] | None = None,
        target_val_indices: dict[str, np.ndarray] | None = None,
        target_test_indices: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.source_train_indices = source_train_indices
        self.source_val_indices = source_val_indices or {
            k: np.array([], dtype=int) for k in source_train_indices
        }
        self.target_train_indices = target_train_indices or {}
        self.target_val_indices = target_val_indices or {
            k: np.array([], dtype=int) for k in (target_train_indices or {})
        }
        self.target_test_indices = target_test_indices or {}


# ---------------------------------------------------------------------------
# Dataset-level splitter (regular DL, dimension=dataset)
# ---------------------------------------------------------------------------


class DatasetLevelSplitResult:
    """Maps dataset aliases to phases for dataset-level splits."""

    def __init__(self, phase_aliases: dict[str, list[str]]) -> None:
        self.phase_aliases = phase_aliases  # "train" -> [alias, ...], etc.


class DatasetLevelSplitter:
    """Splits at the dataset level for cross-dataset regular DL.

    Supports explicit assignment (``train: [ds_a]``, ``test: [ds_b]``)
    or ratio-based automatic partitioning.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.train_aliases: list[str] | None = config.get("train")
        self.val_aliases: list[str] | None = config.get("val")
        self.test_aliases: list[str] | None = config.get("test")

        self.train_ratio = config.get("train_ratio", 0.7)
        self.val_ratio = config.get("val_ratio", 0.0)
        self.test_ratio = config.get("test_ratio", 0.3)
        self.shuffle = config.get("shuffle", True)
        self.seed = config.get("seed")

    def split(self, dataset_aliases: list[str]) -> list[DatasetLevelSplitResult]:
        """Partition dataset aliases into train/val/test.

        Args:
            dataset_aliases: List of all dataset alias strings.

        Returns:
            Single-element list containing a ``DatasetLevelSplitResult``.
        """
        # Explicit mode: user listed aliases per phase
        if self.train_aliases is not None or self.test_aliases is not None:
            train = list(self.train_aliases or [])
            val = list(self.val_aliases or [])
            test = list(self.test_aliases or [])
            # Validate all specified aliases exist
            all_specified = set(train + val + test)
            unknown = all_specified - set(dataset_aliases)
            if unknown:
                raise ConfigError(
                    f"Unknown dataset aliases in split config: {unknown}",
                    hint=f"Available aliases: {dataset_aliases}",
                )
            return [DatasetLevelSplitResult({"train": train, "val": val, "test": test})]

        # Ratio mode: distribute aliases by ratio
        aliases = list(dataset_aliases)
        n = len(aliases)
        if n < 2:
            raise ConfigError(
                "Dataset-level split requires at least 2 datasets.",
                hint="Add more datasets or use a different split dimension.",
            )

        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(aliases)

        n_train = max(1, round(n * self.train_ratio))
        n_val = max(0, round(n * self.val_ratio))
        if n_train + n_val > n:
            n_val = n - n_train

        train = aliases[:n_train]
        val = aliases[n_train:n_train + n_val]
        test = aliases[n_train + n_val:]

        return [DatasetLevelSplitResult({"train": train, "val": val, "test": test})]


# ---------------------------------------------------------------------------
# Intra-dataset UDA splitter
# ---------------------------------------------------------------------------


class IntraDatasetUDASplitter:
    """Intra-dataset UDA: source/target come from the same dataset.

    Splits by subject or session dimension.  Supports four combinations:
    ``(holdout | k-fold) x (transductive | inductive)``.

    - **holdout**: ``target_count`` (priority) or ``target_ratio`` groups
      become the target domain; the rest are source.  Produces 1 fold.
    - **k-fold**: Each group takes a turn as target.  Produces k folds.
    - **transductive**: Target test = target train (all target data).
    - **inductive**: Target data is sub-split into train(/val)/test via
      ``target_split`` config.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.dimension = config.get("dimension", "subject")
        self.strategy = config.get("strategy", "holdout")
        self.variant = config.get("variant", "transductive")
        self.seed = config.get("seed")
        self.shuffle = config.get("shuffle", True)

        # Holdout target sizing
        self.target_count: int | None = config.get("target_count")
        self.target_ratio: float = config.get("target_ratio", 0.2)

        # K-fold
        k = config.get("k-folds", config.get("k_folds", 5))
        self.k = k

        # Source val split
        source_split = config.get("source_split", {})
        self.source_val_ratio: float = source_split.get("val_ratio", 0.0)

        # Inductive target sub-split
        self.target_split: dict[str, Any] | None = config.get("target_split")

    def split(self, data: np.ndarray, alias: str = "main") -> list[UDASplitResult]:
        """Split data into source/target domain partitions.

        Args:
            data: 5-D dataset array ``[sub, sess, rec, ch, samp]``.
            alias: Dataset alias used as the key in result index dicts.

        Returns:
            List of ``UDASplitResult``, one per fold.
        """
        groups = _get_groups(data, self.dimension)
        n = len(groups)

        indices = list(range(n))
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(indices)

        if self.strategy == "holdout":
            folds = self._holdout_folds(indices, n)
        elif self.strategy == "k-fold":
            folds = self._kfold_folds(indices, n)
        else:
            raise ConfigError(
                f"Unknown UDA strategy: '{self.strategy}'",
                hint="Use 'holdout' or 'k-fold'.",
            )

        results = []
        for target_group_idx, source_group_idx in folds:
            results.append(
                self._build_uda_result(groups, source_group_idx, target_group_idx, alias)
            )
        return results

    # -- fold generation helpers --

    def _holdout_folds(
        self, indices: list[int], n: int,
    ) -> list[tuple[list[int], list[int]]]:
        """Return a single (target_groups, source_groups) pair."""
        if self.target_count is not None:
            n_target = self.target_count
        else:
            n_target = max(1, round(n * self.target_ratio))

        if n_target >= n:
            raise ConfigError(
                f"target size ({n_target}) must be < total groups ({n}).",
                hint="Reduce target_count / target_ratio.",
            )

        target_idx = indices[:n_target]
        source_idx = indices[n_target:]
        return [(target_idx, source_idx)]

    def _kfold_folds(
        self, indices: list[int], n: int,
    ) -> list[tuple[list[int], list[int]]]:
        """Return k (target_groups, source_groups) pairs."""
        k = n if (self.k == -1 or self.k == "total") else self.k
        if not isinstance(k, int) or k < 2:
            raise ConfigError(
                f"k-folds must be >= 2 or -1/'total', got {self.k!r}",
            )
        if k > n:
            logger.warning("k=%d > groups=%d, using k=%d (LOOCV)", k, n, n)
            k = n

        fold_size = n // k
        remainder = n % k
        folds = []
        start = 0
        for fold_idx in range(k):
            end = start + fold_size + (1 if fold_idx < remainder else 0)
            target_idx = indices[start:end]
            source_idx = indices[:start] + indices[end:]
            folds.append((target_idx, source_idx))
            start = end
        return folds

    # -- result builder --

    def _build_uda_result(
        self,
        groups: list[np.ndarray],
        source_group_idx: list[int],
        target_group_idx: list[int],
        alias: str,
    ) -> UDASplitResult:
        source_all = (
            np.concatenate([groups[i] for i in source_group_idx])
            if source_group_idx else np.array([], dtype=int)
        )
        target_all = (
            np.concatenate([groups[i] for i in target_group_idx])
            if target_group_idx else np.array([], dtype=int)
        )

        # Source val split
        source_train, source_val = self._split_val(
            source_all, self.source_val_ratio,
        )

        # Target: transductive vs inductive
        if self.variant == "inductive" and self.target_split:
            tgt_train, tgt_val, tgt_test = self._inductive_split(target_all)
        else:
            # Transductive: all target data for both training and testing
            tgt_train = target_all
            tgt_val = np.array([], dtype=int)
            tgt_test = target_all.copy()

        return UDASplitResult(
            source_train_indices={alias: source_train},
            source_val_indices={alias: source_val},
            target_train_indices={alias: tgt_train},
            target_val_indices={alias: tgt_val},
            target_test_indices={alias: tgt_test},
        )

    def _split_val(
        self, indices: np.ndarray, val_ratio: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split off a validation portion from indices."""
        if val_ratio <= 0 or len(indices) == 0:
            return indices, np.array([], dtype=int)
        n_val = max(1, round(len(indices) * val_ratio))
        return indices[:-n_val], indices[-n_val:]

    def _inductive_split(
        self, target_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sub-split target indices into train(/val)/test for inductive UDA."""
        cfg = self.target_split
        train_ratio = cfg.get("train_ratio", 0.7)
        val_ratio = cfg.get("val_ratio", 0.0)
        test_ratio = cfg.get("test_ratio", 0.3)

        n = len(target_indices)
        # Shuffle target indices deterministically
        rng = random.Random(self.seed)
        idx = list(target_indices)
        rng.shuffle(idx)

        n_train = max(1, round(n * train_ratio))
        n_val = max(0, round(n * val_ratio))
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)

        train = np.array(idx[:n_train], dtype=int)
        val = np.array(idx[n_train:n_train + n_val], dtype=int)
        test = np.array(idx[n_train + n_val:], dtype=int)
        return train, val, test


# ---------------------------------------------------------------------------
# Cross-dataset UDA splitter
# ---------------------------------------------------------------------------


class CrossDatasetUDASplitter:
    """Cross-dataset UDA: source and target are different datasets.

    Supports four combinations:
    ``(holdout | k-fold) x (transductive | inductive)``.

    - **holdout**: User specifies ``source_datasets`` / ``target_dataset``.
      Produces 1 fold.
    - **k-fold**: Each dataset takes a turn as target, the rest are source.
      Produces k folds.
    - **transductive**: All target data for unsupervised training and testing.
    - **inductive**: Target data sub-split into train(/val)/test.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.strategy = config.get("strategy", "holdout")
        self.variant = config.get("variant", "transductive")
        self.seed = config.get("seed")
        self.shuffle = config.get("shuffle", True)

        # Holdout explicit assignment
        self.source_datasets: list[str] | None = config.get("source_datasets")
        self.target_dataset: str | None = config.get("target_dataset")

        # K-fold
        k = config.get("k-folds", config.get("k_folds", -1))
        self.k = k

        # Source val
        source_split = config.get("source_split", {})
        self.source_val_ratio: float = source_split.get("val_ratio", 0.0)

        # Inductive target sub-split
        self.target_split: dict[str, Any] | None = config.get("target_split")

    def split(
        self,
        dataset_cache: dict[str, dict],
    ) -> list[UDASplitResult]:
        """Compute UDA splits across datasets.

        Args:
            dataset_cache: alias -> ``{"data": np.ndarray, "labels": ..., "meta": ...}``.
                           Data should be 5-D (original shape) for inductive
                           sub-splitting by dimension, or 3-D (already flat).

        Returns:
            List of ``UDASplitResult``, one per fold.
        """
        aliases = list(dataset_cache.keys())

        if self.strategy == "holdout":
            folds = self._holdout_folds(aliases)
        elif self.strategy == "k-fold":
            folds = self._kfold_folds(aliases)
        else:
            raise ConfigError(
                f"Unknown UDA strategy: '{self.strategy}'",
                hint="Use 'holdout' or 'k-fold'.",
            )

        results = []
        for target_aliases, source_aliases in folds:
            results.append(
                self._build_uda_result(dataset_cache, source_aliases, target_aliases)
            )
        return results

    def _holdout_folds(
        self, aliases: list[str],
    ) -> list[tuple[list[str], list[str]]]:
        if not self.source_datasets or not self.target_dataset:
            raise ConfigError(
                "Cross-dataset holdout UDA requires 'source_datasets' and "
                "'target_dataset' in uda config.",
            )
        unknown = (set(self.source_datasets) | {self.target_dataset}) - set(aliases)
        if unknown:
            raise ConfigError(
                f"Unknown dataset aliases in UDA config: {unknown}",
                hint=f"Available: {aliases}",
            )
        return [([self.target_dataset], list(self.source_datasets))]

    def _kfold_folds(
        self, aliases: list[str],
    ) -> list[tuple[list[str], list[str]]]:
        n = len(aliases)
        k = n if (self.k == -1 or self.k == "total") else self.k
        if k > n:
            logger.warning("k=%d > datasets=%d, using k=%d", k, n, n)
            k = n

        ordered = list(aliases)
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(ordered)

        fold_size = n // k
        remainder = n % k
        folds = []
        start = 0
        for fold_idx in range(k):
            end = start + fold_size + (1 if fold_idx < remainder else 0)
            target = ordered[start:end]
            source = ordered[:start] + ordered[end:]
            folds.append((target, source))
            start = end
        return folds

    def _build_uda_result(
        self,
        dataset_cache: dict[str, dict],
        source_aliases: list[str],
        target_aliases: list[str],
    ) -> UDASplitResult:
        # Source: all data, optionally split val
        src_train: dict[str, np.ndarray] = {}
        src_val: dict[str, np.ndarray] = {}
        for alias in source_aliases:
            n_total = self._flat_len(dataset_cache[alias])
            all_idx = np.arange(n_total, dtype=int)
            if self.source_val_ratio > 0:
                n_val = max(1, round(n_total * self.source_val_ratio))
                src_train[alias] = all_idx[:-n_val]
                src_val[alias] = all_idx[-n_val:]
            else:
                src_train[alias] = all_idx
                src_val[alias] = np.array([], dtype=int)

        # Target: transductive vs inductive
        tgt_train: dict[str, np.ndarray] = {}
        tgt_val: dict[str, np.ndarray] = {}
        tgt_test: dict[str, np.ndarray] = {}
        for alias in target_aliases:
            n_total = self._flat_len(dataset_cache[alias])
            all_idx = np.arange(n_total, dtype=int)

            if self.variant == "inductive" and self.target_split:
                t, v, te = self._inductive_target_split(
                    all_idx, dataset_cache[alias],
                )
                tgt_train[alias] = t
                tgt_val[alias] = v
                tgt_test[alias] = te
            else:
                # Transductive
                tgt_train[alias] = all_idx
                tgt_val[alias] = np.array([], dtype=int)
                tgt_test[alias] = all_idx.copy()

        return UDASplitResult(
            source_train_indices=src_train,
            source_val_indices=src_val,
            target_train_indices=tgt_train,
            target_val_indices=tgt_val,
            target_test_indices=tgt_test,
        )

    def _inductive_target_split(
        self,
        all_idx: np.ndarray,
        cache_entry: dict,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sub-split target data for inductive UDA."""
        cfg = self.target_split
        train_ratio = cfg.get("train_ratio", 0.7)
        val_ratio = cfg.get("val_ratio", 0.0)

        data = cache_entry["data"]
        dimension = cfg.get("dimension", "none")

        # If data is still 5-D and dimension isn't "none", use group-based split
        if data.ndim == 5 and dimension not in ("none", "recording"):
            groups = _get_groups(data, dimension)
            n = len(groups)
            indices = list(range(n))
            rng = random.Random(self.seed)
            rng.shuffle(indices)

            n_train = max(1, round(n * train_ratio))
            n_val = max(0, round(n * val_ratio))
            if n_train + n_val >= n:
                n_val = max(0, n - n_train - 1)

            train_g = [groups[i] for i in indices[:n_train]]
            val_g = [groups[i] for i in indices[n_train:n_train + n_val]]
            test_g = [groups[i] for i in indices[n_train + n_val:]]

            train = np.concatenate(train_g) if train_g else np.array([], dtype=int)
            val = np.concatenate(val_g) if val_g else np.array([], dtype=int)
            test = np.concatenate(test_g) if test_g else np.array([], dtype=int)
            return train, val, test

        # Simple ratio-based split on flat indices
        n = len(all_idx)
        idx = list(all_idx)
        rng = random.Random(self.seed)
        rng.shuffle(idx)

        n_train = max(1, round(n * train_ratio))
        n_val = max(0, round(n * val_ratio))
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)

        train = np.array(idx[:n_train], dtype=int)
        val = np.array(idx[n_train:n_train + n_val], dtype=int)
        test = np.array(idx[n_train + n_val:], dtype=int)
        return train, val, test

    @staticmethod
    def _flat_len(cache_entry: dict) -> int:
        """Total number of samples in a dataset (flat count)."""
        data = cache_entry["data"]
        if data.ndim == 5:
            return data.shape[0] * data.shape[1] * data.shape[2]
        if data.ndim == 3:
            return data.shape[0]
        return data.shape[0]
