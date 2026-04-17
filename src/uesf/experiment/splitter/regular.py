"""Regular splitters — HoldoutSplitter, KFoldSplitter, DatasetLevelSplitter."""

from __future__ import annotations

import numpy as np

from uesf.core.exceptions import SplitError, TypeMismatchError
from uesf.core.logging import get_logger
from uesf.experiment.splitter.base import (
    DatasetLevelSplitResult,
    SplitResult,
)
from uesf.experiment.splitter.grouping import get_groups, require_nontrivial

logger = get_logger("experiment.splitter.regular")


def _shuffle_order(n: int, shuffle: bool, seed: int) -> list[int]:
    order = list(range(n))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(order)
    return order


def _concat_or_empty(arrays: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(arrays) if arrays else np.array([], dtype=int)


class HoldoutSplitter:
    """Holdout Regular split producing a single train/val/test fold.

    Two operating modes:

    - No ``val_split_config``: three-way split at the same ``dimension``.
      ``train_ratio + val_ratio + test_ratio`` must ≈ 1.0.
    - With ``val_split_config``: main split produces train/test only
      (``val_ratio`` must be ``0``), then :class:`ValSplitter` carves val out
      of the train portion on a potentially different dimension.
    """

    def __init__(
        self,
        dimension: str,
        train_ratio: float,
        test_ratio: float,
        val_ratio: float = 0.0,
        val_split_config: dict | None = None,
        shuffle: bool = False,
        seed: int = 42,
    ) -> None:
        self.dimension = dimension
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.val_split_config = val_split_config
        self.shuffle = shuffle
        self.seed = seed

        if val_split_config is not None and val_ratio not in (0, 0.0):
            raise TypeMismatchError(
                "HoldoutSplitter: val_ratio must be 0 when val_split_config is set.",
                hint="Use val_split_config.val_ratio for the independent val split.",
            )

    def split(self, data: np.ndarray) -> list[SplitResult]:
        groups = get_groups(data, self.dimension)
        require_nontrivial(groups, f"HoldoutSplitter(dimension={self.dimension!r})")

        order = _shuffle_order(len(groups), self.shuffle, self.seed)
        n = len(order)

        if self.val_split_config is None:
            n_train = max(1, round(n * self.train_ratio))
            n_val = max(0, round(n * self.val_ratio))
            if n_train + n_val > n:
                n_val = max(0, n - n_train)
            train_groups = [groups[i] for i in order[:n_train]]
            val_groups = [groups[i] for i in order[n_train : n_train + n_val]]
            test_groups = [groups[i] for i in order[n_train + n_val :]]

            return [
                SplitResult(
                    train_indices=_concat_or_empty(train_groups),
                    val_indices=_concat_or_empty(val_groups),
                    test_indices=_concat_or_empty(test_groups),
                    fold_info={"fold_idx": 0},
                )
            ]

        # val_split path — two-way main split, then inner ValSplit on train part
        n_train = max(1, round(n * self.train_ratio))
        if n_train >= n:
            n_train = n - 1
        train_group_pos = order[:n_train]
        test_group_pos = order[n_train:]
        train_groups = [groups[i] for i in train_group_pos]
        test_groups = [groups[i] for i in test_group_pos]

        train_all = _concat_or_empty(train_groups)
        test_all = _concat_or_empty(test_groups)

        train_idx, val_idx = _inner_val_split(
            data=data,
            dimension=self.dimension,
            train_group_pos=train_group_pos,
            train_flat=train_all,
            val_split_config=self.val_split_config,
        )

        return [
            SplitResult(
                train_indices=train_idx,
                val_indices=val_idx,
                test_indices=test_all,
                fold_info={"fold_idx": 0},
            )
        ]


class KFoldSplitter:
    """K-Fold Regular split. ``k == -1`` ⇒ leave-one-out (k = len(groups))."""

    def __init__(
        self,
        dimension: str,
        k: int,
        val_ratio: float = 0.0,
        val_split_config: dict | None = None,
        shuffle: bool = False,
        seed: int = 42,
    ) -> None:
        if k != -1 and (not isinstance(k, int) or k < 2):
            raise TypeMismatchError(
                f"KFoldSplitter: k must be an int >= 2 or -1, got {k!r}",
                hint="Use k >= 2 for k-fold CV or k=-1 for leave-one-out.",
            )
        if val_split_config is not None and val_ratio not in (0, 0.0):
            raise TypeMismatchError(
                "KFoldSplitter: val_ratio must be 0 when val_split_config is set.",
                hint="Use val_split_config.val_ratio for the independent val split.",
            )
        self.dimension = dimension
        self.k = k
        self.val_ratio = val_ratio  # val_ratio_in_train (already converted by ConfigValidator)
        self.val_split_config = val_split_config
        self.shuffle = shuffle
        self.seed = seed

    def split(self, data: np.ndarray) -> list[SplitResult]:
        groups = get_groups(data, self.dimension)
        require_nontrivial(groups, f"KFoldSplitter(dimension={self.dimension!r})")

        n = len(groups)
        k = n if self.k == -1 else self.k
        if k > n:
            raise SplitError(
                f"KFoldSplitter: k={self.k} exceeds number of groups ({n}).",
                hint="Reduce k or add more data along this dimension.",
            )

        order = _shuffle_order(n, self.shuffle, self.seed)
        fold_size = n // k
        remainder = n % k

        results: list[SplitResult] = []
        start = 0
        for fold_idx in range(k):
            end = start + fold_size + (1 if fold_idx < remainder else 0)
            test_pos = order[start:end]
            train_pos = order[:start] + order[end:]

            test_groups = [groups[i] for i in test_pos]
            train_all_groups = [groups[i] for i in train_pos]

            fold_info: dict = {"fold_idx": fold_idx}
            if len(test_pos) == 1:
                fold_info["test_group"] = int(test_pos[0])

            if self.val_split_config is None and self.val_ratio > 0 and len(train_pos) > 1:
                n_val = max(1, int(round(len(train_pos) * self.val_ratio)))
                if n_val >= len(train_pos):
                    n_val = len(train_pos) - 1
                val_pos = train_pos[-n_val:]
                train_pos_final = train_pos[:-n_val]
                train_groups = [groups[i] for i in train_pos_final]
                val_groups = [groups[i] for i in val_pos]
                results.append(
                    SplitResult(
                        train_indices=_concat_or_empty(train_groups),
                        val_indices=_concat_or_empty(val_groups),
                        test_indices=_concat_or_empty(test_groups),
                        fold_info=fold_info,
                    )
                )
            elif self.val_split_config is not None:
                train_flat = _concat_or_empty(train_all_groups)
                train_idx, val_idx = _inner_val_split(
                    data=data,
                    dimension=self.dimension,
                    train_group_pos=train_pos,
                    train_flat=train_flat,
                    val_split_config=self.val_split_config,
                )
                results.append(
                    SplitResult(
                        train_indices=train_idx,
                        val_indices=val_idx,
                        test_indices=_concat_or_empty(test_groups),
                        fold_info=fold_info,
                    )
                )
            else:
                results.append(
                    SplitResult(
                        train_indices=_concat_or_empty(train_all_groups),
                        val_indices=np.array([], dtype=int),
                        test_indices=_concat_or_empty(test_groups),
                        fold_info=fold_info,
                    )
                )
            start = end

        return results


class DatasetLevelSplitter:
    """Alias-level split for Regular ``dimension=dataset``.

    Does not accept 5-D arrays — works on the list of aliases.
    """

    def __init__(
        self,
        strategy: str,
        assign: dict | None = None,
        k: int | None = None,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        if strategy not in ("holdout", "k-fold"):
            raise TypeMismatchError(
                f"DatasetLevelSplitter: unknown strategy '{strategy}'",
                hint="Use 'holdout' or 'k-fold'.",
            )
        if strategy == "holdout" and not assign:
            raise TypeMismatchError(
                "DatasetLevelSplitter: holdout requires explicit 'assign' "
                "(e.g. {'train': [...], 'test': [...]}).",
            )
        if strategy == "k-fold" and k is None:
            raise TypeMismatchError(
                "DatasetLevelSplitter: k-fold requires 'k' argument.",
            )
        self.strategy = strategy
        self.assign = assign
        self.k = k
        self.shuffle = shuffle
        self.seed = seed

    def split(self, aliases: list[str]) -> list[DatasetLevelSplitResult]:
        if self.strategy == "holdout":
            train = list(self.assign.get("train", []))
            test = list(self.assign.get("test", []))
            specified = set(train) | set(test)
            unknown = specified - set(aliases)
            if unknown:
                raise TypeMismatchError(
                    f"DatasetLevelSplitter.assign contains unknown aliases: {sorted(unknown)}",
                    hint=f"Available aliases: {aliases}",
                )
            missing = set(aliases) - specified
            if missing:
                raise TypeMismatchError(
                    f"DatasetLevelSplitter.assign missing aliases: {sorted(missing)}",
                    hint="assign must cover every declared dataset.",
                )
            return [
                DatasetLevelSplitResult(
                    phase_aliases={"train": train, "val": [], "test": test},
                    fold_info={"fold_idx": 0},
                )
            ]

        n = len(aliases)
        if n < 2:
            raise SplitError(
                "DatasetLevelSplitter: k-fold needs at least 2 datasets.",
            )
        k = n if self.k == -1 else int(self.k)
        if k != n:
            raise TypeMismatchError(
                f"DatasetLevelSplitter: k={self.k} must equal number of datasets ({n}) or -1.",
            )

        order = list(range(n))
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(order)

        results: list[DatasetLevelSplitResult] = []
        for fold_idx in range(k):
            test_pos = order[fold_idx]
            test_alias = aliases[test_pos]
            train = [aliases[i] for i in order if i != test_pos]
            results.append(
                DatasetLevelSplitResult(
                    phase_aliases={"train": train, "val": [], "test": [test_alias]},
                    fold_info={"fold_idx": fold_idx, "test_alias": test_alias},
                )
            )
        return results


# ---------------------------------------------------------------------------
# Internal helper for val_split path
# ---------------------------------------------------------------------------


def _inner_val_split(
    *,
    data: np.ndarray,
    dimension: str,
    train_group_pos: list[int],
    train_flat: np.ndarray,
    val_split_config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a ValSplitter on the train portion of a Regular split.

    When the main split dimension is not ``flatten``, we take a clean 5-D
    sub-slice along that dimension and hand it to the ValSplitter
    (which calls ``get_groups`` internally).  When the main dimension is
    ``flatten``, R30 guarantees ``val_split.dimension == 'flatten'`` —
    we delegate by passing ``flat_indices=train_flat``.
    """
    # Deferred import to avoid circular reference.
    from uesf.experiment.splitter.uda import ValSplitter

    inner_dim = val_split_config.get("dimension", dimension)
    inner_seed = val_split_config.get("seed", 0)
    inner_shuffle = val_split_config.get("shuffle", True)
    inner_val_ratio = val_split_config["val_ratio"]

    val_splitter = ValSplitter(
        dimension=inner_dim,
        val_ratio=inner_val_ratio,
        shuffle=inner_shuffle,
        seed=inner_seed,
    )

    if dimension == "flatten":
        sub = val_splitter.split(data, flat_indices=train_flat)
        return sub.train_indices, sub.val_indices

    # Clean sub-slice: take along the main axis (subject/session/recording)
    axis_map = {"subject": 0, "session": 1, "recording": 2}
    if dimension not in axis_map:
        raise TypeMismatchError(
            f"_inner_val_split: unsupported main dimension '{dimension}'",
        )
    axis = axis_map[dimension]

    if dimension == "subject":
        train_data = data[sorted(train_group_pos)]
    elif dimension == "session":
        # Rebuild indexing: order was computed over n_subjects*n_sessions groups.
        n_sub, n_sess = data.shape[:2]
        pairs = [(p // n_sess, p % n_sess) for p in sorted(train_group_pos)]
        # We cannot simply reshape; use a mask approach.
        mask = np.zeros((n_sub, n_sess), dtype=bool)
        for s, ss in pairs:
            mask[s, ss] = True
        # Build a sub array of shape [sum(mask), n_rec, ch, sample] then
        # expand back to 5-D by placing each surviving (s, ss) as its own
        # subject-row with a single session. This preserves get_groups('session')
        # semantics on the sub-slice (one group per kept pair).
        collected = data[mask]  # [K, n_rec, ch, sample]
        train_data = collected[:, None, :, :, :]  # [K, 1, n_rec, ch, sample]
    else:  # recording
        # Each recording is a standalone group; reshape to [K, 1, 1, ch, sample]
        n_sub, n_sess, n_rec = data.shape[:3]
        flat_rec = data.reshape(-1, *data.shape[3:])  # [n_sub*n_sess*n_rec, ch, sample]
        collected = flat_rec[sorted(train_group_pos)]
        train_data = collected[:, None, None, :, :, :]
        # numpy will view as 6-D, so fix explicitly
        train_data = collected.reshape(
            collected.shape[0], 1, 1, collected.shape[-2], collected.shape[-1]
        )

    sub = val_splitter.split(train_data)

    # Map local indices (into sub) back to the original flatten_3d indices.
    sub_local_to_global = train_flat
    train_idx_global = sub_local_to_global[sub.train_indices]
    val_idx_global = sub_local_to_global[sub.val_indices]
    _ = axis  # axis kept for debugging/trace, intentionally unused below
    return train_idx_global, val_idx_global
