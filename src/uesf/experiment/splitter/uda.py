"""UDA splitters — DatasetDomainSplitter, DimensionDomainSplitter,
ValSplitter, UDAOrchestrator.
"""

from __future__ import annotations

import numpy as np

from uesf.core.exceptions import SplitError, TypeMismatchError
from uesf.core.logging import get_logger
from uesf.experiment.splitter.base import (
    DomainSplitResult,
    SplitResult,
    UDASplitResult,
)
from uesf.experiment.splitter.grouping import get_groups, require_nontrivial

logger = get_logger("experiment.splitter.uda")


def _rng_shuffle(n: int, shuffle: bool, seed: int) -> list[int]:
    order = list(range(n))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(order)
    return order


def _concat_or_empty(arrays: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(arrays) if arrays else np.array([], dtype=int)


# ---------------------------------------------------------------------------
# ValSplitter
# ---------------------------------------------------------------------------


class ValSplitter:
    """Carves a validation portion out of a given dataset.

    Input: 5-D data for a single dataset (or a train sub-slice thereof).
    Produces a :class:`SplitResult` with ``test_indices`` always empty.
    """

    def __init__(
        self,
        dimension: str,
        val_ratio: float,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        if not (0 <= val_ratio < 1):
            raise TypeMismatchError(
                f"ValSplitter: val_ratio must be in [0, 1), got {val_ratio}",
            )
        if dimension == "flatten" and not shuffle:
            raise TypeMismatchError(
                "ValSplitter: dimension='flatten' requires shuffle=True (R26).",
            )
        self.dimension = dimension
        self.val_ratio = val_ratio
        self.shuffle = shuffle
        self.seed = seed

    def split(
        self,
        data: np.ndarray,
        flat_indices: np.ndarray | None = None,
    ) -> SplitResult:
        if self.val_ratio == 0:
            if flat_indices is not None:
                return SplitResult(
                    train_indices=np.asarray(flat_indices, dtype=int),
                    val_indices=np.array([], dtype=int),
                )
            total = data.shape[0] * data.shape[1] * data.shape[2]
            return SplitResult(
                train_indices=np.arange(total, dtype=int),
                val_indices=np.array([], dtype=int),
            )

        if flat_indices is not None and self.dimension == "flatten":
            flat = np.asarray(flat_indices, dtype=int)
            order = _rng_shuffle(len(flat), self.shuffle, self.seed)
            n = len(flat)
            n_val = max(1, round(n * self.val_ratio))
            if n_val >= n:
                n_val = n - 1
            train_local = order[: n - n_val]
            val_local = order[n - n_val :]
            return SplitResult(
                train_indices=flat[train_local],
                val_indices=flat[val_local],
            )

        groups = get_groups(data, self.dimension)
        require_nontrivial(groups, f"ValSplitter(dimension={self.dimension!r})")
        order = _rng_shuffle(len(groups), self.shuffle, self.seed)
        n = len(order)
        n_val = max(1, round(n * self.val_ratio))
        if n_val >= n:
            n_val = n - 1
        train_pos = order[: n - n_val]
        val_pos = order[n - n_val :]
        train_groups = [groups[i] for i in train_pos]
        val_groups = [groups[i] for i in val_pos]
        return SplitResult(
            train_indices=_concat_or_empty(train_groups),
            val_indices=_concat_or_empty(val_groups),
        )


# ---------------------------------------------------------------------------
# Domain splitters
# ---------------------------------------------------------------------------


class DatasetDomainSplitter:
    """Domain split on ``dimension=dataset``."""

    def __init__(
        self,
        strategy: str,
        source: list[str] | None = None,
        target: str | None = None,
        k: int | None = None,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        if strategy not in ("holdout", "k-fold"):
            raise TypeMismatchError(
                f"DatasetDomainSplitter: unknown strategy '{strategy}'",
            )
        if strategy == "holdout" and (not source or not target):
            raise TypeMismatchError(
                "DatasetDomainSplitter: holdout requires 'source' list and 'target' alias.",
            )
        if strategy == "k-fold" and k is None:
            raise TypeMismatchError(
                "DatasetDomainSplitter: k-fold requires 'k'.",
            )
        self.strategy = strategy
        self.source = list(source) if source else []
        self.target = target
        self.k = k
        self.shuffle = shuffle
        self.seed = seed

    def split(self, aliases: list[str]) -> list[DomainSplitResult]:
        alias_set = set(aliases)
        if self.strategy == "holdout":
            missing = (set(self.source) | {self.target}) - alias_set
            if missing:
                raise TypeMismatchError(
                    f"DatasetDomainSplitter: unknown aliases {sorted(missing)}",
                    hint=f"Available: {aliases}",
                )
            return [
                DomainSplitResult(
                    source_indices={a: None for a in self.source},
                    target_indices={self.target: None},
                    fold_info={"fold_idx": 0, "target_alias": self.target},
                )
            ]

        n = len(aliases)
        if n < 2:
            raise SplitError(
                "DatasetDomainSplitter: need >= 2 datasets for k-fold.",
            )
        k = n if self.k == -1 else int(self.k)
        if k != n:
            raise TypeMismatchError(
                f"DatasetDomainSplitter: k={self.k} must equal number of datasets ({n}) or -1.",
            )

        order = _rng_shuffle(n, self.shuffle, self.seed)
        results: list[DomainSplitResult] = []
        for fold_idx in range(k):
            tgt_pos = order[fold_idx]
            tgt_alias = aliases[tgt_pos]
            src_aliases = [aliases[i] for i in order if i != tgt_pos]
            results.append(
                DomainSplitResult(
                    source_indices={a: None for a in src_aliases},
                    target_indices={tgt_alias: None},
                    fold_info={"fold_idx": fold_idx, "target_alias": tgt_alias},
                )
            )
        return results


class DimensionDomainSplitter:
    """Domain split on ``dimension=subject|session`` within a single dataset."""

    def __init__(
        self,
        strategy: str,
        dimension: str,
        target_count: int | None = None,
        target_ratio: float | None = None,
        k: int | None = None,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        if strategy not in ("holdout", "k-fold"):
            raise TypeMismatchError(
                f"DimensionDomainSplitter: unknown strategy '{strategy}'",
            )
        if dimension not in ("subject", "session"):
            raise TypeMismatchError(
                f"DimensionDomainSplitter: dimension must be 'subject' or 'session', got '{dimension}'",
            )
        if strategy == "holdout":
            if (target_count is None) and (target_ratio is None):
                raise TypeMismatchError(
                    "DimensionDomainSplitter: holdout requires target_count OR target_ratio.",
                )
            if (target_count is not None) and (target_ratio is not None):
                raise TypeMismatchError(
                    "DimensionDomainSplitter: target_count and target_ratio are mutually exclusive.",
                )
        if strategy == "k-fold" and k is None:
            raise TypeMismatchError(
                "DimensionDomainSplitter: k-fold requires 'k'.",
            )
        self.strategy = strategy
        self.dimension = dimension
        self.target_count = target_count
        self.target_ratio = target_ratio
        self.k = k
        self.shuffle = shuffle
        self.seed = seed

    def split(self, data: np.ndarray, alias: str) -> list[DomainSplitResult]:
        groups = get_groups(data, self.dimension)
        require_nontrivial(groups, f"DimensionDomainSplitter(dimension={self.dimension!r})")
        n = len(groups)
        order = _rng_shuffle(n, self.shuffle, self.seed)

        if self.strategy == "holdout":
            if self.target_count is not None:
                n_target = self.target_count
            else:
                n_target = max(1, round(n * self.target_ratio))
            if n_target <= 0 or n_target >= n:
                raise SplitError(
                    f"DimensionDomainSplitter: target size {n_target} must be in (0, {n}).",
                )
            tgt_pos = order[:n_target]
            src_pos = order[n_target:]
            tgt_idx = _concat_or_empty([groups[i] for i in tgt_pos])
            src_idx = _concat_or_empty([groups[i] for i in src_pos])
            return [
                DomainSplitResult(
                    source_indices={alias: src_idx},
                    target_indices={alias: tgt_idx},
                    fold_info={"fold_idx": 0, "target_groups": sorted(tgt_pos)},
                )
            ]

        k = n if self.k == -1 else int(self.k)
        if k > n:
            raise SplitError(
                f"DimensionDomainSplitter: k={self.k} exceeds groups ({n}).",
            )
        fold_size = n // k
        remainder = n % k

        results: list[DomainSplitResult] = []
        start = 0
        for fold_idx in range(k):
            end = start + fold_size + (1 if fold_idx < remainder else 0)
            tgt_pos = order[start:end]
            src_pos = order[:start] + order[end:]
            tgt_idx = _concat_or_empty([groups[i] for i in tgt_pos])
            src_idx = _concat_or_empty([groups[i] for i in src_pos])
            results.append(
                DomainSplitResult(
                    source_indices={alias: src_idx},
                    target_indices={alias: tgt_idx},
                    fold_info={"fold_idx": fold_idx, "target_groups": sorted(tgt_pos)},
                )
            )
            start = end
        return results


# ---------------------------------------------------------------------------
# UDAOrchestrator
# ---------------------------------------------------------------------------


class UDAOrchestrator:
    """Combines domain split with inner source/target splits into UDASplitResult.

    Layer seeds (per 04 doc §4.7):
      - domain split: ``seed + 0``
      - source inner split: ``seed + 1``
      - target inner split: ``seed + 2``
    """

    def __init__(self, uda_config: dict, seed: int = 42) -> None:
        self.config = uda_config
        self.seed = seed
        self.adaptation = uda_config["adaptation"]
        domain_cfg = uda_config["domain"]

        dom_shuffle = domain_cfg.get("shuffle", True)
        if domain_cfg["dimension"] == "dataset":
            self.domain_splitter: object = DatasetDomainSplitter(
                strategy=domain_cfg["strategy"],
                source=domain_cfg.get("source"),
                target=domain_cfg.get("target"),
                k=domain_cfg.get("k"),
                shuffle=dom_shuffle,
                seed=seed,
            )
        else:
            self.domain_splitter = DimensionDomainSplitter(
                strategy=domain_cfg["strategy"],
                dimension=domain_cfg["dimension"],
                target_count=domain_cfg.get("target_count"),
                target_ratio=domain_cfg.get("target_ratio"),
                k=domain_cfg.get("k"),
                shuffle=dom_shuffle,
                seed=seed,
            )

        source_cfg = uda_config.get("source")
        if source_cfg is not None and "split" in source_cfg:
            s = source_cfg["split"]
            self.source_splitter: ValSplitter | None = ValSplitter(
                dimension=s["dimension"],
                val_ratio=s["val_ratio"],
                shuffle=s.get("shuffle", True),
                seed=seed + 1,
            )
        else:
            self.source_splitter = None

        # target splitter only for inductive — built lazily since it depends
        # on the target dataset's 5-D data shape (for val_split clean slicing).
        self.target_split_cfg = (
            uda_config.get("target", {}).get("split")
            if self.adaptation == "inductive"
            else None
        )

    def split(self, dataset_cache: dict[str, np.ndarray]) -> list[UDASplitResult]:
        aliases = list(dataset_cache.keys())
        domain_cfg = self.config["domain"]

        if domain_cfg["dimension"] == "dataset":
            domain_folds = self.domain_splitter.split(aliases)
        else:
            if len(aliases) != 1:
                raise SplitError(
                    "DimensionDomainSplitter requires exactly one dataset "
                    "(R9/R25). Got: " + repr(aliases),
                )
            alias = aliases[0]
            domain_folds = self.domain_splitter.split(
                dataset_cache[alias], alias=alias
            )

        all_results: list[UDASplitResult] = []
        for d_idx, dfold in enumerate(domain_folds):
            src_train, src_val = self._build_source(dfold, dataset_cache)

            if self.adaptation == "transductive":
                tgt_train, tgt_val, tgt_test = self._build_target_transductive(
                    dfold, dataset_cache
                )
                fold_info = {"domain_fold": d_idx, "inner_fold": 0, **dfold.fold_info}
                all_results.append(
                    UDASplitResult(
                        source_train=src_train,
                        source_val=src_val,
                        target_train=tgt_train,
                        target_val=tgt_val,
                        target_test=tgt_test,
                        fold_info=fold_info,
                    )
                )
                continue

            # inductive: target_splitter may produce multiple inner folds
            tgt_inner_folds = self._build_target_inductive(dfold, dataset_cache)
            for t_idx, (tgt_train, tgt_val, tgt_test) in enumerate(tgt_inner_folds):
                fold_info = {
                    "domain_fold": d_idx,
                    "inner_fold": t_idx,
                    **dfold.fold_info,
                }
                all_results.append(
                    UDASplitResult(
                        source_train=dict(src_train),
                        source_val=dict(src_val),
                        target_train=tgt_train,
                        target_val=tgt_val,
                        target_test=tgt_test,
                        fold_info=fold_info,
                    )
                )
        return all_results

    # -- source --
    def _build_source(
        self, dfold: DomainSplitResult, dataset_cache: dict[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        src_train: dict[str, np.ndarray] = {}
        src_val: dict[str, np.ndarray] = {}
        domain_is_dataset = self.config["domain"]["dimension"] == "dataset"

        for alias in dfold.source_indices:
            data = dataset_cache[alias]
            if domain_is_dataset:
                if self.source_splitter is None:
                    n = data.shape[0] * data.shape[1] * data.shape[2]
                    src_train[alias] = np.arange(n, dtype=int)
                    src_val[alias] = np.array([], dtype=int)
                else:
                    sub = self.source_splitter.split(data)
                    src_train[alias] = sub.train_indices
                    src_val[alias] = sub.val_indices
            else:
                pool = dfold.source_indices[alias]  # flat indices within the dataset
                if self.source_splitter is None:
                    src_train[alias] = np.asarray(pool, dtype=int)
                    src_val[alias] = np.array([], dtype=int)
                else:
                    if self.source_splitter.dimension == "flatten":
                        sub = self.source_splitter.split(data, flat_indices=pool)
                    else:
                        sub = self._split_subset(data, pool, self.source_splitter)
                    src_train[alias] = sub.train_indices
                    src_val[alias] = sub.val_indices
        return src_train, src_val

    # -- target: transductive --
    def _build_target_transductive(
        self, dfold: DomainSplitResult, dataset_cache: dict[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
        train: dict[str, np.ndarray] = {}
        val: dict[str, np.ndarray] = {}
        test: dict[str, np.ndarray] = {}
        domain_is_dataset = self.config["domain"]["dimension"] == "dataset"
        for alias in dfold.target_indices:
            data = dataset_cache[alias]
            if domain_is_dataset:
                n = data.shape[0] * data.shape[1] * data.shape[2]
                all_idx = np.arange(n, dtype=int)
            else:
                all_idx = np.asarray(dfold.target_indices[alias], dtype=int)
            train[alias] = all_idx.copy()
            val[alias] = np.array([], dtype=int)
            test[alias] = all_idx.copy()
        return train, val, test

    # -- target: inductive --
    def _build_target_inductive(
        self, dfold: DomainSplitResult, dataset_cache: dict[str, np.ndarray]
    ) -> list[tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]]:
        cfg = self.target_split_cfg
        if cfg is None:
            raise SplitError(
                "UDAOrchestrator.inductive requires uda.target.split config.",
            )

        # Build a transient Regular splitter on the target dataset.
        from uesf.experiment.splitter.regular import HoldoutSplitter, KFoldSplitter

        strategy = cfg["strategy"]
        dim = cfg["dimension"]
        val_ratio = cfg.get("val_ratio", 0.0)
        val_split_cfg = cfg.get("val_split")
        shuffle = cfg.get("shuffle", True)

        if strategy == "holdout":
            splitter: HoldoutSplitter | KFoldSplitter = HoldoutSplitter(
                dimension=dim,
                train_ratio=cfg["train_ratio"],
                test_ratio=cfg["test_ratio"],
                val_ratio=val_ratio,
                val_split_config=val_split_cfg,
                shuffle=shuffle,
                seed=self.seed + 2,
            )
        elif strategy == "k-fold":
            splitter = KFoldSplitter(
                dimension=dim,
                k=cfg["k"],
                val_ratio=val_ratio,
                val_split_config=val_split_cfg,
                shuffle=shuffle,
                seed=self.seed + 2,
            )
        else:
            raise TypeMismatchError(
                f"UDAOrchestrator: unknown target.split.strategy '{strategy}'",
            )

        domain_is_dataset = self.config["domain"]["dimension"] == "dataset"
        results: list[
            tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]
        ] = []
        # Single target alias in both cross-dataset holdout/k-fold and intra cases.
        target_aliases = list(dfold.target_indices.keys())
        if len(target_aliases) != 1:
            raise SplitError(
                "UDAOrchestrator.inductive expects a single target alias per domain fold, "
                f"got {target_aliases}",
            )
        alias = target_aliases[0]
        data = dataset_cache[alias]

        if domain_is_dataset:
            inner_folds = splitter.split(data)
            for sr in inner_folds:
                results.append(
                    (
                        {alias: sr.train_indices},
                        {alias: sr.val_indices},
                        {alias: sr.test_indices},
                    )
                )
        else:
            # Intra-dataset UDA: the target domain is a subset of `data`.
            # We treat that subset as its own 5-D sub-slice along the domain
            # dimension, then delegate to the splitter.
            domain_dim = self.config["domain"]["dimension"]
            target_flat = np.asarray(dfold.target_indices[alias], dtype=int)
            sub_data, local_to_global = _subslice_along_dim(
                data, domain_dim, target_flat
            )
            inner_folds = splitter.split(sub_data)
            for sr in inner_folds:
                results.append(
                    (
                        {alias: local_to_global[sr.train_indices]},
                        {alias: local_to_global[sr.val_indices]},
                        {alias: local_to_global[sr.test_indices]},
                    )
                )
        return results

    # -- helper: carve a sub-5-D along a domain dimension --
    def _split_subset(
        self,
        data: np.ndarray,
        pool_indices: np.ndarray,
        splitter: ValSplitter,
    ) -> SplitResult:
        """Run a ValSplitter on a sub-pool of a 5-D dataset.

        For intra-dataset UDA, the source domain may be only part of a
        subject/session space. We build a clean 5-D sub-slice (by rearranging
        rows into the first axis) and run the splitter, then map the returned
        local indices back to the global flatten_3d index space.
        """
        domain_dim = self.config["domain"]["dimension"]
        sub_data, local_to_global = _subslice_along_dim(data, domain_dim, pool_indices)
        sub = splitter.split(sub_data)
        return SplitResult(
            train_indices=local_to_global[sub.train_indices],
            val_indices=local_to_global[sub.val_indices],
        )


# ---------------------------------------------------------------------------
# sub-slice helper
# ---------------------------------------------------------------------------


def _subslice_along_dim(
    data: np.ndarray, dimension: str, flat_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Collect units along a given dimension into a clean 5-D sub-slice.

    Returns ``(sub_data, local_to_global)`` where ``sub_data`` is shaped so
    that ``get_groups(sub_data, dimension)`` yields one group per unit in the
    original pool, and ``local_to_global[i]`` maps the i-th flattened sample
    in ``sub_data`` back to its index in ``flat_indices``-space.
    """
    n_sub, n_sess, n_rec = data.shape[:3]
    ch, samp = data.shape[3], data.shape[4]

    # Reshape to [units, ch, samp] with units grouped by the domain dim.
    flat3 = data.reshape(-1, ch, samp)  # [U, ch, samp], U = n_sub*n_sess*n_rec
    flat_indices = np.asarray(flat_indices, dtype=int)

    if dimension == "subject":
        step = n_sess * n_rec
        # Each subject contributes `step` consecutive flat indices.
        subj_ids = np.unique(flat_indices // step)
        blocks = []
        local_to_global: list[np.ndarray] = []
        for sid in subj_ids:
            start = sid * step
            block = flat3[start : start + step]  # [step, ch, samp]
            block = block.reshape(1, n_sess, n_rec, ch, samp)
            blocks.append(block)
            local_to_global.append(np.arange(start, start + step, dtype=int))
        sub_data = np.concatenate(blocks, axis=0)  # [K_subj, n_sess, n_rec, ch, samp]
        return sub_data, np.concatenate(local_to_global)

    if dimension == "session":
        # Each (subj, sess) contributes n_rec consecutive flats.
        sess_ids = np.unique(flat_indices // n_rec)
        blocks = []
        local_to_global = []
        for sid in sess_ids:
            start = sid * n_rec
            block = flat3[start : start + n_rec]  # [n_rec, ch, samp]
            block = block.reshape(1, 1, n_rec, ch, samp)
            blocks.append(block)
            local_to_global.append(np.arange(start, start + n_rec, dtype=int))
        sub_data = np.concatenate(blocks, axis=0)  # [K_sess, 1, n_rec, ch, samp]
        return sub_data, np.concatenate(local_to_global)

    if dimension == "recording":
        # Each recording is its own unit.
        rec_ids = np.unique(flat_indices)
        block = flat3[rec_ids]  # [K, ch, samp]
        sub_data = block.reshape(block.shape[0], 1, 1, ch, samp)
        return sub_data, rec_ids.astype(int)

    if dimension == "flatten":
        # Treat each index as standalone; caller should use flat_indices API
        # on ValSplitter instead. We still support it here for completeness.
        rec_ids = np.asarray(flat_indices, dtype=int)
        block = flat3[rec_ids]
        sub_data = block.reshape(block.shape[0], 1, 1, ch, samp)
        return sub_data, rec_ids

    raise TypeMismatchError(
        f"_subslice_along_dim: unsupported dimension '{dimension}'",
    )
