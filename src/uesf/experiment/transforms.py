"""UESF online transforms and pipeline application.

Transforms run AFTER splitting, following fit-on-train / apply-to-all to
avoid leakage. The :func:`apply_transforms` (Regular) and
:func:`apply_transforms_uda` (UDA) helpers are the unified entry points —
each walks per-step through ``transforms_per_alias`` and dispatches to
scope-specific internals.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from uesf.core.exceptions import ComponentNotFoundError, ConfigError
from uesf.core.logging import get_logger

logger = get_logger("experiment.transforms")


# ---------------------------------------------------------------------------
# Transform registry
# ---------------------------------------------------------------------------


class ZScoreNormalize:
    """Per-feature z-score normalization."""

    def __init__(self, **kwargs: Any) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.eps = kwargs.get("eps", 1e-8)

    def fit(self, data: np.ndarray) -> None:
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("ZScoreNormalize.fit() must be called before transform()")
        return (data - self.mean) / (self.std + self.eps)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.transform(data)


TRANSFORM_REGISTRY: dict[str, type] = {
    "zscore_normalize": ZScoreNormalize,
}


def create_transform(name: str, **kwargs: Any) -> Any:
    if name not in TRANSFORM_REGISTRY:
        raise ComponentNotFoundError(
            f"Unknown transform: '{name}'",
            context={"available": sorted(TRANSFORM_REGISTRY.keys())},
            hint=f"Available transforms: {', '.join(sorted(TRANSFORM_REGISTRY.keys()))}",
        )
    return TRANSFORM_REGISTRY[name](**kwargs)


# ---------------------------------------------------------------------------
# Internal shape helper
# ---------------------------------------------------------------------------


def _flatten_5d(data: np.ndarray) -> np.ndarray:
    """Reshape 5-D [sub, sess, rec, ch, samp] → 3-D [N, ch, samp]."""
    if data.ndim != 5:
        raise ConfigError(
            f"Expected 5-D data, got {data.ndim}-D with shape {data.shape}",
        )
    return data.reshape(-1, data.shape[3], data.shape[4])


def _reshape_5d(flat: np.ndarray, original_shape: tuple[int, ...]) -> np.ndarray:
    return flat.reshape(original_shape)


# ---------------------------------------------------------------------------
# Regular pipeline entry
# ---------------------------------------------------------------------------


def apply_transforms(
    transforms_per_alias: dict[str, list[dict]],
    dataset_cache: dict[str, np.ndarray],
    split_indices: dict[str, dict[str, np.ndarray]],
    fit_phase: str = "train",
) -> None:
    """Apply Regular-mode transforms in place on ``dataset_cache``.

    Walks step-by-step through the pipeline. Every alias present in
    ``transforms_per_alias`` must have an identical list (same length, same
    ``(name, scope)`` tuples per position — enforced by R29). For each step,
    dispatch to ``_apply_step_per_dataset`` or ``_apply_step_global``.
    """
    aliases = list(transforms_per_alias.keys())
    if not aliases:
        return
    reference = transforms_per_alias[aliases[0]]
    n_steps = len(reference)
    for i in range(n_steps):
        step_cfg = reference[i]
        scope = step_cfg.get("scope", "per_dataset")
        if scope == "per_dataset":
            _apply_step_per_dataset(
                step_cfg, dataset_cache, split_indices, fit_phase
            )
        elif scope == "global":
            _apply_step_global(
                step_cfg, dataset_cache, split_indices, fit_phase
            )
        else:
            raise ConfigError(
                f"apply_transforms: unsupported scope '{scope}' "
                "(should have been blocked by R22).",
            )


# ---------------------------------------------------------------------------
# UDA pipeline entry
# ---------------------------------------------------------------------------


def apply_transforms_uda(
    transforms_per_alias: dict[str, list[dict]],
    dataset_cache: dict[str, np.ndarray],
    uda_split: Any,  # UDASplitResult — typed loosely to avoid circular import
    adaptation: str,
) -> None:
    """Apply UDA-mode transforms in place. Always per_dataset (R23).

    Two physical layouts:
      - Cross-dataset UDA: each alias is purely source or purely target.
      - Intra-dataset UDA: a single alias hosts both domains — source and
        target portions must use independent statistics, so we fit/transform
        on specific index subsets rather than overwriting the whole array.
    """
    aliases = list(transforms_per_alias.keys())
    if not aliases:
        return
    reference = transforms_per_alias[aliases[0]]
    n_steps = len(reference)
    src_aliases = set(uda_split.source_train.keys())
    tgt_aliases = set(uda_split.target_train.keys())

    for i in range(n_steps):
        step_cfg = reference[i]
        scope = step_cfg.get("scope", "per_dataset")
        if scope != "per_dataset":
            raise ConfigError(
                f"apply_transforms_uda: scope='{scope}' not permitted in UDA "
                "(should have been blocked by R23).",
            )

        all_aliases = src_aliases | tgt_aliases
        for alias in all_aliases:
            is_source = alias in src_aliases
            is_target = alias in tgt_aliases

            if is_source and is_target:
                # Intra-dataset UDA — two independent fits on disjoint index pools
                _apply_uda_step_on_alias(
                    step_cfg,
                    dataset_cache,
                    alias,
                    fit_indices=uda_split.source_train[alias],
                    apply_indices=[
                        uda_split.source_train[alias],
                        uda_split.source_val.get(alias, np.array([], dtype=int)),
                    ],
                )
                tgt_fit = uda_split.target_train[alias]
                if adaptation == "inductive":
                    tgt_apply = [
                        uda_split.target_train[alias],
                        uda_split.target_val.get(alias, np.array([], dtype=int)),
                        uda_split.target_test.get(alias, np.array([], dtype=int)),
                    ]
                else:
                    # Transductive: target_train == target_test by design; dedupe.
                    tgt_apply = [uda_split.target_train[alias]]
                _apply_uda_step_on_alias(
                    step_cfg,
                    dataset_cache,
                    alias,
                    fit_indices=tgt_fit,
                    apply_indices=tgt_apply,
                )
            elif is_source:
                _apply_uda_step_on_alias(
                    step_cfg,
                    dataset_cache,
                    alias,
                    fit_indices=uda_split.source_train[alias],
                    apply_indices=[
                        uda_split.source_train[alias],
                        uda_split.source_val.get(alias, np.array([], dtype=int)),
                    ],
                )
            else:  # target-only
                tgt_fit = uda_split.target_train[alias]
                if adaptation == "inductive":
                    tgt_apply = [
                        uda_split.target_train[alias],
                        uda_split.target_val.get(alias, np.array([], dtype=int)),
                        uda_split.target_test.get(alias, np.array([], dtype=int)),
                    ]
                else:
                    tgt_apply = [uda_split.target_train[alias]]
                _apply_uda_step_on_alias(
                    step_cfg,
                    dataset_cache,
                    alias,
                    fit_indices=tgt_fit,
                    apply_indices=tgt_apply,
                )


def _apply_uda_step_on_alias(
    step_config: dict,
    dataset_cache: dict[str, np.ndarray],
    alias: str,
    fit_indices: np.ndarray,
    apply_indices: list[np.ndarray],
) -> None:
    """Fit on ``fit_indices`` and transform each index group in ``apply_indices``.

    The in-place edit writes each transformed slice back to the flattened
    5-D cache entry, so non-targeted samples are left untouched — this is
    what lets two independent fits (source + target) coexist on the same alias.
    """
    name = step_config["name"]
    params = step_config.get("params", {}) or {}
    transform = create_transform(name, **params)

    data_5d = dataset_cache[alias]
    original_shape = data_5d.shape
    # Copy off any read-only mmap backing so the later flat[unique] = ...
    # partial writes (independent source/target statistics) are allowed.
    flat = _flatten_5d(data_5d).copy()

    if len(fit_indices) == 0:
        # Fallback — fit on all rows that would be touched.
        non_empty = [idx for idx in apply_indices if len(idx) > 0]
        if non_empty:
            pool = np.unique(np.concatenate(non_empty))
        else:
            pool = np.arange(flat.shape[0])
        transform.fit(flat[pool])
    else:
        transform.fit(flat[fit_indices])

    for idx_group in apply_indices:
        if len(idx_group) == 0:
            continue
        unique = np.unique(idx_group)
        flat[unique] = transform.transform(flat[unique])

    dataset_cache[alias] = _reshape_5d(flat, original_shape)


# ---------------------------------------------------------------------------
# Step-level helpers
# ---------------------------------------------------------------------------


def _apply_step_per_dataset(
    step_config: dict,
    dataset_cache: dict[str, np.ndarray],
    split_indices: dict[str, dict[str, np.ndarray]],
    fit_phase: str,
) -> None:
    """Run one per-dataset transform step in place."""
    name = step_config["name"]
    params = step_config.get("params", {}) or {}

    for alias, phase_indices in split_indices.items():
        data_5d = dataset_cache[alias]
        original_shape = data_5d.shape
        flat = _flatten_5d(data_5d)

        transform = create_transform(name, **params)
        fit_idx = phase_indices.get(fit_phase)
        if fit_idx is None or len(fit_idx) == 0:
            # Test-only dataset (cross-dataset path) — fit on own full data.
            transform.fit(flat)
        else:
            transform.fit(flat[fit_idx])

        transformed = transform.transform(flat)
        dataset_cache[alias] = _reshape_5d(transformed, original_shape)


def _apply_step_global(
    step_config: dict,
    dataset_cache: dict[str, np.ndarray],
    split_indices: dict[str, dict[str, np.ndarray]],
    fit_phase: str,
) -> None:
    """Run one global transform step in place (Regular only)."""
    name = step_config["name"]
    params = step_config.get("params", {}) or {}
    transform = create_transform(name, **params)

    # Collect fit data from every alias that has ``fit_phase`` indices.
    fit_chunks: list[np.ndarray] = []
    for alias, phase_indices in split_indices.items():
        data_5d = dataset_cache[alias]
        flat = _flatten_5d(data_5d)
        fit_idx = phase_indices.get(fit_phase)
        if fit_idx is not None and len(fit_idx) > 0:
            fit_chunks.append(flat[fit_idx])
    if not fit_chunks:
        logger.warning(
            "apply_transforms(global): no fit data collected; skipping step '%s'.", name
        )
        return
    transform.fit(np.concatenate(fit_chunks, axis=0))

    # Transform every alias (including alias without fit_phase — e.g., cross-dataset test).
    for alias in split_indices.keys():
        data_5d = dataset_cache[alias]
        original_shape = data_5d.shape
        flat = _flatten_5d(data_5d)
        dataset_cache[alias] = _reshape_5d(transform.transform(flat), original_shape)
