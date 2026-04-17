"""UESF Cross-Dataset Alignment - channel and label harmonization.

When experiments involve multiple datasets, this module ensures
channel and label spaces are compatible before training.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from uesf.core.exceptions import ConfigError, ShapeMismatchError
from uesf.core.logging import get_logger

logger = get_logger("experiment.alignment")


# ---------------------------------------------------------------------------
# Channel alignment
# ---------------------------------------------------------------------------


class ChannelAligner(ABC):
    """Base interface for channel alignment strategies."""

    @abstractmethod
    def align(
        self,
        datasets: dict[str, tuple[np.ndarray, list[str]]],
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """Align channel dimensions across datasets.

        Args:
            datasets: alias -> (data_array, electrode_list).
                      Data can be 5-D ``[sub, sess, rec, ch, samp]``
                      or 3-D ``[n_total, ch, samp]``.

        Returns:
            (aligned_data_map, common_electrode_list) where
            aligned_data_map is alias -> aligned data array.
        """


class IntersectionAligner(ChannelAligner):
    """Keep only channels common to all datasets.

    The order of the resulting channels follows the first dataset's
    electrode list.
    """

    def align(
        self,
        datasets: dict[str, tuple[np.ndarray, list[str]]],
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        if not datasets:
            raise ConfigError("No datasets provided for channel alignment.")

        aliases = list(datasets.keys())
        first_alias = aliases[0]
        first_electrodes = datasets[first_alias][1]

        # Compute intersection (preserving order of the first dataset)
        common = set(first_electrodes)
        for alias in aliases[1:]:
            common &= set(datasets[alias][1])

        if not common:
            involved = {a: el for a, (_, el) in datasets.items()}
            raise ConfigError(
                "Channel intersection is empty — no channels are shared "
                "across all datasets.",
                context={"electrode_lists": involved},
                hint="Check that your datasets use compatible electrode montages.",
            )

        # Preserve order from the first dataset
        common_electrodes = [ch for ch in first_electrodes if ch in common]

        logger.info(
            "Channel intersection: %d channels (from %s)",
            len(common_electrodes),
            ", ".join(f"{a}={len(datasets[a][1])}" for a in aliases),
        )

        aligned: dict[str, np.ndarray] = {}
        for alias, (data, electrodes) in datasets.items():
            idx = [electrodes.index(ch) for ch in common_electrodes]
            # Channel axis is always the second-to-last axis
            aligned[alias] = np.take(data, idx, axis=-2)

        return aligned, common_electrodes


CHANNEL_ALIGNER_REGISTRY: dict[str, type[ChannelAligner]] = {
    "intersection": IntersectionAligner,
}


def create_channel_aligner(method: str, **kwargs: Any) -> ChannelAligner:
    """Create a channel aligner by method name."""
    if method not in CHANNEL_ALIGNER_REGISTRY:
        raise ConfigError(
            f"Unknown channel alignment method: '{method}'",
            hint=f"Available methods: {', '.join(sorted(CHANNEL_ALIGNER_REGISTRY))}",
        )
    return CHANNEL_ALIGNER_REGISTRY[method](**kwargs)


# ---------------------------------------------------------------------------
# Label alignment
# ---------------------------------------------------------------------------


class LabelAligner:
    """Validates label-space consistency across datasets.

    Checks that all datasets have:
    - The same number of classes
    - Consistent numeric-to-semantic label mappings (when available)
    """

    def validate(self, datasets_meta: dict[str, dict]) -> None:
        """Validate label consistency across datasets.

        Args:
            datasets_meta: alias -> meta dict, each containing
                ``n_classes`` and optionally ``numeric_to_semantic``.

        Raises:
            ConfigError: If label spaces are inconsistent.
        """
        if len(datasets_meta) < 2:
            return

        aliases = list(datasets_meta.keys())
        ref_alias = aliases[0]
        ref_meta = datasets_meta[ref_alias]
        ref_n_classes = ref_meta["n_classes"]

        # Check n_classes consistency
        for alias in aliases[1:]:
            n = datasets_meta[alias]["n_classes"]
            if n != ref_n_classes:
                raise ConfigError(
                    f"Label mismatch: '{ref_alias}' has {ref_n_classes} classes "
                    f"but '{alias}' has {n} classes.",
                    hint="Ensure all datasets have the same number of classes, "
                    "or use label masking to align them.",
                )

        # Check numeric_to_semantic consistency (if available)
        ref_mapping = ref_meta.get("numeric_to_semantic")
        if ref_mapping is None:
            return

        for alias in aliases[1:]:
            mapping = datasets_meta[alias].get("numeric_to_semantic")
            if mapping is None:
                continue
            if mapping != ref_mapping:
                raise ConfigError(
                    f"Label mapping mismatch between '{ref_alias}' and '{alias}'.",
                    context={
                        ref_alias: ref_mapping,
                        alias: mapping,
                    },
                    hint="Ensure numeric_to_semantic mappings are identical "
                    "across datasets (e.g., {0: 'left', 1: 'right'}).",
                )

        logger.info("Label alignment validated: %d classes across %d datasets",
                     ref_n_classes, len(aliases))


# ---------------------------------------------------------------------------
# n_samples / sampling-rate consistency (dev6 addition)
# ---------------------------------------------------------------------------


def check_sample_consistency(datasets_meta: dict[str, dict]) -> None:
    """Verify cross-dataset ``n_samples`` alignment; warn on sampling rate drift.

    Args:
        datasets_meta: alias -> meta dict carrying at least ``n_samples``
            and optionally ``sampling_rate`` / ``sample_rate``.

    Raises:
        ShapeMismatchError: if any two datasets disagree on ``n_samples``.
    """
    if len(datasets_meta) < 2:
        return
    aliases = list(datasets_meta.keys())
    ref_alias = aliases[0]
    ref_n = datasets_meta[ref_alias].get("n_samples")
    if ref_n is None:
        return
    for alias in aliases[1:]:
        n = datasets_meta[alias].get("n_samples")
        if n is None:
            continue
        if n != ref_n:
            raise ShapeMismatchError(
                f"n_samples mismatch: '{ref_alias}'={ref_n} vs '{alias}'={n}",
                hint="Ensure all datasets share the same window length in preprocessing.",
            )

    # Sampling rate drift → warning only
    def _rate(meta: dict) -> Any:
        return meta.get("sampling_rate") or meta.get("sample_rate")

    ref_rate = _rate(datasets_meta[ref_alias])
    if ref_rate is None:
        return
    for alias in aliases[1:]:
        rate = _rate(datasets_meta[alias])
        if rate is not None and rate != ref_rate:
            logger.warning(
                "Sampling rate drift: '%s'=%s Hz vs '%s'=%s Hz — window length may differ.",
                ref_alias, ref_rate, alias, rate,
            )
