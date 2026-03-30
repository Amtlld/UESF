"""UESF Online Transforms - fit-on-train, apply-to-all.

These transforms run AFTER splitting to prevent data leakage.
Statistics are computed from the train split only.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from uesf.core.logging import get_logger

logger = get_logger("experiment.transforms")


class ZScoreNormalize:
    """Z-Score normalization: fit on train, apply to all splits.

    Computes mean and std from the train split, then applies
    (x - mean) / std to all splits.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.eps = kwargs.get("eps", 1e-8)

    def fit(self, data: np.ndarray) -> None:
        """Compute mean and std from training data.

        Args:
            data: Training data array, shape (n_samples, ...).
        """
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        logger.debug("ZScoreNormalize fit: mean shape=%s, std shape=%s", self.mean.shape, self.std.shape)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply the fitted normalization.

        Args:
            data: Data array with same trailing dimensions as fit data.

        Returns:
            Normalized data.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("ZScoreNormalize.fit() must be called before transform()")
        return (data - self.mean) / (self.std + self.eps)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(data)
        return self.transform(data)


TRANSFORM_REGISTRY: dict[str, type] = {
    "zscore_normalize": ZScoreNormalize,
}


def create_transform(name: str, **kwargs: Any) -> ZScoreNormalize:
    """Create a transform instance by name."""
    from uesf.core.exceptions import ComponentNotFoundError

    if name not in TRANSFORM_REGISTRY:
        raise ComponentNotFoundError(
            f"Unknown transform: '{name}'",
            context={"available": sorted(TRANSFORM_REGISTRY.keys())},
            hint=f"Available transforms: {', '.join(sorted(TRANSFORM_REGISTRY.keys()))}",
        )
    return TRANSFORM_REGISTRY[name](**kwargs)
