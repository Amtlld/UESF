"""UESF splitter subpackage.

Public surface:

- Result dataclasses: :class:`SplitResult`, :class:`DatasetLevelSplitResult`,
  :class:`MultiDatasetSplitResult`, :class:`DomainSplitResult`,
  :class:`UDASplitResult`.
- Regular splitters: :class:`HoldoutSplitter`, :class:`KFoldSplitter`,
  :class:`DatasetLevelSplitter`.
- UDA splitters: :class:`ValSplitter`, :class:`DatasetDomainSplitter`,
  :class:`DimensionDomainSplitter`, :class:`UDAOrchestrator`.
- Factories: :func:`create_splitter`, :func:`create_uda_orchestrator`.
- Grouping: :func:`get_groups`.
"""

from __future__ import annotations

from uesf.core.exceptions import TypeMismatchError
from uesf.experiment.splitter.base import (
    DatasetLevelSplitResult,
    DomainSplitResult,
    MultiDatasetSplitResult,
    SplitResult,
    UDASplitResult,
)
from uesf.experiment.splitter.grouping import get_groups
from uesf.experiment.splitter.regular import (
    DatasetLevelSplitter,
    HoldoutSplitter,
    KFoldSplitter,
)
from uesf.experiment.splitter.uda import (
    DatasetDomainSplitter,
    DimensionDomainSplitter,
    UDAOrchestrator,
    ValSplitter,
)

__all__ = [
    "SplitResult",
    "DatasetLevelSplitResult",
    "MultiDatasetSplitResult",
    "DomainSplitResult",
    "UDASplitResult",
    "HoldoutSplitter",
    "KFoldSplitter",
    "DatasetLevelSplitter",
    "ValSplitter",
    "DatasetDomainSplitter",
    "DimensionDomainSplitter",
    "UDAOrchestrator",
    "create_splitter",
    "create_uda_orchestrator",
    "get_groups",
]


def create_splitter(config: dict, seed: int = 42) -> HoldoutSplitter | KFoldSplitter:
    """Create a Regular index-level splitter (dimension != dataset).

    Args:
        config: Regular split block (already normalized). Must contain
            ``strategy`` and ``dimension``. For ``dimension=dataset``, this
            factory is NOT appropriate — callers should construct
            :class:`DatasetLevelSplitter` directly.
        seed: Random seed for the splitter.

    Returns:
        :class:`HoldoutSplitter` or :class:`KFoldSplitter`.
    """
    strategy = config["strategy"]
    dimension = config["dimension"]
    if dimension == "dataset":
        raise TypeMismatchError(
            "create_splitter does not handle dimension='dataset' — "
            "construct DatasetLevelSplitter directly.",
        )
    shuffle = config.get("shuffle", True)
    val_split_config = config.get("val_split")

    if strategy == "holdout":
        return HoldoutSplitter(
            dimension=dimension,
            train_ratio=config["train_ratio"],
            test_ratio=config["test_ratio"],
            val_ratio=config.get("val_ratio", 0.0),
            val_split_config=val_split_config,
            shuffle=shuffle,
            seed=seed,
        )
    if strategy == "k-fold":
        return KFoldSplitter(
            dimension=dimension,
            k=config["k"],
            val_ratio=config.get("val_ratio_in_train", config.get("val_ratio", 0.0)),
            val_split_config=val_split_config,
            shuffle=shuffle,
            seed=seed,
        )
    raise TypeMismatchError(
        f"create_splitter: unknown strategy '{strategy}'",
        hint="Use 'holdout' or 'k-fold'.",
    )


def create_uda_orchestrator(uda_config: dict, seed: int = 42) -> UDAOrchestrator:
    """Create a :class:`UDAOrchestrator` from a normalized UDA config."""
    return UDAOrchestrator(uda_config, seed=seed)
