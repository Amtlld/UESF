"""Preprocessing operator registry."""

from __future__ import annotations

from typing import Callable

from uesf.core.exceptions import ComponentNotFoundError
from uesf.pipeline.operators.data_ops import bandpass_filter, notch_filter, reference, resample
from uesf.pipeline.operators.joint_ops import epoch_normalize, sliding_window
from uesf.pipeline.operators.label_ops import smooth

# Registry: name -> (stream_type, callable)
# stream_type: "data", "label", or "joint"
OPERATOR_REGISTRY: dict[str, tuple[str, Callable]] = {
    "resample": ("data", resample),
    "filter": ("data", bandpass_filter),
    "notch_filter": ("data", notch_filter),
    "reference": ("data", reference),
    "smooth": ("label", smooth),
    "sliding_window": ("joint", sliding_window),
    "epoch_normalize": ("joint", epoch_normalize),
}


def get_operator(name: str) -> tuple[str, Callable]:
    """Look up an operator by name.

    Returns:
        (stream_type, callable)

    Raises:
        ComponentNotFoundError: If operator name is not registered.
    """
    if name not in OPERATOR_REGISTRY:
        raise ComponentNotFoundError(
            f"Unknown preprocessing operator: '{name}'",
            context={"available": sorted(OPERATOR_REGISTRY.keys())},
            hint=f"Available operators: {', '.join(sorted(OPERATOR_REGISTRY.keys()))}",
        )
    return OPERATOR_REGISTRY[name]
