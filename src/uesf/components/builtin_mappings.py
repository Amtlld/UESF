"""Built-in optimizer and scheduler string-to-PyTorch mappings.

Implements Transparent Passthrough: all params from YAML are
directly unpacked to PyTorch constructors via **kwargs.
"""

from __future__ import annotations

from typing import Any

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from uesf.core.exceptions import ComponentNotFoundError

OPTIMIZER_MAP: dict[str, type] = {
    "sgd": optim.SGD,
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "adagrad": optim.Adagrad,
    "adadelta": optim.Adadelta,
    "rmsprop": optim.RMSprop,
    "radam": optim.RAdam,
    "nadam": optim.NAdam,
}

SCHEDULER_MAP: dict[str, type] = {
    "step_lr": lr_scheduler.StepLR,
    "multi_step_lr": lr_scheduler.MultiStepLR,
    "exponential_lr": lr_scheduler.ExponentialLR,
    "linear_lr": lr_scheduler.LinearLR,
    "cosine_annealing_lr": lr_scheduler.CosineAnnealingLR,
    "cosine_annealing_warm_restarts": lr_scheduler.CosineAnnealingWarmRestarts,
    "reduce_lr_on_plateau": lr_scheduler.ReduceLROnPlateau,
    "one_cycle_lr": lr_scheduler.OneCycleLR,
}


def resolve_optimizer(
    name: str,
    model_params: Any,
    opt_params: dict,
) -> optim.Optimizer:
    """Resolve optimizer name to a PyTorch optimizer instance.

    Args:
        name: Optimizer name (e.g., "adam", "adamw").
        model_params: model.parameters() iterable.
        opt_params: Optimizer kwargs from YAML (e.g., {"lr": 0.001}).

    Returns:
        Configured optimizer instance.

    Raises:
        ComponentNotFoundError: If name is not in OPTIMIZER_MAP.
    """
    if name not in OPTIMIZER_MAP:
        raise ComponentNotFoundError(
            f"Unknown optimizer: '{name}'",
            context={"available": sorted(OPTIMIZER_MAP.keys())},
            hint=f"Available optimizers: {', '.join(sorted(OPTIMIZER_MAP.keys()))}",
        )
    return OPTIMIZER_MAP[name](model_params, **opt_params)


def resolve_scheduler(
    name: str,
    optimizer: optim.Optimizer,
    sched_params: dict,
) -> Any:
    """Resolve scheduler name to a PyTorch scheduler instance.

    Args:
        name: Scheduler name (e.g., "cosine_annealing_lr").
        optimizer: The optimizer instance.
        sched_params: Scheduler kwargs from YAML.

    Returns:
        Configured scheduler instance.

    Raises:
        ComponentNotFoundError: If name is not in SCHEDULER_MAP.
    """
    if name not in SCHEDULER_MAP:
        raise ComponentNotFoundError(
            f"Unknown scheduler: '{name}'",
            context={"available": sorted(SCHEDULER_MAP.keys())},
            hint=f"Available schedulers: {', '.join(sorted(SCHEDULER_MAP.keys()))}",
        )
    return SCHEDULER_MAP[name](optimizer, **sched_params)
