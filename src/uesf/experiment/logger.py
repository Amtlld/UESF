"""Training logger protocol + TensorBoard implementation.

Designed so that :class:`Runner` depends only on :class:`TrainingLogger` —
backends plug in via :func:`create_logger`. SummaryWriter is imported lazily
so environments without tensorboard remain lightweight.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import torch

from uesf.core.exceptions import ConfigError


@runtime_checkable
class TrainingLogger(Protocol):
    """Protocol for training-time loggers (context-managed)."""

    def log_scalars(self, tag_value_dict: dict[str, float], step: int) -> None:
        ...

    def log_graph(self, model: torch.nn.Module, input_sample: torch.Tensor) -> None:
        ...

    def close(self) -> None:
        ...

    def __enter__(self) -> "TrainingLogger":
        ...

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        ...


class TensorBoardLogger:
    """``torch.utils.tensorboard.SummaryWriter``-backed logger."""

    def __init__(self, log_dir: Path) -> None:
        from torch.utils.tensorboard import SummaryWriter  # noqa: PLC0415

        self.writer = SummaryWriter(log_dir=str(log_dir))

    def log_scalars(self, tag_value_dict: dict[str, float], step: int) -> None:
        for tag, value in tag_value_dict.items():
            self.writer.add_scalar(tag, value, step)

    def log_graph(self, model: torch.nn.Module, input_sample: torch.Tensor) -> None:
        self.writer.add_graph(model, input_sample)

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()

    def __enter__(self) -> "TensorBoardLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def create_logger(config: dict, log_dir: Path) -> TrainingLogger | None:
    """Create a :class:`TrainingLogger` from a ``training.logging`` block.

    Returns ``None`` when the block is empty / backend not specified —
    callers should gracefully omit all logging in that case.
    """
    if not config:
        return None
    backend = config.get("backend")
    if backend is None:
        return None
    if backend == "tensorboard":
        try:
            return TensorBoardLogger(log_dir)
        except ImportError as exc:
            raise ConfigError(
                "TensorBoard is not installed.",
                hint="Install with: uv pip install uesf[tensorboard]",
            ) from exc
    raise ConfigError(
        f"Unsupported logging backend: {backend}",
        hint="Currently only 'tensorboard' is supported.",
    )
