"""UESF BaseTrainer - base class for all trainers.

Trainers own the entire optimization loop: forward pass, loss computation,
gradient management, and optimizer stepping. The Runner only delegates
batch dicts and collects logging output.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Optional

import torch
import torch.nn as nn


class BaseTrainer:
    """Base class for all UESF trainers."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        **kwargs,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.config = kwargs

    def configure_optimizers(
        self,
    ) -> Optional[tuple[torch.optim.Optimizer, Any]]:
        """Override to provide custom optimizer and scheduler.

        Returns:
            None to use experiment YAML config, or
            (optimizer, scheduler) tuple for custom optimization.
        """
        return None

    @abstractmethod
    def training_step(
        self,
        batch: dict[str, tuple[torch.Tensor, torch.Tensor]],
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, Any]:
        """Execute one training step.

        The trainer is FULLY responsible for:
        1. Forward pass + loss computation
        2. optimizer.zero_grad()
        3. loss.backward()
        4. optimizer.step()

        Args:
            batch: Multi-channel dict {channel_name: (data, labels)}.
            batch_idx: Current batch index.
            optimizer: The optimizer instance.

        Returns:
            Dict with loggable scalar values (detached tensors or floats).
        """
        raise NotImplementedError

    @abstractmethod
    def validation_step(
        self,
        batch: dict[str, tuple[torch.Tensor, torch.Tensor]],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        """Execute one validation/test step.

        Args:
            batch: Multi-channel dict {channel_name: (data, labels)}.
            batch_idx: Current batch index.

        Returns:
            Dict with "preds" and "targets" tensors.
        """
        raise NotImplementedError
