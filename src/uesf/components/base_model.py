"""UESF BaseModel - base class for all EEG models.

All custom models must inherit from BaseModel and implement forward().
Dataset metadata (n_channels, n_samples, n_classes) is auto-injected
by the framework at instantiation time.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """Base class for all UESF models."""

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_classes: int,
        electrode_list: Optional[list[str]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.electrode_list = electrode_list

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor, typically (batch, channels, samples).

        Returns:
            Output tensor, typically (batch, n_classes) logits.
        """
        raise NotImplementedError

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Optional: extract intermediate features.

        Override in subclasses for feature extraction tasks.
        Default implementation raises NotImplementedError.
        """
        raise NotImplementedError("extract_features not implemented for this model")
