"""UESF DummyModel and DummyTrainer for testing and EMBEDDED defaults."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from uesf.components.base_model import BaseModel
from uesf.components.base_trainer import BaseTrainer


class DummyModel(BaseModel):
    """Simple fully-connected model for testing."""

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_classes: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(n_channels, n_samples, n_classes, **kwargs)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(n_channels * n_samples, n_classes)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.fc(self.flatten(x))


class DummyTrainer(BaseTrainer):
    """Simple cross-entropy trainer for testing."""

    def training_step(
        self,
        batch: dict[str, tuple[torch.Tensor, torch.Tensor]],
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, Any]:
        total_loss = 0.0
        n = 0
        for channel_name, (data, labels) in batch.items():
            data = data.to(self.device)
            labels = labels.to(self.device)
            output = self.model(data)
            loss = nn.functional.cross_entropy(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n += 1
        return {"loss": total_loss / max(n, 1)}

    def validation_step(
        self,
        batch: dict[str, tuple[torch.Tensor, torch.Tensor]],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        all_preds = []
        all_targets = []
        for channel_name, (data, labels) in batch.items():
            data = data.to(self.device)
            output = self.model(data)
            all_preds.append(output.argmax(dim=1))
            all_targets.append(labels)
        return {
            "preds": torch.cat(all_preds),
            "targets": torch.cat(all_targets),
        }
