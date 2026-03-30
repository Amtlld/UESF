"""Tests for BaseTrainer interface."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from uesf.components.base_trainer import BaseTrainer


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


class ConcreteTrainer(BaseTrainer):
    """Minimal concrete implementation for testing."""

    def training_step(self, batch, batch_idx, optimizer):
        for channel_name, (data, labels) in batch.items():
            data = data.to(self.device)
            labels = labels.to(self.device)
            output = self.model(data)
            loss = nn.functional.cross_entropy(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return {"loss": loss.item()}

    def validation_step(self, batch, batch_idx):
        all_preds, all_targets = [], []
        for channel_name, (data, labels) in batch.items():
            data = data.to(self.device)
            output = self.model(data)
            all_preds.append(output.argmax(dim=1))
            all_targets.append(labels)
        return {
            "preds": torch.cat(all_preds),
            "targets": torch.cat(all_targets),
        }


class TestBaseTrainerInterface:
    def test_model_moved_to_device(self):
        model = DummyModel()
        device = torch.device("cpu")
        trainer = ConcreteTrainer(model, device)
        assert trainer.device == device

    def test_config_stored(self):
        model = DummyModel()
        trainer = ConcreteTrainer(model, torch.device("cpu"), lr=0.001, epochs=10)
        assert trainer.config["lr"] == 0.001
        assert trainer.config["epochs"] == 10

    def test_configure_optimizers_returns_none_by_default(self):
        model = DummyModel()
        trainer = ConcreteTrainer(model, torch.device("cpu"))
        assert trainer.configure_optimizers() is None

    def test_training_step(self):
        model = DummyModel()
        device = torch.device("cpu")
        trainer = ConcreteTrainer(model, device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        batch = {"main": (torch.randn(4, 10), torch.randint(0, 2, (4,)))}
        result = trainer.training_step(batch, 0, optimizer)
        assert "loss" in result
        assert isinstance(result["loss"], float)

    def test_validation_step(self):
        model = DummyModel()
        device = torch.device("cpu")
        trainer = ConcreteTrainer(model, device)

        batch = {"main": (torch.randn(4, 10), torch.randint(0, 2, (4,)))}
        result = trainer.validation_step(batch, 0)
        assert "preds" in result
        assert "targets" in result
        assert result["preds"].shape == (4,)
        assert result["targets"].shape == (4,)

    def test_training_step_not_implemented_raises(self):
        """BaseTrainer.training_step raises NotImplementedError if not overridden."""

        class IncompleteTrainer(BaseTrainer):
            def validation_step(self, batch, batch_idx):
                return {}

        trainer = IncompleteTrainer(DummyModel(), torch.device("cpu"))
        with pytest.raises(NotImplementedError):
            trainer.training_step({}, 0, None)

    def test_custom_configure_optimizers(self):
        class CustomTrainer(ConcreteTrainer):
            def configure_optimizers(self):
                opt = torch.optim.Adam(self.model.parameters(), lr=0.001)
                return opt, None

        model = DummyModel()
        trainer = CustomTrainer(model, torch.device("cpu"))
        result = trainer.configure_optimizers()
        assert result is not None
        opt, sched = result
        assert isinstance(opt, torch.optim.Adam)
        assert sched is None
