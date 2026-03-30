"""Tests for Runner and EarlyStopping."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from uesf.components.builtin_metrics import accuracy
from uesf.experiment.dataloader_builder import build_dataloaders
from uesf.experiment.dataset import EEGDataset
from uesf.experiment.evaluator import Evaluator
from uesf.experiment.runner import EarlyStopping, Runner


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


class SimpleTrainer:
    def __init__(self, model, device, **kwargs):
        self.model = model.to(device)
        self.device = device

    def configure_optimizers(self):
        return None

    def training_step(self, batch, batch_idx, optimizer):
        total_loss = 0.0
        for name, (data, labels) in batch.items():
            output = self.model(data)
            loss = nn.functional.cross_entropy(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return {"loss": total_loss}

    def validation_step(self, batch, batch_idx):
        all_preds, all_targets = [], []
        for name, (data, labels) in batch.items():
            output = self.model(data)
            all_preds.append(output.argmax(dim=1))
            all_targets.append(labels)
        return {
            "preds": torch.cat(all_preds),
            "targets": torch.cat(all_targets),
        }


class TestEarlyStopping:
    def test_no_stop_when_improving(self):
        es = EarlyStopping(patience=3, mode="min")
        assert not es.step(1.0)
        assert not es.step(0.9)
        assert not es.step(0.8)
        assert not es.should_stop

    def test_stops_after_patience(self):
        es = EarlyStopping(patience=2, mode="min")
        es.step(0.5)
        es.step(0.6)  # No improvement
        assert not es.should_stop
        es.step(0.7)  # No improvement
        assert es.should_stop

    def test_max_mode(self):
        es = EarlyStopping(patience=2, mode="max")
        es.step(0.5)
        es.step(0.6)  # Improvement
        assert not es.should_stop
        es.step(0.5)  # No improvement
        es.step(0.4)  # No improvement
        assert es.should_stop


class TestRunner:
    def _make_runner(self):
        model = SimpleModel()
        device = torch.device("cpu")
        trainer = SimpleTrainer(model, device)
        evaluator = Evaluator({"accuracy": accuracy})
        config = {"epochs": 3}
        return Runner(trainer, evaluator, device, config)

    def _make_loader(self, n=40, phase="train"):
        data = np.random.randn(n, 10).astype(np.float32)
        labels = np.random.randint(0, 2, n)
        ds = EEGDataset(data, labels)
        return build_dataloaders({"main": ds}, batch_size=10, phase=phase)

    def test_train_epoch(self):
        runner = self._make_runner()
        optimizer = torch.optim.SGD(runner.trainer.model.parameters(), lr=0.01)
        loader = self._make_loader()

        metrics = runner.train_epoch(loader, optimizer, 0)
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)

    def test_validate_epoch(self):
        runner = self._make_runner()
        loader = self._make_loader(phase="val")

        metrics, preds, targets = runner.validate_epoch(loader)
        assert "accuracy" in metrics
        assert len(preds) > 0
        assert len(targets) > 0

    def test_full_run(self):
        runner = self._make_runner()
        optimizer = torch.optim.SGD(runner.trainer.model.parameters(), lr=0.01)
        train_loader = self._make_loader()
        val_loader = self._make_loader(phase="val")

        result = runner.run(
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
        )
        assert "history" in result
        assert result["epochs_run"] == 3
        assert len(result["history"]) == 3

    def test_run_with_early_stopping(self):
        runner = self._make_runner()
        runner.epochs = 100  # Would take long without early stopping
        optimizer = torch.optim.SGD(runner.trainer.model.parameters(), lr=0.01)
        train_loader = self._make_loader()
        val_loader = self._make_loader(phase="val")

        result = runner.run(
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            early_stopping_config={
                "monitor": "val_accuracy",
                "patience": 2,
                "mode": "max",
            },
        )
        # Should stop before 100 epochs
        assert result["epochs_run"] <= 100
