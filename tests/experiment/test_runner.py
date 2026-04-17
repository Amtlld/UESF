"""Tests for Runner and EarlyStopping."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from uesf.components.builtin_metrics import accuracy
from uesf.experiment.dataloader_builder import DataloaderBuilder
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
        builder = DataloaderBuilder()
        return builder.build(
            {"main": data}, {"main": labels}, batch_size=10, shuffle=(phase == "train"),
        )

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


class _StubLogger:
    """Minimal TrainingLogger-like stub that records calls."""

    def __init__(self, fail_on_epoch: int | None = None) -> None:
        self.scalar_calls: list[tuple[dict, int]] = []
        self.closed = False
        self.fail_on_epoch = fail_on_epoch
        self.entered = False

    def log_scalars(self, tag_value_dict, step):
        if self.fail_on_epoch is not None and step == self.fail_on_epoch:
            raise RuntimeError("stub failure")
        self.scalar_calls.append((dict(tag_value_dict), step))

    def log_graph(self, model, input_sample):
        pass

    def close(self):
        self.closed = True

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class TestRunnerLogger:
    def _make_runner_and_loaders(self, epochs=4):
        model = SimpleModel()
        device = torch.device("cpu")
        trainer = SimpleTrainer(model, device)
        evaluator = Evaluator({"accuracy": accuracy})
        runner = Runner(trainer, evaluator, device, {"epochs": epochs})

        def _loader(phase):
            data = np.random.randn(20, 10).astype(np.float32)
            labels = np.random.randint(0, 2, 20)
            return DataloaderBuilder().build(
                {"main": data}, {"main": labels}, batch_size=5, shuffle=(phase == "train")
            )

        return runner, _loader

    def test_logger_invoked_each_epoch_by_default(self):
        runner, make_loader = self._make_runner_and_loaders(epochs=3)
        logger = _StubLogger()
        optimizer = torch.optim.SGD(runner.trainer.model.parameters(), lr=0.01)
        runner.run(
            train_loader=make_loader("train"),
            val_loader=make_loader("val"),
            optimizer=optimizer,
            training_logger=logger,
        )
        assert logger.entered and logger.closed
        assert len(logger.scalar_calls) == 3
        # Each call should include loss, val_accuracy, lr
        tags, _ = logger.scalar_calls[0]
        assert "loss" in tags and "lr" in tags

    def test_log_every_n_epochs_respected(self):
        runner, make_loader = self._make_runner_and_loaders(epochs=5)
        logger = _StubLogger()
        optimizer = torch.optim.SGD(runner.trainer.model.parameters(), lr=0.01)
        runner.run(
            train_loader=make_loader("train"),
            val_loader=make_loader("val"),
            optimizer=optimizer,
            training_logger=logger,
            log_every_n_epochs=2,
        )
        # Only epochs 1 and 3 (0-indexed +1) trigger: (1+1)%2==0, (3+1)%2==0, (5)%2==... wait
        # Rule: (epoch + 1) % n == 0. epochs 0..4 → triggers at epochs 1, 3 → 2 calls.
        assert len(logger.scalar_calls) == 2

    def test_logger_close_on_exception(self):
        runner, make_loader = self._make_runner_and_loaders(epochs=3)
        logger = _StubLogger(fail_on_epoch=1)
        optimizer = torch.optim.SGD(runner.trainer.model.parameters(), lr=0.01)
        with pytest.raises(RuntimeError, match="stub failure"):
            runner.run(
                train_loader=make_loader("train"),
                val_loader=make_loader("val"),
                optimizer=optimizer,
                training_logger=logger,
            )
        assert logger.closed, "logger.close() must run even if training raises"

    def test_no_logger_has_no_effect(self):
        runner, make_loader = self._make_runner_and_loaders(epochs=2)
        optimizer = torch.optim.SGD(runner.trainer.model.parameters(), lr=0.01)
        # Sanity: passing None works identically to omitting.
        result = runner.run(
            train_loader=make_loader("train"),
            val_loader=make_loader("val"),
            optimizer=optimizer,
            training_logger=None,
        )
        assert result["epochs_run"] == 2
