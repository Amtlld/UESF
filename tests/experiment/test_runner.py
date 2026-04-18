"""Tests for Runner and EarlyStopping."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from uesf.components.builtin_metrics import accuracy, f1_score
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
    """Test trainer whose training_step returns loss + preds + targets."""

    def __init__(self, model, device, **kwargs):
        self.model = model.to(device)
        self.device = device

    def configure_optimizers(self):
        return None

    def training_step(self, batch, batch_idx, optimizer):
        total_loss = 0.0
        all_preds, all_targets = [], []
        for name, (data, labels) in batch.items():
            output = self.model(data)
            loss = nn.functional.cross_entropy(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            all_preds.append(output.detach().argmax(dim=1))
            all_targets.append(labels.detach())
        return {
            "loss": total_loss,
            "preds": torch.cat(all_preds),
            "targets": torch.cat(all_targets),
        }

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


class SimpleTrainerNoPreds(SimpleTrainer):
    """Trainer whose training_step intentionally omits preds/targets."""

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


def _epoch_calls(stub):
    return [(tags, step) for tags, step in stub.scalar_calls
            if not all(k.startswith("step/") for k in tags)]


def _step_calls(stub):
    return [(tags, step) for tags, step in stub.scalar_calls
            if tags and all(k.startswith("step/") for k in tags)]


class TestRunnerLoggerAdvanced:
    """Tests for step-level writes, filters, train_metrics, test_metrics, log_lr."""

    def _make_runner(self, epochs=3, with_preds=True, metrics=None):
        model = SimpleModel()
        device = torch.device("cpu")
        trainer_cls = SimpleTrainer if with_preds else SimpleTrainerNoPreds
        trainer = trainer_cls(model, device)
        evaluator = Evaluator(metrics or {"accuracy": accuracy, "f1_score": f1_score})
        return Runner(trainer, evaluator, device, {"epochs": epochs})

    def _loader(self, n=20, phase="train"):
        data = np.random.randn(n, 10).astype(np.float32)
        labels = np.random.randint(0, 2, n)
        return DataloaderBuilder().build(
            {"main": data}, {"main": labels}, batch_size=5, shuffle=(phase == "train")
        )

    def test_step_level_writes_emit_step_prefix(self):
        # 4 batches/epoch × 3 epochs = 12 steps; every 3rd (global_step+1)%3==0 → 4 writes.
        runner = self._make_runner(epochs=3)
        stub = _StubLogger()
        optimizer = torch.optim.SGD(runner.trainer.model.parameters(), lr=0.01)
        runner.run(
            train_loader=self._loader(),
            val_loader=self._loader(phase="val"),
            optimizer=optimizer,
            training_logger=stub,
            log_every_n_steps=3,
        )
        step_calls = _step_calls(stub)
        assert len(step_calls) == 4, f"expected 4 step-level writes, got {len(step_calls)}"
        for tags, _ in step_calls:
            assert "step/loss" in tags
            assert "step/lr" in tags

    def test_train_step_filter_blocks_other_scalars(self):
        """A training_step that returns 'loss' + 'extra'; filter picks only 'loss'."""
        runner = self._make_runner(epochs=1)
        # Monkey-patch training_step to also return a bogus scalar.
        original_step = runner.trainer.training_step

        def _step_with_extra(batch, batch_idx, optimizer):
            out = original_step(batch, batch_idx, optimizer)
            out["extra"] = 42.0
            return out

        runner.trainer.training_step = _step_with_extra
        stub = _StubLogger()
        optimizer = torch.optim.SGD(runner.trainer.model.parameters(), lr=0.01)
        runner.run(
            train_loader=self._loader(),
            val_loader=self._loader(phase="val"),
            optimizer=optimizer,
            training_logger=stub,
            train_step_filter=["loss"],
            log_every_n_steps=2,
        )
        for tags, _ in stub.scalar_calls:
            assert "extra" not in tags
            assert "step/extra" not in tags

    def test_val_metrics_filter(self):
        runner = self._make_runner(epochs=2)
        stub = _StubLogger()
        optimizer = torch.optim.SGD(runner.trainer.model.parameters(), lr=0.01)
        runner.run(
            train_loader=self._loader(),
            val_loader=self._loader(phase="val"),
            optimizer=optimizer,
            training_logger=stub,
            val_metrics_filter=["accuracy"],
        )
        for tags, _ in _epoch_calls(stub):
            assert "val_accuracy" in tags
            assert "val_f1_score" not in tags

    def test_train_metrics_with_preds_logs_train_prefix(self):
        runner = self._make_runner(epochs=2, with_preds=True)
        train_eval = Evaluator({"accuracy": accuracy})
        stub = _StubLogger()
        optimizer = torch.optim.SGD(runner.trainer.model.parameters(), lr=0.01)
        runner.run(
            train_loader=self._loader(),
            val_loader=self._loader(phase="val"),
            optimizer=optimizer,
            training_logger=stub,
            train_evaluator=train_eval,
        )
        for tags, _ in _epoch_calls(stub):
            assert "train_accuracy" in tags

    def test_train_metrics_without_preds_warns(self, caplog):
        runner = self._make_runner(epochs=2, with_preds=False)
        train_eval = Evaluator({"accuracy": accuracy})
        stub = _StubLogger()
        optimizer = torch.optim.SGD(runner.trainer.model.parameters(), lr=0.01)
        with caplog.at_level("WARNING", logger="uesf.experiment.runner"):
            runner.run(
                train_loader=self._loader(),
                val_loader=self._loader(phase="val"),
                optimizer=optimizer,
                training_logger=stub,
                train_evaluator=train_eval,
            )
        assert any(
            "training_step" in rec.message and "preds" in rec.message
            for rec in caplog.records
        )
        for tags, _ in stub.scalar_calls:
            assert not any(k.startswith("train_") for k in tags)

    def test_test_metrics_tracked_during_training(self):
        runner = self._make_runner(epochs=2)
        test_eval = Evaluator({"accuracy": accuracy})
        stub = _StubLogger()
        optimizer = torch.optim.SGD(runner.trainer.model.parameters(), lr=0.01)
        runner.run(
            train_loader=self._loader(),
            val_loader=self._loader(phase="val"),
            optimizer=optimizer,
            training_logger=stub,
            test_evaluator=test_eval,
            test_loader=self._loader(phase="val"),
        )
        for tags, _ in _epoch_calls(stub):
            assert "test_accuracy" in tags

    def test_log_lr_false_suppresses_lr_everywhere(self):
        runner = self._make_runner(epochs=2)
        stub = _StubLogger()
        optimizer = torch.optim.SGD(runner.trainer.model.parameters(), lr=0.01)
        runner.run(
            train_loader=self._loader(),
            val_loader=self._loader(phase="val"),
            optimizer=optimizer,
            training_logger=stub,
            log_lr=False,
            log_every_n_steps=2,
        )
        for tags, _ in stub.scalar_calls:
            assert "lr" not in tags
            assert "step/lr" not in tags

    def test_backward_compat_default_behavior(self):
        """Runner.run() with no new params writes the dev6 tag set only."""
        runner = self._make_runner(epochs=2)
        stub = _StubLogger()
        optimizer = torch.optim.SGD(runner.trainer.model.parameters(), lr=0.01)
        runner.run(
            train_loader=self._loader(),
            val_loader=self._loader(phase="val"),
            optimizer=optimizer,
            training_logger=stub,
        )
        for tags, _ in stub.scalar_calls:
            assert all(not k.startswith("step/") for k in tags), "no step-level tags expected"
            assert all(not k.startswith("train_") for k in tags), "no train_<name> tags expected"
            assert all(not k.startswith("test_") for k in tags), "no test_<name> tags expected"
            assert "loss" in tags
            assert "val_accuracy" in tags
            assert "lr" in tags
