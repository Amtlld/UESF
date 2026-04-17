"""Tests for TrainingLogger protocol + TensorBoardLogger + create_logger."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from uesf.core.exceptions import ConfigError
from uesf.experiment.logger import (
    TensorBoardLogger,
    TrainingLogger,
    create_logger,
)


@pytest.fixture
def fake_tensorboard(monkeypatch):
    """Install a fake torch.utils.tensorboard.SummaryWriter."""
    fake_mod = ModuleType("torch.utils.tensorboard")
    writer = MagicMock()
    fake_mod.SummaryWriter = MagicMock(return_value=writer)
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", fake_mod)
    return writer


class TestTensorBoardLogger:
    def test_log_scalars_forwards_each_tag(self, tmp_path, fake_tensorboard):
        logger = TensorBoardLogger(tmp_path)
        logger.log_scalars({"loss": 0.5, "val_acc": 0.8}, step=3)
        calls = fake_tensorboard.add_scalar.call_args_list
        assert len(calls) == 2
        tags = {c.args[0] for c in calls}
        assert tags == {"loss", "val_acc"}

    def test_context_manager_flush_close_on_exit(self, tmp_path, fake_tensorboard):
        with TensorBoardLogger(tmp_path) as logger:
            logger.log_scalars({"a": 1.0}, step=0)
        fake_tensorboard.flush.assert_called_once()
        fake_tensorboard.close.assert_called_once()

    def test_context_manager_closes_on_exception(self, tmp_path, fake_tensorboard):
        with pytest.raises(RuntimeError):
            with TensorBoardLogger(tmp_path):
                raise RuntimeError("boom")
        fake_tensorboard.close.assert_called_once()

    def test_log_graph_calls_writer(self, tmp_path, fake_tensorboard):
        import torch

        logger = TensorBoardLogger(tmp_path)
        model = torch.nn.Linear(4, 2)
        inp = torch.zeros(1, 4)
        logger.log_graph(model, inp)
        fake_tensorboard.add_graph.assert_called_once_with(model, inp)


class TestCreateLogger:
    def test_returns_none_when_empty(self, tmp_path):
        assert create_logger({}, tmp_path) is None
        assert create_logger({"backend": None}, tmp_path) is None

    def test_returns_tensorboard_instance(self, tmp_path, fake_tensorboard):
        logger = create_logger({"backend": "tensorboard"}, tmp_path)
        assert isinstance(logger, TensorBoardLogger)

    def test_unknown_backend_raises(self, tmp_path):
        with pytest.raises(ConfigError, match="Unsupported logging backend"):
            create_logger({"backend": "wandb"}, tmp_path)

    def test_import_error_wrapped(self, tmp_path, monkeypatch):
        def raise_import_error(*args, **kwargs):
            raise ImportError("no tensorboard")

        # Patch the init path so the lazy import throws.
        import uesf.experiment.logger as logger_module

        class _BadLogger:
            def __init__(self, *a, **k):
                raise_import_error()

        monkeypatch.setattr(logger_module, "TensorBoardLogger", _BadLogger)
        with pytest.raises(ConfigError, match="TensorBoard"):
            create_logger({"backend": "tensorboard"}, tmp_path)


class TestProtocolConformance:
    def test_tensorboard_conforms_to_protocol(self, tmp_path, fake_tensorboard):
        logger = TensorBoardLogger(tmp_path)
        # runtime_checkable Protocol requires presence of methods
        assert isinstance(logger, TrainingLogger)
