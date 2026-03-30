"""Tests for builtin optimizer and scheduler mappings."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from uesf.components.builtin_mappings import (
    OPTIMIZER_MAP,
    SCHEDULER_MAP,
    resolve_optimizer,
    resolve_scheduler,
)
from uesf.core.exceptions import ComponentNotFoundError


class TestOptimizerMap:
    def test_all_entries_present(self):
        expected = {"sgd", "adam", "adamw", "adagrad", "adadelta", "rmsprop", "radam", "nadam"}
        assert set(OPTIMIZER_MAP.keys()) == expected

    def test_resolve_adam(self):
        model = nn.Linear(10, 2)
        opt = resolve_optimizer("adam", model.parameters(), {"lr": 0.001})
        assert isinstance(opt, torch.optim.Adam)

    def test_resolve_sgd_with_momentum(self):
        model = nn.Linear(10, 2)
        opt = resolve_optimizer("sgd", model.parameters(), {"lr": 0.01, "momentum": 0.9})
        assert isinstance(opt, torch.optim.SGD)

    def test_resolve_adamw(self):
        model = nn.Linear(10, 2)
        opt = resolve_optimizer("adamw", model.parameters(), {"lr": 0.001, "weight_decay": 0.01})
        assert isinstance(opt, torch.optim.AdamW)

    def test_unknown_optimizer_raises(self):
        model = nn.Linear(10, 2)
        with pytest.raises(ComponentNotFoundError, match="Unknown optimizer"):
            resolve_optimizer("nonexistent", model.parameters(), {})

    def test_transparent_passthrough(self):
        """All kwargs are passed directly to PyTorch constructor."""
        model = nn.Linear(10, 2)
        opt = resolve_optimizer("adam", model.parameters(), {"lr": 0.01, "betas": (0.9, 0.99)})
        assert opt.defaults["lr"] == 0.01
        assert opt.defaults["betas"] == (0.9, 0.99)


class TestSchedulerMap:
    def test_all_entries_present(self):
        expected = {
            "step_lr", "multi_step_lr", "exponential_lr", "linear_lr",
            "cosine_annealing_lr", "cosine_annealing_warm_restarts",
            "reduce_lr_on_plateau", "one_cycle_lr",
        }
        assert set(SCHEDULER_MAP.keys()) == expected

    def test_resolve_step_lr(self):
        model = nn.Linear(10, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        sched = resolve_scheduler("step_lr", opt, {"step_size": 10})
        assert isinstance(sched, torch.optim.lr_scheduler.StepLR)

    def test_resolve_cosine_annealing(self):
        model = nn.Linear(10, 2)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        sched = resolve_scheduler("cosine_annealing_lr", opt, {"T_max": 50})
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_resolve_reduce_on_plateau(self):
        model = nn.Linear(10, 2)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        sched = resolve_scheduler("reduce_lr_on_plateau", opt, {"patience": 5})
        assert isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau)

    def test_unknown_scheduler_raises(self):
        model = nn.Linear(10, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        with pytest.raises(ComponentNotFoundError, match="Unknown scheduler"):
            resolve_scheduler("nonexistent", opt, {})

    def test_transparent_passthrough(self):
        model = nn.Linear(10, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        sched = resolve_scheduler("step_lr", opt, {"step_size": 5, "gamma": 0.5})
        assert sched.step_size == 5
        assert sched.gamma == 0.5
