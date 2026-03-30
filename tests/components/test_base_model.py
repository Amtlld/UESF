"""Tests for BaseModel interface."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from uesf.components.base_model import BaseModel


class ConcreteModel(BaseModel):
    """Minimal concrete implementation for testing."""

    def __init__(self, n_channels, n_samples, n_classes, **kwargs):
        super().__init__(n_channels, n_samples, n_classes, **kwargs)
        self.fc = nn.Linear(n_channels * n_samples, n_classes)

    def forward(self, x, **kwargs):
        batch = x.shape[0]
        return self.fc(x.reshape(batch, -1))


class TestBaseModelInterface:
    def test_stores_dimensions(self):
        model = ConcreteModel(32, 500, 4)
        assert model.n_channels == 32
        assert model.n_samples == 500
        assert model.n_classes == 4

    def test_electrode_list_default_none(self):
        model = ConcreteModel(32, 500, 4)
        assert model.electrode_list is None

    def test_electrode_list_passed(self):
        electrodes = ["Fp1", "Fp2", "C3", "C4"]
        model = ConcreteModel(4, 500, 2, electrode_list=electrodes)
        assert model.electrode_list == electrodes

    def test_extra_kwargs_ignored(self):
        model = ConcreteModel(32, 500, 4, dropout=0.5, hidden_size=128)
        assert model.n_channels == 32

    def test_is_nn_module(self):
        model = ConcreteModel(32, 500, 4)
        assert isinstance(model, nn.Module)

    def test_forward_pass(self):
        model = ConcreteModel(32, 500, 4)
        x = torch.randn(8, 32, 500)
        out = model(x)
        assert out.shape == (8, 4)

    def test_extract_features_raises_by_default(self):
        model = ConcreteModel(32, 500, 4)
        with pytest.raises(NotImplementedError):
            model.extract_features(torch.randn(1, 32, 500))

    def test_forward_not_implemented_raises(self):
        """BaseModel.forward raises NotImplementedError if not overridden."""

        class IncompleteModel(BaseModel):
            pass  # Does not override forward()

        model = IncompleteModel(32, 500, 4)
        with pytest.raises(NotImplementedError):
            model(torch.randn(1, 32, 500))
