"""Tests for online transforms."""

from __future__ import annotations

import numpy as np
import pytest

from uesf.experiment.transforms import ZScoreNormalize, create_transform


class TestZScoreNormalize:
    def test_fit_transform(self):
        data = np.random.randn(100, 32, 500)
        zscore = ZScoreNormalize()
        result = zscore.fit_transform(data)
        assert result.shape == data.shape
        # After z-score, mean should be ~0, std ~1
        np.testing.assert_allclose(result.mean(axis=0), 0.0, atol=1e-6)
        np.testing.assert_allclose(result.std(axis=0), 1.0, atol=0.05)

    def test_fit_on_train_apply_to_test(self):
        train = np.random.randn(80, 10) * 5 + 10
        test = np.random.randn(20, 10) * 5 + 10

        zscore = ZScoreNormalize()
        zscore.fit(train)

        train_normalized = zscore.transform(train)
        test_normalized = zscore.transform(test)

        # Train should be near zero mean
        np.testing.assert_allclose(train_normalized.mean(axis=0), 0.0, atol=1e-6)
        # Test should NOT be exactly zero mean (different distribution)
        assert test_normalized.shape == test.shape

    def test_transform_without_fit_raises(self):
        zscore = ZScoreNormalize()
        with pytest.raises(RuntimeError, match="fit"):
            zscore.transform(np.random.randn(10, 5))


class TestCreateTransform:
    def test_create_zscore(self):
        t = create_transform("zscore_normalize")
        assert isinstance(t, ZScoreNormalize)

    def test_unknown_transform(self):
        from uesf.core.exceptions import ComponentNotFoundError
        with pytest.raises(ComponentNotFoundError, match="Unknown transform"):
            create_transform("nonexistent")
