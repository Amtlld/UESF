"""Tests for dataset splitters."""

from __future__ import annotations

import numpy as np
import pytest

from uesf.experiment.splitter import (
    HoldoutSplitter,
    KFoldSplitter,
    _get_groups,
    create_splitter,
)


class TestHoldoutSplitter:
    def test_basic_split(self):
        data = np.random.randn(100, 32, 500)  # 100 samples
        splitter = HoldoutSplitter({
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "dimension": "none",
            "shuffle": False,
        })
        results = splitter.split(data)
        assert len(results) == 1

        r = results[0]
        total = len(r.train_indices) + len(r.val_indices) + len(r.test_indices)
        assert total == 100
        assert len(r.train_indices) == 70
        assert len(r.val_indices) == 15
        assert len(r.test_indices) == 15

    def test_no_overlap(self):
        data = np.random.randn(50, 32, 500)
        splitter = HoldoutSplitter({"dimension": "none", "seed": 42})
        r = splitter.split(data)[0]

        all_indices = set(r.train_indices) | set(r.val_indices) | set(r.test_indices)
        assert len(all_indices) == 50

    def test_subject_dimension(self):
        # 10 subjects, 5 recordings each
        data = np.random.randn(10, 5, 32, 500)
        splitter = HoldoutSplitter({
            "train_ratio": 0.6,
            "val_ratio": 0.2,
            "test_ratio": 0.2,
            "dimension": "subject",
            "shuffle": False,
        })
        r = splitter.split(data)[0]
        # Each subject has 5 samples
        total = len(r.train_indices) + len(r.val_indices) + len(r.test_indices)
        assert total == 50  # 10 subjects * 5 recordings

    def test_deterministic_with_seed(self):
        data = np.random.randn(20, 32, 500)
        cfg = {"dimension": "none", "seed": 123}
        r1 = HoldoutSplitter(cfg).split(data)[0]
        r2 = HoldoutSplitter(cfg).split(data)[0]
        np.testing.assert_array_equal(r1.train_indices, r2.train_indices)


class TestKFoldSplitter:
    def test_basic_kfold(self):
        data = np.random.randn(100, 32, 500)
        splitter = KFoldSplitter({
            "k-folds": 5,
            "dimension": "none",
            "shuffle": False,
        })
        results = splitter.split(data)
        assert len(results) == 5

        # Each sample should appear in exactly one test set
        all_test = np.concatenate([r.test_indices for r in results])
        assert len(all_test) == 100
        assert len(set(all_test)) == 100

    def test_loocv(self):
        data = np.random.randn(10, 32, 500)
        splitter = KFoldSplitter({
            "k-folds": -1,
            "dimension": "none",
            "shuffle": False,
        })
        results = splitter.split(data)
        assert len(results) == 10
        for r in results:
            assert len(r.test_indices) == 1
            assert len(r.train_indices) == 9

    def test_loocv_with_total(self):
        data = np.random.randn(5, 32, 500)
        splitter = KFoldSplitter({
            "k-folds": "total",
            "dimension": "none",
            "shuffle": False,
        })
        results = splitter.split(data)
        assert len(results) == 5

    def test_subject_isolation(self):
        # 6 subjects, each with 3 recordings
        data = np.random.randn(6, 3, 32, 500)
        splitter = KFoldSplitter({
            "k-folds": 3,
            "dimension": "subject",
            "shuffle": False,
        })
        results = splitter.split(data)
        assert len(results) == 3

        # Check that subjects are not split across folds
        for r in results:
            test_subjects = set(idx // 3 for idx in r.test_indices)
            train_subjects = set(idx // 3 for idx in r.train_indices)
            assert test_subjects.isdisjoint(train_subjects)

    def test_val_ratio_in_train(self):
        data = np.random.randn(20, 32, 500)
        splitter = KFoldSplitter({
            "k-folds": 4,
            "dimension": "none",
            "shuffle": False,
            "val_ratio_in_train": 0.2,
        })
        results = splitter.split(data)
        for r in results:
            assert len(r.val_indices) > 0
            total = len(r.train_indices) + len(r.val_indices) + len(r.test_indices)
            assert total == 20


class TestGetGroups:
    def test_none_dimension(self):
        data = np.random.randn(10, 32, 500)
        groups = _get_groups(data, "none")
        assert len(groups) == 10
        for g in groups:
            assert len(g) == 1

    def test_subject_dimension(self):
        data = np.random.randn(5, 3, 32, 500)
        groups = _get_groups(data, "subject")
        assert len(groups) == 5
        for g in groups:
            assert len(g) == 3


class TestCreateSplitter:
    def test_holdout(self):
        s = create_splitter({"strategy": "holdout"})
        assert isinstance(s, HoldoutSplitter)

    def test_kfold(self):
        s = create_splitter({"strategy": "k-fold"})
        assert isinstance(s, KFoldSplitter)

    def test_unknown(self):
        from uesf.core.exceptions import ConfigError
        with pytest.raises(ConfigError, match="Unknown"):
            create_splitter({"strategy": "unknown"})
