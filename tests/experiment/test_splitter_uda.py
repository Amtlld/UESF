"""Tests for UDA splitters (ValSplitter / Domain splitters / UDAOrchestrator)."""

from __future__ import annotations

import numpy as np
import pytest

from uesf.core.exceptions import SplitError, TypeMismatchError
from uesf.experiment.splitter import (
    DatasetDomainSplitter,
    DimensionDomainSplitter,
    UDAOrchestrator,
    ValSplitter,
    create_uda_orchestrator,
)


def _make_data(sub=4, sess=1, rec=3, ch=3, samp=8):
    return np.empty((sub, sess, rec, ch, samp))


# ---------------------------------------------------------------------------
# ValSplitter
# ---------------------------------------------------------------------------


class TestValSplitter:
    def test_basic_split_subject_dim(self):
        data = _make_data(sub=5, sess=1, rec=4)  # 5 subject groups × 4 rec = 20 samples
        sp = ValSplitter(dimension="subject", val_ratio=0.2, shuffle=False)
        result = sp.split(data)
        assert len(result.train_indices) + len(result.val_indices) == 20
        assert len(result.test_indices) == 0
        # val has exactly round(5 * 0.2) = 1 subject group = 4 samples
        assert len(result.val_indices) == 4

    def test_val_ratio_zero_returns_all_train(self):
        data = _make_data(sub=4, sess=1, rec=2)
        result = ValSplitter(dimension="subject", val_ratio=0.0).split(data)
        assert len(result.train_indices) == 8
        assert len(result.val_indices) == 0

    def test_flatten_requires_shuffle_true(self):
        with pytest.raises(TypeMismatchError, match="flatten"):
            ValSplitter(dimension="flatten", val_ratio=0.2, shuffle=False)

    def test_flatten_with_flat_indices(self):
        flat = np.arange(10, dtype=int)
        sp = ValSplitter(dimension="flatten", val_ratio=0.3, shuffle=True, seed=0)
        result = sp.split(data=np.empty((1, 1, 1, 1, 1)), flat_indices=flat)
        assert len(result.train_indices) + len(result.val_indices) == 10
        assert len(set(result.train_indices.tolist()) | set(result.val_indices.tolist())) == 10

    def test_val_ratio_range_validated(self):
        with pytest.raises(TypeMismatchError, match="val_ratio"):
            ValSplitter(dimension="subject", val_ratio=1.0)

    def test_deterministic(self):
        data = _make_data(sub=5, sess=1, rec=3)
        cfg = dict(dimension="subject", val_ratio=0.2, shuffle=True, seed=42)
        r1 = ValSplitter(**cfg).split(data)
        r2 = ValSplitter(**cfg).split(data)
        np.testing.assert_array_equal(r1.val_indices, r2.val_indices)


# ---------------------------------------------------------------------------
# DatasetDomainSplitter
# ---------------------------------------------------------------------------


class TestDatasetDomainSplitter:
    def test_holdout_explicit(self):
        sp = DatasetDomainSplitter(
            strategy="holdout", source=["a", "b"], target="c"
        )
        (r,) = sp.split(["a", "b", "c"])
        assert list(r.source_indices.keys()) == ["a", "b"]
        assert list(r.target_indices.keys()) == ["c"]
        assert r.fold_info["target_alias"] == "c"

    def test_holdout_unknown_alias(self):
        sp = DatasetDomainSplitter(strategy="holdout", source=["a"], target="nope")
        with pytest.raises(TypeMismatchError, match="unknown aliases"):
            sp.split(["a", "b"])

    def test_kfold_loocv(self):
        sp = DatasetDomainSplitter(strategy="k-fold", k=-1, shuffle=False)
        folds = sp.split(["a", "b", "c"])
        assert len(folds) == 3
        targets = [list(f.target_indices.keys())[0] for f in folds]
        assert sorted(targets) == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# DimensionDomainSplitter
# ---------------------------------------------------------------------------


class TestDimensionDomainSplitter:
    def test_holdout_target_count(self):
        data = _make_data(sub=5, sess=1, rec=4)
        sp = DimensionDomainSplitter(
            strategy="holdout", dimension="subject", target_count=2, shuffle=False
        )
        (r,) = sp.split(data, alias="main")
        # target = first 2 subjects × 4 rec
        assert len(r.target_indices["main"]) == 8
        assert len(r.source_indices["main"]) == 12

    def test_holdout_target_ratio(self):
        data = _make_data(sub=10, sess=1, rec=1)
        sp = DimensionDomainSplitter(
            strategy="holdout", dimension="subject", target_ratio=0.3, shuffle=False
        )
        (r,) = sp.split(data, alias="main")
        assert len(r.target_indices["main"]) == 3

    def test_holdout_rejects_target_count_and_ratio_both(self):
        with pytest.raises(TypeMismatchError, match="mutually exclusive"):
            DimensionDomainSplitter(
                strategy="holdout",
                dimension="subject",
                target_count=1,
                target_ratio=0.2,
            )

    def test_kfold_loocv(self):
        data = _make_data(sub=4, sess=1, rec=3)
        folds = DimensionDomainSplitter(
            strategy="k-fold", dimension="subject", k=-1, shuffle=False
        ).split(data, alias="ds")
        assert len(folds) == 4
        for f in folds:
            assert len(f.target_indices["ds"]) == 3
            assert len(f.source_indices["ds"]) == 9

    def test_subject_isolation(self):
        data = _make_data(sub=6, sess=1, rec=2)
        (r,) = DimensionDomainSplitter(
            strategy="holdout",
            dimension="subject",
            target_count=2,
            shuffle=False,
        ).split(data, alias="m")
        src = set(r.source_indices["m"].tolist())
        tgt = set(r.target_indices["m"].tolist())
        assert not src & tgt

    def test_target_size_bounds(self):
        data = _make_data(sub=3, sess=1, rec=1)
        sp = DimensionDomainSplitter(
            strategy="holdout", dimension="subject", target_count=3, shuffle=False
        )
        with pytest.raises(SplitError, match="must be in"):
            sp.split(data, alias="x")


# ---------------------------------------------------------------------------
# UDAOrchestrator — transductive (cross-dataset + intra)
# ---------------------------------------------------------------------------


def _cache(*specs) -> dict[str, np.ndarray]:
    c: dict[str, np.ndarray] = {}
    for alias, sub, sess, rec in specs:
        c[alias] = np.empty((sub, sess, rec, 3, 8))
    return c


class TestUDAOrchestratorTransductive:
    def test_cross_dataset_transductive_holdout_no_source_split(self):
        cache = _cache(("a", 2, 1, 3), ("b", 3, 1, 2))
        cfg = {
            "domain": {
                "strategy": "holdout",
                "dimension": "dataset",
                "source": ["a"],
                "target": "b",
            },
            "adaptation": "transductive",
        }
        orch = UDAOrchestrator(cfg, seed=0)
        folds = orch.split(cache)
        assert len(folds) == 1
        f = folds[0]
        # source: full a (2*1*3 = 6), no val
        assert len(f.source_train["a"]) == 6
        assert len(f.source_val["a"]) == 0
        # target: full b, transductive copy
        assert len(f.target_train["b"]) == 6
        assert len(f.target_test["b"]) == 6
        np.testing.assert_array_equal(f.target_train["b"], f.target_test["b"])
        assert len(f.target_val["b"]) == 0
        # distinct copies (not aliasing)
        assert f.target_train["b"] is not f.target_test["b"]

    def test_cross_dataset_transductive_holdout_with_source_split(self):
        cache = _cache(("a", 5, 1, 2), ("b", 2, 1, 2))
        cfg = {
            "domain": {
                "strategy": "holdout",
                "dimension": "dataset",
                "source": ["a"],
                "target": "b",
            },
            "adaptation": "transductive",
            "source": {
                "split": {"dimension": "subject", "val_ratio": 0.2, "shuffle": False}
            },
        }
        orch = UDAOrchestrator(cfg, seed=0)
        (f,) = orch.split(cache)
        total_src = len(f.source_train["a"]) + len(f.source_val["a"])
        assert total_src == 10
        assert len(f.source_val["a"]) > 0

    def test_cross_dataset_transductive_kfold(self):
        cache = _cache(("a", 2, 1, 2), ("b", 2, 1, 2), ("c", 2, 1, 2))
        cfg = {
            "domain": {
                "strategy": "k-fold",
                "dimension": "dataset",
                "k": -1,
                "shuffle": False,
            },
            "adaptation": "transductive",
        }
        folds = UDAOrchestrator(cfg, seed=0).split(cache)
        assert len(folds) == 3
        # each dataset is target exactly once
        tgts = sorted(list(f.target_train.keys())[0] for f in folds)
        assert tgts == ["a", "b", "c"]

    def test_intra_dataset_transductive_subject(self):
        cache = _cache(("main", 6, 1, 2))
        cfg = {
            "domain": {
                "strategy": "holdout",
                "dimension": "subject",
                "target_count": 2,
                "shuffle": False,
            },
            "adaptation": "transductive",
        }
        (f,) = UDAOrchestrator(cfg, seed=0).split(cache)
        # domain fold: target = first 2 subjects × 2 rec = 4 rows
        assert len(f.target_train["main"]) == 4
        np.testing.assert_array_equal(f.target_train["main"], f.target_test["main"])
        # source: remaining 4 subjects × 2 rec = 8 rows
        assert len(f.source_train["main"]) == 8


# ---------------------------------------------------------------------------
# UDAOrchestrator — inductive (incl. nested fold flattening)
# ---------------------------------------------------------------------------


class TestUDAOrchestratorInductive:
    def test_cross_dataset_inductive_holdout(self):
        cache = _cache(("a", 2, 1, 4), ("b", 2, 1, 10))
        cfg = {
            "domain": {
                "strategy": "holdout",
                "dimension": "dataset",
                "source": ["a"],
                "target": "b",
            },
            "adaptation": "inductive",
            "target": {
                "split": {
                    "strategy": "holdout",
                    "dimension": "recording",
                    "train_ratio": 0.7,
                    "test_ratio": 0.3,
                    "shuffle": False,
                }
            },
        }
        (f,) = UDAOrchestrator(cfg, seed=0).split(cache)
        n_total = 2 * 1 * 10  # 20
        assert len(f.target_train["b"]) + len(f.target_test["b"]) == n_total
        assert len(f.target_val["b"]) == 0
        assert len(f.target_test["b"]) > 0
        # source is full a
        assert len(f.source_train["a"]) == 8

    def test_cross_dataset_inductive_nested_kfold_flattened(self):
        cache = _cache(("a", 2, 1, 4), ("b", 2, 1, 4), ("c", 2, 1, 6))
        cfg = {
            "domain": {
                "strategy": "k-fold",
                "dimension": "dataset",
                "k": -1,
                "shuffle": False,
            },
            "adaptation": "inductive",
            "target": {
                "split": {
                    "strategy": "k-fold",
                    "dimension": "recording",
                    "k": 2,
                    "shuffle": False,
                }
            },
        }
        folds = UDAOrchestrator(cfg, seed=0).split(cache)
        # 3 domain folds × 2 target inner folds = 6
        assert len(folds) == 6
        # fold_info carries both indices
        for f in folds:
            assert "domain_fold" in f.fold_info
            assert "inner_fold" in f.fold_info
        # source split is shared across inner folds for same domain fold
        by_domain: dict[int, list] = {}
        for f in folds:
            by_domain.setdefault(f.fold_info["domain_fold"], []).append(f)
        for group in by_domain.values():
            assert len(group) == 2
            ref = group[0].source_train
            for other in group[1:]:
                for alias in ref:
                    np.testing.assert_array_equal(ref[alias], other.source_train[alias])

    def test_intra_dataset_inductive_holdout(self):
        cache = _cache(("main", 5, 1, 4))  # 5 subjects × 4 recs
        cfg = {
            "domain": {
                "strategy": "holdout",
                "dimension": "subject",
                "target_count": 1,
                "shuffle": False,
            },
            "adaptation": "inductive",
            "target": {
                "split": {
                    "strategy": "holdout",
                    "dimension": "recording",
                    "train_ratio": 0.75,
                    "test_ratio": 0.25,
                    "shuffle": False,
                }
            },
        }
        (f,) = UDAOrchestrator(cfg, seed=0).split(cache)
        # target = first subject (4 rows)
        total_tgt = (
            len(f.target_train["main"])
            + len(f.target_val["main"])
            + len(f.target_test["main"])
        )
        assert total_tgt == 4
        # source = remaining 4 subjects × 4 rec = 16 rows
        assert len(f.source_train["main"]) == 16

    def test_source_block_omitted(self):
        cache = _cache(("a", 3, 1, 2), ("b", 3, 1, 2))
        cfg = {
            "domain": {
                "strategy": "holdout",
                "dimension": "dataset",
                "source": ["a"],
                "target": "b",
            },
            "adaptation": "transductive",
        }
        (f,) = UDAOrchestrator(cfg, seed=0).split(cache)
        assert len(f.source_val["a"]) == 0  # no val by default
        assert len(f.source_train["a"]) == 6


# ---------------------------------------------------------------------------
# create_uda_orchestrator
# ---------------------------------------------------------------------------


class TestCreateUDAOrchestrator:
    def test_returns_orchestrator(self):
        orch = create_uda_orchestrator(
            {
                "domain": {
                    "strategy": "holdout",
                    "dimension": "dataset",
                    "source": ["a"],
                    "target": "b",
                },
                "adaptation": "transductive",
            }
        )
        assert isinstance(orch, UDAOrchestrator)
