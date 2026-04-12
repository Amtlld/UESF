"""Tests for UDA splitters and DatasetLevelSplitter."""

from __future__ import annotations

import numpy as np
import pytest

from uesf.core.exceptions import ConfigError
from uesf.experiment.splitter import (
    CrossDatasetUDASplitter,
    DatasetLevelSplitter,
    IntraDatasetUDASplitter,
    UDASplitResult,
    create_splitter,
)


def _make_data(sub=4, sess=2, rec=5, chan=32, samp=500):
    return np.empty((sub, sess, rec, chan, samp))


def _flat_total(data):
    return data.shape[0] * data.shape[1] * data.shape[2]


# -----------------------------------------------------------------------
# UDASplitResult
# -----------------------------------------------------------------------


class TestUDASplitResult:
    def test_defaults(self):
        r = UDASplitResult(source_train_indices={"a": np.array([0, 1])})
        assert len(r.source_val_indices["a"]) == 0
        assert len(r.target_train_indices) == 0
        assert len(r.target_test_indices) == 0

    def test_full(self):
        r = UDASplitResult(
            source_train_indices={"a": np.array([0, 1])},
            source_val_indices={"a": np.array([2])},
            target_train_indices={"a": np.array([3, 4])},
            target_val_indices={"a": np.array([5])},
            target_test_indices={"a": np.array([6])},
        )
        assert len(r.source_train_indices["a"]) == 2
        assert len(r.target_val_indices["a"]) == 1


# -----------------------------------------------------------------------
# IntraDatasetUDASplitter
# -----------------------------------------------------------------------


class TestIntraDatasetUDASplitter:
    # --- Transductive + Holdout ---

    def test_transductive_holdout_basic(self):
        data = _make_data(sub=5, sess=1, rec=4)  # 20 units, 5 groups
        splitter = IntraDatasetUDASplitter({
            "dimension": "subject",
            "strategy": "holdout",
            "variant": "transductive",
            "target_count": 1,
            "seed": 42,
        })
        results = splitter.split(data, alias="ds")
        assert len(results) == 1

        r = results[0]
        src = r.source_train_indices["ds"]
        tgt_train = r.target_train_indices["ds"]
        tgt_test = r.target_test_indices["ds"]

        # 4 source subjects * 4 recordings = 16
        assert len(src) == 16
        # 1 target subject * 4 recordings = 4
        assert len(tgt_train) == 4
        # Transductive: test == train
        np.testing.assert_array_equal(tgt_train, tgt_test)
        # No overlap
        assert len(set(src) & set(tgt_train)) == 0

    def test_transductive_holdout_target_ratio(self):
        data = _make_data(sub=10, sess=1, rec=1)  # 10 groups
        splitter = IntraDatasetUDASplitter({
            "dimension": "subject",
            "strategy": "holdout",
            "variant": "transductive",
            "target_ratio": 0.3,
            "seed": 0,
        })
        results = splitter.split(data)
        r = results[0]
        # 3 target subjects
        assert len(r.target_train_indices["main"]) == 3

    def test_transductive_holdout_source_val(self):
        data = _make_data(sub=10, sess=1, rec=2)  # 20 units
        splitter = IntraDatasetUDASplitter({
            "dimension": "subject",
            "strategy": "holdout",
            "variant": "transductive",
            "target_count": 2,
            "source_split": {"val_ratio": 0.2},
            "seed": 42,
        })
        results = splitter.split(data)
        r = results[0]
        assert len(r.source_val_indices["main"]) > 0
        total_src = len(r.source_train_indices["main"]) + len(r.source_val_indices["main"])
        assert total_src == 16  # 8 subjects * 2 recordings

    # --- Transductive + K-fold ---

    def test_transductive_kfold_loocv(self):
        data = _make_data(sub=5, sess=1, rec=3)  # 15 units, 5 groups
        splitter = IntraDatasetUDASplitter({
            "dimension": "subject",
            "strategy": "k-fold",
            "k-folds": -1,
            "variant": "transductive",
            "seed": 42,
        })
        results = splitter.split(data)
        assert len(results) == 5

        all_target = []
        for r in results:
            tgt = r.target_train_indices["main"]
            assert len(tgt) == 3  # 1 subject * 3 recordings
            np.testing.assert_array_equal(tgt, r.target_test_indices["main"])
            all_target.extend(tgt.tolist())

        # Every recording appears as target exactly once
        assert sorted(all_target) == list(range(15))

    def test_transductive_kfold_k3(self):
        data = _make_data(sub=6, sess=1, rec=1)
        splitter = IntraDatasetUDASplitter({
            "dimension": "subject",
            "strategy": "k-fold",
            "k-folds": 3,
            "variant": "transductive",
            "seed": 42,
        })
        results = splitter.split(data)
        assert len(results) == 3
        for r in results:
            assert len(r.target_train_indices["main"]) == 2
            assert len(r.source_train_indices["main"]) == 4

    # --- Inductive + Holdout ---

    def test_inductive_holdout(self):
        data = _make_data(sub=5, sess=1, rec=10)  # 50 units
        splitter = IntraDatasetUDASplitter({
            "dimension": "subject",
            "strategy": "holdout",
            "variant": "inductive",
            "target_count": 1,
            "target_split": {
                "dimension": "recording",
                "train_ratio": 0.7,
                "test_ratio": 0.3,
            },
            "seed": 42,
        })
        results = splitter.split(data)
        assert len(results) == 1

        r = results[0]
        tgt_train = r.target_train_indices["main"]
        tgt_test = r.target_test_indices["main"]

        # Target has 10 recordings: ~7 train, ~3 test
        assert len(tgt_train) + len(tgt_test) == 10
        assert len(tgt_train) >= 1
        assert len(tgt_test) >= 1
        # No overlap between target train and test
        assert len(set(tgt_train) & set(tgt_test)) == 0

    def test_inductive_holdout_with_target_val(self):
        data = _make_data(sub=5, sess=1, rec=10)
        splitter = IntraDatasetUDASplitter({
            "dimension": "subject",
            "strategy": "holdout",
            "variant": "inductive",
            "target_count": 1,
            "target_split": {
                "train_ratio": 0.6,
                "val_ratio": 0.2,
                "test_ratio": 0.2,
            },
            "seed": 42,
        })
        results = splitter.split(data)
        r = results[0]

        tgt_train = r.target_train_indices["main"]
        tgt_val = r.target_val_indices["main"]
        tgt_test = r.target_test_indices["main"]

        assert len(tgt_train) + len(tgt_val) + len(tgt_test) == 10
        assert len(tgt_val) >= 1

    # --- Inductive + K-fold ---

    def test_inductive_kfold_loocv(self):
        data = _make_data(sub=4, sess=1, rec=10)  # 40 units
        splitter = IntraDatasetUDASplitter({
            "dimension": "subject",
            "strategy": "k-fold",
            "k-folds": -1,
            "variant": "inductive",
            "target_split": {
                "train_ratio": 0.7,
                "test_ratio": 0.3,
            },
            "seed": 42,
        })
        results = splitter.split(data)
        assert len(results) == 4

        for r in results:
            tgt_train = r.target_train_indices["main"]
            tgt_test = r.target_test_indices["main"]
            # Each subject has 10 recordings split into ~7 train + ~3 test
            assert len(tgt_train) + len(tgt_test) == 10
            assert len(set(tgt_train) & set(tgt_test)) == 0

    # --- Subject isolation ---

    def test_subject_isolation(self):
        """Source and target indices should not share subjects."""
        n_sub, n_sess, n_rec = 8, 1, 5
        data = _make_data(sub=n_sub, sess=n_sess, rec=n_rec)
        splitter = IntraDatasetUDASplitter({
            "dimension": "subject",
            "strategy": "holdout",
            "variant": "transductive",
            "target_count": 2,
            "shuffle": False,
        })
        r = splitter.split(data)[0]

        src_subjects = {idx // n_rec for idx in r.source_train_indices["main"]}
        tgt_subjects = {idx // n_rec for idx in r.target_train_indices["main"]}
        assert len(src_subjects & tgt_subjects) == 0

    # --- Session dimension ---

    def test_session_dimension(self):
        data = _make_data(sub=3, sess=4, rec=2)  # 12 groups (3*4)
        splitter = IntraDatasetUDASplitter({
            "dimension": "session",
            "strategy": "holdout",
            "variant": "transductive",
            "target_count": 2,
            "seed": 42,
        })
        r = splitter.split(data)[0]
        total = len(r.source_train_indices["main"]) + len(r.target_train_indices["main"])
        assert total == 24  # 3*4*2

    # --- Error cases ---

    def test_target_count_too_large(self):
        data = _make_data(sub=3, sess=1, rec=1)
        splitter = IntraDatasetUDASplitter({
            "dimension": "subject",
            "strategy": "holdout",
            "target_count": 3,
        })
        with pytest.raises(ConfigError, match="target size"):
            splitter.split(data)

    def test_deterministic(self):
        data = _make_data(sub=6, sess=1, rec=3)
        cfg = {
            "dimension": "subject",
            "strategy": "holdout",
            "variant": "transductive",
            "target_count": 2,
            "seed": 123,
        }
        r1 = IntraDatasetUDASplitter(cfg).split(data)[0]
        r2 = IntraDatasetUDASplitter(cfg).split(data)[0]
        np.testing.assert_array_equal(
            r1.source_train_indices["main"],
            r2.source_train_indices["main"],
        )


# -----------------------------------------------------------------------
# CrossDatasetUDASplitter
# -----------------------------------------------------------------------


def _make_cache(*specs):
    """Create a fake dataset_cache. Each spec is (alias, n_sub, n_sess, n_rec)."""
    cache = {}
    for alias, ns, nse, nr in specs:
        cache[alias] = {
            "data": np.empty((ns, nse, nr, 32, 100)),
            "labels": np.zeros(ns * nse * nr),
            "meta": {"n_channels": 32, "n_samples": 100, "n_classes": 2},
        }
    return cache


class TestCrossDatasetUDASplitter:
    # --- Transductive + Holdout ---

    def test_transductive_holdout(self):
        cache = _make_cache(("src", 3, 1, 5), ("tgt", 2, 1, 5))
        splitter = CrossDatasetUDASplitter({
            "strategy": "holdout",
            "variant": "transductive",
            "source_datasets": ["src"],
            "target_dataset": "tgt",
        })
        results = splitter.split(cache)
        assert len(results) == 1

        r = results[0]
        assert len(r.source_train_indices["src"]) == 15  # 3*1*5
        assert len(r.target_train_indices["tgt"]) == 10  # 2*1*5
        np.testing.assert_array_equal(
            r.target_train_indices["tgt"],
            r.target_test_indices["tgt"],
        )

    def test_transductive_holdout_multi_source(self):
        cache = _make_cache(("s1", 2, 1, 3), ("s2", 3, 1, 2), ("tgt", 1, 1, 5))
        splitter = CrossDatasetUDASplitter({
            "strategy": "holdout",
            "variant": "transductive",
            "source_datasets": ["s1", "s2"],
            "target_dataset": "tgt",
        })
        r = splitter.split(cache)[0]
        assert "s1" in r.source_train_indices
        assert "s2" in r.source_train_indices
        assert len(r.source_train_indices["s1"]) == 6
        assert len(r.source_train_indices["s2"]) == 6

    def test_transductive_holdout_source_val(self):
        cache = _make_cache(("src", 5, 1, 10), ("tgt", 2, 1, 5))
        splitter = CrossDatasetUDASplitter({
            "strategy": "holdout",
            "variant": "transductive",
            "source_datasets": ["src"],
            "target_dataset": "tgt",
            "source_split": {"val_ratio": 0.2},
        })
        r = splitter.split(cache)[0]
        assert len(r.source_val_indices["src"]) > 0
        total = len(r.source_train_indices["src"]) + len(r.source_val_indices["src"])
        assert total == 50

    # --- Transductive + K-fold ---

    def test_transductive_kfold(self):
        cache = _make_cache(("a", 2, 1, 3), ("b", 2, 1, 3), ("c", 2, 1, 3))
        splitter = CrossDatasetUDASplitter({
            "strategy": "k-fold",
            "k-folds": -1,
            "variant": "transductive",
            "seed": 42,
        })
        results = splitter.split(cache)
        assert len(results) == 3

        # Each dataset appears as target exactly once
        target_aliases = set()
        for r in results:
            ta = list(r.target_train_indices.keys())
            assert len(ta) == 1
            target_aliases.add(ta[0])
        assert target_aliases == {"a", "b", "c"}

    # --- Inductive + Holdout ---

    def test_inductive_holdout(self):
        cache = _make_cache(("src", 3, 1, 5), ("tgt", 2, 1, 10))
        splitter = CrossDatasetUDASplitter({
            "strategy": "holdout",
            "variant": "inductive",
            "source_datasets": ["src"],
            "target_dataset": "tgt",
            "target_split": {
                "dimension": "recording",
                "train_ratio": 0.7,
                "test_ratio": 0.3,
            },
            "seed": 42,
        })
        r = splitter.split(cache)[0]

        tgt_train = r.target_train_indices["tgt"]
        tgt_test = r.target_test_indices["tgt"]
        assert len(tgt_train) + len(tgt_test) == 20  # 2*1*10
        assert len(tgt_test) >= 1
        assert len(set(tgt_train) & set(tgt_test)) == 0

    def test_inductive_holdout_target_val(self):
        cache = _make_cache(("src", 2, 1, 5), ("tgt", 2, 1, 20))
        splitter = CrossDatasetUDASplitter({
            "strategy": "holdout",
            "variant": "inductive",
            "source_datasets": ["src"],
            "target_dataset": "tgt",
            "target_split": {
                "train_ratio": 0.6,
                "val_ratio": 0.2,
                "test_ratio": 0.2,
            },
            "seed": 42,
        })
        r = splitter.split(cache)[0]
        assert len(r.target_val_indices["tgt"]) > 0

    # --- Inductive + K-fold ---

    def test_inductive_kfold(self):
        cache = _make_cache(("a", 2, 1, 5), ("b", 2, 1, 5), ("c", 2, 1, 5))
        splitter = CrossDatasetUDASplitter({
            "strategy": "k-fold",
            "k-folds": -1,
            "variant": "inductive",
            "target_split": {
                "train_ratio": 0.7,
                "test_ratio": 0.3,
            },
            "seed": 42,
        })
        results = splitter.split(cache)
        assert len(results) == 3
        for r in results:
            for alias in r.target_train_indices:
                tgt_total = (
                    len(r.target_train_indices[alias])
                    + len(r.target_test_indices[alias])
                )
                assert tgt_total == 10  # 2*1*5

    # --- Error cases ---

    def test_holdout_missing_source(self):
        cache = _make_cache(("a", 1, 1, 1))
        splitter = CrossDatasetUDASplitter({
            "strategy": "holdout",
            "variant": "transductive",
        })
        with pytest.raises(ConfigError, match="source_datasets"):
            splitter.split(cache)

    def test_holdout_unknown_alias(self):
        cache = _make_cache(("a", 1, 1, 1))
        splitter = CrossDatasetUDASplitter({
            "strategy": "holdout",
            "variant": "transductive",
            "source_datasets": ["a"],
            "target_dataset": "nonexistent",
        })
        with pytest.raises(ConfigError, match="Unknown dataset aliases"):
            splitter.split(cache)


# -----------------------------------------------------------------------
# DatasetLevelSplitter
# -----------------------------------------------------------------------


class TestDatasetLevelSplitter:
    def test_explicit_assignment(self):
        splitter = DatasetLevelSplitter({
            "dimension": "dataset",
            "train": ["ds_a", "ds_b"],
            "test": ["ds_c"],
        })
        results = splitter.split(["ds_a", "ds_b", "ds_c"])
        assert len(results) == 1
        p = results[0].phase_aliases
        assert p["train"] == ["ds_a", "ds_b"]
        assert p["test"] == ["ds_c"]
        assert p["val"] == []

    def test_explicit_with_val(self):
        splitter = DatasetLevelSplitter({
            "dimension": "dataset",
            "train": ["a"],
            "val": ["b"],
            "test": ["c"],
        })
        p = splitter.split(["a", "b", "c"])[0].phase_aliases
        assert p["val"] == ["b"]

    def test_explicit_unknown_alias(self):
        splitter = DatasetLevelSplitter({
            "dimension": "dataset",
            "train": ["a"],
            "test": ["z"],
        })
        with pytest.raises(ConfigError, match="Unknown"):
            splitter.split(["a", "b"])

    def test_ratio_split(self):
        splitter = DatasetLevelSplitter({
            "dimension": "dataset",
            "train_ratio": 0.66,
            "test_ratio": 0.34,
            "shuffle": False,
        })
        results = splitter.split(["ds_a", "ds_b", "ds_c"])
        p = results[0].phase_aliases
        # 3 datasets, ~66% train = 2, ~34% test = 1
        assert len(p["train"]) == 2
        assert len(p["test"]) == 1

    def test_ratio_too_few_datasets(self):
        splitter = DatasetLevelSplitter({
            "dimension": "dataset",
            "train_ratio": 0.5,
            "test_ratio": 0.5,
        })
        with pytest.raises(ConfigError, match="at least 2"):
            splitter.split(["only_one"])

    def test_deterministic_with_seed(self):
        cfg = {"dimension": "dataset", "train_ratio": 0.5, "test_ratio": 0.5, "seed": 42}
        r1 = DatasetLevelSplitter(cfg).split(["a", "b", "c", "d"])[0]
        r2 = DatasetLevelSplitter(cfg).split(["a", "b", "c", "d"])[0]
        assert r1.phase_aliases == r2.phase_aliases


# -----------------------------------------------------------------------
# create_splitter factory extension
# -----------------------------------------------------------------------


class TestCreateSplitterExtended:
    def test_uda_intra(self):
        s = create_splitter(mode="uda", uda_config={
            "type": "intra-dataset", "dimension": "subject",
        })
        assert isinstance(s, IntraDatasetUDASplitter)

    def test_uda_cross(self):
        s = create_splitter(mode="uda", uda_config={
            "type": "cross-dataset",
            "source_datasets": ["a"],
            "target_dataset": "b",
        })
        assert isinstance(s, CrossDatasetUDASplitter)

    def test_uda_unknown_type(self):
        with pytest.raises(ConfigError, match="Unknown UDA type"):
            create_splitter(mode="uda", uda_config={"type": "unknown"})

    def test_uda_missing_config(self):
        with pytest.raises(ConfigError, match="uda_config"):
            create_splitter(mode="uda")

    def test_dataset_level(self):
        s = create_splitter({"dimension": "dataset", "train": ["a"], "test": ["b"]})
        assert isinstance(s, DatasetLevelSplitter)

    def test_backward_compat_holdout(self):
        from uesf.experiment.splitter import HoldoutSplitter
        s = create_splitter({"strategy": "holdout"})
        assert isinstance(s, HoldoutSplitter)

    def test_backward_compat_kfold(self):
        from uesf.experiment.splitter import KFoldSplitter
        s = create_splitter({"strategy": "k-fold"})
        assert isinstance(s, KFoldSplitter)
