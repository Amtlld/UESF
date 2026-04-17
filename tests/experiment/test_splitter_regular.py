"""Tests for Regular splitters (Holdout / KFold / DatasetLevel)."""

from __future__ import annotations

import numpy as np
import pytest

from uesf.core.exceptions import SplitError, TypeMismatchError
from uesf.experiment.splitter import (
    DatasetLevelSplitResult,
    DatasetLevelSplitter,
    HoldoutSplitter,
    KFoldSplitter,
    SplitResult,
    create_splitter,
)


def _make_data(sub=5, sess=2, rec=4, ch=3, samp=8):
    return np.empty((sub, sess, rec, ch, samp))


def _assign_sets(result: SplitResult, total: int) -> dict[int, str]:
    asg: dict[int, str] = {}
    for label, idx in [
        ("train", result.train_indices),
        ("val", result.val_indices),
        ("test", result.test_indices),
    ]:
        for i in idx.tolist():
            assert i not in asg, f"index {i} appears in multiple phases"
            asg[int(i)] = label
    assert len(asg) == total
    return asg


class TestHoldoutSplitter:
    def test_three_way_ratios_shuffle_false(self):
        # 5 subjects * 2 sess * 4 rec = 40 units; dim=subject → 5 groups
        data = _make_data(sub=5, sess=2, rec=4)
        sp = HoldoutSplitter(
            dimension="subject",
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            shuffle=False,
            seed=0,
        )
        (r,) = sp.split(data)
        assigned = _assign_sets(r, 40)
        # sequential: first 3 subjects → train, next 1 → val, last 1 → test
        assert {assigned[i] for i in range(0, 3 * 8)} == {"train"}
        assert {assigned[i] for i in range(3 * 8, 4 * 8)} == {"val"}
        assert {assigned[i] for i in range(4 * 8, 5 * 8)} == {"test"}
        assert r.fold_info == {"fold_idx": 0}

    def test_subject_isolation(self):
        data = _make_data(sub=6, sess=2, rec=3)
        sp = HoldoutSplitter(
            dimension="subject",
            train_ratio=0.5,
            val_ratio=0.25,
            test_ratio=0.25,
            shuffle=True,
            seed=123,
        )
        (r,) = sp.split(data)
        # Every subject's 6 rows must all belong to the same phase
        samples_per_sub = 2 * 3
        assigned = _assign_sets(r, 36)
        for s in range(6):
            start, end = s * samples_per_sub, (s + 1) * samples_per_sub
            labels = {assigned[i] for i in range(start, end)}
            assert len(labels) == 1

    def test_val_split_inner_different_dim(self):
        # main dim=subject, val_split dim=session
        data = _make_data(sub=4, sess=3, rec=2)
        sp = HoldoutSplitter(
            dimension="subject",
            train_ratio=0.75,
            test_ratio=0.25,
            val_split_config={"dimension": "session", "val_ratio": 0.25, "shuffle": False},
            shuffle=False,
            seed=1,
        )
        (r,) = sp.split(data)
        # 4 subjects → 3 train + 1 test; train=3*3*2=18 units, val carved from
        # those 3 subjects at session granularity
        assert len(r.test_indices) == 6  # 1 subject × 3 sess × 2 rec
        assert len(r.train_indices) + len(r.val_indices) == 18
        assert len(r.val_indices) > 0

    def test_rejects_val_ratio_with_val_split(self):
        with pytest.raises(TypeMismatchError, match="val_ratio must be 0"):
            HoldoutSplitter(
                dimension="subject",
                train_ratio=0.7,
                test_ratio=0.2,
                val_ratio=0.1,
                val_split_config={"dimension": "subject", "val_ratio": 0.1},
            )

    def test_raises_when_only_one_group(self):
        data = _make_data(sub=1, sess=2, rec=3)
        sp = HoldoutSplitter(dimension="subject", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        with pytest.raises(SplitError, match="need at least 2"):
            sp.split(data)

    def test_deterministic_with_seed(self):
        data = _make_data(sub=4, sess=1, rec=3)
        cfg = dict(dimension="subject", train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, shuffle=True, seed=7)
        (r1,) = HoldoutSplitter(**cfg).split(data)
        (r2,) = HoldoutSplitter(**cfg).split(data)
        np.testing.assert_array_equal(r1.train_indices, r2.train_indices)


class TestKFoldSplitter:
    def test_basic_kfold_each_sample_in_exactly_one_test(self):
        data = _make_data(sub=5, sess=2, rec=3)  # 5 subject groups
        sp = KFoldSplitter(dimension="subject", k=5, shuffle=False, seed=0)
        folds = sp.split(data)
        assert len(folds) == 5
        all_test = np.concatenate([f.test_indices for f in folds])
        assert len(all_test) == 5 * 2 * 3
        assert len(set(all_test.tolist())) == len(all_test)

    def test_loocv_k_minus_one(self):
        data = _make_data(sub=4, sess=1, rec=2)  # 4 subjects
        folds = KFoldSplitter(dimension="subject", k=-1, shuffle=False).split(data)
        assert len(folds) == 4
        for f in folds:
            assert len(f.test_indices) == 2  # one subject × 2 recordings
            assert "test_group" in f.fold_info

    def test_val_ratio_in_train(self):
        data = _make_data(sub=5, sess=1, rec=4)  # 5 subject groups
        sp = KFoldSplitter(dimension="subject", k=5, val_ratio=0.25, shuffle=False)
        folds = sp.split(data)
        for f in folds:
            assert len(f.val_indices) > 0
            total = len(f.train_indices) + len(f.val_indices) + len(f.test_indices)
            assert total == 5 * 4

    def test_val_split_inner(self):
        # main dim=subject, val_split dim=recording
        data = _make_data(sub=5, sess=1, rec=4)  # 5 groups × 4 recs
        sp = KFoldSplitter(
            dimension="subject",
            k=5,
            val_split_config={"dimension": "recording", "val_ratio": 0.2, "shuffle": False},
            shuffle=False,
            seed=1,
        )
        folds = sp.split(data)
        assert len(folds) == 5
        for f in folds:
            assert len(f.val_indices) > 0
            # val is carved only from the 4-subject training pool
            assert not set(f.val_indices.tolist()) & set(f.test_indices.tolist())

    def test_k_must_be_valid(self):
        with pytest.raises(TypeMismatchError):
            KFoldSplitter(dimension="subject", k=0)
        with pytest.raises(TypeMismatchError):
            KFoldSplitter(dimension="subject", k=1)

    def test_k_greater_than_groups_raises(self):
        data = _make_data(sub=3, sess=1, rec=1)
        with pytest.raises(SplitError, match="exceeds number of groups"):
            KFoldSplitter(dimension="subject", k=5).split(data)

    def test_fold_info_has_fold_idx(self):
        data = _make_data(sub=4, sess=1, rec=2)
        folds = KFoldSplitter(dimension="subject", k=4, shuffle=False).split(data)
        for i, f in enumerate(folds):
            assert f.fold_info["fold_idx"] == i

    def test_deterministic_with_seed(self):
        data = _make_data(sub=6, sess=1, rec=2)
        cfg = dict(dimension="subject", k=3, shuffle=True, seed=42)
        f1 = KFoldSplitter(**cfg).split(data)
        f2 = KFoldSplitter(**cfg).split(data)
        for a, b in zip(f1, f2):
            np.testing.assert_array_equal(a.test_indices, b.test_indices)


class TestDatasetLevelSplitter:
    def test_holdout_assign(self):
        sp = DatasetLevelSplitter(
            strategy="holdout",
            assign={"train": ["a", "b"], "test": ["c"]},
        )
        (r,) = sp.split(["a", "b", "c"])
        assert isinstance(r, DatasetLevelSplitResult)
        assert r.phase_aliases == {"train": ["a", "b"], "val": [], "test": ["c"]}
        assert r.fold_info == {"fold_idx": 0}

    def test_holdout_missing_alias_raises(self):
        sp = DatasetLevelSplitter(strategy="holdout", assign={"train": ["a"], "test": ["b"]})
        with pytest.raises(TypeMismatchError, match="missing"):
            sp.split(["a", "b", "c"])

    def test_holdout_unknown_alias_raises(self):
        sp = DatasetLevelSplitter(strategy="holdout", assign={"train": ["a"], "test": ["z"]})
        with pytest.raises(TypeMismatchError, match="unknown"):
            sp.split(["a", "b"])

    def test_kfold_loocv(self):
        sp = DatasetLevelSplitter(strategy="k-fold", k=-1, shuffle=False)
        folds = sp.split(["a", "b", "c"])
        assert len(folds) == 3
        # each alias is test exactly once
        test_aliases = [f.phase_aliases["test"][0] for f in folds]
        assert sorted(test_aliases) == ["a", "b", "c"]

    def test_kfold_k_must_match_datasets(self):
        sp = DatasetLevelSplitter(strategy="k-fold", k=2, shuffle=False)
        with pytest.raises(TypeMismatchError, match="must equal"):
            sp.split(["a", "b", "c"])

    def test_kfold_deterministic(self):
        sp1 = DatasetLevelSplitter(strategy="k-fold", k=-1, shuffle=True, seed=99)
        sp2 = DatasetLevelSplitter(strategy="k-fold", k=-1, shuffle=True, seed=99)
        assert [f.phase_aliases for f in sp1.split(["a", "b", "c", "d"])] == [
            f.phase_aliases for f in sp2.split(["a", "b", "c", "d"])
        ]


class TestCreateSplitterFactory:
    def test_returns_holdout(self):
        s = create_splitter(
            {
                "strategy": "holdout",
                "dimension": "subject",
                "train_ratio": 0.7,
                "val_ratio": 0.15,
                "test_ratio": 0.15,
                "shuffle": True,
            }
        )
        assert isinstance(s, HoldoutSplitter)

    def test_returns_kfold(self):
        s = create_splitter(
            {"strategy": "k-fold", "dimension": "subject", "k": 3, "shuffle": True}
        )
        assert isinstance(s, KFoldSplitter)

    def test_dataset_dim_not_supported(self):
        with pytest.raises(TypeMismatchError, match="dataset"):
            create_splitter({"strategy": "holdout", "dimension": "dataset"})

    def test_unknown_strategy(self):
        with pytest.raises(TypeMismatchError, match="unknown strategy"):
            create_splitter({"strategy": "foo", "dimension": "subject"})
