"""Tests for dataset splitters."""

from __future__ import annotations

import numpy as np
import pytest

from uesf.core.exceptions import ConfigError
from uesf.experiment.splitter import (
    HoldoutSplitter,
    KFoldSplitter,
    _get_groups,
    create_splitter,
)


# Helper: create 5-D dummy data [sub, sess, rec, chan, sample]
def _make_data(sub=4, sess=2, rec=5, chan=32, sample=500):
    return np.empty((sub, sess, rec, chan, sample))


def _assign_sets(result, total):
    """Map each flat index to the set it belongs to ('train'/'val'/'test')."""
    assignment = {}
    for label, indices in [
        ("train", result.train_indices),
        ("val", result.val_indices),
        ("test", result.test_indices),
    ]:
        for idx in indices:
            assignment[int(idx)] = label
    assert len(assignment) == total, "indices must cover all units without overlap"
    return assignment


def _group_key_subject(idx, n_sess, n_rec):
    """Flat index → subject id."""
    return idx // (n_sess * n_rec)


def _group_key_session(idx, n_sess, n_rec):
    """Flat index → (subject, session) id (unique int)."""
    return idx // n_rec


class TestHoldoutSplitter:
    def test_basic_split(self):
        # 4 sub * 2 sess * 5 rec = 40 sample units
        data = _make_data(sub=4, sess=2, rec=5)
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
        assert total == 40

    def test_no_overlap(self):
        data = _make_data(sub=5, sess=2, rec=5)  # 50 units
        splitter = HoldoutSplitter({"dimension": "none", "seed": 42})
        r = splitter.split(data)[0]

        all_indices = set(r.train_indices) | set(r.val_indices) | set(r.test_indices)
        assert len(all_indices) == 50

    def test_subject_isolation(self):
        """Subject split: all recordings of a subject stay in the same set."""
        n_sub, n_sess, n_rec = 10, 3, 4  # 120 units
        data = _make_data(sub=n_sub, sess=n_sess, rec=n_rec)
        r = HoldoutSplitter({
            "train_ratio": 0.6, "val_ratio": 0.2, "test_ratio": 0.2,
            "dimension": "subject", "shuffle": False,
        }).split(data)[0]
        assignment = _assign_sets(r, n_sub * n_sess * n_rec)

        # Property 2: every index of the same subject maps to the same set
        subject_sets: dict[int, set[str]] = {}
        for idx, label in assignment.items():
            s = _group_key_subject(idx, n_sess, n_rec)
            subject_sets.setdefault(s, set()).add(label)
        for s, labels in subject_sets.items():
            assert len(labels) == 1, f"subject {s} leaked across sets: {labels}"

        # Property 1: subjects are actually distributed across sets
        all_labels = {next(iter(v)) for v in subject_sets.values()}
        assert len(all_labels) >= 2, "split did not distribute subjects across sets"

    def test_session_isolation(self):
        """Session split: recordings of the same session stay together,
        but different sessions of the same subject CAN be in different sets."""
        n_sub, n_sess, n_rec = 4, 5, 3  # 60 units
        data = _make_data(sub=n_sub, sess=n_sess, rec=n_rec)
        r = HoldoutSplitter({
            "train_ratio": 0.6, "val_ratio": 0.2, "test_ratio": 0.2,
            "dimension": "session", "shuffle": False,
        }).split(data)[0]
        assignment = _assign_sets(r, n_sub * n_sess * n_rec)

        # Property 2: every index of the same (sub, sess) maps to the same set
        session_sets: dict[int, set[str]] = {}
        for idx, label in assignment.items():
            key = _group_key_session(idx, n_sess, n_rec)
            session_sets.setdefault(key, set()).add(label)
        for key, labels in session_sets.items():
            assert len(labels) == 1, f"session {key} leaked across sets: {labels}"

        # Property 1: sessions are actually distributed across sets
        all_labels = {next(iter(v)) for v in session_sets.values()}
        assert len(all_labels) >= 2, "split did not distribute sessions across sets"

        # Split is ONLY at session level: same subject appears in multiple sets
        subject_labels: dict[int, set[str]] = {}
        for idx, label in assignment.items():
            s = _group_key_subject(idx, n_sess, n_rec)
            subject_labels.setdefault(s, set()).add(label)
        subjects_in_multiple = [s for s, lb in subject_labels.items() if len(lb) > 1]
        assert len(subjects_in_multiple) > 0, (
            "session-level split should not fully isolate subjects"
        )

    def test_deterministic_with_seed(self):
        data = _make_data(sub=4, sess=1, rec=5)  # 20 units
        cfg = {"dimension": "none", "seed": 123}
        r1 = HoldoutSplitter(cfg).split(data)[0]
        r2 = HoldoutSplitter(cfg).split(data)[0]
        np.testing.assert_array_equal(r1.train_indices, r2.train_indices)

    def test_invalid_ratio_sum(self):
        with pytest.raises(ConfigError, match="sum to 1.0"):
            HoldoutSplitter({
                "train_ratio": 0.8,
                "val_ratio": 0.8,
                "test_ratio": 0.8,
            })


class TestKFoldSplitter:
    def test_basic_kfold(self):
        data = _make_data(sub=10, sess=2, rec=5)  # 100 units
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
        data = _make_data(sub=10, sess=1, rec=1)  # 10 units
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
        data = _make_data(sub=5, sess=1, rec=1)  # 5 units
        splitter = KFoldSplitter({
            "k-folds": "total",
            "dimension": "none",
            "shuffle": False,
        })
        results = splitter.split(data)
        assert len(results) == 5

    def test_subject_isolation(self):
        """K-fold on subject: each fold keeps all of a subject's data together."""
        n_sub, n_sess, n_rec = 6, 2, 3  # 36 units, 6 per subject
        data = _make_data(sub=n_sub, sess=n_sess, rec=n_rec)
        results = KFoldSplitter({
            "k-folds": 3, "dimension": "subject", "shuffle": False,
        }).split(data)
        assert len(results) == 3

        for r in results:
            assignment = _assign_sets(r, n_sub * n_sess * n_rec)
            # Property 2: same subject never in both train and test
            subject_sets: dict[int, set[str]] = {}
            for idx, label in assignment.items():
                s = _group_key_subject(idx, n_sess, n_rec)
                subject_sets.setdefault(s, set()).add(label)
            for s, labels in subject_sets.items():
                assert len(labels) == 1, f"fold leaked subject {s}: {labels}"

            # Property 1: both train and test have subjects
            all_labels = {next(iter(v)) for v in subject_sets.values()}
            assert "train" in all_labels and "test" in all_labels

    def test_session_isolation(self):
        """K-fold on session: sessions are isolated, subjects are not."""
        # 5 subjects * 3 sessions = 15 groups, k=4 → fold sizes 4,4,4,3
        # which forces folds to cross subject boundaries
        n_sub, n_sess, n_rec = 5, 3, 2  # 30 units
        total = n_sub * n_sess * n_rec
        data = _make_data(sub=n_sub, sess=n_sess, rec=n_rec)
        results = KFoldSplitter({
            "k-folds": 4, "dimension": "session", "shuffle": False,
        }).split(data)

        any_subject_crossed = False
        for r in results:
            assignment = _assign_sets(r, total)

            # Property 2: same session never split across train/test
            session_sets: dict[int, set[str]] = {}
            for idx, label in assignment.items():
                key = _group_key_session(idx, n_sess, n_rec)
                session_sets.setdefault(key, set()).add(label)
            for key, labels in session_sets.items():
                assert len(labels) == 1, f"fold leaked session {key}: {labels}"

            # Property 1: both train and test present
            all_labels = {next(iter(v)) for v in session_sets.values()}
            assert "train" in all_labels and "test" in all_labels

            # Collect: does any fold have a subject in both train and test?
            subject_labels: dict[int, set[str]] = {}
            for idx, label in assignment.items():
                s = _group_key_subject(idx, n_sess, n_rec)
                subject_labels.setdefault(s, set()).add(label)
            if any(len(lb) > 1 for lb in subject_labels.values()):
                any_subject_crossed = True

        # Split is ONLY at session level: at least one fold has same subject
        # in both train and test
        assert any_subject_crossed, (
            "session-level fold should not fully isolate subjects"
        )

    def test_val_ratio_in_train(self):
        data = _make_data(sub=4, sess=1, rec=5)  # 20 units
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

    def test_invalid_k(self):
        with pytest.raises(ConfigError, match="k-folds"):
            KFoldSplitter({"k-folds": 0})
        with pytest.raises(ConfigError, match="k-folds"):
            KFoldSplitter({"k-folds": 1})


class TestGetGroups:
    def test_none_dimension(self):
        data = _make_data(sub=2, sess=1, rec=5)  # 10 units
        groups = _get_groups(data, "none")
        assert len(groups) == 10
        for g in groups:
            assert len(g) == 1

    def test_subject_dimension(self):
        data = _make_data(sub=5, sess=1, rec=3)  # 5 groups of 3
        groups = _get_groups(data, "subject")
        assert len(groups) == 5
        for g in groups:
            assert len(g) == 3

    def test_session_dimension(self):
        data = _make_data(sub=3, sess=2, rec=4)  # 6 groups of 4
        groups = _get_groups(data, "session")
        assert len(groups) == 6
        for g in groups:
            assert len(g) == 4

    def test_recording_dimension(self):
        data = _make_data(sub=2, sess=2, rec=3)  # 12 groups of 1
        groups = _get_groups(data, "recording")
        assert len(groups) == 12
        for g in groups:
            assert len(g) == 1

    def test_rejects_non_5d(self):
        with pytest.raises(ConfigError, match="5-D"):
            _get_groups(np.empty((10, 32, 500)), "none")

    def test_unknown_dimension(self):
        with pytest.raises(ConfigError, match="Unknown"):
            _get_groups(_make_data(), "trial")


class TestCreateSplitter:
    def test_holdout(self):
        s = create_splitter({"strategy": "holdout"})
        assert isinstance(s, HoldoutSplitter)

    def test_kfold(self):
        s = create_splitter({"strategy": "k-fold"})
        assert isinstance(s, KFoldSplitter)

    def test_unknown(self):
        with pytest.raises(ConfigError, match="Unknown"):
            create_splitter({"strategy": "unknown"})
