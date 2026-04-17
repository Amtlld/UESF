"""Tests for the 5-D dimension grouping helper."""

from __future__ import annotations

import numpy as np
import pytest

from uesf.core.exceptions import TypeMismatchError
from uesf.experiment.splitter import get_groups


def _make_data(sub=4, sess=2, rec=5, ch=3, samp=8):
    return np.empty((sub, sess, rec, ch, samp))


class TestGetGroups:
    def test_subject(self):
        groups = get_groups(_make_data(sub=5, sess=2, rec=3), "subject")
        assert len(groups) == 5
        # Each subject group spans 2 * 3 = 6 flat indices, contiguous
        for i, g in enumerate(groups):
            assert g.tolist() == list(range(i * 6, (i + 1) * 6))

    def test_session(self):
        groups = get_groups(_make_data(sub=3, sess=2, rec=4), "session")
        assert len(groups) == 6  # 3 * 2
        for g in groups:
            assert len(g) == 4

    def test_recording(self):
        groups = get_groups(_make_data(sub=2, sess=2, rec=3), "recording")
        assert len(groups) == 12
        for g in groups:
            assert len(g) == 1

    def test_flatten(self):
        groups = get_groups(_make_data(sub=2, sess=3, rec=2), "flatten")
        assert len(groups) == 12  # same total as recording, semantics differ downstream
        for g in groups:
            assert len(g) == 1

    def test_covers_all_indices_no_overlap(self):
        data = _make_data(sub=3, sess=2, rec=3)
        for dim in ("subject", "session", "recording", "flatten"):
            groups = get_groups(data, dim)
            union = np.sort(np.concatenate(groups))
            assert np.array_equal(union, np.arange(3 * 2 * 3))

    def test_rejects_non_5d(self):
        with pytest.raises(TypeMismatchError, match="5-D"):
            get_groups(np.empty((10, 32, 500)), "subject")

    def test_rejects_unknown_dim(self):
        with pytest.raises(TypeMismatchError, match="Unknown dimension"):
            get_groups(_make_data(), "trial")
