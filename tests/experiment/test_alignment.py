"""Tests for cross-dataset alignment module."""

from __future__ import annotations

import numpy as np
import pytest

from uesf.core.exceptions import ConfigError
from uesf.experiment.alignment import (
    IntersectionAligner,
    LabelAligner,
    create_channel_aligner,
)


class TestIntersectionAligner:
    def test_basic_intersection(self):
        """Common channels are retained, others dropped."""
        aligner = IntersectionAligner()
        ds = {
            "a": (
                np.random.randn(2, 1, 3, 4, 100),  # 4 channels
                ["Fp1", "Fp2", "C3", "C4"],
            ),
            "b": (
                np.random.randn(3, 1, 2, 5, 100),  # 5 channels
                ["Fp2", "C3", "C4", "P3", "P4"],
            ),
        }
        aligned, common = aligner.align(ds)

        assert common == ["Fp2", "C3", "C4"]
        assert aligned["a"].shape[-2] == 3
        assert aligned["b"].shape[-2] == 3
        # Remaining dimensions untouched
        assert aligned["a"].shape[:3] == (2, 1, 3)
        assert aligned["b"].shape[:3] == (3, 1, 2)

    def test_preserves_first_dataset_order(self):
        """Intersection order follows the first dataset's electrode list."""
        aligner = IntersectionAligner()
        ds = {
            "a": (np.zeros((1, 1, 1, 3, 10)), ["C4", "C3", "Fp1"]),
            "b": (np.zeros((1, 1, 1, 3, 10)), ["Fp1", "C3", "C4"]),
        }
        _, common = aligner.align(ds)
        assert common == ["C4", "C3", "Fp1"]

    def test_three_datasets(self):
        """Intersection across three datasets."""
        aligner = IntersectionAligner()
        ds = {
            "a": (np.zeros((1, 1, 1, 4, 10)), ["Fp1", "Fp2", "C3", "C4"]),
            "b": (np.zeros((1, 1, 1, 3, 10)), ["Fp2", "C3", "Pz"]),
            "c": (np.zeros((1, 1, 1, 2, 10)), ["Fp2", "C3"]),
        }
        aligned, common = aligner.align(ds)
        assert common == ["Fp2", "C3"]
        for v in aligned.values():
            assert v.shape[-2] == 2

    def test_empty_intersection_raises(self):
        """No shared channels → ConfigError."""
        aligner = IntersectionAligner()
        ds = {
            "a": (np.zeros((1, 1, 1, 2, 10)), ["Fp1", "Fp2"]),
            "b": (np.zeros((1, 1, 1, 2, 10)), ["C3", "C4"]),
        }
        with pytest.raises(ConfigError, match="empty"):
            aligner.align(ds)

    def test_no_datasets_raises(self):
        aligner = IntersectionAligner()
        with pytest.raises(ConfigError, match="No datasets"):
            aligner.align({})

    def test_works_on_3d_data(self):
        """Alignment should also work on already-flattened 3-D data."""
        aligner = IntersectionAligner()
        ds = {
            "a": (np.random.randn(20, 4, 100), ["Fp1", "Fp2", "C3", "C4"]),
            "b": (np.random.randn(15, 3, 100), ["Fp2", "C3", "Pz"]),
        }
        aligned, common = aligner.align(ds)
        assert common == ["Fp2", "C3"]
        assert aligned["a"].shape == (20, 2, 100)
        assert aligned["b"].shape == (15, 2, 100)

    def test_data_values_correct(self):
        """Verify the correct channels are selected by value."""
        aligner = IntersectionAligner()
        data_a = np.zeros((1, 1, 1, 3, 2))
        data_a[0, 0, 0, 0, :] = [1, 2]  # Fp1
        data_a[0, 0, 0, 1, :] = [3, 4]  # C3
        data_a[0, 0, 0, 2, :] = [5, 6]  # C4

        data_b = np.zeros((1, 1, 1, 2, 2))
        data_b[0, 0, 0, 0, :] = [10, 20]  # C3
        data_b[0, 0, 0, 1, :] = [30, 40]  # C4

        ds = {
            "a": (data_a, ["Fp1", "C3", "C4"]),
            "b": (data_b, ["C3", "C4"]),
        }
        aligned, common = aligner.align(ds)
        assert common == ["C3", "C4"]
        np.testing.assert_array_equal(aligned["a"][0, 0, 0, 0], [3, 4])  # C3
        np.testing.assert_array_equal(aligned["a"][0, 0, 0, 1], [5, 6])  # C4
        np.testing.assert_array_equal(aligned["b"][0, 0, 0, 0], [10, 20])  # C3


class TestLabelAligner:
    def test_consistent_labels_pass(self):
        """No error when labels match."""
        aligner = LabelAligner()
        meta = {
            "a": {"n_classes": 2, "numeric_to_semantic": {0: "left", 1: "right"}},
            "b": {"n_classes": 2, "numeric_to_semantic": {0: "left", 1: "right"}},
        }
        aligner.validate(meta)  # Should not raise

    def test_class_count_mismatch(self):
        aligner = LabelAligner()
        meta = {
            "a": {"n_classes": 2},
            "b": {"n_classes": 3},
        }
        with pytest.raises(ConfigError, match="Label mismatch"):
            aligner.validate(meta)

    def test_mapping_mismatch(self):
        aligner = LabelAligner()
        meta = {
            "a": {"n_classes": 2, "numeric_to_semantic": {0: "left", 1: "right"}},
            "b": {"n_classes": 2, "numeric_to_semantic": {0: "right", 1: "left"}},
        }
        with pytest.raises(ConfigError, match="mapping mismatch"):
            aligner.validate(meta)

    def test_single_dataset_skipped(self):
        """Validation is a no-op with fewer than 2 datasets."""
        aligner = LabelAligner()
        aligner.validate({"a": {"n_classes": 2}})  # Should not raise

    def test_missing_mapping_ok(self):
        """If one dataset has no mapping, skip mapping check."""
        aligner = LabelAligner()
        meta = {
            "a": {"n_classes": 2, "numeric_to_semantic": {0: "left", 1: "right"}},
            "b": {"n_classes": 2},
        }
        aligner.validate(meta)  # Should not raise


class TestCreateChannelAligner:
    def test_intersection(self):
        a = create_channel_aligner("intersection")
        assert isinstance(a, IntersectionAligner)

    def test_unknown(self):
        with pytest.raises(ConfigError, match="Unknown"):
            create_channel_aligner("spherical_spline")
