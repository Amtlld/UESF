"""Tests for apply_transforms / apply_transforms_uda."""

from __future__ import annotations

import numpy as np

from uesf.experiment.splitter import UDASplitResult
from uesf.experiment.transforms import apply_transforms, apply_transforms_uda


def _make_data(sub=3, sess=1, rec=2, ch=2, samp=4, mean_shift=0.0, std=1.0):
    data = np.random.RandomState(0).randn(sub, sess, rec, ch, samp).astype(np.float32)
    return data * std + mean_shift


class TestApplyTransformsRegular:
    def test_single_dataset_per_dataset_fits_on_train(self):
        data = _make_data(sub=5, sess=1, rec=2, mean_shift=10.0, std=3.0)
        cache = {"a": data.copy()}
        split = {"a": {"train": np.arange(0, 6), "val": np.arange(6, 8), "test": np.arange(8, 10)}}
        steps = [{"name": "zscore_normalize", "scope": "per_dataset"}]

        apply_transforms({"a": steps}, cache, split)

        flat = cache["a"].reshape(10, 2, 4)
        train_mean = flat[split["a"]["train"]].mean(axis=0)
        np.testing.assert_allclose(train_mean, 0.0, atol=1e-5)

    def test_multi_dataset_per_dataset_uses_own_stats(self):
        d1 = _make_data(sub=3, sess=1, rec=2, mean_shift=5.0)
        d2 = _make_data(sub=3, sess=1, rec=2, mean_shift=-5.0)
        cache = {"a": d1.copy(), "b": d2.copy()}
        split = {
            "a": {"train": np.arange(0, 4), "val": np.array([], int), "test": np.array([], int)},
            "b": {"train": np.arange(0, 4), "val": np.array([], int), "test": np.array([], int)},
        }
        steps = [{"name": "zscore_normalize", "scope": "per_dataset"}]
        apply_transforms({"a": steps, "b": steps}, cache, split)

        # Each dataset's own train mean is ~0 after transform
        for alias in ("a", "b"):
            flat = cache[alias].reshape(6, 2, 4)
            np.testing.assert_allclose(
                flat[split[alias]["train"]].mean(axis=0), 0.0, atol=1e-5
            )

    def test_cross_dataset_test_only_alias_fits_on_self(self):
        """dimension=dataset: the test-only alias has no 'train' — fit on self."""
        d1 = _make_data(sub=3, sess=1, rec=2, mean_shift=4.0)
        d_test = _make_data(sub=2, sess=1, rec=2, mean_shift=-10.0)
        cache = {"a": d1.copy(), "t": d_test.copy()}
        split = {
            "a": {"train": np.arange(0, 4), "val": np.arange(4, 6)},
            # test alias has only 'test' phase, no 'train'
            "t": {"test": np.arange(0, 4)},
        }
        steps = [{"name": "zscore_normalize", "scope": "per_dataset"}]
        apply_transforms({"a": steps, "t": steps}, cache, split)

        flat_test = cache["t"].reshape(4, 2, 4)
        np.testing.assert_allclose(flat_test.mean(axis=0), 0.0, atol=1e-5)

    def test_global_scope_merges_train_for_fit(self):
        d1 = _make_data(sub=3, sess=1, rec=2, mean_shift=5.0, std=2.0)
        d2 = _make_data(sub=3, sess=1, rec=2, mean_shift=-5.0, std=2.0)
        cache = {"a": d1.copy(), "b": d2.copy()}
        split = {
            "a": {"train": np.arange(0, 6), "val": np.array([], int), "test": np.array([], int)},
            "b": {"train": np.arange(0, 6), "val": np.array([], int), "test": np.array([], int)},
        }
        steps = [{"name": "zscore_normalize", "scope": "global"}]
        apply_transforms({"a": steps, "b": steps}, cache, split)

        # After global fit, the combined-train mean across both datasets ≈ 0
        merged = np.concatenate(
            [cache["a"].reshape(6, 2, 4), cache["b"].reshape(6, 2, 4)], axis=0
        )
        np.testing.assert_allclose(merged.mean(axis=0), 0.0, atol=1e-5)

    def test_no_transforms_is_noop(self):
        cache = {"a": np.arange(30).reshape(3, 1, 2, 1, 5).astype(np.float32)}
        original = cache["a"].copy()
        apply_transforms({"a": []}, cache, {"a": {"train": np.arange(6)}})
        np.testing.assert_array_equal(cache["a"], original)


class TestApplyTransformsUDA:
    def _make_uda(self, inductive=False):
        data = _make_data(sub=4, sess=1, rec=2, mean_shift=10.0, std=3.0)
        cache = {"main": data.copy()}
        split = UDASplitResult(
            source_train={"main": np.array([0, 1, 2, 3], int)},
            source_val={"main": np.array([], int)},
            target_train={"main": np.array([4, 5, 6, 7], int) if not inductive else np.array([4, 5], int)},
            target_val={"main": np.array([], int) if not inductive else np.array([6], int)},
            target_test={"main": np.array([4, 5, 6, 7], int) if not inductive else np.array([7], int)},
        )
        return cache, split

    def test_per_dataset_transductive(self):
        cache, split = self._make_uda(inductive=False)
        steps = [{"name": "zscore_normalize", "scope": "per_dataset"}]

        apply_transforms_uda(
            {"main": steps}, cache, split, adaptation="transductive"
        )

        flat = cache["main"].reshape(8, 2, 4)
        # Target transductive: fit on target_train = full target — target_train mean ≈ 0
        np.testing.assert_allclose(
            flat[split.target_train["main"]].mean(axis=0), 0.0, atol=1e-5
        )

    def test_per_dataset_inductive_target_train_fit(self):
        cache, split = self._make_uda(inductive=True)
        steps = [{"name": "zscore_normalize", "scope": "per_dataset"}]

        apply_transforms_uda(
            {"main": steps}, cache, split, adaptation="inductive"
        )

        flat = cache["main"].reshape(8, 2, 4)
        # Target_train mean ~ 0 after fitting on target_train
        np.testing.assert_allclose(
            flat[split.target_train["main"]].mean(axis=0), 0.0, atol=1e-5
        )
