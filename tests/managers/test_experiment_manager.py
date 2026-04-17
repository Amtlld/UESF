"""Smoke tests for ExperimentManager dev6 plumbing.

E2E coverage lives in tests/e2e/test_full_workflow.py. This module focuses
on the new dataclasses, the config pipeline integration, and the partial /
failed-fold aggregation path that's tricky to provoke through e2e only.
"""

from __future__ import annotations

import numpy as np

from uesf.managers.experiment_manager import (
    ExperimentContext,
    ExperimentResult,
    FoldResult,
    RegularExecutionStrategy,
    _aggregate_folds,
)


class TestDataclasses:
    def test_fold_result_defaults(self):
        fr = FoldResult(metrics={"accuracy": 0.9})
        assert fr.failed is False
        assert fr.error is None
        assert fr.fold_info == {}

    def test_experiment_result_fields(self):
        result = ExperimentResult(
            fold_results=[FoldResult(metrics={"x": 1.0})],
            aggregated_metrics={"x": {"mean": 1.0, "std": 0.0}},
            aggregation_mode="mean_std",
        )
        assert result.aggregation_mode == "mean_std"


class TestAggregateFolds:
    def test_mean_std_skips_failed(self):
        folds = [
            FoldResult(metrics={"accuracy": 0.8}),
            FoldResult(metrics={}, failed=True, error="boom"),
            FoldResult(metrics={"accuracy": 1.0}),
        ]
        agg = _aggregate_folds(folds, mode="mean_std", metric_funcs={})
        assert abs(agg["accuracy"]["mean"] - 0.9) < 1e-9

    def test_all_failed_returns_empty(self):
        folds = [FoldResult(metrics={}, failed=True)] * 3
        agg = _aggregate_folds(folds, mode="mean_std", metric_funcs={})
        assert agg == {}


class TestRegularStrategySingleDataset:
    def test_runs_with_stub_model(self, tmp_path, monkeypatch):
        """Drive RegularExecutionStrategy directly with an in-memory context."""
        import torch.nn as nn

        from uesf.components.builtin_metrics import accuracy
        from uesf.components.dummy import DummyModel, DummyTrainer

        # 4 subjects × 1 session × 2 rec × 2 ch × 4 samp
        data = np.random.RandomState(0).randn(4, 1, 2, 2, 4).astype(np.float32)
        labels = np.random.randint(0, 2, 8).astype(np.int64)

        ctx = ExperimentContext(
            config={
                "seed": 0,
                "split": {
                    "strategy": "holdout",
                    "dimension": "subject",
                    "train_ratio": 0.5,
                    "val_ratio": 0.25,
                    "test_ratio": 0.25,
                    "shuffle": False,
                },
                "datasets": {"ds": {"name": "ds"}},
                "training": {
                    "epochs": 1,
                    "batch_size": 2,
                    "optimizer": {"name": "adam", "params": {"lr": 0.01}},
                },
                "evaluation": {"metrics": ["accuracy"]},
            },
            dataset_cache={"ds": data},
            labels_cache={"ds": labels},
            metadata_cache={"ds": {"n_channels": 2, "n_samples": 4, "n_classes": 2}},
            experiment_id=1,
            output_dir=tmp_path,
            device=__import__("torch").device("cpu"),
            project_dir=tmp_path,
            seed=0,
            model_cls=DummyModel,
            model_params={},
            trainer_cls=DummyTrainer,
            trainer_params={},
            metric_funcs={"accuracy": accuracy},
            num_workers=0,
        )
        _ = nn  # keep unused import warning silenced

        strategy = RegularExecutionStrategy()
        fold_results = strategy.run(ctx)
        assert len(fold_results) == 1
        fr = fold_results[0]
        assert fr.failed is False
        # fold_info comes from SplitResult
        assert "fold_idx" in fr.fold_info
        # Per-fold directory created
        assert (tmp_path / "fold_0").exists()
