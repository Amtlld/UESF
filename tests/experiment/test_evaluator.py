"""Tests for Evaluator."""

from __future__ import annotations

import torch

from uesf.components.builtin_metrics import accuracy, f1_score
from uesf.experiment.evaluator import Evaluator


class TestEvaluator:
    def test_compute_epoch_metrics(self):
        evaluator = Evaluator({"accuracy": accuracy, "f1_score": f1_score})

        preds = [torch.tensor([0, 1, 0]), torch.tensor([1, 0, 1])]
        targets = [torch.tensor([0, 1, 0]), torch.tensor([1, 0, 1])]

        metrics = evaluator.compute_epoch_metrics(preds, targets)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_score"] == 1.0

    def test_empty_preds(self):
        evaluator = Evaluator({"accuracy": accuracy})
        metrics = evaluator.compute_epoch_metrics([], [])
        assert metrics == {}

    def test_metric_failure_handled(self):
        def bad_metric(preds, targets):
            raise ValueError("intentional error")

        evaluator = Evaluator({"bad": bad_metric, "accuracy": accuracy})
        preds = [torch.tensor([0, 1])]
        targets = [torch.tensor([0, 1])]

        metrics = evaluator.compute_epoch_metrics(preds, targets)
        assert metrics["bad"] is None
        assert metrics["accuracy"] == 1.0


class TestAggregation:
    def test_mean_std_aggregation(self):
        fold_results = [
            {"accuracy": 0.9, "f1_score": 0.85},
            {"accuracy": 0.8, "f1_score": 0.75},
            {"accuracy": 0.85, "f1_score": 0.8},
        ]
        result = Evaluator.aggregate_fold_results(fold_results, mode="mean_std")
        assert "accuracy" in result
        assert "mean" in result["accuracy"]
        assert "std" in result["accuracy"]
        assert abs(result["accuracy"]["mean"] - 0.85) < 1e-6

    def test_concat_aggregation(self):
        fold_preds = [
            [torch.tensor([0, 1])],
            [torch.tensor([1, 0])],
        ]
        fold_targets = [
            [torch.tensor([0, 1])],
            [torch.tensor([1, 0])],
        ]

        result = Evaluator.aggregate_fold_results(
            [{}, {}],
            mode="concat",
            fold_preds=fold_preds,
            fold_targets=fold_targets,
            metric_funcs={"accuracy": accuracy},
        )
        assert result["accuracy"] == 1.0
