"""Tests for built-in evaluation metrics."""

from __future__ import annotations

import torch

from uesf.components.builtin_metrics import (
    BUILTIN_METRICS,
    accuracy,
    auroc,
    confusion_matrix,
    f1_score,
    precision,
    recall,
)


class TestAccuracy:
    def test_perfect_accuracy(self):
        preds = torch.tensor([0, 1, 2, 0, 1])
        targets = torch.tensor([0, 1, 2, 0, 1])
        assert accuracy(preds, targets) == 1.0

    def test_zero_accuracy(self):
        preds = torch.tensor([0, 0, 0])
        targets = torch.tensor([1, 1, 1])
        assert accuracy(preds, targets) == 0.0

    def test_partial_accuracy(self):
        preds = torch.tensor([0, 1, 0, 1])
        targets = torch.tensor([0, 1, 1, 0])
        assert accuracy(preds, targets) == 0.5

    def test_logits_input(self):
        preds = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
        targets = torch.tensor([0, 1, 0])
        assert accuracy(preds, targets) == 1.0


class TestF1Score:
    def test_perfect_f1(self):
        preds = torch.tensor([0, 1, 0, 1])
        targets = torch.tensor([0, 1, 0, 1])
        assert f1_score(preds, targets) == 1.0

    def test_macro_average(self):
        preds = torch.tensor([0, 1, 0, 1])
        targets = torch.tensor([0, 1, 0, 1])
        result = f1_score(preds, targets, average="macro")
        assert result == 1.0

    def test_micro_average(self):
        preds = torch.tensor([0, 1, 0, 1])
        targets = torch.tensor([0, 1, 0, 1])
        result = f1_score(preds, targets, average="micro")
        assert result == 1.0

    def test_weighted_average(self):
        preds = torch.tensor([0, 1, 0, 1])
        targets = torch.tensor([0, 1, 0, 1])
        result = f1_score(preds, targets, average="weighted")
        assert result == 1.0


class TestPrecision:
    def test_perfect_precision(self):
        preds = torch.tensor([0, 1, 0, 1])
        targets = torch.tensor([0, 1, 0, 1])
        assert precision(preds, targets) == 1.0

    def test_zero_precision(self):
        preds = torch.tensor([1, 1, 1])
        targets = torch.tensor([0, 0, 0])
        # class 0: prec=0 (0 TP), class 1: prec=0 (0 TP, 3 FP)
        assert precision(preds, targets) == 0.0


class TestRecall:
    def test_perfect_recall(self):
        preds = torch.tensor([0, 1, 0, 1])
        targets = torch.tensor([0, 1, 0, 1])
        assert recall(preds, targets) == 1.0

    def test_zero_recall(self):
        preds = torch.tensor([1, 1, 1])
        targets = torch.tensor([0, 0, 0])
        # class 0: recall=0, class 1: recall=nan->0
        assert recall(preds, targets) == 0.0


class TestAUROC:
    def test_perfect_binary(self):
        scores = torch.tensor([0.9, 0.8, 0.2, 0.1])
        targets = torch.tensor([1, 1, 0, 0])
        result = auroc(scores, targets)
        assert result == 1.0

    def test_random_binary(self):
        scores = torch.tensor([0.5, 0.5, 0.5, 0.5])
        targets = torch.tensor([1, 0, 1, 0])
        result = auroc(scores, targets)
        assert 0.0 <= result <= 1.0

    def test_multiclass_logits(self):
        preds = torch.tensor([
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ])
        targets = torch.tensor([0, 1, 2])
        result = auroc(preds, targets)
        assert result == 1.0


class TestConfusionMatrix:
    def test_perfect_classification(self):
        preds = torch.tensor([0, 1, 2, 0, 1, 2])
        targets = torch.tensor([0, 1, 2, 0, 1, 2])
        result = confusion_matrix(preds, targets)
        assert result["labels"] == [0, 1, 2]
        assert result["matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]

    def test_with_errors(self):
        preds = torch.tensor([0, 1, 0])
        targets = torch.tensor([0, 0, 1])
        result = confusion_matrix(preds, targets)
        # Row=true, Col=pred
        # Class 0: 1 correct, 1 predicted as class 1
        # Class 1: 1 predicted as class 0, 0 correct
        assert result["matrix"] == [[1, 1], [1, 0]]

    def test_from_logits(self):
        preds = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
        targets = torch.tensor([0, 1])
        result = confusion_matrix(preds, targets)
        assert result["matrix"] == [[1, 0], [0, 1]]


class TestBuiltinMetricsRegistry:
    def test_all_metrics_present(self):
        expected = {"accuracy", "f1_score", "precision", "recall", "auroc", "confusion_matrix"}
        assert set(BUILTIN_METRICS.keys()) == expected

    def test_all_callable(self):
        for name, func in BUILTIN_METRICS.items():
            assert callable(func), f"{name} is not callable"
