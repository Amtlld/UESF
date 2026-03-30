"""Built-in evaluation metrics for UESF.

All metrics follow the unified signature:
    def metric_func(preds: Tensor, targets: Tensor, **kwargs) -> float | dict

preds and targets are epoch-level aggregated tensors (not per-batch).
"""

from __future__ import annotations

from typing import Any

import torch


def accuracy(preds: torch.Tensor, targets: torch.Tensor, **kwargs: Any) -> float:
    """Compute classification accuracy.

    Args:
        preds: Predicted class indices or logits (N,) or (N, C).
        targets: Ground truth class indices (N,).

    Returns:
        Accuracy as a float in [0, 1].
    """
    if preds.dim() > 1:
        preds = preds.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total if total > 0 else 0.0


def f1_score(
    preds: torch.Tensor,
    targets: torch.Tensor,
    average: str = "macro",
    **kwargs: Any,
) -> float:
    """Compute F1 score.

    Args:
        preds: Predicted class indices or logits.
        targets: Ground truth class indices.
        average: 'macro', 'micro', or 'weighted'.

    Returns:
        F1 score as a float.
    """
    if preds.dim() > 1:
        preds = preds.argmax(dim=1)

    classes = torch.unique(torch.cat([preds, targets]))

    if average == "micro":
        tp = (preds == targets).sum().float()
        total = targets.numel()
        # micro F1 = micro precision = micro recall = accuracy
        return (tp / total).item() if total > 0 else 0.0

    f1s = []
    weights = []
    for c in classes:
        tp = ((preds == c) & (targets == c)).sum().float()
        fp = ((preds == c) & (targets != c)).sum().float()
        fn = ((preds != c) & (targets == c)).sum().float()

        prec = tp / (tp + fp) if (tp + fp) > 0 else torch.tensor(0.0)
        rec = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.0)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else torch.tensor(0.0)

        f1s.append(f1.item())
        weights.append((targets == c).sum().item())

    if not f1s:
        return 0.0

    if average == "weighted":
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0
        return sum(f * w for f, w in zip(f1s, weights)) / total_weight

    # macro
    return sum(f1s) / len(f1s)


def precision(
    preds: torch.Tensor,
    targets: torch.Tensor,
    average: str = "macro",
    **kwargs: Any,
) -> float:
    """Compute precision score.

    Args:
        preds: Predicted class indices or logits.
        targets: Ground truth class indices.
        average: 'macro', 'micro', or 'weighted'.

    Returns:
        Precision as a float.
    """
    if preds.dim() > 1:
        preds = preds.argmax(dim=1)

    classes = torch.unique(torch.cat([preds, targets]))

    if average == "micro":
        tp = (preds == targets).sum().float()
        total_pred = targets.numel()
        return (tp / total_pred).item() if total_pred > 0 else 0.0

    precisions = []
    weights = []
    for c in classes:
        tp = ((preds == c) & (targets == c)).sum().float()
        fp = ((preds == c) & (targets != c)).sum().float()
        prec = tp / (tp + fp) if (tp + fp) > 0 else torch.tensor(0.0)
        precisions.append(prec.item())
        weights.append((targets == c).sum().item())

    if not precisions:
        return 0.0

    if average == "weighted":
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0
        return sum(p * w for p, w in zip(precisions, weights)) / total_weight

    return sum(precisions) / len(precisions)


def recall(
    preds: torch.Tensor,
    targets: torch.Tensor,
    average: str = "macro",
    **kwargs: Any,
) -> float:
    """Compute recall score.

    Args:
        preds: Predicted class indices or logits.
        targets: Ground truth class indices.
        average: 'macro', 'micro', or 'weighted'.

    Returns:
        Recall as a float.
    """
    if preds.dim() > 1:
        preds = preds.argmax(dim=1)

    classes = torch.unique(torch.cat([preds, targets]))

    if average == "micro":
        tp = (preds == targets).sum().float()
        total = targets.numel()
        return (tp / total).item() if total > 0 else 0.0

    recalls = []
    weights = []
    for c in classes:
        tp = ((preds == c) & (targets == c)).sum().float()
        fn = ((preds != c) & (targets == c)).sum().float()
        rec = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.0)
        recalls.append(rec.item())
        weights.append((targets == c).sum().item())

    if not recalls:
        return 0.0

    if average == "weighted":
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0
        return sum(r * w for r, w in zip(recalls, weights)) / total_weight

    return sum(recalls) / len(recalls)


def auroc(
    preds: torch.Tensor,
    targets: torch.Tensor,
    **kwargs: Any,
) -> float:
    """Compute Area Under ROC Curve (binary or macro-averaged multi-class).

    Args:
        preds: Predicted probabilities or logits. For binary: (N,) or (N, 2).
               For multi-class: (N, C) with softmax/logit scores.
        targets: Ground truth class indices (N,).

    Returns:
        AUROC as a float.
    """
    if preds.dim() == 1:
        # Binary case
        return _binary_auroc(preds, targets)

    n_classes = preds.shape[1]
    if n_classes == 2:
        return _binary_auroc(preds[:, 1], targets)

    # Multi-class: macro-average one-vs-rest
    aurocs = []
    for c in range(n_classes):
        binary_targets = (targets == c).long()
        if binary_targets.sum() == 0 or binary_targets.sum() == len(binary_targets):
            continue  # Skip classes not present in targets
        aurocs.append(_binary_auroc(preds[:, c], binary_targets))

    return sum(aurocs) / len(aurocs) if aurocs else 0.0


def _binary_auroc(scores: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute binary AUROC using the trapezoidal rule."""
    sorted_indices = torch.argsort(scores, descending=True)
    sorted_targets = targets[sorted_indices].float()

    n_pos = sorted_targets.sum().item()
    n_neg = len(sorted_targets) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.0

    tpr = torch.cumsum(sorted_targets, dim=0) / n_pos
    fpr = torch.cumsum(1 - sorted_targets, dim=0) / n_neg

    # Prepend (0, 0)
    tpr = torch.cat([torch.tensor([0.0]), tpr])
    fpr = torch.cat([torch.tensor([0.0]), fpr])

    # Trapezoidal rule
    auc = torch.trapezoid(tpr, fpr).item()
    return auc


def confusion_matrix(
    preds: torch.Tensor,
    targets: torch.Tensor,
    **kwargs: Any,
) -> dict[str, Any]:
    """Compute confusion matrix.

    Args:
        preds: Predicted class indices or logits.
        targets: Ground truth class indices.

    Returns:
        Dict with 'matrix' (list of lists) and 'labels' (list of class indices).
    """
    if preds.dim() > 1:
        preds = preds.argmax(dim=1)

    classes = torch.unique(torch.cat([preds, targets])).sort().values
    n_classes = len(classes)
    class_to_idx = {c.item(): i for i, c in enumerate(classes)}

    matrix = torch.zeros(n_classes, n_classes, dtype=torch.long)
    for p, t in zip(preds, targets):
        pi = class_to_idx[p.item()]
        ti = class_to_idx[t.item()]
        matrix[ti, pi] += 1

    return {
        "matrix": matrix.tolist(),
        "labels": [c.item() for c in classes],
    }


# Registry of built-in metrics for name resolution
BUILTIN_METRICS: dict[str, callable] = {
    "accuracy": accuracy,
    "f1_score": f1_score,
    "precision": precision,
    "recall": recall,
    "auroc": auroc,
    "confusion_matrix": confusion_matrix,
}
