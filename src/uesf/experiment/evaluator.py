"""UESF Evaluator - epoch-level metric aggregation.

Collects per-batch predictions and targets, then computes metrics
on the full epoch-level concatenated tensors. Supports concat and
mean_std aggregation for K-Fold results.
"""

from __future__ import annotations

from typing import Any

import torch

from uesf.core.logging import get_logger

logger = get_logger("experiment.evaluator")


class Evaluator:
    """Evaluates epoch-level predictions against targets.

    Args:
        metric_funcs: Dict mapping metric names to callable functions.
            Each function has signature: (preds, targets, **kwargs) -> float | dict
    """

    def __init__(self, metric_funcs: dict[str, callable]) -> None:
        self.metric_funcs = metric_funcs

    def compute_epoch_metrics(
        self,
        all_preds: list[torch.Tensor],
        all_targets: list[torch.Tensor],
    ) -> dict[str, Any]:
        """Compute all metrics on epoch-level aggregated tensors.

        Args:
            all_preds: List of per-batch prediction tensors.
            all_targets: List of per-batch target tensors.

        Returns:
            Dict mapping metric names to computed values.
        """
        if not all_preds:
            return {}

        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)

        results = {}
        for name, func in self.metric_funcs.items():
            try:
                results[name] = func(preds, targets)
            except Exception as exc:
                logger.warning("Metric '%s' failed: %s", name, exc)
                results[name] = None

        return results

    @staticmethod
    def aggregate_fold_results(
        fold_results: list[dict[str, Any]],
        mode: str = "concat",
        fold_preds: list[list[torch.Tensor]] | None = None,
        fold_targets: list[list[torch.Tensor]] | None = None,
        metric_funcs: dict[str, callable] | None = None,
    ) -> dict[str, Any]:
        """Aggregate results across K folds.

        Args:
            fold_results: List of per-fold metric dicts (used for mean_std mode).
            mode: "concat" (recommended) or "mean_std".
            fold_preds: Per-fold prediction tensors (for concat mode).
            fold_targets: Per-fold target tensors (for concat mode).
            metric_funcs: Metric functions (for concat mode recomputation).

        Returns:
            Aggregated result dict.
        """
        if mode == "concat" and fold_preds and fold_targets and metric_funcs:
            # Concatenate all folds and recompute
            all_preds = []
            all_targets = []
            for fp in fold_preds:
                all_preds.extend(fp)
            for ft in fold_targets:
                all_targets.extend(ft)

            evaluator = Evaluator(metric_funcs)
            return evaluator.compute_epoch_metrics(all_preds, all_targets)

        if mode == "mean_std":
            if not fold_results:
                return {}

            # Collect numeric metrics only
            aggregated = {}
            metric_names = set()
            for fr in fold_results:
                for k, v in fr.items():
                    if isinstance(v, (int, float)):
                        metric_names.add(k)

            for name in metric_names:
                values = [fr[name] for fr in fold_results if name in fr and isinstance(fr[name], (int, float))]
                if values:
                    mean = sum(values) / len(values)
                    std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
                    aggregated[name] = {"mean": mean, "std": std}

            return aggregated

        # Fallback: return last fold
        return fold_results[-1] if fold_results else {}
