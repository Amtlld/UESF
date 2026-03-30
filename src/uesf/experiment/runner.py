"""UESF Runner - training loop orchestration.

The Runner is deliberately thin: it delegates batch processing to the Trainer
and metric computation to the Evaluator. The Runner handles:
- Epoch loop
- Gradient clipping
- Early stopping
- Checkpoint saving
- State machine (PENDING → RUNNING → COMPLETED/FAILED)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from uesf.core.logging import get_logger

logger = get_logger("experiment.runner")


class EarlyStopping:
    """Early stopping monitor."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value: float | None = None
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Check if training should stop.

        Returns:
            True if training should stop.
        """
        if self.best_value is None:
            self.best_value = value
            return False

        if self.mode == "min":
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True

        return False


class Runner:
    """Orchestrates the training loop.

    Args:
        trainer: The Trainer instance (owns forward/backward/optimizer steps).
        evaluator: The Evaluator instance for metric computation.
        device: The torch device.
        config: Training configuration dict from experiment YAML.
    """

    def __init__(
        self,
        trainer: Any,
        evaluator: Any,
        device: torch.device,
        config: dict[str, Any],
    ) -> None:
        self.trainer = trainer
        self.evaluator = evaluator
        self.device = device
        self.config = config

        self.epochs = config.get("epochs", 10)
        self.gradient_clip = config.get("gradient_clip")

    def train_epoch(
        self,
        train_loader: Any,
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ) -> dict[str, float]:
        """Run one training epoch.

        Returns:
            Dict of averaged training metrics over the epoch.
        """
        self.trainer.model.train()
        epoch_metrics: dict[str, list[float]] = {}

        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            batch = _move_batch_to_device(batch, self.device)

            step_result = self.trainer.training_step(batch, batch_idx, optimizer)

            # Gradient clipping (after backward, before optimizer.step happens in trainer)
            if self.gradient_clip:
                max_norm = self.gradient_clip.get("max_norm", 1.0)
                norm_type = self.gradient_clip.get("norm_type", 2)
                nn.utils.clip_grad_norm_(
                    self.trainer.model.parameters(),
                    max_norm=max_norm,
                    norm_type=norm_type,
                )

            for key, value in step_result.items():
                if isinstance(value, (int, float)):
                    epoch_metrics.setdefault(key, []).append(value)

        # Average training metrics
        return {k: sum(v) / len(v) for k, v in epoch_metrics.items()}

    @torch.no_grad()
    def validate_epoch(
        self,
        val_loader: Any,
    ) -> tuple[dict[str, Any], list[torch.Tensor], list[torch.Tensor]]:
        """Run one validation epoch.

        Returns:
            Tuple of (computed metrics dict, all_preds list, all_targets list).
        """
        self.trainer.model.eval()
        all_preds = []
        all_targets = []

        for batch_idx, batch in enumerate(val_loader):
            batch = _move_batch_to_device(batch, self.device)
            step_result = self.trainer.validation_step(batch, batch_idx)

            if "preds" in step_result and step_result["preds"] is not None:
                all_preds.append(step_result["preds"].detach().cpu())
            if "targets" in step_result and step_result["targets"] is not None:
                all_targets.append(step_result["targets"].detach().cpu())

        metrics = self.evaluator.compute_epoch_metrics(all_preds, all_targets)
        return metrics, all_preds, all_targets

    def run(
        self,
        train_loader: Any,
        val_loader: Any | None,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None = None,
        checkpoint_dir: Path | None = None,
        checkpoint_metric: str | None = None,
        early_stopping_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the full training loop.

        Returns:
            Dict with training history and best metrics.
        """
        early_stopper = None
        if early_stopping_config:
            early_stopper = EarlyStopping(
                patience=early_stopping_config.get("patience", 10),
                min_delta=early_stopping_config.get("min_delta", 0.0),
                mode=early_stopping_config.get("mode", "min"),
            )

        best_metric_value = None
        best_metrics = {}
        history: list[dict[str, Any]] = []

        for epoch in range(self.epochs):
            # Training
            train_metrics = self.train_epoch(train_loader, optimizer, epoch)

            # Validation
            val_metrics = {}
            if val_loader and len(val_loader) > 0:
                val_metrics, _, _ = self.validate_epoch(val_loader)
                # Prefix val metrics
                val_metrics = {f"val_{k}": v for k, v in val_metrics.items()}

            # Scheduler step
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    monitor = early_stopping_config.get("monitor", "val_loss") if early_stopping_config else "val_loss"
                    monitor_val = val_metrics.get(monitor, train_metrics.get("loss", 0.0))
                    if isinstance(monitor_val, (int, float)):
                        scheduler.step(monitor_val)
                else:
                    scheduler.step()

            # Combine metrics
            epoch_result = {**train_metrics, **val_metrics, "epoch": epoch}
            history.append(epoch_result)

            logger.info("Epoch %d/%d - %s", epoch + 1, self.epochs, _format_metrics(epoch_result))

            # Checkpoint saving
            if checkpoint_dir and checkpoint_metric and checkpoint_metric in val_metrics:
                metric_val = val_metrics[checkpoint_metric]
                if isinstance(metric_val, (int, float)):
                    if best_metric_value is None or metric_val > best_metric_value:
                        best_metric_value = metric_val
                        best_metrics = {**train_metrics, **val_metrics}
                        checkpoint_dir.mkdir(parents=True, exist_ok=True)
                        torch.save(
                            self.trainer.model.state_dict(),
                            checkpoint_dir / "best_model.pt",
                        )
                        logger.info("Saved best checkpoint (%s=%.4f)", checkpoint_metric, metric_val)

            # Early stopping
            if early_stopper and early_stopping_config:
                monitor = early_stopping_config["monitor"]
                combined = {**train_metrics, **val_metrics}
                if monitor in combined and isinstance(combined[monitor], (int, float)):
                    if early_stopper.step(combined[monitor]):
                        logger.info("Early stopping triggered at epoch %d", epoch + 1)
                        break

        return {
            "history": history,
            "best_metrics": best_metrics or (history[-1] if history else {}),
            "epochs_run": len(history),
        }


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move a multi-channel batch dict to the target device."""
    moved = {}
    for name, (data, labels) in batch.items():
        moved[name] = (data.to(device), labels.to(device))
    return moved


def _format_metrics(metrics: dict[str, Any]) -> str:
    """Format metrics dict for logging."""
    parts = []
    for k, v in metrics.items():
        if k == "epoch":
            continue
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        elif isinstance(v, dict):
            parts.append(f"{k}={json.dumps(v)}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)
