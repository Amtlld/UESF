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

import contextlib
import json
from collections import deque
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from uesf.core.logging import get_logger
from uesf.experiment.logger import TrainingLogger

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

        # Per-run logging context — overridden in run(); defaulted here so
        # train_epoch() can be called in isolation (tests) without errors.
        self._training_logger: TrainingLogger | None = None
        self._log_every_n_steps: int | None = None
        self._train_step_filter: list[str] | None = None
        self._log_lr: bool = True
        self._train_evaluator: Any | None = None
        self._train_preds: list[torch.Tensor] = []
        self._train_targets: list[torch.Tensor] = []
        self._train_missing_pred_count: int = 0

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
            batch = _move_batch_to_device(batch, self.device)

            step_result = self.trainer.training_step(batch, batch_idx, optimizer)

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

            if self._train_evaluator is not None:
                preds = step_result.get("preds")
                targets = step_result.get("targets")
                if isinstance(preds, torch.Tensor) and isinstance(targets, torch.Tensor):
                    self._train_preds.append(preds.detach().cpu())
                    self._train_targets.append(targets.detach().cpu())
                else:
                    self._train_missing_pred_count += 1

            if self._training_logger is not None and self._log_every_n_steps:
                global_step = epoch * len(train_loader) + batch_idx
                if (global_step + 1) % self._log_every_n_steps == 0:
                    step_scalars: dict[str, float] = {}
                    for key, value in step_result.items():
                        if isinstance(value, (int, float)) and _keep(
                            key, self._train_step_filter
                        ):
                            step_scalars[f"step/{key}"] = float(value)
                    if self._log_lr:
                        step_scalars["step/lr"] = float(
                            optimizer.param_groups[0]["lr"]
                        )
                    if step_scalars:
                        self._training_logger.log_scalars(
                            step_scalars, step=global_step
                        )

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
        training_logger: TrainingLogger | None = None,
        log_every_n_epochs: int = 1,
        log_every_n_steps: int | None = None,
        train_step_filter: list[str] | None = None,
        val_metrics_filter: list[str] | None = None,
        train_evaluator: Any | None = None,
        test_evaluator: Any | None = None,
        test_loader: Any | None = None,
        log_lr: bool = True,
        last_n_test_aggregate: int | None = None,
    ) -> dict[str, Any]:
        """Run the full training loop.

        Args:
            training_logger: Optional :class:`TrainingLogger`; wrapped in a
                context manager so ``close()`` still runs on exception.
            log_every_n_epochs: Epoch-level logging cadence.
            log_every_n_steps: Step-level (per-batch) logging cadence; ``None``
                disables step-level writes.
            train_step_filter: Whitelist of ``training_step`` return keys; when
                given, only matching scalars are written (both step- and
                epoch-level). ``None`` keeps every numeric scalar.
            val_metrics_filter: Whitelist (bare names) over
                ``evaluation.metrics``; ``None`` keeps all.
            train_evaluator: Optional :class:`Evaluator` computed on training
                ``preds``/``targets`` collected from ``training_step``. Tags
                are ``train_<name>``.
            test_evaluator: Optional :class:`Evaluator` computed on
                ``test_loader`` each ``log_every_n_epochs``. Tags are
                ``test_<name>``. Caller is responsible for warning about
                data-leakage risk before invoking.
            test_loader: DataLoader used only when ``test_evaluator`` is set
                or ``last_n_test_aggregate`` is enabled.
            log_lr: Whether to write learning rate scalars.
            last_n_test_aggregate: When set to a positive int N, runs
                ``test_loader`` at the end of every epoch and keeps the
                predictions/targets from the last N epochs (a sliding window
                via ``deque(maxlen=N)``). The collected tensors are returned
                under ``last_n_test_preds`` / ``last_n_test_targets`` so the
                caller can concatenate them and recompute aggregated metrics.
                Requires ``test_loader`` to be a non-empty DataLoader.

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

        if last_n_test_aggregate is not None and last_n_test_aggregate >= 1:
            if test_loader is None or len(test_loader) == 0:
                raise ValueError(
                    "last_n_test_aggregate is set but test_loader is empty/None — "
                    "evaluation.test_with={'last': N} needs a non-empty test set.",
                )
            last_n_window: deque[tuple[list[torch.Tensor], list[torch.Tensor]]] | None = deque(
                maxlen=last_n_test_aggregate,
            )
        else:
            last_n_window = None

        best_metric_value = None
        best_metrics: dict[str, Any] = {}
        history: list[dict[str, Any]] = []

        self._training_logger = training_logger
        self._log_every_n_steps = log_every_n_steps
        self._train_step_filter = train_step_filter
        self._log_lr = log_lr
        self._train_evaluator = train_evaluator

        logger_cm = training_logger if training_logger is not None else contextlib.nullcontext()

        self.trainer.on_fit_start(train_loader, self.epochs)

        with logger_cm:
            for epoch in range(self.epochs):
                self._train_preds = []
                self._train_targets = []
                self._train_missing_pred_count = 0

                train_metrics = self.train_epoch(train_loader, optimizer, epoch)

                val_metrics: dict[str, Any] = {}
                if val_loader and len(val_loader) > 0:
                    val_metrics, _, _ = self.validate_epoch(val_loader)
                    val_metrics = {f"val_{k}": v for k, v in val_metrics.items()}

                train_computed: dict[str, Any] = {}
                if train_evaluator is not None:
                    if self._train_preds and self._train_targets:
                        raw_train = train_evaluator.compute_epoch_metrics(
                            self._train_preds, self._train_targets
                        )
                        train_computed = {f"train_{k}": v for k, v in raw_train.items()}
                    elif epoch == 0 and self._train_missing_pred_count > 0:
                        logger.warning(
                            "training.logging.train_metrics is configured but "
                            "training_step did not return 'preds'/'targets' — "
                            "skipping train-side metric computation. Update your "
                            "Trainer to return these tensors to enable train metrics."
                        )

                should_epoch_log = (epoch + 1) % log_every_n_epochs == 0

                test_computed: dict[str, Any] = {}
                want_test_eval = (
                    test_evaluator is not None
                    and test_loader is not None
                    and len(test_loader) > 0
                    and should_epoch_log
                )
                if want_test_eval or last_n_window is not None:
                    test_preds_epoch, test_targets_epoch = self._collect_loader(
                        test_loader,
                    )
                    if want_test_eval:
                        raw_test = test_evaluator.compute_epoch_metrics(
                            test_preds_epoch, test_targets_epoch,
                        )
                        test_computed = {f"test_{k}": v for k, v in raw_test.items()}
                    if last_n_window is not None:
                        last_n_window.append((test_preds_epoch, test_targets_epoch))

                if scheduler is not None:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        monitor = (
                            early_stopping_config.get("metric", "val_loss")
                            if early_stopping_config
                            else "val_loss"
                        )
                        monitor_val = val_metrics.get(monitor, train_metrics.get("loss", 0.0))
                        if isinstance(monitor_val, (int, float)):
                            scheduler.step(monitor_val)
                    else:
                        scheduler.step()

                epoch_result = {
                    **train_metrics,
                    **train_computed,
                    **val_metrics,
                    **test_computed,
                    "epoch": epoch,
                }
                history.append(epoch_result)

                logger.info(
                    "Epoch %d/%d - %s",
                    epoch + 1,
                    self.epochs,
                    _format_metrics(epoch_result),
                )

                if training_logger is not None and should_epoch_log:
                    scalars: dict[str, float] = {}
                    for k, v in train_metrics.items():
                        if isinstance(v, (int, float)) and _keep(k, train_step_filter):
                            scalars[k] = float(v)
                    for k, v in train_computed.items():
                        if isinstance(v, (int, float)):
                            scalars[k] = float(v)
                    for k, v in val_metrics.items():
                        if isinstance(v, (int, float)):
                            bare = k[4:] if k.startswith("val_") else k
                            if _keep(bare, val_metrics_filter):
                                scalars[k] = float(v)
                    for k, v in test_computed.items():
                        if isinstance(v, (int, float)):
                            scalars[k] = float(v)
                    if log_lr:
                        scalars["lr"] = float(optimizer.param_groups[0]["lr"])
                    training_logger.log_scalars(scalars, step=epoch)

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
                            logger.info(
                                "Saved best checkpoint (%s=%.4f)",
                                checkpoint_metric,
                                metric_val,
                            )

                if early_stopper and early_stopping_config:
                    monitor = early_stopping_config["metric"]
                    combined = {**train_metrics, **val_metrics}
                    if monitor in combined and isinstance(combined[monitor], (int, float)):
                        if early_stopper.step(combined[monitor]):
                            logger.info(
                                "Early stopping triggered at epoch %d", epoch + 1
                            )
                            break

        last_n_test_preds: list[torch.Tensor] = []
        last_n_test_targets: list[torch.Tensor] = []
        last_n_epochs_used = 0
        if last_n_window is not None:
            last_n_epochs_used = len(last_n_window)
            for preds_epoch, targets_epoch in last_n_window:
                last_n_test_preds.extend(preds_epoch)
                last_n_test_targets.extend(targets_epoch)
            if (
                last_n_test_aggregate is not None
                and last_n_epochs_used < last_n_test_aggregate
            ):
                logger.warning(
                    "evaluation.test_with={'last': %d} but only %d epoch(s) ran "
                    "(early stop / shorter schedule) — aggregating over the "
                    "available epochs.",
                    last_n_test_aggregate,
                    last_n_epochs_used,
                )

        return {
            "history": history,
            "best_metrics": best_metrics or (history[-1] if history else {}),
            "epochs_run": len(history),
            "last_n_test_preds": last_n_test_preds,
            "last_n_test_targets": last_n_test_targets,
            "last_n_epochs_used": last_n_epochs_used,
        }

    @torch.no_grad()
    def _collect_loader(
        self,
        loader: Any,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Run forward pass over ``loader`` and return per-batch preds/targets.

        Identical data path to :meth:`validate_epoch` but skips metric
        computation — used when the caller wants raw tensors for downstream
        aggregation (e.g. last-N-epoch test concatenation).
        """
        self.trainer.model.eval()
        preds_list: list[torch.Tensor] = []
        targets_list: list[torch.Tensor] = []
        for batch_idx, batch in enumerate(loader):
            batch = _move_batch_to_device(batch, self.device)
            step_result = self.trainer.validation_step(batch, batch_idx)
            if "preds" in step_result and step_result["preds"] is not None:
                preds_list.append(step_result["preds"].detach().cpu())
            if "targets" in step_result and step_result["targets"] is not None:
                targets_list.append(step_result["targets"].detach().cpu())
        return preds_list, targets_list

    @torch.no_grad()
    def _evaluate_on_loader(
        self,
        loader: Any,
        evaluator: Any,
    ) -> tuple[dict[str, Any], list[torch.Tensor], list[torch.Tensor]]:
        """Run an eval pass on ``loader`` with a caller-supplied evaluator.

        Mirrors :meth:`validate_epoch` but lets the caller swap evaluators —
        useful for computing a different metric subset on the test set during
        training.
        """
        self.trainer.model.eval()
        preds_list: list[torch.Tensor] = []
        targets_list: list[torch.Tensor] = []
        for batch_idx, batch in enumerate(loader):
            batch = _move_batch_to_device(batch, self.device)
            step_result = self.trainer.validation_step(batch, batch_idx)
            if "preds" in step_result and step_result["preds"] is not None:
                preds_list.append(step_result["preds"].detach().cpu())
            if "targets" in step_result and step_result["targets"] is not None:
                targets_list.append(step_result["targets"].detach().cpu())
        metrics = evaluator.compute_epoch_metrics(preds_list, targets_list)
        return metrics, preds_list, targets_list


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move a multi-channel batch dict to the target device."""
    moved = {}
    for name, (data, labels) in batch.items():
        moved[name] = (data.to(device), labels.to(device))
    return moved


def _keep(name: str, whitelist: list[str] | None) -> bool:
    """``None`` whitelist lets everything through; otherwise membership check."""
    return whitelist is None or name in whitelist


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
