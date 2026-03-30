"""UESF Experiment Manager - orchestrates the full experiment lifecycle.

Handles: add, remove, run, query experiments.
The run() method orchestrates the complete pipeline:
  config load → component init → split → transform → train → evaluate → save
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from uesf.components.builtin_mappings import resolve_optimizer, resolve_scheduler
from uesf.core.config import ConfigManager
from uesf.core.database import DatabaseManager
from uesf.core.exceptions import (
    ComponentNotFoundError,
    YAMLParseError,
)
from uesf.core.logging import get_logger
from uesf.experiment.dataloader_builder import build_dataloaders
from uesf.experiment.dataset import EEGDataset
from uesf.experiment.evaluator import Evaluator
from uesf.experiment.runner import Runner
from uesf.experiment.splitter import create_splitter
from uesf.experiment.transforms import create_transform
from uesf.managers.metric_manager import MetricManager
from uesf.managers.model_manager import ModelManager
from uesf.managers.project_manager import ProjectManager
from uesf.managers.trainer_manager import TrainerManager

logger = get_logger("manager.experiment")


class ExperimentManager:
    """Manages experiment lifecycle."""

    def __init__(
        self,
        db: DatabaseManager,
        config: ConfigManager,
        project_manager: ProjectManager,
        model_manager: ModelManager,
        trainer_manager: TrainerManager,
        metric_manager: MetricManager,
    ) -> None:
        self.db = db
        self.config = config
        self.project_manager = project_manager
        self.model_manager = model_manager
        self.trainer_manager = trainer_manager
        self.metric_manager = metric_manager

    def add(
        self,
        project_dir: Path,
        experiment_name: str | None = None,
        from_existing: str | None = None,
        description: str | None = None,
    ) -> Path:
        """Add a new experiment configuration file.

        Args:
            project_dir: Project directory.
            experiment_name: Name for the experiment. Auto-generated if None.
            from_existing: Copy config from an existing experiment.
            description: Experiment description.

        Returns:
            Path to the created experiment YAML file.
        """
        project_config = self.project_manager.load(project_dir)
        project_name = project_config["project-name"]
        project_dir = Path(project_dir).resolve()

        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"{project_name}_{timestamp}"

        experiments_dir = project_dir / "experiments"
        experiments_dir.mkdir(exist_ok=True)
        yml_path = experiments_dir / f"{experiment_name}.yml"

        if from_existing:
            src = experiments_dir / f"{from_existing}.yml"
            if not src.exists():
                raise ComponentNotFoundError(
                    f"Source experiment '{from_existing}' not found at '{src}'",
                    hint="Check the experiment name.",
                )
            content = yaml.safe_load(src.read_text(encoding="utf-8"))
            content["name"] = experiment_name
            if description:
                content["description"] = description
            yml_path.write_text(
                yaml.dump(content, default_flow_style=False, allow_unicode=True),
                encoding="utf-8",
            )
        else:
            template = _experiment_template(experiment_name, description)
            yml_path.write_text(template, encoding="utf-8")

        logger.info("Created experiment '%s' at '%s'", experiment_name, yml_path)
        return yml_path

    def list(self, project_dir: Path) -> list[dict[str, Any]]:
        """List all experiments for a project."""
        project_config = self.project_manager.load(project_dir)
        project_name = project_config["project-name"]
        return self.db.fetch_all(
            "SELECT * FROM experiments WHERE project_name = ? ORDER BY created_at DESC",
            (project_name,),
        )

    def remove(
        self,
        project_dir: Path,
        experiment_name: str,
        results_only: bool = False,
    ) -> None:
        """Remove an experiment or just its results."""
        project_config = self.project_manager.load(project_dir)
        project_name = project_config["project-name"]
        project_dir = Path(project_dir).resolve()

        # Remove results directory
        results_dir = project_dir / "experiments" / "results" / experiment_name
        if results_dir.exists():
            shutil.rmtree(results_dir)
            logger.info("Removed results for '%s'", experiment_name)

        if not results_only:
            # Remove YAML config
            yml_path = project_dir / "experiments" / f"{experiment_name}.yml"
            if yml_path.exists():
                yml_path.unlink()

            # Remove DB record
            self.db.execute(
                "DELETE FROM experiments WHERE project_name = ? AND experiment_name = ?",
                (project_name, experiment_name),
            )
            self.db.commit()
            logger.info("Removed experiment '%s'", experiment_name)

    def run(self, project_dir: Path, experiment_name: str) -> dict[str, Any]:
        """Execute an experiment.

        This is the main orchestration method that runs the full pipeline:
        1. Load configs (project + experiment YAML)
        2. Initialize components (model, trainer, metrics)
        3. Load data, split, transform
        4. Train and evaluate
        5. Save results to database

        Args:
            project_dir: Project directory.
            experiment_name: Name of the experiment to run.

        Returns:
            Experiment results dict.
        """
        project_dir = Path(project_dir).resolve()
        project_config = self.project_manager.load(project_dir)
        project_name = project_config["project-name"]

        # Load experiment config
        exp_yml = project_dir / "experiments" / f"{experiment_name}.yml"
        if not exp_yml.exists():
            raise ComponentNotFoundError(
                f"Experiment config not found: '{exp_yml}'",
                hint=f"Run 'uesf experiment add' to create '{experiment_name}'.",
            )

        try:
            exp_config = yaml.safe_load(exp_yml.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise YAMLParseError(f"Invalid experiment YAML: {exc}") from exc

        # Create DB record
        exp_record_id = self._create_db_record(project_name, experiment_name, exp_config)
        self._update_status(exp_record_id, "RUNNING")

        try:
            results = self._execute(project_dir, project_config, exp_config, experiment_name)

            self._update_status(exp_record_id, "COMPLETED", results=results)
            logger.info("Experiment '%s' completed successfully", experiment_name)
            return results

        except Exception as exc:
            self._update_status(exp_record_id, "FAILED", error=str(exc))
            logger.error("Experiment '%s' failed: %s", experiment_name, exc)
            raise

    def _execute(
        self,
        project_dir: Path,
        project_config: dict,
        exp_config: dict,
        experiment_name: str,
    ) -> dict[str, Any]:
        """Internal execution logic."""
        seed = exp_config.get("seed", 42)
        torch.manual_seed(seed)
        np.random.seed(seed)

        device = torch.device(self.config.get("default_device"))
        training_config = exp_config.get("training", {})
        eval_config = exp_config.get("evaluation", {})

        # 1. Initialize model
        model_config = exp_config.get("model", {})
        model_name = model_config.get("name")
        model_params = model_config.get("params", {})

        model_resolution = self.project_manager.resolve_component(
            model_name, "models", project_config, project_dir,
        )

        # Load model class and get dataset info for auto-injection
        model_cls = self.model_manager.load_class(
            model_name,
            entrypoint=model_resolution.get("entrypoint"),
            project_dir=project_dir,
        )

        # 2. Initialize trainer
        trainer_config = exp_config.get("trainer", {})
        trainer_name = trainer_config.get("name")
        trainer_params = trainer_config.get("params", {})

        trainer_resolution = self.project_manager.resolve_component(
            trainer_name, "trainers", project_config, project_dir,
        )
        trainer_cls = self.trainer_manager.load_class(
            trainer_name,
            entrypoint=trainer_resolution.get("entrypoint"),
            project_dir=project_dir,
        )

        # 3. Load metrics
        metric_names = eval_config.get("metrics", ["accuracy"])
        metric_funcs = {}
        for mname in metric_names:
            try:
                self.project_manager.resolve_component(mname, "metrics", project_config, project_dir)
            except ComponentNotFoundError:
                pass
            metric_funcs[mname] = self.metric_manager.load_metric(mname, project_dir=project_dir)

        # 4. Load datasets, split, transform, train
        datasets_config = exp_config.get("datasets", {})
        dataloaders_config = exp_config.get("dataloaders", {})
        k_fold_aggregation = eval_config.get("k_fold_aggregation", "concat")

        # Process each dataset alias
        all_split_data = {}
        for alias, ds_cfg in datasets_config.items():
            ds_name = ds_cfg["name"]
            split_cfg = ds_cfg.get("split", {"strategy": "holdout"})
            split_cfg["seed"] = seed

            # Load preprocessed data
            data, labels, dataset_meta = self._load_dataset(ds_name)

            # Create splitter and split
            splitter = create_splitter(split_cfg)
            splits = splitter.split(data)

            # Apply transforms per fold
            transforms_config = ds_cfg.get("transforms", [])

            for fold_idx, split_result in enumerate(splits):
                fold_key = (alias, fold_idx)
                si = split_result
                train_data = data[si.train_indices] if len(si.train_indices) > 0 else np.array([])
                train_labels = labels[si.train_indices] if len(si.train_indices) > 0 else np.array([])
                val_data = data[si.val_indices] if len(si.val_indices) > 0 else np.array([])
                val_labels = labels[si.val_indices] if len(si.val_indices) > 0 else np.array([])
                test_data = data[si.test_indices] if len(si.test_indices) > 0 else np.array([])
                test_labels = labels[si.test_indices] if len(si.test_indices) > 0 else np.array([])

                # Apply online transforms
                for t_cfg in transforms_config:
                    t_name = t_cfg["name"]
                    t_params = t_cfg.get("params", {})
                    transform = create_transform(t_name, **t_params)

                    if len(train_data) > 0:
                        transform.fit(train_data)
                        train_data = transform.transform(train_data)
                        if len(val_data) > 0:
                            val_data = transform.transform(val_data)
                        if len(test_data) > 0:
                            test_data = transform.transform(test_data)

                all_split_data[fold_key] = {
                    "train": (train_data, train_labels),
                    "val": (val_data, val_labels),
                    "test": (test_data, test_labels),
                    "meta": dataset_meta,
                }

        # Determine number of folds
        n_folds = max(fold_idx + 1 for (_, fold_idx) in all_split_data.keys())

        # Get dataset metadata for model instantiation
        first_alias = next(iter(datasets_config.keys()))
        first_meta = all_split_data[(first_alias, 0)]["meta"]

        # Run training per fold
        all_fold_results = []
        all_fold_preds = []
        all_fold_targets = []

        for fold_idx in range(n_folds):
            logger.info("=== Fold %d/%d ===", fold_idx + 1, n_folds)

            # Instantiate fresh model per fold
            model = model_cls(
                n_channels=first_meta["n_channels"],
                n_samples=first_meta["n_samples"],
                n_classes=first_meta["n_classes"],
                **model_params,
            )

            trainer = trainer_cls(model, device, **trainer_params)
            evaluator = Evaluator(metric_funcs)

            # Setup optimizer/scheduler
            custom_optim = trainer.configure_optimizers()
            if custom_optim is not None:
                logger.warning("Trainer.configure_optimizers() returned non-None; ignoring YAML optimizer/scheduler.")
                optimizer, scheduler = custom_optim
            else:
                opt_config = training_config.get("optimizer", {"name": "adam", "params": {"lr": 0.001}})
                optimizer = resolve_optimizer(
                    opt_config["name"],
                    model.parameters(),
                    opt_config.get("params", {}),
                )
                sched_config = training_config.get("scheduler")
                scheduler = None
                if sched_config:
                    scheduler = resolve_scheduler(
                        sched_config["name"],
                        optimizer,
                        sched_config.get("params", {}),
                    )

            # Build dataloaders for this fold
            train_datasets, val_datasets, test_datasets = self._build_phase_datasets(
                dataloaders_config, all_split_data, fold_idx,
            )

            batch_size = training_config.get("batch_size", 32)
            num_workers = int(self.config.get("num_workers"))

            dl_kw = {"batch_size": batch_size, "num_workers": num_workers}
            train_loader = build_dataloaders(train_datasets, phase="train", **dl_kw)
            val_loader = (
                build_dataloaders(val_datasets, phase="val", **dl_kw) if val_datasets else None
            )
            test_loader = (
                build_dataloaders(test_datasets, phase="test", **dl_kw) if test_datasets else None
            )

            # Checkpoint dir
            checkpoint_dir = project_dir / "experiments" / "results" / experiment_name / "checkpoints"
            if n_folds > 1:
                checkpoint_dir = checkpoint_dir / f"fold_{fold_idx}"

            # Run training
            runner = Runner(trainer, evaluator, device, training_config)
            run_result = runner.run(
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoint_dir=checkpoint_dir,
                checkpoint_metric=exp_config.get("logging", {}).get("checkpoint_metric"),
                early_stopping_config=training_config.get("early_stopping"),
            )

            # Test evaluation
            test_metrics = {}
            fold_test_preds = []
            fold_test_targets = []
            if test_loader and len(test_loader) > 0:
                test_metrics, fold_test_preds, fold_test_targets = runner.validate_epoch(test_loader)
                test_metrics = {f"test_{k}": v for k, v in test_metrics.items()}

            fold_result = {**run_result["best_metrics"], **test_metrics, "epochs_run": run_result["epochs_run"]}
            all_fold_results.append(fold_result)
            all_fold_preds.append(fold_test_preds)
            all_fold_targets.append(fold_test_targets)

        # Aggregate results across folds
        if n_folds > 1:
            final_results = Evaluator.aggregate_fold_results(
                all_fold_results,
                mode=k_fold_aggregation,
                fold_preds=all_fold_preds,
                fold_targets=all_fold_targets,
                metric_funcs=metric_funcs,
            )
            final_results["fold_results"] = all_fold_results
        else:
            final_results = all_fold_results[0] if all_fold_results else {}

        final_results["n_folds"] = n_folds

        return final_results

    def _load_dataset(self, name: str) -> tuple[np.ndarray, np.ndarray, dict]:
        """Load a preprocessed or masked dataset from disk."""
        # Try preprocessed first
        row = self.db.fetch_one("SELECT * FROM preprocessed_datasets WHERE name = ?", (name,))
        if row is None:
            row = self.db.fetch_one("SELECT * FROM masked_datasets WHERE name = ?", (name,))
        if row is None:
            raise ComponentNotFoundError(
                f"Dataset '{name}' not found",
                hint="Register or preprocess the dataset first.",
            )

        data_path = Path(row["data_path"])
        labels_path = Path(row["labels_path"])

        data = np.load(str(data_path))
        labels = np.load(str(labels_path))

        # Flatten to (n_total_samples, channels, samples) for DataLoader
        original_shape = data.shape
        if data.ndim == 5:
            # [subject, session, recording, channel, sample]
            n_sub, n_sess, n_rec, n_ch, n_samp = data.shape
            data = data.reshape(-1, n_ch, n_samp)
            labels = labels.reshape(-1)
        elif data.ndim == 4:
            # [subject, recording, channel, sample]
            n_sub, n_rec, n_ch, n_samp = data.shape
            data = data.reshape(-1, n_ch, n_samp)
            labels = labels.reshape(-1)
        elif data.ndim == 3:
            n_ch, n_samp = data.shape[-2], data.shape[-1]
        else:
            n_ch = data.shape[-1] if data.ndim >= 2 else 1
            n_samp = 1

        n_classes = len(np.unique(labels))

        meta = {
            "n_channels": n_ch if data.ndim >= 2 else data.shape[-1],
            "n_samples": n_samp if data.ndim >= 2 else 1,
            "n_classes": n_classes,
            "original_shape": original_shape,
        }

        return data, labels, meta

    def _build_phase_datasets(
        self,
        dataloaders_config: dict,
        all_split_data: dict,
        fold_idx: int,
    ) -> tuple[dict, dict, dict]:
        """Build channel-mapped EEGDataset dicts for each phase."""
        train_datasets = {}
        val_datasets = {}
        test_datasets = {}

        for phase_name, phase_config in [
            ("train", dataloaders_config.get("train", {})),
            ("val", dataloaders_config.get("val", {})),
            ("test", dataloaders_config.get("test", {})),
        ]:
            target = {"train": train_datasets, "val": val_datasets, "test": test_datasets}[phase_name]
            for channel_name, mapping in (phase_config or {}).items():
                alias, split_phase = mapping.rsplit(".", 1)
                fold_key = (alias, fold_idx)
                if fold_key not in all_split_data:
                    continue

                split_data = all_split_data[fold_key]
                data, labels = split_data[split_phase]
                if len(data) > 0:
                    target[channel_name] = EEGDataset(data, labels)

        # Fallback: if no dataloaders config, use first dataset alias with default channels
        if not train_datasets and not dataloaders_config:
            for alias in all_split_data:
                if alias[1] == fold_idx:
                    split_data = all_split_data[alias]
                    feat, lab = split_data["train"]
                    if len(feat) > 0:
                        train_datasets["main"] = EEGDataset(feat, lab)
                    feat, lab = split_data["val"]
                    if len(feat) > 0:
                        val_datasets["main"] = EEGDataset(feat, lab)
                    feat, lab = split_data["test"]
                    if len(feat) > 0:
                        test_datasets["main"] = EEGDataset(feat, lab)
                    break

        return train_datasets, val_datasets, test_datasets

    def _create_db_record(
        self,
        project_name: str,
        experiment_name: str,
        config: dict,
    ) -> int:
        """Create or update an experiment record in the database."""
        existing = self.db.fetch_one(
            "SELECT id FROM experiments WHERE project_name = ? AND experiment_name = ?",
            (project_name, experiment_name),
        )
        if existing:
            self.db.execute(
                """UPDATE experiments SET config = ?, status = 'PENDING',
                   updated_at = CURRENT_TIMESTAMP WHERE id = ?""",
                (json.dumps(config, ensure_ascii=False), existing["id"]),
            )
            self.db.commit()
            return existing["id"]

        with self.db.transaction() as cursor:
            cursor.execute(
                """INSERT INTO experiments (project_name, experiment_name, config, status)
                   VALUES (?, ?, ?, 'PENDING')""",
                (project_name, experiment_name, json.dumps(config, ensure_ascii=False)),
            )
            return cursor.lastrowid

    def _update_status(
        self,
        record_id: int,
        status: str,
        results: dict | None = None,
        error: str | None = None,
    ) -> None:
        """Update experiment status in the database."""
        results_json = json.dumps(results, ensure_ascii=False, default=str) if results else None
        if error:
            results_json = json.dumps({"error": error}, ensure_ascii=False)

        self.db.execute(
            """UPDATE experiments SET status = ?, results = ?,
               updated_at = CURRENT_TIMESTAMP WHERE id = ?""",
            (status, results_json, record_id),
        )
        self.db.commit()

    def query(
        self,
        project_name: str | None = None,
        metrics: list[str] | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query experiment results.

        Args:
            project_name: Filter by project name (None for all).
            metrics: List of metric names to include in results.
            status: Filter by status ('COMPLETED', 'FAILED', etc).

        Returns:
            List of experiment result dicts.
        """
        query = "SELECT * FROM experiments WHERE 1=1"
        params: list = []

        if project_name:
            query += " AND project_name = ?"
            params.append(project_name)
        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY created_at DESC"
        rows = self.db.fetch_all(query, tuple(params))

        results = []
        for row in rows:
            entry = dict(row)
            if row["results"]:
                try:
                    result_data = json.loads(row["results"])
                    if metrics:
                        entry["selected_metrics"] = {
                            m: result_data.get(m, result_data.get(f"test_{m}"))
                            for m in metrics
                        }
                    entry["parsed_results"] = result_data
                except json.JSONDecodeError:
                    pass
            results.append(entry)

        return results


def _experiment_template(name: str, description: str | None = None) -> str:
    """Generate a blank experiment YAML template."""
    return f"""\
name: {name}
description: "{description or ''}"
seed: 42

model:
  name: ""  # Model name from project.yml or global registry
  params: {{}}

trainer:
  name: ""  # Trainer name from project.yml or global registry
  params: {{}}

datasets:
  main_dataset:
    name: ""  # Preprocessed or masked dataset name
    split:
      strategy: holdout
      dimension: subject
      shuffle: true
      train_ratio: 0.7
      val_ratio: 0.15
      test_ratio: 0.15
    transforms:
      - name: zscore_normalize
        fit_on: train
        apply_to: all

dataloaders:
  train:
    main: "main_dataset.train"
  val:
    main: "main_dataset.val"
  test:
    main: "main_dataset.test"

training:
  epochs: 100
  batch_size: 32
  optimizer:
    name: adam
    params:
      lr: 0.001
  # gradient_clip:
  #   max_norm: 1.0
  # scheduler:
  #   name: cosine_annealing_lr
  #   params:
  #     T_max: 50
  # early_stopping:
  #   monitor: val_accuracy
  #   patience: 10
  #   mode: max

evaluation:
  metrics:
    - accuracy
    - f1_score
  k_fold_aggregation: concat

logging:
  use_wandb: false
  checkpoint_metric: val_accuracy
"""
