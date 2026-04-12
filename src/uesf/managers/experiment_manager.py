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
    ConfigError,
    YAMLParseError,
)
from uesf.core.logging import get_logger
from uesf.experiment.alignment import LabelAligner, create_channel_aligner
from uesf.experiment.dataloader_builder import build_dataloaders
from uesf.experiment.dataset import EEGDataset
from uesf.experiment.evaluator import Evaluator
from uesf.experiment.runner import Runner
from uesf.experiment.splitter import (
    DatasetLevelSplitter,
    UDASplitResult,
    create_splitter,
)
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
                yaml.dump(content, default_flow_style=False, allow_unicode=True, sort_keys=False),
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
            results = self._execute(project_dir, project_config, exp_config, experiment_name, exp_record_id)

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
        exp_record_id: int,
    ) -> dict[str, Any]:
        """Internal execution logic — dispatches to mode-specific methods."""
        seed = exp_config.get("seed", 42)
        torch.manual_seed(seed)
        np.random.seed(seed)

        device = torch.device(self.config.get("default_device"))
        training_config = exp_config.get("training", {})
        eval_config = exp_config.get("evaluation", {})

        # --- Component initialisation (shared by all modes) ---
        model_cls, model_params, model_id = self._init_model(
            exp_config, project_config, project_dir,
        )
        trainer_cls, trainer_params, trainer_id = self._init_trainer(
            exp_config, project_config, project_dir,
        )
        metric_funcs = self._init_metrics(eval_config, project_config, project_dir)

        # Link component IDs to experiment record
        self.db.execute(
            """UPDATE experiments SET model_id = ?, trainer_id = ?,
               updated_at = CURRENT_TIMESTAMP WHERE id = ?""",
            (model_id, trainer_id, exp_record_id),
        )
        self.db.commit()

        # --- Load all datasets ---
        datasets_config = exp_config.get("datasets", {})
        dataset_cache = self._load_all_datasets(datasets_config, seed)

        # --- Cross-dataset alignment ---
        alignment_config = exp_config.get("alignment", {})
        if len(dataset_cache) > 1 and alignment_config:
            self._apply_alignment(dataset_cache, alignment_config)

        # Shared kwargs passed to every execution mode
        ctx = {
            "project_dir": project_dir,
            "exp_config": exp_config,
            "experiment_name": experiment_name,
            "seed": seed,
            "device": device,
            "training_config": training_config,
            "eval_config": eval_config,
            "model_cls": model_cls,
            "model_params": model_params,
            "trainer_cls": trainer_cls,
            "trainer_params": trainer_params,
            "metric_funcs": metric_funcs,
            "dataset_cache": dataset_cache,
            "datasets_config": datasets_config,
        }

        # --- Mode dispatch ---
        mode = exp_config.get("mode", "regular")

        if mode == "uda":
            return self._execute_uda(**ctx)

        # Regular mode — check for dataset-level split
        top_split = exp_config.get("split", {})
        if top_split.get("dimension") == "dataset":
            return self._execute_dataset_split(top_split=top_split, **ctx)

        return self._execute_regular(**ctx)

    # ------------------------------------------------------------------
    # Component initialisation helpers
    # ------------------------------------------------------------------

    def _init_model(
        self, exp_config: dict, project_config: dict, project_dir: Path,
    ) -> tuple[type, dict, int | None]:
        model_config = exp_config.get("model", {})
        model_name = model_config.get("name")
        model_params = model_config.get("params", {})

        model_resolution = self.project_manager.resolve_component(
            model_name, "models", project_config, project_dir,
        )

        model_id = None
        if model_resolution["source"] == "PROJECT" and model_resolution.get("entrypoint"):
            entrypoint = model_resolution["entrypoint"]
            try:
                self.model_manager.get(model_name)
                model_record = self.model_manager.detect_and_reregister(
                    model_name, entrypoint, project_dir,
                )
            except ComponentNotFoundError:
                model_record = self.model_manager.register(
                    model_name, entrypoint, project_dir,
                )
            model_id = model_record["id"]
        elif model_resolution.get("record"):
            model_id = model_resolution["record"]["id"]

        model_cls = self.model_manager.load_class(
            model_name,
            entrypoint=model_resolution.get("entrypoint"),
            project_dir=project_dir,
        )
        return model_cls, model_params, model_id

    def _init_trainer(
        self, exp_config: dict, project_config: dict, project_dir: Path,
    ) -> tuple[type, dict, int | None]:
        trainer_config = exp_config.get("trainer", {})
        trainer_name = trainer_config.get("name")
        trainer_params = trainer_config.get("params", {})

        trainer_resolution = self.project_manager.resolve_component(
            trainer_name, "trainers", project_config, project_dir,
        )

        trainer_id = None
        if trainer_resolution["source"] == "PROJECT" and trainer_resolution.get("entrypoint"):
            entrypoint = trainer_resolution["entrypoint"]
            try:
                self.trainer_manager.get(trainer_name)
                trainer_record = self.trainer_manager.detect_and_reregister(
                    trainer_name, entrypoint, project_dir,
                )
            except ComponentNotFoundError:
                trainer_record = self.trainer_manager.register(
                    trainer_name, entrypoint, project_dir,
                )
            trainer_id = trainer_record["id"]
        elif trainer_resolution.get("record"):
            trainer_id = trainer_resolution["record"]["id"]

        trainer_cls = self.trainer_manager.load_class(
            trainer_name,
            entrypoint=trainer_resolution.get("entrypoint"),
            project_dir=project_dir,
        )
        return trainer_cls, trainer_params, trainer_id

    def _init_metrics(
        self, eval_config: dict, project_config: dict, project_dir: Path,
    ) -> dict[str, Any]:
        metric_names = eval_config.get("metrics", ["accuracy"])
        metric_funcs = {}
        for mname in metric_names:
            metric_resolution = None
            try:
                metric_resolution = self.project_manager.resolve_component(
                    mname, "metrics", project_config, project_dir,
                )
            except ComponentNotFoundError:
                pass

            if (
                metric_resolution
                and metric_resolution["source"] == "PROJECT"
                and metric_resolution.get("entrypoint")
            ):
                entrypoint = metric_resolution["entrypoint"]
                try:
                    self.metric_manager.get(mname)
                    self.metric_manager.detect_and_reregister(
                        mname, entrypoint, project_dir,
                    )
                except ComponentNotFoundError:
                    self.metric_manager.register(
                        mname, entrypoint, project_dir,
                    )

            metric_funcs[mname] = self.metric_manager.load_metric(mname, project_dir=project_dir)
        return metric_funcs

    # ------------------------------------------------------------------
    # Dataset loading & alignment
    # ------------------------------------------------------------------

    def _load_all_datasets(
        self, datasets_config: dict, seed: int,
    ) -> dict[str, dict]:
        """Load every dataset listed in the config into a cache dict.

        Each entry: ``{data, labels, meta, transforms_config}``.
        Data is kept in original 5-D shape for splitters.
        """
        dataset_cache: dict[str, dict] = {}
        for alias, ds_cfg in datasets_config.items():
            ds_name = ds_cfg["name"]
            data, labels, meta = self._load_dataset(ds_name)
            dataset_cache[alias] = {
                "data": data,
                "labels": labels,
                "meta": meta,
                "transforms_config": ds_cfg.get("transforms", []),
            }
        return dataset_cache

    def _apply_alignment(
        self, dataset_cache: dict[str, dict], alignment_config: dict,
    ) -> None:
        """Apply channel and label alignment in-place on dataset_cache."""
        # Channel alignment
        ch_config = alignment_config.get("channels", {})
        if ch_config:
            method = ch_config.get("method", "intersection")
            aligner = create_channel_aligner(method)
            input_map = {}
            for alias, cache in dataset_cache.items():
                electrodes = cache["meta"].get("electrode_list", [])
                if not electrodes:
                    raise ConfigError(
                        f"Dataset '{alias}' has no electrode_list metadata — "
                        "cannot perform channel alignment.",
                        hint="Re-preprocess the dataset with electrode info.",
                    )
                input_map[alias] = (cache["data"], electrodes)

            aligned_data, common_electrodes = aligner.align(input_map)
            for alias in dataset_cache:
                dataset_cache[alias]["data"] = aligned_data[alias]
                dataset_cache[alias]["meta"]["electrode_list"] = common_electrodes
                dataset_cache[alias]["meta"]["n_channels"] = len(common_electrodes)

        # Label validation
        label_config = alignment_config.get("labels", {})
        if label_config.get("check_consistency", True):
            label_aligner = LabelAligner()
            label_aligner.validate({
                alias: cache["meta"] for alias, cache in dataset_cache.items()
            })

    # ------------------------------------------------------------------
    # Regular execution (backward-compatible path)
    # ------------------------------------------------------------------

    def _execute_regular(self, **ctx: Any) -> dict[str, Any]:
        """Original single/multi-dataset execution with per-dataset splits."""
        dataset_cache = ctx["dataset_cache"]
        datasets_config = ctx["datasets_config"]
        seed = ctx["seed"]
        dataloaders_config = ctx["exp_config"].get("dataloaders", {})
        k_fold_aggregation = ctx["eval_config"].get("k_fold_aggregation", "concat")

        # Compute per-dataset splits
        for alias, cache in dataset_cache.items():
            ds_cfg = datasets_config[alias]
            split_cfg = ds_cfg.get("split", {"strategy": "holdout"})
            split_cfg["seed"] = seed
            splitter = create_splitter(split_cfg)
            cache["splits"] = splitter.split(cache["data"])
            # Flatten to 3-D for downstream indexing
            self._flatten_cache(cache)

        first_alias = next(iter(datasets_config.keys()))
        n_folds = len(dataset_cache[first_alias]["splits"])
        first_meta = dataset_cache[first_alias]["meta"]

        return self._train_folds(
            n_folds=n_folds,
            first_meta=first_meta,
            fold_data_fn=lambda fold_idx: self._regular_fold_data(
                dataset_cache, dataloaders_config, fold_idx,
            ),
            k_fold_aggregation=k_fold_aggregation,
            **ctx,
        )

    def _regular_fold_data(
        self,
        dataset_cache: dict[str, dict],
        dataloaders_config: dict,
        fold_idx: int,
    ) -> tuple[dict, dict, dict]:
        """Build phase datasets for one fold of a regular experiment."""
        fold_split_data: dict[tuple, dict] = {}
        for alias, cache in dataset_cache.items():
            data = cache["data"]
            labels = cache["labels"]
            si = cache["splits"][fold_idx]

            train_data = data[si.train_indices] if len(si.train_indices) > 0 else np.array([])
            train_labels = labels[si.train_indices] if len(si.train_indices) > 0 else np.array([])
            val_data = data[si.val_indices] if len(si.val_indices) > 0 else np.array([])
            val_labels = labels[si.val_indices] if len(si.val_indices) > 0 else np.array([])
            test_data = data[si.test_indices] if len(si.test_indices) > 0 else np.array([])
            test_labels = labels[si.test_indices] if len(si.test_indices) > 0 else np.array([])

            # Transforms: fit on train
            for t_cfg in cache["transforms_config"]:
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

            fold_split_data[(alias, fold_idx)] = {
                "train": (train_data, train_labels),
                "val": (val_data, val_labels),
                "test": (test_data, test_labels),
                "meta": cache["meta"],
            }

        return self._build_phase_datasets(dataloaders_config, fold_split_data, fold_idx)

    # ------------------------------------------------------------------
    # Dataset-level split execution
    # ------------------------------------------------------------------

    def _execute_dataset_split(self, *, top_split: dict, **ctx: Any) -> dict[str, Any]:
        """Regular DL with dimension=dataset: entire datasets as train/test."""
        dataset_cache = ctx["dataset_cache"]
        seed = ctx["seed"]
        k_fold_aggregation = ctx["eval_config"].get("k_fold_aggregation", "concat")

        top_split["seed"] = seed
        splitter = create_splitter(top_split)
        assert isinstance(splitter, DatasetLevelSplitter)
        ds_splits = splitter.split(list(dataset_cache.keys()))

        # Flatten all datasets
        for cache in dataset_cache.values():
            self._flatten_cache(cache)

        n_folds = len(ds_splits)
        first_meta = next(iter(dataset_cache.values()))["meta"]

        def fold_data_fn(fold_idx: int) -> tuple[dict, dict, dict]:
            phase_aliases = ds_splits[fold_idx].phase_aliases
            return self._dataset_split_fold_data(
                dataset_cache, phase_aliases, ctx.get("datasets_config", {}),
            )

        return self._train_folds(
            n_folds=n_folds,
            first_meta=first_meta,
            fold_data_fn=fold_data_fn,
            k_fold_aggregation=k_fold_aggregation,
            **ctx,
        )

    def _dataset_split_fold_data(
        self,
        dataset_cache: dict[str, dict],
        phase_aliases: dict[str, list[str]],
        datasets_config: dict,
    ) -> tuple[dict, dict, dict]:
        """Build phase datasets for a dataset-level split fold."""
        train_datasets: dict[str, EEGDataset] = {}
        val_datasets: dict[str, EEGDataset] = {}
        test_datasets: dict[str, EEGDataset] = {}

        # Collect transforms config from all datasets in the training phase
        transforms_config = []
        for alias in phase_aliases.get("train", []):
            transforms_config = dataset_cache[alias].get("transforms_config", [])
            if transforms_config:
                break

        # Fit transforms on concatenated training data
        transform_instances = []
        if transforms_config:
            train_data_parts = [
                dataset_cache[a]["data"] for a in phase_aliases.get("train", [])
                if len(dataset_cache[a]["data"]) > 0
            ]
            if train_data_parts:
                concat_train = np.concatenate(train_data_parts)
                for t_cfg in transforms_config:
                    t = create_transform(t_cfg["name"], **t_cfg.get("params", {}))
                    t.fit(concat_train)
                    concat_train = t.transform(concat_train)
                    transform_instances.append(t)

        for phase, target_dict in [
            ("train", train_datasets), ("val", val_datasets), ("test", test_datasets),
        ]:
            parts_data, parts_labels = [], []
            for alias in phase_aliases.get(phase, []):
                d = dataset_cache[alias]["data"]
                l = dataset_cache[alias]["labels"]
                if len(d) > 0:
                    for t in transform_instances:
                        d = t.transform(d)
                    parts_data.append(d)
                    parts_labels.append(l)

            if parts_data:
                target_dict["main"] = EEGDataset(
                    np.concatenate(parts_data),
                    np.concatenate(parts_labels),
                )

        return train_datasets, val_datasets, test_datasets

    # ------------------------------------------------------------------
    # UDA execution
    # ------------------------------------------------------------------

    def _execute_uda(self, **ctx: Any) -> dict[str, Any]:
        """Execute a UDA experiment (cross-dataset or intra-dataset)."""
        exp_config = ctx["exp_config"]
        dataset_cache = ctx["dataset_cache"]
        seed = ctx["seed"]
        k_fold_aggregation = ctx["eval_config"].get("k_fold_aggregation", "concat")

        uda_config = exp_config.get("uda", {})
        uda_config["seed"] = seed
        splitter = create_splitter(mode="uda", uda_config=uda_config)

        uda_type = uda_config["type"]

        if uda_type == "intra-dataset":
            # Single dataset — split on its 5-D data
            alias = next(iter(dataset_cache.keys()))
            uda_splits: list[UDASplitResult] = splitter.split(
                dataset_cache[alias]["data"], alias=alias,
            )
            # Flatten after splitting
            self._flatten_cache(dataset_cache[alias])
        else:
            # Cross-dataset — pass full cache (5-D data intact for inductive split)
            uda_splits = splitter.split(dataset_cache)
            # Flatten all datasets after splitting
            for cache in dataset_cache.values():
                self._flatten_cache(cache)

        n_folds = len(uda_splits)
        first_meta = next(iter(dataset_cache.values()))["meta"]

        def fold_data_fn(fold_idx: int) -> tuple[dict, dict, dict]:
            return self._uda_fold_data(
                dataset_cache, uda_splits[fold_idx], uda_config,
            )

        return self._train_folds(
            n_folds=n_folds,
            first_meta=first_meta,
            fold_data_fn=fold_data_fn,
            k_fold_aggregation=k_fold_aggregation,
            **ctx,
        )

    def _uda_fold_data(
        self,
        dataset_cache: dict[str, dict],
        uda_split: UDASplitResult,
        uda_config: dict,
    ) -> tuple[dict, dict, dict]:
        """Build phase datasets for one fold of a UDA experiment."""
        # Gather source data
        src_train_d, src_train_l = self._gather_domain_data(
            dataset_cache, uda_split.source_train_indices,
        )
        src_val_d, src_val_l = self._gather_domain_data(
            dataset_cache, uda_split.source_val_indices,
        )

        # Gather target data
        tgt_train_d, tgt_train_l = self._gather_domain_data(
            dataset_cache, uda_split.target_train_indices,
        )
        tgt_val_d, tgt_val_l = self._gather_domain_data(
            dataset_cache, uda_split.target_val_indices,
        )
        tgt_test_d, tgt_test_l = self._gather_domain_data(
            dataset_cache, uda_split.target_test_indices,
        )

        # Transforms: fit on SOURCE train, apply to all
        transforms_config = self._collect_transforms(dataset_cache)
        transform_instances = []
        for t_cfg in transforms_config:
            t = create_transform(t_cfg["name"], **t_cfg.get("params", {}))
            if len(src_train_d) > 0:
                t.fit(src_train_d)
                src_train_d = t.transform(src_train_d)
                if len(src_val_d) > 0:
                    src_val_d = t.transform(src_val_d)
                if len(tgt_train_d) > 0:
                    tgt_train_d = t.transform(tgt_train_d)
                if len(tgt_val_d) > 0:
                    tgt_val_d = t.transform(tgt_val_d)
                if len(tgt_test_d) > 0:
                    tgt_test_d = t.transform(tgt_test_d)
            transform_instances.append(t)

        # Build EEGDataset dicts
        train_datasets: dict[str, EEGDataset] = {}
        val_datasets: dict[str, EEGDataset] = {}
        test_datasets: dict[str, EEGDataset] = {}

        if len(src_train_d) > 0:
            train_datasets["source"] = EEGDataset(src_train_d, src_train_l)
        if len(tgt_train_d) > 0:
            train_datasets["target"] = EEGDataset(tgt_train_d, tgt_train_l)
        if len(src_val_d) > 0:
            val_datasets["source_val"] = EEGDataset(src_val_d, src_val_l)
        if len(tgt_val_d) > 0:
            val_datasets["target_val"] = EEGDataset(tgt_val_d, tgt_val_l)
        if len(tgt_test_d) > 0:
            test_datasets["main"] = EEGDataset(tgt_test_d, tgt_test_l)

        return train_datasets, val_datasets, test_datasets

    @staticmethod
    def _gather_domain_data(
        dataset_cache: dict[str, dict],
        index_map: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Concatenate data from one or more datasets using index arrays."""
        all_data, all_labels = [], []
        for alias, indices in index_map.items():
            if len(indices) == 0:
                continue
            cache = dataset_cache[alias]
            all_data.append(cache["data"][indices])
            all_labels.append(cache["labels"][indices])
        if all_data:
            return np.concatenate(all_data), np.concatenate(all_labels)
        return np.array([]), np.array([])

    @staticmethod
    def _collect_transforms(dataset_cache: dict[str, dict]) -> list[dict]:
        """Get the first non-empty transforms_config from any cached dataset."""
        for cache in dataset_cache.values():
            cfg = cache.get("transforms_config", [])
            if cfg:
                return cfg
        return []

    # ------------------------------------------------------------------
    # Shared training loop
    # ------------------------------------------------------------------

    def _train_folds(
        self,
        *,
        n_folds: int,
        first_meta: dict,
        fold_data_fn,
        k_fold_aggregation: str,
        **ctx: Any,
    ) -> dict[str, Any]:
        """Run training across folds — shared by all execution modes."""
        training_config = ctx["training_config"]
        model_cls = ctx["model_cls"]
        model_params = ctx["model_params"]
        trainer_cls = ctx["trainer_cls"]
        trainer_params = ctx["trainer_params"]
        metric_funcs = ctx["metric_funcs"]
        device = ctx["device"]
        project_dir = ctx["project_dir"]
        experiment_name = ctx["experiment_name"]
        exp_config = ctx["exp_config"]

        batch_size = training_config.get("batch_size", 32)
        num_workers = int(self.config.get("num_workers"))
        dl_kw = {"batch_size": batch_size, "num_workers": num_workers}

        all_fold_results = []
        all_fold_preds = []
        all_fold_targets = []

        for fold_idx in range(n_folds):
            logger.info("=== Fold %d/%d ===", fold_idx + 1, n_folds)

            train_datasets, val_datasets, test_datasets = fold_data_fn(fold_idx)

            # Fresh model per fold
            model = model_cls(
                n_channels=first_meta["n_channels"],
                n_samples=first_meta["n_samples"],
                n_classes=first_meta["n_classes"],
                **model_params,
            )

            trainer = trainer_cls(model, device, **trainer_params)
            evaluator = Evaluator(metric_funcs)

            # Optimizer / scheduler
            custom_optim = trainer.configure_optimizers()
            if custom_optim is not None:
                logger.warning(
                    "Trainer.configure_optimizers() returned non-None; "
                    "ignoring YAML optimizer/scheduler.",
                )
                optimizer, scheduler = custom_optim
            else:
                opt_config = training_config.get(
                    "optimizer", {"name": "adam", "params": {"lr": 0.001}},
                )
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

            # Build dataloaders
            train_loader = build_dataloaders(train_datasets, phase="train", **dl_kw)
            val_loader = (
                build_dataloaders(val_datasets, phase="val", **dl_kw) if val_datasets else None
            )
            test_loader = (
                build_dataloaders(test_datasets, phase="test", **dl_kw) if test_datasets else None
            )

            # Checkpoint dir
            checkpoint_dir = (
                project_dir / "experiments" / "results" / experiment_name / "checkpoints"
            )
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
            fold_test_preds: list = []
            fold_test_targets: list = []
            if test_loader and len(test_loader) > 0:
                test_metrics, fold_test_preds, fold_test_targets = runner.validate_epoch(
                    test_loader,
                )
                test_metrics = {f"test_{k}": v for k, v in test_metrics.items()}

            fold_result = {
                **run_result["best_metrics"],
                **test_metrics,
                "epochs_run": run_result["epochs_run"],
            }
            all_fold_results.append(fold_result)
            all_fold_preds.append(fold_test_preds)
            all_fold_targets.append(fold_test_targets)

            # Free per-fold resources
            del train_loader, val_loader, test_loader
            del train_datasets, val_datasets, test_datasets
            del model, trainer, runner, optimizer
            if scheduler is not None:
                del scheduler
            if device.type == "cuda":
                torch.cuda.empty_cache()

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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_cache(cache: dict) -> None:
        """Flatten 5-D data to 3-D ``(n_total, ch, samp)`` in place."""
        data = cache["data"]
        if data.ndim > 3:
            n_ch = cache["meta"]["n_channels"]
            n_samp = cache["meta"]["n_samples"]
            cache["data"] = data.reshape(-1, n_ch, n_samp)
            cache["labels"] = cache["labels"].reshape(-1)

    def _load_dataset(self, name: str) -> tuple[np.ndarray, np.ndarray, dict]:
        """Load a preprocessed or masked dataset from disk.

        Returns the original-shape data (not flattened) so the splitter can
        correctly interpret dimensions.  The caller is responsible for
        flattening to (n_total, channels, samples) after splitting.
        """
        # Try preprocessed first
        row = self.db.fetch_one("SELECT * FROM preprocessed_datasets WHERE name = ?", (name,))
        if row is None:
            row = self.db.fetch_one("SELECT * FROM masked_datasets WHERE name = ?", (name,))
        if row is None:
            raise ComponentNotFoundError(
                f"Dataset '{name}' not found",
                hint="Register or preprocess the dataset first.",
            )

        data_dir = Path(row["data_dir_path"])
        data = np.load(str(data_dir / "eeg_data.npy")).astype(np.float32)
        labels = np.load(str(data_dir / "labels.npy"))

        # Extract channel/sample metadata without flattening
        if data.ndim >= 3:
            n_ch, n_samp = data.shape[-2], data.shape[-1]
        elif data.ndim == 2:
            n_ch, n_samp = data.shape[-1], 1
        else:
            n_ch, n_samp = 1, 1

        n_classes = len(np.unique(labels))

        # Extract electrode_list and numeric_to_semantic from DB row
        electrode_list = None
        if row.get("electrode_list"):
            try:
                electrode_list = json.loads(row["electrode_list"])
            except (json.JSONDecodeError, TypeError):
                pass

        numeric_to_semantic = None
        if row.get("numeric_to_semantic"):
            try:
                numeric_to_semantic = json.loads(row["numeric_to_semantic"])
            except (json.JSONDecodeError, TypeError):
                pass

        meta = {
            "n_channels": n_ch,
            "n_samples": n_samp,
            "n_classes": n_classes,
            "original_shape": data.shape,
            "electrode_list": electrode_list,
            "numeric_to_semantic": numeric_to_semantic,
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
# mode: regular  # "regular" (default) or "uda"

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

# --- UDA example (uncomment to use) ---
# mode: uda
# uda:
#   type: intra-dataset        # or "cross-dataset"
#   dimension: subject          # for intra-dataset: "subject" or "session"
#   strategy: k-fold            # "holdout" or "k-fold"
#   k-folds: -1                 # -1 for LOOCV
#   variant: transductive       # "transductive" or "inductive"
#   # target_count: 1           # for holdout
#   # source_split:
#   #   val_ratio: 0.15
#   # target_split:             # for inductive
#   #   dimension: recording
#   #   train_ratio: 0.7
#   #   val_ratio: 0.1
#   #   test_ratio: 0.2
#
# --- Cross-dataset alignment (for multi-dataset experiments) ---
# alignment:
#   channels:
#     method: intersection
#   labels:
#     check_consistency: true
"""
