"""UESF Experiment Manager — orchestrates the full experiment lifecycle.

Dev6 architecture (see docs/version_design/0.1.0.dev6_design/):

  ExperimentManager.run()
      │
      ├─ load config + DB record
      ├─ ConfigValidator.normalize() + validate()
      ├─ build ExperimentContext
      └─ ExperimentExecutor.execute(ctx)
             ├─ RegularExecutionStrategy  (mode=regular)
             └─ UDAExecutionStrategy       (mode=uda)
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
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
    UESFException,
    YAMLParseError,
)
from uesf.core.logging import get_logger
from uesf.experiment.alignment import (
    LabelAligner,
    check_sample_consistency,
    create_channel_aligner,
)
from uesf.experiment.config_schema import ConfigValidator
from uesf.experiment.dataloader_builder import (
    DataloaderBuilder,
    get_sample_input,
    prepare_channel_data,
    prepare_uda_channel_data,
)
from uesf.experiment.evaluator import Evaluator
from uesf.experiment.label_mapping import apply_label_mapping
from uesf.experiment.logger import create_logger
from uesf.experiment.runner import Runner
from uesf.experiment.splitter import (
    DatasetLevelSplitter,
    MultiDatasetSplitResult,
    SplitResult,
    UDAOrchestrator,
    ValSplitter,
    create_splitter,
)
from uesf.experiment.transforms import apply_transforms, apply_transforms_uda
from uesf.managers.metric_manager import MetricManager
from uesf.managers.model_manager import ModelManager
from uesf.managers.project_manager import ProjectManager
from uesf.managers.trainer_manager import TrainerManager

logger = get_logger("manager.experiment")


# ---------------------------------------------------------------------------
# Dataclasses (dev6 §3.3.1)
# ---------------------------------------------------------------------------


@dataclass
class ExperimentContext:
    """Runtime context passed from ExperimentManager to the Executor."""

    config: dict
    dataset_cache: dict[str, np.ndarray]  # alias → 5-D data
    labels_cache: dict[str, np.ndarray]  # alias → flattened labels
    metadata_cache: dict[str, dict]
    experiment_id: int
    output_dir: Path
    device: torch.device
    project_dir: Path
    seed: int
    model_cls: type
    model_params: dict
    trainer_cls: type
    trainer_params: dict
    metric_funcs: dict
    num_workers: int = 0


@dataclass
class FoldResult:
    metrics: dict[str, Any]
    fold_info: dict = field(default_factory=dict)
    predictions: Any = None
    targets: Any = None
    failed: bool = False
    error: str | None = None


@dataclass
class ExperimentResult:
    fold_results: list[FoldResult]
    aggregated_metrics: dict[str, Any]
    aggregation_mode: str


# ---------------------------------------------------------------------------
# ExperimentManager
# ---------------------------------------------------------------------------


class ExperimentManager:
    """Experiment lifecycle: add / list / remove / run / query."""

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

    # ------------------------------------------------------------------
    # add / list / remove
    # ------------------------------------------------------------------

    def add(
        self,
        project_dir: Path,
        experiment_name: str | None = None,
        from_existing: str | None = None,
        description: str | None = None,
    ) -> Path:
        project_config = self.project_manager.load(project_dir)
        project_name = project_config["project-name"]
        project_dir = Path(project_dir).resolve()

        if experiment_name is None:
            experiment_name = f"{project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

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
            content["experiment_name"] = experiment_name
            if description:
                content["description"] = description
            yml_path.write_text(
                yaml.dump(content, default_flow_style=False, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
        else:
            yml_path.write_text(
                _experiment_template(experiment_name, description), encoding="utf-8"
            )

        logger.info("Created experiment '%s' at '%s'", experiment_name, yml_path)
        return yml_path

    def list(self, project_dir: Path) -> list[dict[str, Any]]:
        project_config = self.project_manager.load(project_dir)
        return self.db.fetch_all(
            "SELECT * FROM experiments WHERE project_name = ? ORDER BY created_at DESC",
            (project_config["project-name"],),
        )

    def remove(
        self,
        project_dir: Path,
        experiment_name: str,
        results_only: bool = False,
    ) -> None:
        project_config = self.project_manager.load(project_dir)
        project_name = project_config["project-name"]
        project_dir = Path(project_dir).resolve()

        results_dir = project_dir / "experiments" / "results" / experiment_name
        if results_dir.exists():
            shutil.rmtree(results_dir)
            logger.info("Removed results for '%s'", experiment_name)

        if not results_only:
            yml_path = project_dir / "experiments" / f"{experiment_name}.yml"
            if yml_path.exists():
                yml_path.unlink()
            self.db.execute(
                "DELETE FROM experiments WHERE project_name = ? AND experiment_name = ?",
                (project_name, experiment_name),
            )
            self.db.commit()
            logger.info("Removed experiment '%s'", experiment_name)

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    def run(self, project_dir: Path, experiment_name: str) -> dict[str, Any]:
        project_dir = Path(project_dir).resolve()
        project_config = self.project_manager.load(project_dir)
        project_name = project_config["project-name"]

        exp_yml = project_dir / "experiments" / f"{experiment_name}.yml"
        if not exp_yml.exists():
            raise ComponentNotFoundError(
                f"Experiment config not found: '{exp_yml}'",
                hint=f"Run 'uesf experiment add' to create '{experiment_name}'.",
            )
        try:
            raw_config = yaml.safe_load(exp_yml.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise YAMLParseError(f"Invalid experiment YAML: {exc}") from exc

        cfg = ConfigValidator.normalize(raw_config)
        ConfigValidator.validate(cfg)

        exp_record_id = self._create_db_record(project_name, experiment_name, cfg)
        self._update_status(exp_record_id, "RUNNING")

        try:
            seed = cfg["seed"]
            torch.manual_seed(seed)
            np.random.seed(seed)

            device_name = cfg.get("device") or self.config.get("default_device")
            device = torch.device(device_name)

            model_cls, model_params, model_id = self._init_model(
                cfg, project_config, project_dir
            )
            trainer_cls, trainer_params, trainer_id = self._init_trainer(
                cfg, project_config, project_dir
            )
            metric_funcs = self._init_metrics(cfg, project_config, project_dir)

            self.db.execute(
                """UPDATE experiments SET model_id = ?, trainer_id = ?,
                   updated_at = CURRENT_TIMESTAMP WHERE id = ?""",
                (model_id, trainer_id, exp_record_id),
            )
            self.db.commit()

            dataset_cache, labels_cache, metadata_cache = self._load_all_datasets(
                cfg["datasets"]
            )
            apply_label_mapping(cfg["datasets"], labels_cache, metadata_cache)
            if cfg.get("alignment") and len(dataset_cache) > 1:
                self._apply_alignment(
                    dataset_cache, metadata_cache, cfg["alignment"]
                )
                check_sample_consistency(metadata_cache)

            output_dir = (
                project_dir / "experiments" / "results" / experiment_name
            )
            output_dir.mkdir(parents=True, exist_ok=True)

            ctx = ExperimentContext(
                config=cfg,
                dataset_cache=dataset_cache,
                labels_cache=labels_cache,
                metadata_cache=metadata_cache,
                experiment_id=exp_record_id,
                output_dir=output_dir,
                device=device,
                project_dir=project_dir,
                seed=seed,
                model_cls=model_cls,
                model_params=model_params,
                trainer_cls=trainer_cls,
                trainer_params=trainer_params,
                metric_funcs=metric_funcs,
                num_workers=int(self.config.get("num_workers")),
            )

            executor = ExperimentExecutor()
            exp_result = executor.execute(ctx)
            results_dict = self._serialize(exp_result)

            if exp_result.fold_results and not any(f.failed for f in exp_result.fold_results):
                self._update_status(exp_record_id, "COMPLETED", results=results_dict)
                logger.info("Experiment '%s' completed successfully", experiment_name)
            elif exp_result.fold_results and all(f.failed for f in exp_result.fold_results):
                self._update_status(exp_record_id, "FAILED", results=results_dict)
            else:
                self._update_status(exp_record_id, "PARTIAL", results=results_dict)

            return results_dict

        except UESFException as exc:
            self._update_status(exp_record_id, "FAILED", error=str(exc))
            logger.error("Experiment '%s' failed: %s", experiment_name, exc)
            raise
        except Exception as exc:
            self._update_status(exp_record_id, "FAILED", error=str(exc))
            logger.error("Experiment '%s' failed: %s", experiment_name, exc)
            raise

    # ------------------------------------------------------------------
    # Component resolution (unchanged from dev5)
    # ------------------------------------------------------------------

    def _init_model(self, exp_config, project_config, project_dir):
        model_config = exp_config.get("model", {})
        model_name = model_config.get("name")
        model_params = model_config.get("params", {}) or {}

        resolution = self.project_manager.resolve_component(
            model_name, "models", project_config, project_dir
        )
        model_id = None
        if resolution["source"] == "PROJECT" and resolution.get("entrypoint"):
            entry = resolution["entrypoint"]
            try:
                self.model_manager.get(model_name)
                rec = self.model_manager.detect_and_reregister(model_name, entry, project_dir)
            except ComponentNotFoundError:
                rec = self.model_manager.register(model_name, entry, project_dir)
            model_id = rec["id"]
        elif resolution.get("record"):
            model_id = resolution["record"]["id"]

        model_cls = self.model_manager.load_class(
            model_name,
            entrypoint=resolution.get("entrypoint"),
            project_dir=project_dir,
        )
        return model_cls, model_params, model_id

    def _init_trainer(self, exp_config, project_config, project_dir):
        trainer_config = exp_config.get("trainer") or {"name": "dummy_trainer"}
        trainer_name = trainer_config["name"]
        trainer_params = trainer_config.get("params", {}) or {}

        resolution = self.project_manager.resolve_component(
            trainer_name, "trainers", project_config, project_dir
        )
        trainer_id = None
        if resolution["source"] == "PROJECT" and resolution.get("entrypoint"):
            entry = resolution["entrypoint"]
            try:
                self.trainer_manager.get(trainer_name)
                rec = self.trainer_manager.detect_and_reregister(trainer_name, entry, project_dir)
            except ComponentNotFoundError:
                rec = self.trainer_manager.register(trainer_name, entry, project_dir)
            trainer_id = rec["id"]
        elif resolution.get("record"):
            trainer_id = resolution["record"]["id"]

        trainer_cls = self.trainer_manager.load_class(
            trainer_name,
            entrypoint=resolution.get("entrypoint"),
            project_dir=project_dir,
        )
        return trainer_cls, trainer_params, trainer_id

    def _init_metrics(self, cfg, project_config, project_dir):
        eval_config = cfg.get("evaluation") or {}
        logging_cfg = (cfg.get("training") or {}).get("logging") or {}

        # Union of eval.metrics + logging.train_metrics + logging.test_metrics
        # so one resolved pool backs every downstream Evaluator.
        metric_names: list[str] = list(eval_config.get("metrics") or ["accuracy"])
        for extra_key in ("train_metrics", "test_metrics"):
            for name in logging_cfg.get(extra_key) or []:
                if name not in metric_names:
                    metric_names.append(name)

        funcs: dict[str, Any] = {}
        for mname in metric_names:
            resolution = None
            try:
                resolution = self.project_manager.resolve_component(
                    mname, "metrics", project_config, project_dir
                )
            except ComponentNotFoundError:
                pass
            if (
                resolution
                and resolution["source"] == "PROJECT"
                and resolution.get("entrypoint")
            ):
                entry = resolution["entrypoint"]
                try:
                    self.metric_manager.get(mname)
                    self.metric_manager.detect_and_reregister(mname, entry, project_dir)
                except ComponentNotFoundError:
                    self.metric_manager.register(mname, entry, project_dir)
            funcs[mname] = self.metric_manager.load_metric(mname, project_dir=project_dir)
        return funcs

    # ------------------------------------------------------------------
    # Dataset loading / alignment
    # ------------------------------------------------------------------

    def _load_all_datasets(
        self, datasets_config: dict
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, dict]]:
        dataset_cache: dict[str, np.ndarray] = {}
        labels_cache: dict[str, np.ndarray] = {}
        metadata_cache: dict[str, dict] = {}
        for alias, ds_cfg in datasets_config.items():
            data, labels, meta = self._load_dataset(ds_cfg["name"])
            dataset_cache[alias] = data  # 5-D
            labels_cache[alias] = labels.reshape(-1)
            metadata_cache[alias] = meta
        return dataset_cache, labels_cache, metadata_cache

    def _load_dataset(self, name: str) -> tuple[np.ndarray, np.ndarray, dict]:
        row = self.db.fetch_one("SELECT * FROM preprocessed_datasets WHERE name = ?", (name,))
        if row is None:
            row = self.db.fetch_one("SELECT * FROM masked_datasets WHERE name = ?", (name,))
        if row is None:
            raise ComponentNotFoundError(
                f"Dataset '{name}' not found",
                hint="Register or preprocess the dataset first.",
            )

        data_dir = Path(row["data_dir_path"])
        # mmap the EEG array so the ~GB-scale tensor lives in OS page cache,
        # not process RSS; transforms that need RAM-resident data will copy
        # on demand (see _apply_uda_step_on_alias) or allocate fresh output
        # arrays (see _apply_step_per_dataset).
        data = np.load(str(data_dir / "eeg_data.npy"), mmap_mode="r")
        if data.dtype != np.float32:
            data = np.asarray(data, dtype=np.float32)
        labels = np.load(str(data_dir / "labels.npy"))

        n_ch = data.shape[-2] if data.ndim >= 3 else data.shape[-1]
        n_samp = data.shape[-1] if data.ndim >= 3 else 1
        n_classes = len(np.unique(labels))

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

    def _apply_alignment(
        self,
        dataset_cache: dict[str, np.ndarray],
        metadata_cache: dict[str, dict],
        alignment_config: dict,
    ) -> None:
        channel = alignment_config.get("channel")
        if channel:
            aligner = create_channel_aligner(channel)
            input_map = {}
            for alias, data in dataset_cache.items():
                electrodes = metadata_cache[alias].get("electrode_list") or []
                if not electrodes:
                    raise ConfigError(
                        f"Dataset '{alias}' has no electrode_list metadata — "
                        "cannot perform channel alignment.",
                        hint="Re-preprocess the dataset with electrode info.",
                    )
                input_map[alias] = (data, electrodes)
            aligned, common = aligner.align(input_map)
            for alias, arr in aligned.items():
                dataset_cache[alias] = arr
                metadata_cache[alias]["electrode_list"] = common
                metadata_cache[alias]["n_channels"] = len(common)

        if alignment_config.get("label", True):
            LabelAligner().validate(metadata_cache)

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _create_db_record(self, project_name, experiment_name, config):
        existing = self.db.fetch_one(
            "SELECT id FROM experiments WHERE project_name = ? AND experiment_name = ?",
            (project_name, experiment_name),
        )
        if existing:
            self.db.execute(
                """UPDATE experiments SET config = ?, status = 'PENDING',
                   updated_at = CURRENT_TIMESTAMP WHERE id = ?""",
                (json.dumps(config, ensure_ascii=False, default=str), existing["id"]),
            )
            self.db.commit()
            return existing["id"]
        with self.db.transaction() as cursor:
            cursor.execute(
                """INSERT INTO experiments (project_name, experiment_name, config, status)
                   VALUES (?, ?, ?, 'PENDING')""",
                (project_name, experiment_name, json.dumps(config, ensure_ascii=False, default=str)),
            )
            return cursor.lastrowid

    def _update_status(self, record_id, status, results=None, error=None):
        results_json = (
            json.dumps(results, ensure_ascii=False, default=str) if results else None
        )
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
                    parsed = json.loads(row["results"])
                    if metrics:
                        entry["selected_metrics"] = {
                            m: parsed.get(m, parsed.get(f"test_{m}"))
                            for m in metrics
                        }
                    entry["parsed_results"] = parsed
                except json.JSONDecodeError:
                    pass
            results.append(entry)
        return results

    # ------------------------------------------------------------------
    # Result serialization
    # ------------------------------------------------------------------

    def _serialize(self, result: ExperimentResult) -> dict[str, Any]:
        out: dict[str, Any] = {
            "n_folds": len(result.fold_results),
            "aggregation_mode": result.aggregation_mode,
            "aggregated_metrics": result.aggregated_metrics,
            "fold_results": [
                {
                    "metrics": fr.metrics,
                    "fold_info": fr.fold_info,
                    "failed": fr.failed,
                    "error": fr.error,
                }
                for fr in result.fold_results
            ],
        }
        # Surface first-fold test_* metrics at top level for convenience.
        if result.fold_results:
            first = result.fold_results[0]
            for k, v in first.metrics.items():
                out.setdefault(k, v)
        return out


# ---------------------------------------------------------------------------
# Executor + Strategies
# ---------------------------------------------------------------------------


class ExperimentExecutor:
    def execute(self, ctx: ExperimentContext) -> ExperimentResult:
        mode = ctx.config.get("mode", "regular")
        strategy: RegularExecutionStrategy | UDAExecutionStrategy
        if mode == "uda":
            strategy = UDAExecutionStrategy()
        else:
            strategy = RegularExecutionStrategy()
        fold_results = strategy.run(ctx)

        eval_cfg = ctx.config.get("evaluation", {})
        aggregation_mode = eval_cfg.get("k_fold_aggregation", "concat")
        aggregated = _aggregate_folds(fold_results, aggregation_mode, ctx.metric_funcs)
        return ExperimentResult(
            fold_results=fold_results,
            aggregated_metrics=aggregated,
            aggregation_mode=aggregation_mode,
        )


def _aggregate_folds(
    fold_results: list[FoldResult], mode: str, metric_funcs: dict
) -> dict[str, Any]:
    successful = [f for f in fold_results if not f.failed]
    if not successful:
        return {}
    if mode == "mean_std" or any(f.predictions is None for f in successful):
        # mean ± std of metrics
        agg: dict[str, Any] = {}
        names = set()
        for f in successful:
            for k, v in f.metrics.items():
                if isinstance(v, (int, float)):
                    names.add(k)
        for name in names:
            vals = [f.metrics[name] for f in successful if isinstance(f.metrics.get(name), (int, float))]
            if vals:
                mean = sum(vals) / len(vals)
                std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
                agg[name] = {"mean": mean, "std": std}
        return agg
    # concat
    all_preds: list = []
    all_targets: list = []
    for f in successful:
        all_preds.extend(f.predictions)
        all_targets.extend(f.targets)
    evaluator = Evaluator(metric_funcs)
    return evaluator.compute_epoch_metrics(all_preds, all_targets)


# ---------------------------------------------------------------------------
# Regular Strategy
# ---------------------------------------------------------------------------


class RegularExecutionStrategy:
    def run(self, ctx: ExperimentContext) -> list[FoldResult]:
        split_cfg = ctx.config["split"]
        dimension = split_cfg["dimension"]
        if dimension == "dataset":
            return self._run_cross_dataset(ctx, split_cfg)
        return self._run_single_dataset(ctx, split_cfg)

    def _run_single_dataset(self, ctx, split_cfg) -> list[FoldResult]:
        alias = next(iter(ctx.dataset_cache.keys()))
        splitter = create_splitter(split_cfg, seed=ctx.seed)
        folds = splitter.split(ctx.dataset_cache[alias])

        results: list[FoldResult] = []
        for fold_idx, sr in enumerate(folds):
            results.append(
                self._run_single_fold(
                    ctx=ctx,
                    fold_idx=fold_idx,
                    split_result=sr,
                    phases=("train", "val", "test"),
                    fold_info=sr.fold_info,
                )
            )
        return results

    def _run_cross_dataset(self, ctx, split_cfg) -> list[FoldResult]:
        strategy = split_cfg["strategy"]
        shuffle = split_cfg.get("shuffle", True)
        if strategy == "holdout":
            splitter = DatasetLevelSplitter(
                strategy="holdout",
                assign=split_cfg.get("assign"),
                shuffle=shuffle,
                seed=ctx.seed,
            )
        else:
            splitter = DatasetLevelSplitter(
                strategy="k-fold",
                k=split_cfg.get("k"),
                shuffle=shuffle,
                seed=ctx.seed,
            )
        ds_folds = splitter.split(list(ctx.dataset_cache.keys()))

        val_split_cfg = split_cfg.get("val_split")
        results: list[FoldResult] = []
        for fold_idx, ds_fold in enumerate(ds_folds):
            phase_indices: dict[str, dict[str, np.ndarray]] = {"train": {}, "val": {}, "test": {}}
            train_aliases = ds_fold.phase_aliases.get("train", [])
            test_aliases = ds_fold.phase_aliases.get("test", [])

            for ta in train_aliases:
                data = ctx.dataset_cache[ta]
                total = data.shape[0] * data.shape[1] * data.shape[2]
                all_idx = np.arange(total, dtype=int)
                if val_split_cfg is not None:
                    vs = ValSplitter(
                        dimension=val_split_cfg["dimension"],
                        val_ratio=val_split_cfg["val_ratio"],
                        shuffle=val_split_cfg.get("shuffle", True),
                        seed=ctx.seed + 1,
                    )
                    sub = vs.split(data)
                    phase_indices["train"][ta] = sub.train_indices
                    phase_indices["val"][ta] = sub.val_indices
                else:
                    phase_indices["train"][ta] = all_idx
            for te in test_aliases:
                data = ctx.dataset_cache[te]
                total = data.shape[0] * data.shape[1] * data.shape[2]
                phase_indices["test"][te] = np.arange(total, dtype=int)

            msr = MultiDatasetSplitResult(
                phase_indices=phase_indices,
                fold_info=dict(ds_fold.fold_info),
            )
            results.append(
                self._run_single_fold(
                    ctx=ctx,
                    fold_idx=fold_idx,
                    split_result=msr,
                    phases=("train", "val", "test"),
                    fold_info=msr.fold_info,
                )
            )
        return results

    def _run_single_fold(
        self,
        ctx: ExperimentContext,
        fold_idx: int,
        split_result,
        phases: tuple[str, ...],
        fold_info: dict,
    ) -> FoldResult:
        fold_dir = ctx.output_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Apply transforms (mutates dataset_cache snapshot)
            cache_snapshot = _snapshot_cache(ctx.dataset_cache)
            transforms_per_alias = _collect_transforms_regular(ctx.config, split_result)
            split_indices = _split_indices_for_transforms(split_result, ctx.dataset_cache)
            apply_transforms(
                transforms_per_alias,
                ctx.dataset_cache,
                split_indices,
            )

            # Build dataloaders
            builder = DataloaderBuilder(num_workers=ctx.num_workers)
            batch_size = ctx.config.get("training", {}).get("batch_size", 32)
            phase_loaders = {}
            for phase in phases:
                cdata, clabels = prepare_channel_data(
                    split_result, ctx.dataset_cache, ctx.labels_cache, phase
                )
                if not cdata:
                    phase_loaders[phase] = None
                else:
                    phase_loaders[phase] = builder.build(
                        cdata, clabels, batch_size=batch_size, shuffle=(phase == "train"),
                    )

            result = _train_and_evaluate(
                ctx=ctx,
                train_loader=phase_loaders.get("train"),
                val_loader=phase_loaders.get("val"),
                test_loader=phase_loaders.get("test"),
                fold_dir=fold_dir,
            )
            return FoldResult(
                metrics=result["metrics"],
                fold_info=fold_info,
                predictions=result["predictions"],
                targets=result["targets"],
            )
        except Exception as exc:  # noqa: BLE001 - fold-level isolation
            logger.exception("Fold %d failed: %s", fold_idx, exc)
            return FoldResult(
                metrics={},
                fold_info=fold_info,
                failed=True,
                error=str(exc),
            )
        finally:
            # Restore pre-transform data for next fold
            ctx.dataset_cache.clear()
            ctx.dataset_cache.update(cache_snapshot)


# ---------------------------------------------------------------------------
# UDA Strategy
# ---------------------------------------------------------------------------


class UDAExecutionStrategy:
    def run(self, ctx: ExperimentContext) -> list[FoldResult]:
        uda_cfg = ctx.config["uda"]
        orchestrator = UDAOrchestrator(uda_cfg, seed=ctx.seed)
        folds = orchestrator.split(ctx.dataset_cache)
        adaptation = uda_cfg["adaptation"]

        results: list[FoldResult] = []
        for fold_idx, uda_split in enumerate(folds):
            fold_dir = ctx.output_dir / f"fold_{fold_idx}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            try:
                cache_snapshot = _snapshot_cache(ctx.dataset_cache)
                transforms_per_alias = _collect_transforms_uda(ctx.config, ctx.dataset_cache)
                apply_transforms_uda(
                    transforms_per_alias,
                    ctx.dataset_cache,
                    uda_split,
                    adaptation=adaptation,
                )

                builder = DataloaderBuilder(num_workers=ctx.num_workers)
                batch_size = ctx.config.get("training", {}).get("batch_size", 32)
                phases = prepare_uda_channel_data(
                    uda_split, ctx.dataset_cache, ctx.labels_cache
                )

                def _build(phase):
                    cdata, clabels = phases[phase]
                    if not cdata:
                        return None
                    return builder.build(
                        cdata, clabels, batch_size=batch_size, shuffle=(phase == "train"),
                    )

                result = _train_and_evaluate(
                    ctx=ctx,
                    train_loader=_build("train"),
                    val_loader=_build("val"),
                    test_loader=_build("test"),
                    fold_dir=fold_dir,
                )
                results.append(
                    FoldResult(
                        metrics=result["metrics"],
                        fold_info=uda_split.fold_info,
                        predictions=result["predictions"],
                        targets=result["targets"],
                    )
                )
            except Exception as exc:  # noqa: BLE001 - fold-level isolation
                logger.exception("UDA fold %d failed: %s", fold_idx, exc)
                results.append(
                    FoldResult(
                        metrics={},
                        fold_info=uda_split.fold_info,
                        failed=True,
                        error=str(exc),
                    )
                )
            finally:
                ctx.dataset_cache.clear()
                ctx.dataset_cache.update(cache_snapshot)
        return results


# ---------------------------------------------------------------------------
# Shared train/eval helper
# ---------------------------------------------------------------------------


def _train_and_evaluate(
    ctx: ExperimentContext,
    train_loader,
    val_loader,
    test_loader,
    fold_dir: Path,
) -> dict[str, Any]:
    if train_loader is None:
        raise ConfigError("train_loader is empty — no training samples available.")

    first_meta = next(iter(ctx.metadata_cache.values()))
    model = ctx.model_cls(
        n_channels=first_meta["n_channels"],
        n_samples=first_meta["n_samples"],
        n_classes=first_meta["n_classes"],
        **ctx.model_params,
    )
    trainer = ctx.trainer_cls(model, ctx.device, **ctx.trainer_params)
    evaluator = Evaluator(ctx.metric_funcs)

    training_config = ctx.config.get("training", {})

    custom_optim = trainer.configure_optimizers()
    if custom_optim is not None:
        optimizer, scheduler = custom_optim
    else:
        opt_config = training_config.get(
            "optimizer", {"name": "adam", "params": {"lr": 0.001}}
        )
        optimizer = resolve_optimizer(
            opt_config["name"], model.parameters(), opt_config.get("params", {})
        )
        sched_config = training_config.get("scheduler")
        scheduler = None
        if sched_config:
            scheduler = resolve_scheduler(
                sched_config["name"], optimizer, sched_config.get("params", {})
            )

    logging_cfg = training_config.get("logging") or {}
    training_logger = None
    log_every_n = 1
    log_every_n_steps: int | None = None
    train_step_filter: list[str] | None = None
    val_metrics_filter: list[str] | None = None
    train_metrics_names: list[str] = []
    test_metrics_names: list[str] = []
    log_lr = True
    if isinstance(logging_cfg, dict) and logging_cfg:
        training_logger = create_logger(logging_cfg, fold_dir / "tb_logs")
        log_every_n = int(logging_cfg.get("log_every_n_epochs", 1))
        log_every_n_steps = logging_cfg.get("log_every_n_steps")
        train_step_filter = logging_cfg.get("train_step_scalars")
        val_metrics_filter = logging_cfg.get("val_metrics")
        train_metrics_names = list(logging_cfg.get("train_metrics") or [])
        test_metrics_names = list(logging_cfg.get("test_metrics") or [])
        log_lr = bool(logging_cfg.get("log_lr", True))
        if training_logger is not None and logging_cfg.get("log_graph"):
            try:
                sample = get_sample_input(train_loader)
                training_logger.log_graph(model.to(ctx.device), sample.to(ctx.device))
            except Exception as exc:  # noqa: BLE001
                logger.warning("log_graph failed: %s", exc)

    train_evaluator = (
        Evaluator({n: ctx.metric_funcs[n] for n in train_metrics_names})
        if train_metrics_names
        else None
    )
    test_evaluator = (
        Evaluator({n: ctx.metric_funcs[n] for n in test_metrics_names})
        if test_metrics_names
        else None
    )

    if test_metrics_names:
        logger.warning(
            "training.logging.test_metrics is set — tracking test-set metrics "
            "DURING training creates data-leakage risk via model selection / "
            "hyperparameter tuning. Prefer val_metrics for monitoring; only "
            "enable test_metrics for controlled ablation / analysis where you "
            "accept the risk."
        )

    ckpt_cfg = training_config.get("checkpoint")
    checkpoint_dir = None
    checkpoint_metric = None
    if isinstance(ckpt_cfg, dict):
        checkpoint_dir = fold_dir / "checkpoints"
        checkpoint_metric = ckpt_cfg.get("metric")

    epochs = training_config.get("epochs", 1)
    runner = Runner(trainer, evaluator, ctx.device, {"epochs": epochs, **training_config})

    run_result = runner.run(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_dir=checkpoint_dir,
        checkpoint_metric=checkpoint_metric,
        early_stopping_config=training_config.get("early_stopping"),
        training_logger=training_logger,
        log_every_n_epochs=log_every_n,
        log_every_n_steps=log_every_n_steps,
        train_step_filter=train_step_filter,
        val_metrics_filter=val_metrics_filter,
        train_evaluator=train_evaluator,
        test_evaluator=test_evaluator,
        test_loader=test_loader,
        log_lr=log_lr,
    )

    metrics = {**run_result["best_metrics"]}
    predictions: list = []
    targets: list = []
    if test_loader is not None and len(test_loader) > 0:
        eval_cfg = ctx.config.get("evaluation") or {}
        test_with = eval_cfg.get("test_with", "last")
        if test_with == "best":
            best_ckpt = (
                checkpoint_dir / "best_model.pt" if checkpoint_dir is not None else None
            )
            if best_ckpt is not None and best_ckpt.exists():
                trainer.model.load_state_dict(
                    torch.load(best_ckpt, map_location=ctx.device)
                )
                logger.info("Loaded best checkpoint for test evaluation: %s", best_ckpt)
            else:
                logger.warning(
                    "evaluation.test_with='best' but no best checkpoint was saved "
                    "during training (e.g. validation never produced %r) — falling "
                    "back to the model state at training end.",
                    checkpoint_metric,
                )
        test_metrics, predictions, targets = runner.validate_epoch(test_loader)
        for k, v in test_metrics.items():
            metrics[f"test_{k}"] = v
    metrics["epochs_run"] = run_result["epochs_run"]

    # Help GC
    del model, trainer, runner, optimizer
    if scheduler is not None:
        del scheduler
    if ctx.device.type == "cuda":
        torch.cuda.empty_cache()

    return {"metrics": metrics, "predictions": predictions, "targets": targets}


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def _snapshot_cache(cache: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    # Shallow dict copy — saves array references, not data. transforms never
    # write to cache[alias] in place; they allocate fresh arrays and rebind
    # the dict entry (see transforms.py). Restoring via dict reassignment
    # drops the fold's transformed arrays and reinstates the originals.
    return dict(cache)


def _collect_transforms_regular(config: dict, split_result) -> dict[str, list[dict]]:
    datasets = config["datasets"]
    out: dict[str, list[dict]] = {}
    if isinstance(split_result, MultiDatasetSplitResult):
        for alias in split_result.phase_indices.get("train", {}).keys():
            out[alias] = datasets[alias].get("transforms") or []
        for alias in split_result.phase_indices.get("test", {}).keys():
            out.setdefault(alias, datasets[alias].get("transforms") or [])
    else:
        alias = next(iter(datasets.keys()))
        out[alias] = datasets[alias].get("transforms") or []
    return out


def _collect_transforms_uda(config: dict, dataset_cache: dict) -> dict[str, list[dict]]:
    return {alias: config["datasets"][alias].get("transforms") or [] for alias in dataset_cache}


def _split_indices_for_transforms(split_result, dataset_cache):
    if isinstance(split_result, MultiDatasetSplitResult):
        out: dict[str, dict[str, np.ndarray]] = {}
        for phase, aliases_map in split_result.phase_indices.items():
            for alias, idx in aliases_map.items():
                out.setdefault(alias, {})[phase] = idx
        return out
    # Single dataset
    alias = next(iter(dataset_cache.keys()))
    assert isinstance(split_result, SplitResult)
    return {
        alias: {
            "train": split_result.train_indices,
            "val": split_result.val_indices,
            "test": split_result.test_indices,
        }
    }


# ---------------------------------------------------------------------------
# Experiment template (dev6 YAML)
# ---------------------------------------------------------------------------


def _experiment_template(name: str, description: str | None = None) -> str:
    return f"""\
experiment_name: {name}
description: "{description or ''}"
seed: 42

# mode: regular | uda    (default: regular)

datasets:
  main:
    name: ""              # preprocessed / masked dataset name
    # transforms:         # optional, fit-on-train, applied to all phases
    #   - name: zscore_normalize
    #     scope: per_dataset   # per_dataset | global  (default: per_dataset)

split:
  strategy: holdout       # holdout | k-fold
  dimension: subject      # subject | session | recording | flatten | dataset
  shuffle: true
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  # val_split:            # optional — independent validation split
  #   dimension: recording
  #   val_ratio: 0.1

model:
  name: ""                # model name (project-local or global)
  params: {{}}

# trainer:
#   name: ""
#   params: {{}}

training:
  epochs: 50
  batch_size: 32
  optimizer:
    name: adam
    params: {{ lr: 0.001 }}
  # early_stopping:
  #   metric: val_accuracy
  #   patience: 10
  #   mode: max
  # checkpoint:
  #   metric: val_accuracy
  #   mode: max
  # logging:
  #   backend: tensorboard
  #   log_every_n_epochs: 1
  #   log_graph: false

evaluation:
  metrics: [accuracy, f1_score]
  k_fold_aggregation: concat

# --- UDA example (uncomment to use) ---
# mode: uda
# uda:
#   domain:
#     strategy: holdout
#     dimension: subject
#     target_count: 1
#   adaptation: transductive
#   source:
#     split:
#       dimension: recording
#       val_ratio: 0.2
#   # target:
#   #   split:
#   #     strategy: holdout
#   #     dimension: recording
#   #     train_ratio: 0.7
#   #     test_ratio: 0.3

# --- Cross-dataset alignment (multi-dataset experiments) ---
# alignment:
#   channel: intersection
#   label: true
"""


# For backward compatibility of imports
__all__ = [
    "ExperimentManager",
    "ExperimentExecutor",
    "ExperimentContext",
    "FoldResult",
    "ExperimentResult",
    "RegularExecutionStrategy",
    "UDAExecutionStrategy",
]
