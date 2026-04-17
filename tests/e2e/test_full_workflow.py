"""E2E tests: full workflow from preprocessed dataset to experiment results.

Uses the dev6 YAML surface — top-level `split`, `mode: regular | uda`,
`training.logging`, auto-wired channel mapping (no `dataloaders` section).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from uesf.core.config import ConfigManager
from uesf.managers.experiment_manager import ExperimentManager
from uesf.managers.metric_manager import MetricManager
from uesf.managers.model_manager import ModelManager
from uesf.managers.project_manager import ProjectManager
from uesf.managers.trainer_manager import TrainerManager


@pytest.fixture
def setup_env(uesf_home, db):
    config = ConfigManager(db, uesf_home)
    project_mgr = ProjectManager(db, config)
    model_mgr = ModelManager(db, config)
    trainer_mgr = TrainerManager(db, config)
    metric_mgr = MetricManager(db, config)
    exp_mgr = ExperimentManager(db, config, project_mgr, model_mgr, trainer_mgr, metric_mgr)
    return {
        "db": db,
        "config": config,
        "project_mgr": project_mgr,
        "model_mgr": model_mgr,
        "trainer_mgr": trainer_mgr,
        "metric_mgr": metric_mgr,
        "exp_mgr": exp_mgr,
    }


def _create_fake_preprocessed(
    db, data_dir: Path, name: str,
    n_subjects=4, n_sessions=2, n_recordings=5,
    n_channels=8, n_timepoints=32, n_classes=2,
):
    ds_dir = data_dir / name
    ds_dir.mkdir(parents=True)
    data = np.random.randn(
        n_subjects, n_sessions, n_recordings, n_channels, n_timepoints
    ).astype(np.float32)
    total = n_subjects * n_sessions * n_recordings
    labels = np.random.randint(0, n_classes, total).astype(np.int64)
    np.save(str(ds_dir / "eeg_data.npy"), data)
    np.save(str(ds_dir / "labels.npy"), labels)

    shape_str = f"[{n_subjects}, {n_sessions}, {n_recordings}, {n_channels}, {n_timepoints}]"
    db.execute(
        """INSERT INTO preprocessed_datasets
           (name, data_dir_path, data_shape, numeric_to_semantic,
            n_subjects, n_channels, n_samples)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (name, str(ds_dir), shape_str,
         '{"0": "class_0", "1": "class_1"}',
         n_subjects, n_channels, n_timepoints),
    )
    db.commit()
    return ds_dir


def _create_project_with_dummy(project_dir: Path, dataset_names: list[str]):
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "experiments").mkdir(exist_ok=True)

    (project_dir / "model.py").write_text(
        "from uesf.components.dummy import DummyModel\n", encoding="utf-8"
    )
    (project_dir / "trainer.py").write_text(
        "from uesf.components.dummy import DummyTrainer\n", encoding="utf-8"
    )

    project_config = {
        "project-name": "test_project",
        "description": "E2E test project",
        "preprocessed_datasets": dataset_names,
        "models": {
            "dummy_model": {
                "entrypoint": f"{project_dir / 'model.py'}:DummyModel",
            }
        },
        "trainers": {
            "dummy_trainer": {
                "entrypoint": f"{project_dir / 'trainer.py'}:DummyTrainer",
            }
        },
    }
    yml = project_dir / "project.yml"
    yml.write_text(yaml.dump(project_config, default_flow_style=False), encoding="utf-8")
    return yml


def _write_experiment(project_dir: Path, experiment_name: str, exp_config: dict) -> Path:
    (project_dir / "experiments").mkdir(exist_ok=True)
    yml_path = project_dir / "experiments" / f"{experiment_name}.yml"
    yml_path.write_text(yaml.dump(exp_config, default_flow_style=False), encoding="utf-8")
    return yml_path


class TestFullWorkflow:
    def test_regular_holdout_subject(self, setup_env, tmp_path):
        env = setup_env
        _create_fake_preprocessed(env["db"], tmp_path / "data", "ds1")
        _create_project_with_dummy(tmp_path / "project", ["ds1"])

        exp_config = {
            "experiment_name": "reg_holdout",
            "seed": 42,
            "datasets": {"main": {"name": "ds1"}},
            "split": {
                "strategy": "holdout",
                "dimension": "subject",
                "train_ratio": 0.5,
                "val_ratio": 0.25,
                "test_ratio": 0.25,
                "shuffle": True,
            },
            "model": {"name": "dummy_model", "params": {}},
            "trainer": {"name": "dummy_trainer", "params": {}},
            "training": {
                "epochs": 2,
                "batch_size": 4,
                "optimizer": {"name": "adam", "params": {"lr": 0.01}},
            },
            "evaluation": {"metrics": ["accuracy"], "k_fold_aggregation": "concat"},
        }
        _write_experiment(tmp_path / "project", "reg_holdout", exp_config)

        results = env["exp_mgr"].run(tmp_path / "project", "reg_holdout")
        assert results["n_folds"] == 1
        assert results["fold_results"][0]["failed"] is False
        assert "test_accuracy" in results["aggregated_metrics"] or "test_accuracy" in results

        experiments = env["exp_mgr"].query(project_name="test_project", status="COMPLETED")
        assert len(experiments) == 1

    def test_regular_kfold_subject_with_val_split(self, setup_env, tmp_path):
        env = setup_env
        _create_fake_preprocessed(
            env["db"], tmp_path / "data", "kf_ds", n_subjects=4, n_sessions=2, n_recordings=4,
        )
        _create_project_with_dummy(tmp_path / "project", ["kf_ds"])

        exp_config = {
            "experiment_name": "reg_kfold",
            "seed": 0,
            "datasets": {"main": {"name": "kf_ds"}},
            "split": {
                "strategy": "k-fold",
                "dimension": "subject",
                "k": 4,
                "val_split": {
                    "dimension": "recording",
                    "val_ratio": 0.25,
                    "shuffle": False,
                },
                "shuffle": False,
            },
            "model": {"name": "dummy_model", "params": {}},
            "trainer": {"name": "dummy_trainer", "params": {}},
            "training": {
                "epochs": 1,
                "batch_size": 4,
                "optimizer": {"name": "adam", "params": {"lr": 0.01}},
            },
            "evaluation": {"metrics": ["accuracy"], "k_fold_aggregation": "mean_std"},
        }
        _write_experiment(tmp_path / "project", "reg_kfold", exp_config)

        results = env["exp_mgr"].run(tmp_path / "project", "reg_kfold")
        assert results["n_folds"] == 4
        for fr in results["fold_results"]:
            assert fr["failed"] is False
            assert "fold_idx" in fr["fold_info"]

    def test_experiment_add_and_list(self, setup_env, tmp_path):
        env = setup_env
        project_dir = tmp_path / "proj"
        env["project_mgr"].init(project_dir)
        yml = env["exp_mgr"].add(project_dir, experiment_name="my_exp")
        assert yml.exists() and "my_exp" in yml.name

    def test_missing_dataset_fails_with_failed_status(self, setup_env, tmp_path):
        env = setup_env
        project_dir = tmp_path / "project"
        _create_project_with_dummy(project_dir, ["nonexistent"])

        exp_config = {
            "experiment_name": "fail_exp",
            "seed": 42,
            "datasets": {"main": {"name": "nonexistent"}},
            "split": {
                "strategy": "holdout",
                "dimension": "subject",
                "train_ratio": 0.7,
                "val_ratio": 0.15,
                "test_ratio": 0.15,
            },
            "model": {"name": "dummy_model", "params": {}},
            "trainer": {"name": "dummy_trainer", "params": {}},
            "training": {"epochs": 1, "batch_size": 4, "optimizer": {"name": "adam", "params": {"lr": 0.01}}},
            "evaluation": {"metrics": ["accuracy"]},
        }
        _write_experiment(project_dir, "fail_exp", exp_config)

        with pytest.raises(Exception):
            env["exp_mgr"].run(project_dir, "fail_exp")

        failed = env["exp_mgr"].query(project_name="test_project", status="FAILED")
        assert len(failed) == 1
        assert failed[0]["status"] == "FAILED"
