"""E2E test: full workflow from preprocessed dataset to experiment results.

This test simulates a complete UESF workflow:
1. Create fake preprocessed dataset (.npy files + DB record)
2. Initialize a project with model/trainer
3. Create experiment config
4. Run the experiment
5. Query results
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
    """Set up a complete environment for E2E testing."""
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


def _create_fake_preprocessed(db, data_dir: Path, name: str, n_samples=60, n_channels=8, n_timepoints=50, n_classes=2):
    """Create a fake preprocessed dataset with .npy files and DB record."""
    ds_dir = data_dir / name
    ds_dir.mkdir(parents=True)

    # Generate random data: (n_samples, n_channels, n_timepoints)
    data = np.random.randn(n_samples, n_channels, n_timepoints).astype(np.float32)
    labels = np.random.randint(0, n_classes, n_samples).astype(np.int64)

    np.save(str(ds_dir / "eeg_data.npy"), data)
    np.save(str(ds_dir / "labels.npy"), labels)

    db.execute(
        """INSERT INTO preprocessed_datasets
           (name, data_dir_path, data_shape, numeric_to_semantic,
            n_subjects, n_channels, n_samples)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (name, str(ds_dir), f"[{n_samples}, {n_channels}, {n_timepoints}]",
         '{"0": "class_0", "1": "class_1"}', n_samples, n_channels, n_timepoints),
    )
    db.commit()

    return ds_dir


def _create_project_with_dummy(project_dir: Path, dataset_name: str):
    """Create a project with DummyModel and DummyTrainer pointing to built-in code."""
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "experiments").mkdir(exist_ok=True)

    # Write model file
    model_file = project_dir / "model.py"
    model_file.write_text(
        'from uesf.components.dummy import DummyModel\n',
        encoding="utf-8",
    )

    # Write trainer file
    trainer_file = project_dir / "trainer.py"
    trainer_file.write_text(
        'from uesf.components.dummy import DummyTrainer\n',
        encoding="utf-8",
    )

    # Write project.yml
    project_config = {
        "project-name": "test_project",
        "description": "E2E test project",
        "preprocessed_datasets": [dataset_name],
        "models": {
            "dummy_model": {
                "entrypoint": f"{model_file}:DummyModel",
            },
        },
        "trainers": {
            "dummy_trainer": {
                "entrypoint": f"{trainer_file}:DummyTrainer",
            },
        },
    }

    yml_path = project_dir / "project.yml"
    yml_path.write_text(
        yaml.dump(project_config, default_flow_style=False),
        encoding="utf-8",
    )

    return yml_path


def _create_experiment_yml(project_dir: Path, dataset_name: str, experiment_name: str = "test_exp"):
    """Create a minimal experiment YAML file."""
    exp_config = {
        "name": experiment_name,
        "description": "E2E test experiment",
        "seed": 42,
        "model": {
            "name": "dummy_model",
            "params": {},
        },
        "trainer": {
            "name": "dummy_trainer",
            "params": {},
        },
        "datasets": {
            "main_dataset": {
                "name": dataset_name,
                "split": {
                    "strategy": "holdout",
                    "dimension": "none",
                    "shuffle": True,
                    "train_ratio": 0.6,
                    "val_ratio": 0.2,
                    "test_ratio": 0.2,
                },
                "transforms": [
                    {"name": "zscore_normalize", "fit_on": "train", "apply_to": "all"},
                ],
            },
        },
        "dataloaders": {
            "train": {"main": "main_dataset.train"},
            "val": {"main": "main_dataset.val"},
            "test": {"main": "main_dataset.test"},
        },
        "training": {
            "epochs": 3,
            "batch_size": 8,
            "optimizer": {
                "name": "adam",
                "params": {"lr": 0.01},
            },
        },
        "evaluation": {
            "metrics": ["accuracy", "f1_score"],
            "k_fold_aggregation": "concat",
        },
        "logging": {
            "use_wandb": False,
            "checkpoint_metric": "val_accuracy",
        },
    }

    yml_path = project_dir / "experiments" / f"{experiment_name}.yml"
    yml_path.write_text(
        yaml.dump(exp_config, default_flow_style=False),
        encoding="utf-8",
    )
    return yml_path


class TestFullWorkflow:
    def test_end_to_end(self, setup_env, tmp_path):
        """Complete workflow: create dataset → project → experiment → run → query."""
        env = setup_env
        data_dir = tmp_path / "data"
        project_dir = tmp_path / "project"
        dataset_name = "fake_preprocessed"

        # 1. Create fake preprocessed dataset
        _create_fake_preprocessed(env["db"], data_dir, dataset_name)

        # 2. Create project
        _create_project_with_dummy(project_dir, dataset_name)

        # 3. Create experiment config
        _create_experiment_yml(project_dir, dataset_name)

        # 4. Run experiment
        results = env["exp_mgr"].run(project_dir, "test_exp")

        # 5. Verify results
        assert "epochs_run" in results
        assert results["epochs_run"] == 3

        # Check test metrics exist
        assert "test_accuracy" in results or "val_accuracy" in results

        # 6. Query results
        experiments = env["exp_mgr"].query(
            project_name="test_project",
            status="COMPLETED",
        )
        assert len(experiments) == 1
        assert experiments[0]["experiment_name"] == "test_exp"
        assert experiments[0]["status"] == "COMPLETED"

    def test_kfold_workflow(self, setup_env, tmp_path):
        """K-Fold cross-validation workflow."""
        env = setup_env
        data_dir = tmp_path / "data"
        project_dir = tmp_path / "project"
        dataset_name = "kfold_dataset"

        _create_fake_preprocessed(env["db"], data_dir, dataset_name, n_samples=30)
        _create_project_with_dummy(project_dir, dataset_name)

        # Create K-Fold experiment config
        exp_config = {
            "name": "kfold_exp",
            "seed": 42,
            "model": {"name": "dummy_model", "params": {}},
            "trainer": {"name": "dummy_trainer", "params": {}},
            "datasets": {
                "main_dataset": {
                    "name": dataset_name,
                    "split": {
                        "strategy": "k-fold",
                        "dimension": "none",
                        "k-folds": 3,
                        "shuffle": True,
                    },
                },
            },
            "dataloaders": {
                "train": {"main": "main_dataset.train"},
                "test": {"main": "main_dataset.test"},
            },
            "training": {"epochs": 2, "batch_size": 8, "optimizer": {"name": "adam", "params": {"lr": 0.01}}},
            "evaluation": {"metrics": ["accuracy"], "k_fold_aggregation": "mean_std"},
        }

        (project_dir / "experiments").mkdir(exist_ok=True)
        yml_path = project_dir / "experiments" / "kfold_exp.yml"
        yml_path.write_text(yaml.dump(exp_config, default_flow_style=False), encoding="utf-8")

        results = env["exp_mgr"].run(project_dir, "kfold_exp")
        assert results["n_folds"] == 3
        assert "fold_results" in results

    def test_experiment_add_and_list(self, setup_env, tmp_path):
        """Test experiment add and list CLI-level operations."""
        env = setup_env
        project_dir = tmp_path / "proj"
        env["project_mgr"].init(project_dir)

        # Add experiment
        yml_path = env["exp_mgr"].add(project_dir, experiment_name="my_exp")
        assert yml_path.exists()
        assert "my_exp" in yml_path.name

    def test_failed_experiment_records_status(self, setup_env, tmp_path):
        """Experiment that fails should have FAILED status in DB."""
        env = setup_env
        project_dir = tmp_path / "project"
        _create_project_with_dummy(project_dir, "nonexistent_dataset")

        # Create experiment with nonexistent dataset
        _create_experiment_yml(project_dir, "nonexistent_dataset", "fail_exp")

        with pytest.raises(Exception):
            env["exp_mgr"].run(project_dir, "fail_exp")

        # Check DB record
        experiments = env["exp_mgr"].query(project_name="test_project", status="FAILED")
        assert len(experiments) == 1
        assert experiments[0]["status"] == "FAILED"
