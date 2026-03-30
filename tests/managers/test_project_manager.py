"""Tests for ProjectManager."""

from __future__ import annotations

import pytest
import yaml

from uesf.core.config import ConfigManager
from uesf.core.exceptions import ComponentNotFoundError, MissingRequiredKeyError, YAMLParseError
from uesf.managers.project_manager import ProjectManager


@pytest.fixture
def config(db, uesf_home):
    return ConfigManager(db, uesf_home)


@pytest.fixture
def manager(db, config):
    return ProjectManager(db, config)


class TestInit:
    def test_creates_project_yml(self, manager, tmp_path):
        yml_path = manager.init(tmp_path / "myproject")
        assert yml_path.exists()
        data = yaml.safe_load(yml_path.read_text())
        assert data["project-name"] == "myproject"

    def test_creates_experiments_dir(self, manager, tmp_path):
        project_dir = tmp_path / "proj"
        manager.init(project_dir)
        assert (project_dir / "experiments").is_dir()

    def test_does_not_overwrite_existing(self, manager, tmp_path):
        project_dir = tmp_path / "existing"
        project_dir.mkdir()
        yml_path = project_dir / "project.yml"
        yml_path.write_text("project-name: original\n", encoding="utf-8")

        result = manager.init(project_dir)
        data = yaml.safe_load(result.read_text())
        assert data["project-name"] == "original"


class TestLoad:
    def test_load_valid(self, manager, tmp_path):
        yml = tmp_path / "project.yml"
        yml.write_text(
            "project-name: test\ndescription: A test project\n",
            encoding="utf-8",
        )
        data = manager.load(tmp_path)
        assert data["project-name"] == "test"
        assert data["description"] == "A test project"

    def test_load_missing_yml(self, manager, tmp_path):
        with pytest.raises(ComponentNotFoundError, match="project.yml not found"):
            manager.load(tmp_path)

    def test_load_invalid_yaml(self, manager, tmp_path):
        yml = tmp_path / "project.yml"
        yml.write_text(":\n  :\n    - [invalid", encoding="utf-8")
        with pytest.raises(YAMLParseError):
            manager.load(tmp_path)

    def test_load_missing_project_name(self, manager, tmp_path):
        yml = tmp_path / "project.yml"
        yml.write_text("description: no name\n", encoding="utf-8")
        with pytest.raises(MissingRequiredKeyError, match="project-name"):
            manager.load(tmp_path)

    def test_load_not_a_mapping(self, manager, tmp_path):
        yml = tmp_path / "project.yml"
        yml.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(YAMLParseError, match="mapping"):
            manager.load(tmp_path)


class TestInfo:
    def test_info_returns_summary(self, manager, tmp_path):
        yml = tmp_path / "project.yml"
        yml.write_text(
            yaml.dump({
                "project-name": "test_proj",
                "description": "My project",
                "preprocessed_datasets": ["ds1", "ds2"],
                "models": {"m1": {"entrypoint": "m1.py:M1"}},
                "trainers": {"t1": {"entrypoint": "t1.py:T1"}},
                "metrics": {"acc": {"entrypoint": "acc.py:accuracy"}},
            }),
            encoding="utf-8",
        )
        info = manager.info(tmp_path)
        assert info["project_name"] == "test_proj"
        assert info["preprocessed_datasets"] == ["ds1", "ds2"]
        assert info["models"] == ["m1"]
        assert info["trainers"] == ["t1"]
        assert info["metrics"] == ["acc"]

    def test_info_empty_sections(self, manager, tmp_path):
        yml = tmp_path / "project.yml"
        yml.write_text("project-name: empty\n", encoding="utf-8")
        info = manager.info(tmp_path)
        assert info["models"] == []
        assert info["trainers"] == []
        assert info["metrics"] == []


class TestResolveComponent:
    def test_resolve_project_level(self, manager, tmp_path):
        config = {
            "project-name": "test",
            "models": {"my_model": {"entrypoint": "./src/model.py:MyModel"}},
        }
        result = manager.resolve_component("my_model", "models", config, tmp_path)
        assert result["source"] == "PROJECT"
        assert result["name"] == "my_model"
        assert result["entrypoint"] == "./src/model.py:MyModel"

    def test_resolve_global(self, manager, db, tmp_path):
        # Insert a global model into db
        db.execute(
            "INSERT INTO models (name, description, model_type, is_obsolete) VALUES (?, ?, ?, ?)",
            ("global_model", "A global model", "GLOBAL", 0),
        )
        db.commit()

        config = {"project-name": "test"}
        result = manager.resolve_component("global_model", "models", config, tmp_path)
        assert result["source"] == "GLOBAL"

    def test_resolve_not_found(self, manager, tmp_path):
        config = {"project-name": "test"}
        with pytest.raises(ComponentNotFoundError, match="not found"):
            manager.resolve_component("missing", "models", config, tmp_path)

    def test_project_shadows_global(self, manager, db, tmp_path):
        # Insert a global model
        db.execute(
            "INSERT INTO models (name, description, model_type, is_obsolete) VALUES (?, ?, ?, ?)",
            ("shared_name", "Global version", "GLOBAL", 0),
        )
        db.commit()

        config = {
            "project-name": "test",
            "models": {"shared_name": {"entrypoint": "./local.py:Model"}},
        }
        result = manager.resolve_component("shared_name", "models", config, tmp_path)
        assert result["source"] == "PROJECT"

    def test_resolve_trainer(self, manager, db, tmp_path):
        db.execute(
            "INSERT INTO trainers (name, description, trainer_type, is_obsolete) VALUES (?, ?, ?, ?)",
            ("global_trainer", "A trainer", "GLOBAL", 0),
        )
        db.commit()

        config = {"project-name": "test"}
        result = manager.resolve_component("global_trainer", "trainers", config, tmp_path)
        assert result["source"] == "GLOBAL"

    def test_resolve_metric(self, manager, db, tmp_path):
        db.execute(
            "INSERT INTO metrics (name, description, metric_type, is_obsolete) VALUES (?, ?, ?, ?)",
            ("global_metric", "A metric", "GLOBAL", 0),
        )
        db.commit()

        config = {"project-name": "test"}
        result = manager.resolve_component("global_metric", "metrics", config, tmp_path)
        assert result["source"] == "GLOBAL"
