"""Tests for TrainerManager."""

from __future__ import annotations

from pathlib import Path

import pytest

from uesf.core.config import ConfigManager
from uesf.core.exceptions import ComponentNotFoundError, InterfaceViolationError
from uesf.managers.trainer_manager import TrainerManager


@pytest.fixture
def config(db, uesf_home):
    return ConfigManager(db, uesf_home)


@pytest.fixture
def manager(db, config):
    return TrainerManager(db, config)


@pytest.fixture
def trainer_source(tmp_path):
    """Create a dummy trainer source file."""
    src = tmp_path / "my_trainer.py"
    src.write_text(
        'class MyTrainer:\n'
        '    def __init__(self, model, device, **kwargs):\n'
        '        self.model = model\n'
        '    def training_step(self, batch, batch_idx, optimizer):\n'
        '        return {"loss": 0.0}\n'
        '    def validation_step(self, batch, batch_idx):\n'
        '        return {"preds": None, "targets": None}\n',
        encoding="utf-8",
    )
    return src


class TestAddGlobal:
    def test_add_global_success(self, manager, trainer_source, uesf_home):
        record = manager.add_global(trainer_source, "test_trainer", description="A test trainer")
        assert record["name"] == "test_trainer"
        assert record["trainer_type"] == "GLOBAL"
        assert record["description"] == "A test trainer"
        assert record["source_code_snapshot"] is not None

        dest = uesf_home / "trainers" / "test_trainer.py"
        assert dest.exists()

    def test_add_global_nonexistent_file(self, manager):
        with pytest.raises(ComponentNotFoundError, match="not found"):
            manager.add_global(Path("/nonexistent/trainer.py"), "bad")


class TestGetAndList:
    def test_get_existing(self, manager, trainer_source):
        manager.add_global(trainer_source, "t1")
        record = manager.get("t1")
        assert record["name"] == "t1"

    def test_get_nonexistent(self, manager):
        with pytest.raises(ComponentNotFoundError, match="not found"):
            manager.get("nonexistent")

    def test_list_empty(self, manager):
        assert manager.list() == []

    def test_list_multiple(self, manager, trainer_source):
        manager.add_global(trainer_source, "t1")
        manager.add_global(trainer_source, "t2")
        trainers = manager.list()
        names = [t["name"] for t in trainers]
        assert sorted(names) == ["t1", "t2"]

    def test_list_hides_obsolete_by_default(self, db, manager, trainer_source):
        manager.add_global(trainer_source, "t1")
        db.execute("UPDATE trainers SET is_obsolete = 1 WHERE name = ?", ("t1",))
        db.commit()
        assert manager.list(show_obsolete=False) == []
        assert len(manager.list(show_obsolete=True)) == 1


class TestRemove:
    def test_remove_global_deletes_file(self, manager, trainer_source, uesf_home):
        manager.add_global(trainer_source, "t1")
        dest = uesf_home / "trainers" / "t1.py"
        assert dest.exists()

        manager.remove("t1")
        assert not dest.exists()

        with pytest.raises(ComponentNotFoundError):
            manager.get("t1")

    def test_remove_nonexistent(self, manager):
        with pytest.raises(ComponentNotFoundError):
            manager.remove("nonexistent")


class TestEdit:
    def test_edit_description(self, manager, trainer_source):
        manager.add_global(trainer_source, "t1", description="old")
        record = manager.edit("t1", description="new description")
        assert record["description"] == "new description"

    def test_edit_nonexistent(self, manager):
        with pytest.raises(ComponentNotFoundError):
            manager.edit("nonexistent", description="x")


class TestRegister:
    def test_register_success(self, manager, tmp_path):
        src = tmp_path / "proj_trainer.py"
        src.write_text("class ProjTrainer:\n    pass\n", encoding="utf-8")
        record = manager.register("proj_trainer", f"{src}:ProjTrainer", tmp_path)
        assert record["name"] == "proj_trainer"
        assert record["trainer_type"] == "REGISTERED"

    def test_register_bad_entrypoint(self, manager, tmp_path):
        with pytest.raises(InterfaceViolationError, match="Invalid entrypoint"):
            manager.register("bad", "no_colon", tmp_path)

    def test_register_missing_file(self, manager, tmp_path):
        with pytest.raises(ComponentNotFoundError, match="not found"):
            manager.register("bad", "missing.py:Class", tmp_path)


class TestDetectAndReregister:
    def test_no_change_returns_same(self, manager, tmp_path):
        src = tmp_path / "stable.py"
        src.write_text("class Stable:\n    pass\n", encoding="utf-8")
        manager.register("stable", f"{src}:Stable", tmp_path)
        record = manager.detect_and_reregister("stable", f"{src}:Stable", tmp_path)
        assert record["name"] == "stable"

    def test_source_change_archives_old(self, manager, tmp_path):
        src = tmp_path / "changing.py"
        src.write_text("class Changing:\n    version = 1\n", encoding="utf-8")
        manager.register("changing", f"{src}:Changing", tmp_path)
        old_record = manager.get("changing")

        src.write_text("class Changing:\n    version = 2\n", encoding="utf-8")
        new_record = manager.detect_and_reregister("changing", f"{src}:Changing", tmp_path)
        assert new_record["name"] == "changing"
        assert new_record["id"] != old_record["id"]

        all_trainers = manager.list(show_obsolete=True)
        archived = [t for t in all_trainers if t["is_obsolete"]]
        assert len(archived) == 1
        assert archived[0]["name"].startswith("changing_")

    def test_non_registered_type_unchanged(self, manager, trainer_source):
        manager.add_global(trainer_source, "global_t")
        record = manager.detect_and_reregister("global_t", f"{trainer_source}:MyTrainer", trainer_source.parent)
        assert record["trainer_type"] == "GLOBAL"


class TestLoadClass:
    def test_load_from_entrypoint(self, manager, tmp_path):
        src = tmp_path / "loadable.py"
        src.write_text("class LoadableTrainer:\n    pass\n", encoding="utf-8")
        cls = manager.load_class("loadable", entrypoint=f"{src}:LoadableTrainer", project_dir=tmp_path)
        assert cls.__name__ == "LoadableTrainer"

    def test_load_from_db_record(self, manager, trainer_source):
        manager.add_global(trainer_source, "my_trainer")
        cls = manager.load_class("my_trainer")
        assert cls.__name__ == "MyTrainer"
