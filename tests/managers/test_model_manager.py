"""Tests for ModelManager."""

from __future__ import annotations

from pathlib import Path

import pytest

from uesf.core.config import ConfigManager
from uesf.core.exceptions import ComponentNotFoundError, InterfaceViolationError
from uesf.managers.model_manager import ModelManager, _import_class, _parse_entrypoint


@pytest.fixture
def config(db, uesf_home):
    return ConfigManager(db, uesf_home)


@pytest.fixture
def manager(db, config):
    return ModelManager(db, config)


@pytest.fixture
def model_source(tmp_path):
    """Create a dummy model source file."""
    src = tmp_path / "my_model.py"
    src.write_text(
        'import torch.nn as nn\n'
        'class MyModel(nn.Module):\n'
        '    def __init__(self):\n'
        '        super().__init__()\n'
        '        self.fc = nn.Linear(10, 2)\n'
        '    def forward(self, x):\n'
        '        return self.fc(x)\n',
        encoding="utf-8",
    )
    return src


class TestAddGlobal:
    def test_add_global_success(self, manager, model_source, uesf_home):
        record = manager.add_global(model_source, "test_model", description="A test model")
        assert record["name"] == "test_model"
        assert record["model_type"] == "GLOBAL"
        assert record["description"] == "A test model"
        assert record["source_code_snapshot"] is not None

        # Check file was copied
        dest = uesf_home / "models" / "test_model.py"
        assert dest.exists()

    def test_add_global_nonexistent_file(self, manager):
        with pytest.raises(ComponentNotFoundError, match="not found"):
            manager.add_global(Path("/nonexistent/model.py"), "bad_model")


class TestGetAndList:
    def test_get_existing(self, manager, model_source):
        manager.add_global(model_source, "m1")
        record = manager.get("m1")
        assert record["name"] == "m1"

    def test_get_nonexistent(self, manager):
        with pytest.raises(ComponentNotFoundError, match="not found"):
            manager.get("nonexistent")

    def test_list_empty(self, manager):
        assert manager.list() == []

    def test_list_multiple(self, manager, model_source):
        manager.add_global(model_source, "m1")
        manager.add_global(model_source, "m2")
        models = manager.list()
        names = [m["name"] for m in models]
        assert sorted(names) == ["m1", "m2"]

    def test_list_hides_obsolete_by_default(self, db, manager, model_source):
        manager.add_global(model_source, "m1")
        db.execute("UPDATE models SET is_obsolete = 1 WHERE name = ?", ("m1",))
        db.commit()
        assert manager.list(show_obsolete=False) == []
        assert len(manager.list(show_obsolete=True)) == 1


class TestRemove:
    def test_remove_global_deletes_file(self, manager, model_source, uesf_home):
        manager.add_global(model_source, "m1")
        dest = uesf_home / "models" / "m1.py"
        assert dest.exists()

        manager.remove("m1")
        assert not dest.exists()

        with pytest.raises(ComponentNotFoundError):
            manager.get("m1")

    def test_remove_nonexistent(self, manager):
        with pytest.raises(ComponentNotFoundError):
            manager.remove("nonexistent")


class TestEdit:
    def test_edit_description(self, manager, model_source):
        manager.add_global(model_source, "m1", description="old")
        record = manager.edit("m1", description="new description")
        assert record["description"] == "new description"

    def test_edit_nonexistent(self, manager):
        with pytest.raises(ComponentNotFoundError):
            manager.edit("nonexistent", description="x")


class TestRegister:
    def test_register_success(self, manager, tmp_path):
        src = tmp_path / "proj_model.py"
        src.write_text(
            'class ProjModel:\n    pass\n',
            encoding="utf-8",
        )
        record = manager.register("proj_model", f"{src}:ProjModel", tmp_path)
        assert record["name"] == "proj_model"
        assert record["model_type"] == "REGISTERED"

    def test_register_bad_entrypoint(self, manager, tmp_path):
        with pytest.raises(InterfaceViolationError, match="Invalid entrypoint"):
            manager.register("bad", "no_colon_here", tmp_path)

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

        # Modify source
        src.write_text("class Changing:\n    version = 2\n", encoding="utf-8")

        new_record = manager.detect_and_reregister("changing", f"{src}:Changing", tmp_path)
        assert new_record["name"] == "changing"
        assert new_record["id"] != old_record["id"]

        # Old record should be archived
        all_models = manager.list(show_obsolete=True)
        archived = [m for m in all_models if m["is_obsolete"]]
        assert len(archived) == 1
        assert archived[0]["name"].startswith("changing_")

    def test_non_registered_type_unchanged(self, manager, model_source):
        manager.add_global(model_source, "global_m")
        record = manager.detect_and_reregister("global_m", f"{model_source}:MyModel", model_source.parent)
        assert record["model_type"] == "GLOBAL"


class TestLoadClass:
    def test_load_from_entrypoint(self, manager, tmp_path):
        src = tmp_path / "loadable.py"
        src.write_text(
            'class LoadableModel:\n    pass\n',
            encoding="utf-8",
        )
        cls = manager.load_class("loadable", entrypoint=f"{src}:LoadableModel", project_dir=tmp_path)
        assert cls.__name__ == "LoadableModel"

    def test_load_from_db_record(self, manager, model_source):
        manager.add_global(model_source, "my_model")
        cls = manager.load_class("my_model")
        assert cls.__name__ == "MyModel"


class TestParseEntrypoint:
    def test_valid_entrypoint(self, tmp_path):
        src = tmp_path / "mod.py"
        src.write_text("class Cls: pass", encoding="utf-8")
        file_path, class_name = _parse_entrypoint(f"{src}:Cls", tmp_path)
        assert file_path == src
        assert class_name == "Cls"

    def test_relative_path(self, tmp_path):
        src = tmp_path / "sub" / "mod.py"
        src.parent.mkdir(parents=True)
        src.write_text("class Cls: pass", encoding="utf-8")
        file_path, class_name = _parse_entrypoint("sub/mod.py:Cls", tmp_path)
        assert file_path == src
        assert class_name == "Cls"

    def test_no_colon_raises(self, tmp_path):
        with pytest.raises(InterfaceViolationError, match="Invalid entrypoint"):
            _parse_entrypoint("no_colon", tmp_path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ComponentNotFoundError, match="not found"):
            _parse_entrypoint("missing.py:Cls", tmp_path)


class TestImportClass:
    def test_import_success(self, tmp_path):
        src = tmp_path / "importable.py"
        src.write_text("class Good:\n    value = 42\n", encoding="utf-8")
        cls = _import_class(src, "Good")
        assert cls.value == 42

    def test_import_missing_class(self, tmp_path):
        src = tmp_path / "partial.py"
        src.write_text("class Exists: pass\n", encoding="utf-8")
        with pytest.raises(InterfaceViolationError, match="not found"):
            _import_class(src, "Missing")
