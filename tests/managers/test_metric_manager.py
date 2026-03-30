"""Tests for MetricManager."""

from __future__ import annotations

from pathlib import Path

import pytest

from uesf.core.config import ConfigManager
from uesf.core.exceptions import ComponentNotFoundError
from uesf.managers.metric_manager import MetricManager


@pytest.fixture
def config(db, uesf_home):
    return ConfigManager(db, uesf_home)


@pytest.fixture
def manager(db, config):
    return MetricManager(db, config)


@pytest.fixture
def metric_source(tmp_path):
    """Create a dummy metric source file."""
    src = tmp_path / "my_metric.py"
    src.write_text(
        'def my_metric(preds, targets, **kwargs):\n'
        '    return 0.95\n',
        encoding="utf-8",
    )
    return src


class TestAddGlobal:
    def test_add_global_success(self, manager, metric_source, uesf_home):
        record = manager.add_global(metric_source, "test_metric", description="A test metric")
        assert record["name"] == "test_metric"
        assert record["metric_type"] == "GLOBAL"
        assert record["description"] == "A test metric"

        dest = uesf_home / "metrics" / "test_metric.py"
        assert dest.exists()

    def test_add_global_nonexistent_file(self, manager):
        with pytest.raises(ComponentNotFoundError, match="not found"):
            manager.add_global(Path("/nonexistent/metric.py"), "bad")


class TestGetAndList:
    def test_get_existing(self, manager, metric_source):
        manager.add_global(metric_source, "m1")
        record = manager.get("m1")
        assert record["name"] == "m1"

    def test_get_nonexistent(self, manager):
        with pytest.raises(ComponentNotFoundError, match="not found"):
            manager.get("nonexistent")

    def test_list_empty(self, manager):
        assert manager.list() == []

    def test_list_multiple(self, manager, metric_source):
        manager.add_global(metric_source, "m1")
        manager.add_global(metric_source, "m2")
        metrics = manager.list()
        names = [m["name"] for m in metrics]
        assert sorted(names) == ["m1", "m2"]

    def test_list_hides_obsolete_by_default(self, db, manager, metric_source):
        manager.add_global(metric_source, "m1")
        db.execute("UPDATE metrics SET is_obsolete = 1 WHERE name = ?", ("m1",))
        db.commit()
        assert manager.list(show_obsolete=False) == []
        assert len(manager.list(show_obsolete=True)) == 1


class TestRemove:
    def test_remove_global_deletes_file(self, manager, metric_source, uesf_home):
        manager.add_global(metric_source, "m1")
        dest = uesf_home / "metrics" / "m1.py"
        assert dest.exists()
        manager.remove("m1")
        assert not dest.exists()

    def test_remove_nonexistent(self, manager):
        with pytest.raises(ComponentNotFoundError):
            manager.remove("nonexistent")


class TestEdit:
    def test_edit_description(self, manager, metric_source):
        manager.add_global(metric_source, "m1", description="old")
        record = manager.edit("m1", description="new description")
        assert record["description"] == "new description"


class TestRegisterAndReregister:
    def test_register_success(self, manager, tmp_path):
        src = tmp_path / "proj_metric.py"
        src.write_text("def proj_metric(preds, targets): return 0.5\n", encoding="utf-8")
        record = manager.register("proj_metric", f"{src}:proj_metric", tmp_path)
        assert record["metric_type"] == "REGISTERED"

    def test_source_change_archives_old(self, manager, tmp_path):
        src = tmp_path / "changing.py"
        src.write_text("def changing(preds, targets): return 1.0\n", encoding="utf-8")
        manager.register("changing", f"{src}:changing", tmp_path)

        src.write_text("def changing(preds, targets): return 0.5\n", encoding="utf-8")
        new_record = manager.detect_and_reregister("changing", f"{src}:changing", tmp_path)
        assert new_record["name"] == "changing"

        all_metrics = manager.list(show_obsolete=True)
        archived = [m for m in all_metrics if m["is_obsolete"]]
        assert len(archived) == 1


class TestLoadMetric:
    def test_load_builtin(self, manager):
        func = manager.load_metric("accuracy")
        assert callable(func)

    def test_load_from_entrypoint(self, manager, tmp_path):
        src = tmp_path / "custom.py"
        src.write_text("def custom_score(preds, targets): return 0.99\n", encoding="utf-8")
        func = manager.load_metric("custom_score", entrypoint=f"{src}:custom_score", project_dir=tmp_path)
        assert callable(func)

    def test_load_nonexistent(self, manager):
        with pytest.raises(ComponentNotFoundError, match="not found"):
            manager.load_metric("nonexistent_metric")
