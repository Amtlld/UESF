"""Tests for DataManager raw dataset operations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.io import savemat

from tests.fixtures.fake_data import create_fake_raw_dataset, create_inconsistent_dataset
from uesf.core.config import ConfigManager
from uesf.core.exceptions import (
    DatasetNotFoundError,
    MissingRequiredKeyError,
    ShapeMismatchError,
    YAMLParseError,
)
from uesf.managers.data_manager import DataManager


@pytest.fixture
def data_manager(db, uesf_home):
    config = ConfigManager(db, uesf_home)
    return DataManager(db, config)


class TestRegisterRaw:
    def test_register_basic(self, data_manager, tmp_path):
        ds_dir = create_fake_raw_dataset(tmp_path)
        record = data_manager.register_raw(ds_dir)

        assert record["name"] == "test_eeg"
        assert record["n_subjects"] == 3
        assert record["is_imported"] == 0
        assert record["eeg_data_key"] == "data"
        assert record["label_key"] == "label"
        assert str(ds_dir) in record["data_dir_path"]

    def test_register_infers_shape(self, data_manager, tmp_path):
        ds_dir = create_fake_raw_dataset(tmp_path, n_sessions=2, n_recordings=1, n_channels=32, n_samples=500)
        record = data_manager.register_raw(ds_dir)

        data_shape = json.loads(record["data_shape"])
        assert data_shape == [2, 1, 32, 500]

        label_shape = json.loads(record["label_shape"])
        assert label_shape == [2, 1]

    def test_register_stores_snapshot(self, data_manager, tmp_path):
        ds_dir = create_fake_raw_dataset(tmp_path)
        record = data_manager.register_raw(ds_dir)

        snapshot = json.loads(record["raw_info_snapshot"])
        assert snapshot["eeg_data_key"] == "data"
        assert "dimension_info" in snapshot

    def test_register_numeric_to_semantic(self, data_manager, tmp_path):
        ds_dir = create_fake_raw_dataset(tmp_path, n_classes=3)
        record = data_manager.register_raw(ds_dir)

        n2s = json.loads(record["numeric_to_semantic"])
        assert "0" in n2s
        assert "1" in n2s
        assert "2" in n2s

    def test_register_missing_raw_yml(self, data_manager, tmp_path):
        ds_dir = tmp_path / "no_yml"
        ds_dir.mkdir()
        with pytest.raises(YAMLParseError, match="raw.yml not found"):
            data_manager.register_raw(ds_dir)

    def test_register_missing_required_keys(self, data_manager, tmp_path):
        ds_dir = tmp_path / "bad_yml"
        ds_dir.mkdir()
        # Missing eeg_data_key, label_key, etc.
        (ds_dir / "raw.yml").write_text(yaml.dump({"raw": {"name": "bad"}}))
        with pytest.raises(MissingRequiredKeyError, match="missing required keys"):
            data_manager.register_raw(ds_dir)

    def test_register_no_mat_files(self, data_manager, tmp_path):
        ds_dir = tmp_path / "no_mat"
        ds_dir.mkdir()
        (ds_dir / "raw.yml").write_text(yaml.dump({
            "raw": {
                "name": "no_mat",
                "eeg_data_key": "data",
                "label_key": "label",
                "dimension_info": ["channel", "sample"],
                "numeric_to_semantic": {0: "a", 1: "b"},
            }
        }))
        with pytest.raises(DatasetNotFoundError, match="No .mat files"):
            data_manager.register_raw(ds_dir)

    def test_register_inconsistent_shapes(self, data_manager, tmp_path):
        ds_dir = create_inconsistent_dataset(tmp_path)
        with pytest.raises(ShapeMismatchError, match="Inconsistent shapes"):
            data_manager.register_raw(ds_dir)

    def test_register_wrong_eeg_key(self, data_manager, tmp_path):
        ds_dir = tmp_path / "wrong_key"
        ds_dir.mkdir()
        savemat(str(ds_dir / "subject_01.mat"), {"eeg": np.zeros((2, 32, 500)), "label": np.zeros(2)})
        (ds_dir / "raw.yml").write_text(yaml.dump({
            "raw": {
                "name": "wrong_key",
                "eeg_data_key": "data",  # Wrong key, .mat has "eeg"
                "label_key": "label",
                "dimension_info": ["channel", "sample"],
                "numeric_to_semantic": {0: "a"},
            }
        }))
        with pytest.raises(MissingRequiredKeyError, match="Key 'data' not found"):
            data_manager.register_raw(ds_dir)

    def test_register_uses_dirname_as_fallback_name(self, data_manager, tmp_path):
        ds_dir = tmp_path / "my_dataset"
        ds_dir.mkdir()
        savemat(str(ds_dir / "s01.mat"), {"data": np.zeros((2, 32, 500)), "label": np.zeros(2)})
        (ds_dir / "raw.yml").write_text(yaml.dump({
            "raw": {
                "eeg_data_key": "data",
                "label_key": "label",
                "dimension_info": ["channel", "sample"],
                "numeric_to_semantic": {0: "a"},
            }
        }))
        record = data_manager.register_raw(ds_dir)
        assert record["name"] == "my_dataset"


class TestImportRaw:
    def test_import_copies_files(self, data_manager, tmp_path, uesf_home):
        ds_dir = create_fake_raw_dataset(tmp_path, name="import_test")
        record = data_manager.import_raw(ds_dir)

        assert record["is_imported"] == 1
        # Files should be copied to data dir
        import_dir = Path(record["data_dir_path"])
        assert import_dir.exists()
        assert len(list(import_dir.glob("*.mat"))) == 3

    def test_import_preserves_raw_yml(self, data_manager, tmp_path, uesf_home):
        ds_dir = create_fake_raw_dataset(tmp_path, name="import_yml")
        record = data_manager.import_raw(ds_dir)

        import_dir = Path(record["data_dir_path"])
        assert (import_dir / "raw.yml").exists()


class TestListRaw:
    def test_list_empty(self, data_manager):
        assert data_manager.list_raw() == []

    def test_list_returns_all(self, data_manager, tmp_path):
        for i in range(3):
            ds_dir = create_fake_raw_dataset(tmp_path, name=f"ds_{i}")
            data_manager.register_raw(ds_dir)

        datasets = data_manager.list_raw()
        assert len(datasets) == 3
        names = [d["name"] for d in datasets]
        assert sorted(names) == ["ds_0", "ds_1", "ds_2"]


class TestRemoveRaw:
    def test_remove_registered(self, data_manager, tmp_path):
        ds_dir = create_fake_raw_dataset(tmp_path, name="to_remove")
        data_manager.register_raw(ds_dir)

        data_manager.remove_raw("to_remove")
        assert data_manager.list_raw() == []

    def test_remove_imported_deletes_files(self, data_manager, tmp_path, uesf_home):
        ds_dir = create_fake_raw_dataset(tmp_path, name="to_remove_imported")
        record = data_manager.import_raw(ds_dir)

        import_dir = Path(record["data_dir_path"])
        assert import_dir.exists()

        data_manager.remove_raw("to_remove_imported")
        assert not import_dir.exists()

    def test_remove_nonexistent(self, data_manager):
        with pytest.raises(DatasetNotFoundError):
            data_manager.remove_raw("nonexistent")


class TestEditRaw:
    def test_edit_description(self, data_manager, tmp_path):
        ds_dir = create_fake_raw_dataset(tmp_path, name="to_edit")
        data_manager.register_raw(ds_dir)

        updated = data_manager.edit_raw("to_edit", description="Updated description")
        assert updated["description"] == "Updated description"

    def test_edit_sampling_rate(self, data_manager, tmp_path):
        ds_dir = create_fake_raw_dataset(tmp_path, name="to_edit_sr")
        data_manager.register_raw(ds_dir)

        updated = data_manager.edit_raw("to_edit_sr", sampling_rate=512.0)
        assert updated["sampling_rate"] == 512.0

    def test_edit_nonexistent(self, data_manager):
        with pytest.raises(DatasetNotFoundError):
            data_manager.edit_raw("nonexistent", description="test")
