"""Tests for the end-to-end preprocessing pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.fixtures.fake_data import create_fake_raw_dataset
from uesf.core.config import ConfigManager
from uesf.core.exceptions import DatasetNotFoundError
from uesf.managers.data_manager import DataManager
from uesf.pipeline.preprocessor import Preprocessor


@pytest.fixture
def setup(db, uesf_home, tmp_path):
    """Set up preprocessor with a registered raw dataset."""
    config = ConfigManager(db, uesf_home)
    data_mgr = DataManager(db, config)
    preprocessor = Preprocessor(db, config)

    ds_dir = create_fake_raw_dataset(
        tmp_path, name="raw_for_preproc",
        n_subjects=3, n_sessions=2, n_recordings=1,
        n_channels=8, n_samples=500, sampling_rate=250.0,
    )
    data_mgr.register_raw(ds_dir)

    return preprocessor, data_mgr, config


class TestPreprocessor:
    def test_basic_preprocessing(self, setup):
        preprocessor, data_mgr, config = setup

        preprocess_config = {
            "pipeline": {
                "data": [{"name": "filter", "params": {"l_freq": 1.0, "h_freq": 45.0}}],
            }
        }
        record = preprocessor.run(preprocess_config, "raw_for_preproc", "filtered_ds")

        assert record is not None
        assert record["name"] == "filtered_ds"
        assert record["n_subjects"] == 3

        # Verify .npy files exist
        out_dir = Path(record["data_dir_path"])
        assert (out_dir / "eeg_data.npy").exists()
        assert (out_dir / "labels.npy").exists()

    def test_output_shape(self, setup):
        preprocessor, _, _ = setup

        preprocess_config = {"pipeline": {}}
        record = preprocessor.run(preprocess_config, "raw_for_preproc", "passthrough_ds")

        data = np.load(Path(record["data_dir_path"]) / "eeg_data.npy")
        # Shape: (3 subjects, 2 sessions, 1 recording, 8 channels, 500 samples)
        assert data.shape == (3, 2, 1, 8, 500)

    def test_sliding_window_changes_shape(self, setup):
        preprocessor, _, _ = setup

        preprocess_config = {
            "pipeline": {
                "joint": [
                    {"name": "sliding_window", "params": {
                        "window_size_sec": 1.0, "stride_sec": 1.0,
                    }}
                ]
            }
        }
        record = preprocessor.run(preprocess_config, "raw_for_preproc", "windowed_ds")

        data = np.load(Path(record["data_dir_path"]) / "eeg_data.npy")
        # window_size = 1.0 * 250 = 250 samples
        # stride = 1.0 * 250 = 250 samples
        # windows = (500 - 250) // 250 + 1 = 2
        assert data.shape[0] == 3  # subjects
        assert data.shape[-1] == 250  # window samples
        assert data.shape[1] == 2  # sessions
        assert data.shape[2] == 2  # windows (recordings expanded)

    def test_missing_source_dataset(self, setup):
        preprocessor, _, _ = setup
        with pytest.raises(DatasetNotFoundError):
            preprocessor.run({}, "nonexistent_ds", "output")

    def test_config_snapshot_stored(self, setup):
        preprocessor, _, _ = setup

        preprocess_config = {
            "pipeline": {"data": [{"name": "reference", "params": {"type": "CAR"}}]},
        }
        record = preprocessor.run(preprocess_config, "raw_for_preproc", "snapshot_ds")

        snapshot = json.loads(record["preprocess_config_snapshot"])
        assert "pipeline" in snapshot

    def test_inherits_numeric_to_semantic(self, setup):
        preprocessor, data_mgr, _ = setup

        record = preprocessor.run({"pipeline": {}}, "raw_for_preproc", "inherit_ds")
        raw_record = data_mgr.get_raw("raw_for_preproc")

        assert record["numeric_to_semantic"] == raw_record["numeric_to_semantic"]


class TestMaskedDataset:
    def test_create_masked(self, setup):
        preprocessor, data_mgr, config = setup

        # First create a preprocessed dataset
        preprocessor.run({"pipeline": {}}, "raw_for_preproc", "base_ds")

        # Create masked dataset
        mapping = {"class_0": "negative", "class_1": "positive"}
        record = data_mgr.create_masked("base_ds", "masked_ds", mapping)

        assert record["name"] == "masked_ds"
        assert record["n_classes"] == 2

        n2s = json.loads(record["numeric_to_semantic"])
        assert "negative" in n2s.values()
        assert "positive" in n2s.values()

    def test_masked_labels_file_exists(self, setup):
        preprocessor, data_mgr, _ = setup
        preprocessor.run({"pipeline": {}}, "raw_for_preproc", "base_ds2")

        mapping = {"class_0": "a", "class_1": "b"}
        record = data_mgr.create_masked("base_ds2", "masked2", mapping)

        labels_path = Path(record["data_dir_path"]) / "labels.npy"
        assert labels_path.exists()

    def test_list_masked(self, setup):
        preprocessor, data_mgr, _ = setup
        preprocessor.run({"pipeline": {}}, "raw_for_preproc", "base_ds3")
        data_mgr.create_masked("base_ds3", "m1", {"class_0": "a", "class_1": "b"})
        data_mgr.create_masked("base_ds3", "m2", {"class_0": "x", "class_1": "y"})

        masked = data_mgr.list_masked()
        assert len(masked) == 2

    def test_remove_masked(self, setup):
        preprocessor, data_mgr, _ = setup
        preprocessor.run({"pipeline": {}}, "raw_for_preproc", "base_ds4")
        record = data_mgr.create_masked("base_ds4", "to_remove_m", {"class_0": "a", "class_1": "b"})

        data_mgr.remove_masked("to_remove_m")
        assert data_mgr.list_masked() == []
        assert not Path(record["data_dir_path"]).exists()
