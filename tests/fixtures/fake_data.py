"""Helpers for creating fake EEG datasets for testing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from scipy.io import savemat


def create_fake_raw_dataset(
    base_dir: Path,
    name: str = "test_eeg",
    n_subjects: int = 3,
    n_sessions: int = 2,
    n_recordings: int = 1,
    n_channels: int = 32,
    n_samples: int = 500,
    n_classes: int = 2,
    eeg_data_key: str = "data",
    label_key: str = "label",
    sampling_rate: float = 250.0,
) -> Path:
    """Create a fake raw EEG dataset with .mat files and raw.yml.

    Args:
        base_dir: Parent directory for the dataset.
        name: Dataset name.
        n_subjects: Number of subject .mat files.
        Other params: Dataset dimensions.

    Returns:
        Path to the created dataset directory.
    """
    dataset_dir = base_dir / name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Create .mat files
    for i in range(n_subjects):
        data = np.random.randn(n_sessions, n_recordings, n_channels, n_samples).astype(np.float32)
        labels = np.random.randint(0, n_classes, size=(n_sessions, n_recordings)).astype(np.int64)
        savemat(
            str(dataset_dir / f"subject_{i + 1:02d}.mat"),
            {eeg_data_key: data, label_key: labels},
        )

    # Create raw.yml
    numeric_to_semantic = {i: f"class_{i}" for i in range(n_classes)}
    raw_config = {
        "raw": {
            "name": name,
            "description": f"Fake dataset for testing ({name})",
            "eeg_data_key": eeg_data_key,
            "label_key": label_key,
            "sampling_rate": sampling_rate,
            "n_subjects": n_subjects,
            "n_sessions": n_sessions,
            "n_recordings": n_recordings,
            "n_channels": n_channels,
            "n_samples": n_samples,
            "dimension_info": ["session", "recording", "channel", "sample"],
            "numeric_to_semantic": numeric_to_semantic,
        }
    }

    with open(dataset_dir / "raw.yml", "w", encoding="utf-8") as f:
        yaml.dump(raw_config, f, default_flow_style=False)

    return dataset_dir


def create_inconsistent_dataset(base_dir: Path, name: str = "bad_eeg") -> Path:
    """Create a dataset where subject files have different shapes."""
    dataset_dir = base_dir / name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Subject 1: shape (2, 1, 32, 500)
    savemat(
        str(dataset_dir / "subject_01.mat"),
        {
            "data": np.random.randn(2, 1, 32, 500).astype(np.float32),
            "label": np.random.randint(0, 2, size=(2, 1)).astype(np.int64),
        },
    )
    # Subject 2: different shape (2, 1, 64, 500) -- different n_channels
    savemat(
        str(dataset_dir / "subject_02.mat"),
        {
            "data": np.random.randn(2, 1, 64, 500).astype(np.float32),
            "label": np.random.randint(0, 2, size=(2, 1)).astype(np.int64),
        },
    )

    raw_config = {
        "raw": {
            "name": name,
            "eeg_data_key": "data",
            "label_key": "label",
            "dimension_info": ["session", "recording", "channel", "sample"],
            "numeric_to_semantic": {0: "a", 1: "b"},
        }
    }
    with open(dataset_dir / "raw.yml", "w", encoding="utf-8") as f:
        yaml.dump(raw_config, f, default_flow_style=False)

    return dataset_dir
