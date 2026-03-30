"""UESF Data Preprocessor.

Implements subject-wise lazy loading with three-stream pipeline
(data -> label -> joint) execution architecture.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.io import loadmat

from uesf.core.config import ConfigManager
from uesf.core.database import DatabaseManager
from uesf.core.exceptions import (
    DatasetNotFoundError,
    YAMLParseError,
)
from uesf.core.logging import get_logger
from uesf.pipeline.operators import get_operator

logger = get_logger("pipeline")


class Preprocessor:
    """Executes preprocessing pipeline with subject-wise lazy loading."""

    def __init__(self, db: DatabaseManager, config: ConfigManager) -> None:
        self.db = db
        self.config = config

    def run(
        self,
        preprocess_config: dict,
        source_dataset_name: str,
        out_name: str,
    ) -> dict[str, Any]:
        """Execute the preprocessing pipeline.

        Args:
            preprocess_config: Parsed preprocess.yml config dict.
            source_dataset_name: Name of the source raw dataset.
            out_name: Name for the output preprocessed dataset.

        Returns:
            The preprocessed dataset DB record.
        """
        # Resolve source dataset
        raw_record = self.db.fetch_one(
            "SELECT * FROM raw_datasets WHERE name = ?", (source_dataset_name,)
        )
        if raw_record is None:
            raise DatasetNotFoundError(
                f"Source raw dataset '{source_dataset_name}' not found",
                hint="Run 'uesf data raw list' to see available datasets.",
            )

        raw_dir = Path(raw_record["data_dir_path"])
        mat_files = sorted(raw_dir.glob("*.mat"))
        if not mat_files:
            raise DatasetNotFoundError(
                f"No .mat files found in '{raw_dir}'",
                context={"dataset": source_dataset_name, "path": str(raw_dir)},
            )

        # Parse pipeline steps
        pipeline = preprocess_config.get("pipeline", {})
        data_ops = pipeline.get("data", []) or []
        label_ops = pipeline.get("label", []) or []
        joint_ops = pipeline.get("joint", []) or []

        # Prepare output directory
        out_dir = self.config.get_data_dir() / "preprocessed" / out_name
        out_dir.mkdir(parents=True, exist_ok=True)

        sampling_rate = raw_record["sampling_rate"] or 250.0
        eeg_data_key = raw_record["eeg_data_key"]
        label_key = raw_record["label_key"]

        all_data = []
        all_labels = []

        try:
            for mat_file in mat_files:
                logger.info("Processing %s", mat_file.name)

                # Load single subject
                mat_data = loadmat(str(mat_file), squeeze_me=False)
                subject_data = np.array(mat_data[eeg_data_key], dtype=np.float32)
                subject_labels = np.array(mat_data[label_key], dtype=np.int64)

                # Apply data stream operators
                current_sr = sampling_rate
                for op_config in data_ops:
                    op_name = op_config["name"]
                    op_params = op_config.get("params", {})
                    _, op_fn = get_operator(op_name)
                    subject_data, current_sr = op_fn(subject_data, current_sr, op_params)

                # Apply label stream operators
                for op_config in label_ops:
                    op_name = op_config["name"]
                    op_params = op_config.get("params", {})
                    _, op_fn = get_operator(op_name)
                    subject_labels = op_fn(subject_labels, op_params)

                # Apply joint stream operators
                for op_config in joint_ops:
                    op_name = op_config["name"]
                    op_params = op_config.get("params", {})
                    _, op_fn = get_operator(op_name)
                    subject_data, subject_labels, current_sr = op_fn(
                        subject_data, subject_labels, current_sr, op_params
                    )

                all_data.append(subject_data)
                all_labels.append(subject_labels)

                # Release memory for this subject
                del mat_data, subject_data, subject_labels

        except Exception:
            # Strict failure: clean up partial output
            if out_dir.exists():
                shutil.rmtree(out_dir)
            raise

        # Stack all subjects and save
        final_data = np.stack(all_data, axis=0)  # (subjects, ...)
        final_labels = np.stack(all_labels, axis=0)  # (subjects, ...)

        np.save(str(out_dir / "eeg_data.npy"), final_data)
        np.save(str(out_dir / "labels.npy"), final_labels)

        # Determine output metadata
        n_subjects = final_data.shape[0]
        data_shape = list(final_data.shape)
        label_shape = list(final_labels.shape)

        # Infer dimension counts from shape
        # Preprocessed shape: (subjects, sessions, recordings, channels, samples)
        n_sessions = final_data.shape[1] if final_data.ndim > 1 else 1
        n_recordings = final_data.shape[2] if final_data.ndim > 2 else 1
        n_channels = final_data.shape[3] if final_data.ndim > 3 else 1
        n_samples = final_data.shape[4] if final_data.ndim > 4 else final_data.shape[-1]

        # Inherit numeric_to_semantic from source
        numeric_to_semantic = raw_record["numeric_to_semantic"]

        with self.db.transaction() as cursor:
            cursor.execute(
                """INSERT INTO preprocessed_datasets
                   (name, description, source_raw_dataset_id, data_dir_path,
                    n_subjects, sampling_rate, n_sessions, n_recordings,
                    n_channels, n_samples, electrode_list, data_shape,
                    label_shape, numeric_to_semantic, preprocess_config_snapshot)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    out_name,
                    f"Preprocessed from {source_dataset_name}",
                    raw_record["id"],
                    str(out_dir),
                    n_subjects,
                    current_sr,
                    n_sessions,
                    n_recordings,
                    n_channels,
                    n_samples,
                    raw_record.get("electrode_list"),
                    json.dumps(data_shape),
                    json.dumps(label_shape),
                    numeric_to_semantic,
                    json.dumps(preprocess_config),
                ),
            )

        logger.info(
            "Preprocessing complete: '%s' -> '%s' (shape: %s)",
            source_dataset_name, out_name, data_shape,
        )
        return self.db.fetch_one("SELECT * FROM preprocessed_datasets WHERE name = ?", (out_name,))


def parse_preprocess_yml(config_path: Path) -> dict[str, Any]:
    """Parse a preprocess.yml configuration file.

    Returns:
        The preprocess config dict (contents under 'preprocess' key).
    """
    if not config_path.exists():
        raise YAMLParseError(
            f"Preprocessing config not found: '{config_path}'",
            context={"path": str(config_path)},
            hint="Create a preprocess.yml file or specify --config-path.",
        )

    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "preprocess" not in data:
        raise YAMLParseError(
            "preprocess.yml must have a top-level 'preprocess' key",
            context={"path": str(config_path)},
        )

    config = data["preprocess"]
    if not isinstance(config, dict):
        raise YAMLParseError(
            "'preprocess' key must contain a mapping",
            context={"path": str(config_path)},
        )

    return config
