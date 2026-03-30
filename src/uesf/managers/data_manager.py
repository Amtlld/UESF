"""UESF Data Manager - manages raw, preprocessed, and masked datasets."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import yaml
from scipy.io import loadmat

from uesf.core.config import ConfigManager
from uesf.core.database import DatabaseManager
from uesf.core.exceptions import (
    DatasetNotFoundError,
    MissingRequiredKeyError,
    ShapeMismatchError,
    YAMLParseError,
)
from uesf.core.logging import get_logger

logger = get_logger("manager.data")

_RAW_REQUIRED_KEYS = {"eeg_data_key", "label_key", "dimension_info", "numeric_to_semantic"}


class DataManager:
    """Manages all dataset operations: raw, preprocessed, and masked."""

    def __init__(self, db: DatabaseManager, config: ConfigManager) -> None:
        self.db = db
        self.config = config

    # ── Raw Dataset Operations ──────────────────────────────────────────

    def register_raw(self, dataset_path: Path) -> dict[str, Any]:
        """Register a raw dataset (user manages storage).

        Parses raw.yml, validates .mat files, infers shapes,
        and stores metadata in the database.

        Args:
            dataset_path: Path to directory containing raw.yml and .mat files.

        Returns:
            The database record as a dict.
        """
        dataset_path = Path(dataset_path).resolve()
        raw_config = self._parse_raw_yml(dataset_path)
        mat_files = self._find_mat_files(dataset_path)

        data_shape, label_shape = self._infer_shapes(
            mat_files, raw_config["eeg_data_key"], raw_config["label_key"]
        )

        name = raw_config["name"]
        n_subjects = len(mat_files)

        with self.db.transaction() as cursor:
            cursor.execute(
                """INSERT INTO raw_datasets
                   (name, description, is_imported, data_dir_path,
                    eeg_data_key, label_key, n_subjects, sampling_rate,
                    n_sessions, n_recordings, n_channels, n_samples,
                    electrode_list, data_shape, dimension_info, label_shape,
                    numeric_to_semantic, raw_info_snapshot)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    name,
                    raw_config.get("description"),
                    0,  # is_imported = False
                    str(dataset_path),
                    raw_config["eeg_data_key"],
                    raw_config["label_key"],
                    n_subjects,
                    raw_config.get("sampling_rate"),
                    raw_config.get("n_sessions"),
                    raw_config.get("n_recordings"),
                    raw_config.get("n_channels"),
                    raw_config.get("n_samples"),
                    json.dumps(raw_config.get("electrode_list")),
                    json.dumps(list(data_shape)),
                    json.dumps(raw_config["dimension_info"]),
                    json.dumps(list(label_shape)),
                    json.dumps(raw_config["numeric_to_semantic"]),
                    json.dumps(raw_config),
                ),
            )

        logger.info("Registered raw dataset '%s' (%d subjects)", name, n_subjects)
        return self.get_raw(name)

    def import_raw(self, dataset_path: Path) -> dict[str, Any]:
        """Import a raw dataset (copy files to UESF data dir).

        Args:
            dataset_path: Path to directory containing raw.yml and .mat files.

        Returns:
            The database record as a dict.
        """
        dataset_path = Path(dataset_path).resolve()
        raw_config = self._parse_raw_yml(dataset_path)
        name = raw_config["name"]

        data_dir = self.config.get_data_dir() / "raw" / name
        data_dir.mkdir(parents=True, exist_ok=True)

        # Copy .mat files and raw.yml
        mat_files = self._find_mat_files(dataset_path)
        for mat_file in mat_files:
            shutil.copy2(mat_file, data_dir / mat_file.name)

        raw_yml = dataset_path / "raw.yml"
        if raw_yml.exists():
            shutil.copy2(raw_yml, data_dir / "raw.yml")

        data_shape, label_shape = self._infer_shapes(
            mat_files, raw_config["eeg_data_key"], raw_config["label_key"]
        )

        n_subjects = len(mat_files)

        with self.db.transaction() as cursor:
            cursor.execute(
                """INSERT INTO raw_datasets
                   (name, description, is_imported, data_dir_path,
                    eeg_data_key, label_key, n_subjects, sampling_rate,
                    n_sessions, n_recordings, n_channels, n_samples,
                    electrode_list, data_shape, dimension_info, label_shape,
                    numeric_to_semantic, raw_info_snapshot)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    name,
                    raw_config.get("description"),
                    1,  # is_imported = True
                    str(data_dir),
                    raw_config["eeg_data_key"],
                    raw_config["label_key"],
                    n_subjects,
                    raw_config.get("sampling_rate"),
                    raw_config.get("n_sessions"),
                    raw_config.get("n_recordings"),
                    raw_config.get("n_channels"),
                    raw_config.get("n_samples"),
                    json.dumps(raw_config.get("electrode_list")),
                    json.dumps(list(data_shape)),
                    json.dumps(raw_config["dimension_info"]),
                    json.dumps(list(label_shape)),
                    json.dumps(raw_config["numeric_to_semantic"]),
                    json.dumps(raw_config),
                ),
            )

        logger.info("Imported raw dataset '%s' (%d subjects) to %s", name, n_subjects, data_dir)
        return self.get_raw(name)

    def get_raw(self, name: str) -> dict[str, Any]:
        """Get a raw dataset record by name."""
        row = self.db.fetch_one("SELECT * FROM raw_datasets WHERE name = ?", (name,))
        if row is None:
            raise DatasetNotFoundError(
                f"Raw dataset '{name}' not found",
                hint="Run 'uesf data raw list' to see available datasets.",
            )
        return row

    def list_raw(self) -> list[dict[str, Any]]:
        """List all registered raw datasets."""
        return self.db.fetch_all("SELECT * FROM raw_datasets ORDER BY name")

    def remove_raw(self, name: str, delete_preprocessed: bool = False) -> None:
        """Remove a raw dataset and handle cascading.

        Args:
            name: Dataset name to remove.
            delete_preprocessed: If True, delete dependent preprocessed datasets.
                If False, mark them as orphans.
        """
        record = self.get_raw(name)  # Raises if not found

        with self.db.transaction() as cursor:
            if delete_preprocessed:
                # Find all dependent preprocessed datasets
                prep_rows = cursor.execute(
                    "SELECT id, name, data_dir_path FROM preprocessed_datasets WHERE source_raw_dataset_id = ?",
                    (record["id"],),
                ).fetchall()

                for prep in prep_rows:
                    # Delete dependent masked datasets
                    masked_rows = cursor.execute(
                        "SELECT data_dir_path FROM masked_datasets WHERE source_dataset_id = ?",
                        (prep["id"],),
                    ).fetchall()
                    for masked in masked_rows:
                        if masked["data_dir_path"]:
                            masked_dir = Path(masked["data_dir_path"])
                            if masked_dir.exists():
                                shutil.rmtree(masked_dir)
                    cursor.execute("DELETE FROM masked_datasets WHERE source_dataset_id = ?", (prep["id"],))

                    # Delete preprocessed data files
                    if prep["data_dir_path"]:
                        prep_dir = Path(prep["data_dir_path"])
                        if prep_dir.exists():
                            shutil.rmtree(prep_dir)

                cursor.execute(
                    "DELETE FROM preprocessed_datasets WHERE source_raw_dataset_id = ?",
                    (record["id"],),
                )
            else:
                # Mark preprocessed datasets as orphans
                cursor.execute(
                    """UPDATE preprocessed_datasets
                       SET is_orphan = 1, source_raw_dataset_id = NULL,
                           updated_at = CURRENT_TIMESTAMP
                       WHERE source_raw_dataset_id = ?""",
                    (record["id"],),
                )

            # Delete raw dataset files if imported
            if record["is_imported"] and record["data_dir_path"]:
                raw_dir = Path(record["data_dir_path"])
                if raw_dir.exists():
                    shutil.rmtree(raw_dir)

            cursor.execute("DELETE FROM raw_datasets WHERE id = ?", (record["id"],))

        logger.info("Removed raw dataset '%s'", name)

    def edit_raw(self, name: str, **fields: Any) -> dict[str, Any]:
        """Edit metadata fields of a raw dataset.

        Args:
            name: Dataset name to edit.
            **fields: Fields to update (description, sampling_rate, etc.)

        Returns:
            Updated database record.
        """
        self.get_raw(name)  # Verify exists

        allowed = {"description", "sampling_rate", "n_sessions", "n_recordings",
                    "n_channels", "n_samples", "electrode_list"}
        update_fields = {k: v for k, v in fields.items() if k in allowed and v is not None}
        if not update_fields:
            return self.get_raw(name)

        set_clauses = []
        values = []
        for k, v in update_fields.items():
            set_clauses.append(f"{k} = ?")
            values.append(json.dumps(v) if isinstance(v, (list, dict)) else v)

        set_clauses.append("updated_at = CURRENT_TIMESTAMP")
        values.append(name)

        self.db.execute(
            f"UPDATE raw_datasets SET {', '.join(set_clauses)} WHERE name = ?",
            tuple(values),
        )
        self.db.commit()

        logger.info("Updated raw dataset '%s': %s", name, list(update_fields.keys()))
        return self.get_raw(name)

    # ── Internal Helpers ────────────────────────────────────────────────

    def _parse_raw_yml(self, dataset_path: Path) -> dict[str, Any]:
        """Parse and validate raw.yml from a dataset directory."""
        yml_path = dataset_path / "raw.yml"
        if not yml_path.exists():
            raise YAMLParseError(
                f"raw.yml not found in '{dataset_path}'",
                context={"path": str(dataset_path)},
                hint="Create a raw.yml file in the dataset directory.",
            )

        with open(yml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict) or "raw" not in data:
            raise YAMLParseError(
                "raw.yml must have a top-level 'raw' key",
                context={"path": str(yml_path)},
                hint="Wrap your config under 'raw:' key.",
            )

        raw_config = data["raw"]
        if not isinstance(raw_config, dict):
            raise YAMLParseError(
                "'raw' key must contain a mapping",
                context={"path": str(yml_path)},
            )

        # Check required keys
        missing = _RAW_REQUIRED_KEYS - set(raw_config.keys())
        if missing:
            raise MissingRequiredKeyError(
                f"raw.yml is missing required keys: {', '.join(sorted(missing))}",
                context={"path": str(yml_path), "missing_keys": sorted(missing)},
                hint=f"Add the following keys to raw.yml: {', '.join(sorted(missing))}",
            )

        # Ensure name exists (use directory name as fallback)
        if "name" not in raw_config:
            raw_config["name"] = dataset_path.name

        # Normalize numeric_to_semantic keys to strings
        n2s = raw_config["numeric_to_semantic"]
        raw_config["numeric_to_semantic"] = {str(k): v for k, v in n2s.items()}

        return raw_config

    def _find_mat_files(self, dataset_path: Path) -> list[Path]:
        """Find all .mat files in a dataset directory."""
        mat_files = sorted(dataset_path.glob("*.mat"))
        if not mat_files:
            raise DatasetNotFoundError(
                f"No .mat files found in '{dataset_path}'",
                context={"path": str(dataset_path)},
                hint="Ensure the dataset directory contains subject_*.mat files.",
            )
        return mat_files

    def _infer_shapes(
        self, mat_files: list[Path], eeg_data_key: str, label_key: str
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Infer and validate data/label shapes from .mat files.

        All subjects must have identical shapes. Returns the common
        (data_shape, label_shape).
        """
        data_shape: tuple[int, ...] | None = None
        label_shape: tuple[int, ...] | None = None
        mismatches: list[str] = []

        for mat_file in mat_files:
            mat_data = loadmat(str(mat_file), squeeze_me=False)

            if eeg_data_key not in mat_data:
                raise MissingRequiredKeyError(
                    f"Key '{eeg_data_key}' not found in '{mat_file.name}'",
                    context={"file": str(mat_file), "available_keys": [k for k in mat_data if not k.startswith("__")]},
                    hint=f"Check that eeg_data_key='{eeg_data_key}' matches the .mat file contents.",
                )
            if label_key not in mat_data:
                raise MissingRequiredKeyError(
                    f"Key '{label_key}' not found in '{mat_file.name}'",
                    context={"file": str(mat_file), "available_keys": [k for k in mat_data if not k.startswith("__")]},
                    hint=f"Check that label_key='{label_key}' matches the .mat file contents.",
                )

            current_data_shape = tuple(mat_data[eeg_data_key].shape)
            current_label_shape = tuple(mat_data[label_key].shape)

            if data_shape is None:
                data_shape = current_data_shape
                label_shape = current_label_shape
            else:
                if current_data_shape != data_shape:
                    mismatches.append(
                        f"{mat_file.name}: data_shape={current_data_shape} (expected {data_shape})"
                    )
                if current_label_shape != label_shape:
                    mismatches.append(
                        f"{mat_file.name}: label_shape={current_label_shape} (expected {label_shape})"
                    )

        if mismatches:
            raise ShapeMismatchError(
                "Inconsistent shapes across subject files",
                context={"mismatches": mismatches},
                hint="Ensure all subject .mat files have identical data and label dimensions.",
            )

        assert data_shape is not None and label_shape is not None
        return data_shape, label_shape
