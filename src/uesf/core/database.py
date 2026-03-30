"""UESF SQLite database manager.

Handles database initialization, schema creation, migration, and
provides a transaction context manager for atomic operations.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

from uesf.core.logging import get_logger

logger = get_logger("db")

CURRENT_SCHEMA_VERSION = 1

# --- DDL Statements ---

_DDL_RAW_DATASETS = """
CREATE TABLE IF NOT EXISTS raw_datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    is_imported BOOLEAN,
    data_dir_path TEXT,
    eeg_data_key TEXT NOT NULL,
    label_key TEXT NOT NULL,
    n_subjects INTEGER,
    sampling_rate REAL,
    n_sessions INTEGER,
    n_recordings INTEGER,
    n_channels INTEGER,
    n_samples INTEGER,
    electrode_list TEXT,
    data_shape TEXT,
    dimension_info TEXT NOT NULL,
    label_shape TEXT,
    numeric_to_semantic TEXT NOT NULL,
    raw_info_snapshot TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

_DDL_PREPROCESSED_DATASETS = """
CREATE TABLE IF NOT EXISTS preprocessed_datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    source_raw_dataset_id INTEGER,
    data_dir_path TEXT,
    n_subjects INTEGER,
    sampling_rate REAL,
    n_sessions INTEGER,
    n_recordings INTEGER,
    n_channels INTEGER,
    n_samples INTEGER,
    electrode_list TEXT,
    data_shape TEXT,
    dimension_info TEXT NOT NULL DEFAULT '["subject", "session", "recording", "channel", "sample"]',
    label_shape TEXT,
    numeric_to_semantic TEXT NOT NULL,
    preprocess_config_snapshot TEXT,
    is_orphan BOOLEAN DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(source_raw_dataset_id) REFERENCES raw_datasets(id)
);
"""

_DDL_MASKED_DATASETS = """
CREATE TABLE IF NOT EXISTS masked_datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    source_dataset_id INTEGER,
    data_dir_path TEXT,
    label_mapping TEXT NOT NULL,
    numeric_to_semantic TEXT NOT NULL,
    n_classes INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(source_dataset_id) REFERENCES preprocessed_datasets(id)
);
"""

_DDL_TRAINERS = """
CREATE TABLE IF NOT EXISTS trainers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    trainer_path TEXT,
    trainer_type TEXT,
    is_obsolete BOOLEAN DEFAULT 0,
    source_code_snapshot TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

_DDL_MODELS = """
CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    model_path TEXT,
    model_type TEXT,
    is_obsolete BOOLEAN DEFAULT 0,
    source_code_snapshot TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

_DDL_METRICS = """
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    metric_path TEXT,
    metric_type TEXT,
    is_obsolete BOOLEAN DEFAULT 0,
    source_code_snapshot TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

_DDL_EXPERIMENTS = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_name TEXT NOT NULL,
    experiment_name TEXT NOT NULL,
    description TEXT,
    model_id INTEGER,
    trainer_id INTEGER,
    config TEXT,
    results TEXT,
    status TEXT DEFAULT 'PENDING',
    environment_snapshot TEXT,
    checkpoint_dir_path TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(model_id) REFERENCES models(id),
    FOREIGN KEY(trainer_id) REFERENCES trainers(id)
);
"""

_DDL_EXPERIMENTS_INDICES = """
CREATE INDEX IF NOT EXISTS idx_experiments_project_name ON experiments(project_name);
CREATE INDEX IF NOT EXISTS idx_experiments_experiment_name ON experiments(experiment_name);
CREATE INDEX IF NOT EXISTS idx_experiments_model_id ON experiments(model_id);
CREATE INDEX IF NOT EXISTS idx_experiments_trainer_id ON experiments(trainer_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_experiments_project_experiment ON experiments(project_name, experiment_name);
"""

_DDL_CONFIGS = """
CREATE TABLE IF NOT EXISTS configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT UNIQUE NOT NULL,
    value TEXT NOT NULL,
    description TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

_DDL_SCHEMA_VERSIONS = """
CREATE TABLE IF NOT EXISTS schema_versions (
    version INTEGER PRIMARY KEY,
    description TEXT,
    applied_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

_ALL_DDL = [
    _DDL_RAW_DATASETS,
    _DDL_PREPROCESSED_DATASETS,
    _DDL_MASKED_DATASETS,
    _DDL_TRAINERS,
    _DDL_MODELS,
    _DDL_METRICS,
    _DDL_EXPERIMENTS,
    _DDL_EXPERIMENTS_INDICES,
    _DDL_CONFIGS,
    _DDL_SCHEMA_VERSIONS,
]

# Default config seed data
_DEFAULT_CONFIGS = [
    ("data_dir", "~/.uesf/data", "UESF-managed datasets storage directory"),
    ("default_device", "cpu", "Default compute device (e.g., cpu, cuda:0)"),
    ("num_workers", "4", "DataLoader worker processes"),
    ("log_level", "INFO", "Framework logging level (DEBUG, INFO, WARNING, ERROR)"),
]


class DatabaseManager:
    """Manages the UESF SQLite database lifecycle."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            uesf_home = Path(os.environ.get("UESF_HOME", Path.home() / ".uesf"))
            uesf_home.mkdir(parents=True, exist_ok=True)
            db_path = uesf_home / "uesf.db"

        self.db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None

    def get_connection(self) -> sqlite3.Connection:
        """Get or create a database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA journal_mode = WAL")
        return self._conn

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def initialize(self) -> None:
        """Create all tables and seed default config data.

        Safe to call multiple times (uses IF NOT EXISTS).
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            for ddl in _ALL_DDL:
                cursor.executescript(ddl)

            # Seed default configs (only if table is empty)
            cursor.execute("SELECT COUNT(*) FROM configs")
            if cursor.fetchone()[0] == 0:
                cursor.executemany(
                    "INSERT INTO configs (key, value, description) VALUES (?, ?, ?)",
                    _DEFAULT_CONFIGS,
                )

            # Record schema version
            cursor.execute("SELECT COUNT(*) FROM schema_versions WHERE version = ?", (CURRENT_SCHEMA_VERSION,))
            if cursor.fetchone()[0] == 0:
                cursor.execute(
                    "INSERT INTO schema_versions (version, description) VALUES (?, ?)",
                    (CURRENT_SCHEMA_VERSION, "Initial schema"),
                )

            conn.commit()
            logger.debug("Database initialized (schema v%d)", CURRENT_SCHEMA_VERSION)
        except Exception:
            conn.rollback()
            raise

    def execute(self, sql: str, params: tuple | dict = ()) -> sqlite3.Cursor:
        """Execute a single SQL statement."""
        conn = self.get_connection()
        logger.debug("SQL: %s | params: %s", sql.strip()[:200], params)
        return conn.execute(sql, params)

    def executemany(self, sql: str, params_seq: list) -> sqlite3.Cursor:
        """Execute a SQL statement against multiple parameter sets."""
        conn = self.get_connection()
        return conn.executemany(sql, params_seq)

    def commit(self) -> None:
        """Commit the current transaction."""
        conn = self.get_connection()
        conn.commit()

    def rollback(self) -> None:
        """Roll back the current transaction."""
        conn = self.get_connection()
        conn.rollback()

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Cursor, None, None]:
        """Context manager for atomic database operations.

        Automatically commits on success or rolls back on exception.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def fetch_one(self, sql: str, params: tuple | dict = ()) -> dict[str, Any] | None:
        """Execute query and return first row as dict, or None."""
        cursor = self.execute(sql, params)
        row = cursor.fetchone()
        return dict(row) if row else None

    def fetch_all(self, sql: str, params: tuple | dict = ()) -> list[dict[str, Any]]:
        """Execute query and return all rows as list of dicts."""
        cursor = self.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]

    def get_table_names(self) -> list[str]:
        """Return list of all user table names in the database."""
        rows = self.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        return [r["name"] for r in rows]
