"""Tests for UESF database manager."""

import sqlite3

import pytest

from uesf.core.database import CURRENT_SCHEMA_VERSION, DatabaseManager


class TestDatabaseInitialization:
    def test_creates_all_tables(self, db):
        tables = db.get_table_names()
        expected = [
            "configs",
            "experiments",
            "masked_datasets",
            "metrics",
            "models",
            "preprocessed_datasets",
            "raw_datasets",
            "schema_versions",
            "trainers",
        ]
        assert sorted(tables) == expected

    def test_seeds_default_configs(self, db):
        rows = db.fetch_all("SELECT key, value FROM configs ORDER BY key")
        keys = [r["key"] for r in rows]
        assert "data_dir" in keys
        assert "default_device" in keys
        assert "num_workers" in keys
        assert "log_level" in keys

    def test_default_config_values(self, db):
        configs = {r["key"]: r["value"] for r in db.fetch_all("SELECT key, value FROM configs")}
        assert configs["data_dir"] == "<uesf-home>/data"
        assert configs["default_device"] == "cpu"
        assert configs["num_workers"] == "4"
        assert configs["log_level"] == "INFO"

    def test_records_schema_version(self, db):
        row = db.fetch_one("SELECT version FROM schema_versions WHERE version = ?", (CURRENT_SCHEMA_VERSION,))
        assert row is not None
        assert row["version"] == CURRENT_SCHEMA_VERSION

    def test_idempotent_initialization(self, db):
        # Calling initialize again should not fail or duplicate data
        db.initialize()
        rows = db.fetch_all("SELECT key FROM configs")
        assert len(rows) == 4


class TestDatabaseOperations:
    def test_execute_insert_and_fetch(self, db):
        db.execute(
            "INSERT INTO raw_datasets (name, eeg_data_key, label_key, dimension_info, numeric_to_semantic) "
            "VALUES (?, ?, ?, ?, ?)",
            ("test_ds", "data", "label", '["session"]', '{"0": "a"}'),
        )
        db.commit()

        row = db.fetch_one("SELECT * FROM raw_datasets WHERE name = ?", ("test_ds",))
        assert row is not None
        assert row["name"] == "test_ds"
        assert row["eeg_data_key"] == "data"

    def test_fetch_all(self, db):
        for i in range(3):
            db.execute(
                "INSERT INTO raw_datasets (name, eeg_data_key, label_key, dimension_info, numeric_to_semantic) "
                "VALUES (?, ?, ?, ?, ?)",
                (f"ds_{i}", "data", "label", '["session"]', '{"0": "a"}'),
            )
        db.commit()

        rows = db.fetch_all("SELECT name FROM raw_datasets ORDER BY name")
        assert len(rows) == 3
        assert [r["name"] for r in rows] == ["ds_0", "ds_1", "ds_2"]

    def test_fetch_one_returns_none_for_missing(self, db):
        row = db.fetch_one("SELECT * FROM raw_datasets WHERE name = ?", ("nonexistent",))
        assert row is None


class TestTransactions:
    def test_transaction_commits_on_success(self, db):
        with db.transaction() as cursor:
            cursor.execute(
                "INSERT INTO raw_datasets (name, eeg_data_key, label_key, dimension_info, numeric_to_semantic) "
                "VALUES (?, ?, ?, ?, ?)",
                ("tx_test", "data", "label", '["session"]', '{"0": "a"}'),
            )

        row = db.fetch_one("SELECT name FROM raw_datasets WHERE name = ?", ("tx_test",))
        assert row is not None

    def test_transaction_rolls_back_on_error(self, db):
        with pytest.raises(ValueError):
            with db.transaction() as cursor:
                cursor.execute(
                    "INSERT INTO raw_datasets (name, eeg_data_key, label_key, dimension_info, numeric_to_semantic) "
                    "VALUES (?, ?, ?, ?, ?)",
                    ("rollback_test", "data", "label", '["session"]', '{"0": "a"}'),
                )
                raise ValueError("intentional error")

        row = db.fetch_one("SELECT name FROM raw_datasets WHERE name = ?", ("rollback_test",))
        assert row is None

    def test_unique_constraint(self, db):
        db.execute(
            "INSERT INTO raw_datasets (name, eeg_data_key, label_key, dimension_info, numeric_to_semantic) "
            "VALUES (?, ?, ?, ?, ?)",
            ("unique_ds", "data", "label", '["session"]', '{"0": "a"}'),
        )
        db.commit()

        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO raw_datasets (name, eeg_data_key, label_key, dimension_info, numeric_to_semantic) "
                "VALUES (?, ?, ?, ?, ?)",
                ("unique_ds", "data", "label", '["session"]', '{"0": "a"}'),
            )


class TestDiskDatabase:
    def test_creates_db_file(self, db_on_disk, uesf_home):
        db_path = uesf_home / "uesf.db"
        assert db_path.exists()

    def test_persistence_across_connections(self, uesf_home):
        db_path = uesf_home / "uesf.db"

        db1 = DatabaseManager(db_path)
        db1.initialize()
        db1.execute(
            "INSERT INTO raw_datasets (name, eeg_data_key, label_key, dimension_info, numeric_to_semantic) "
            "VALUES (?, ?, ?, ?, ?)",
            ("persist_test", "data", "label", '["session"]', '{"0": "a"}'),
        )
        db1.commit()
        db1.close()

        db2 = DatabaseManager(db_path)
        row = db2.fetch_one("SELECT name FROM raw_datasets WHERE name = ?", ("persist_test",))
        assert row is not None
        db2.close()
