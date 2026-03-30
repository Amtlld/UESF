"""UESF Metric Manager - manages EMBEDDED, REGISTERED, and GLOBAL metrics."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

from uesf.components.builtin_metrics import BUILTIN_METRICS
from uesf.core.config import ConfigManager
from uesf.core.database import DatabaseManager
from uesf.core.exceptions import ComponentNotFoundError
from uesf.core.logging import get_logger
from uesf.managers.model_manager import _import_class, _parse_entrypoint

logger = get_logger("manager.metric")


class MetricManager:
    """Manages metric lifecycle: add, list, remove, load, auto-reregister."""

    def __init__(self, db: DatabaseManager, config: ConfigManager) -> None:
        self.db = db
        self.config = config

    def add_global(
        self,
        source_path: Path,
        name: str,
        description: str | None = None,
    ) -> dict[str, Any]:
        """Add a global metric from a source file."""
        source_path = Path(source_path).resolve()
        if not source_path.exists():
            raise ComponentNotFoundError(
                f"Metric source file not found: '{source_path}'",
                hint="Check the file path.",
            )

        source_code = source_path.read_text(encoding="utf-8")

        metrics_dir = self.config.uesf_home / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        dest = metrics_dir / f"{name}.py"
        shutil.copy2(source_path, dest)

        with self.db.transaction() as cursor:
            cursor.execute(
                """INSERT INTO metrics (name, description, metric_path, metric_type, source_code_snapshot)
                   VALUES (?, ?, ?, ?, ?)""",
                (name, description, str(dest), "GLOBAL", source_code),
            )

        logger.info("Added global metric '%s' from '%s'", name, source_path)
        return self.get(name)

    def get(self, name: str) -> dict[str, Any]:
        """Get a metric record by name."""
        row = self.db.fetch_one("SELECT * FROM metrics WHERE name = ?", (name,))
        if row is None:
            raise ComponentNotFoundError(
                f"Metric '{name}' not found",
                hint="Run 'uesf metric list' to see available metrics.",
            )
        return row

    def list(self, show_obsolete: bool = False) -> list[dict[str, Any]]:
        """List all metrics."""
        if show_obsolete:
            return self.db.fetch_all("SELECT * FROM metrics ORDER BY name")
        return self.db.fetch_all("SELECT * FROM metrics WHERE is_obsolete = 0 ORDER BY name")

    def remove(self, name: str) -> None:
        """Remove a metric."""
        record = self.get(name)
        if record["metric_type"] == "GLOBAL" and record["metric_path"]:
            path = Path(record["metric_path"])
            if path.exists():
                path.unlink()

        self.db.execute("DELETE FROM metrics WHERE id = ?", (record["id"],))
        self.db.commit()
        logger.info("Removed metric '%s'", name)

    def edit(self, name: str, description: str | None = None) -> dict[str, Any]:
        """Edit metric metadata."""
        self.get(name)
        if description is not None:
            self.db.execute(
                "UPDATE metrics SET description = ?, updated_at = CURRENT_TIMESTAMP WHERE name = ?",
                (description, name),
            )
            self.db.commit()
        return self.get(name)

    def register(self, name: str, entrypoint: str, project_dir: Path) -> dict[str, Any]:
        """Register a project-level metric (REGISTERED type)."""
        file_path, _ = _parse_entrypoint(entrypoint, project_dir)
        source_code = file_path.read_text(encoding="utf-8")

        with self.db.transaction() as cursor:
            cursor.execute(
                """INSERT INTO metrics (name, description, metric_path, metric_type, source_code_snapshot)
                   VALUES (?, ?, ?, ?, ?)""",
                (name, f"Registered from {entrypoint}", str(file_path), "REGISTERED", source_code),
            )

        logger.info("Registered metric '%s' from '%s'", name, entrypoint)
        return self.get(name)

    def detect_and_reregister(self, name: str, entrypoint: str, project_dir: Path) -> dict[str, Any]:
        """Check if source code has changed and re-register if needed."""
        record = self.get(name)
        if record["metric_type"] != "REGISTERED":
            return record

        file_path, _ = _parse_entrypoint(entrypoint, project_dir)
        current_code = file_path.read_text(encoding="utf-8")
        current_hash = hashlib.sha256(current_code.encode()).hexdigest()
        stored_hash = hashlib.sha256(
            (record["source_code_snapshot"] or "").encode()
        ).hexdigest()

        if current_hash == stored_hash:
            return record

        archive_name = f"{name}_{stored_hash[:8]}"
        logger.warning("Metric '%s' source changed. Archiving as '%s'", name, archive_name)

        with self.db.transaction() as cursor:
            cursor.execute(
                """UPDATE metrics SET name = ?, is_obsolete = 1, updated_at = CURRENT_TIMESTAMP
                   WHERE id = ?""",
                (archive_name, record["id"]),
            )
            cursor.execute(
                """INSERT INTO metrics (name, description, metric_path, metric_type, source_code_snapshot)
                   VALUES (?, ?, ?, ?, ?)""",
                (name, f"Re-registered from {entrypoint}", str(file_path), "REGISTERED", current_code),
            )

        return self.get(name)

    def load_metric(
        self,
        name: str,
        entrypoint: str | None = None,
        project_dir: Path | None = None,
    ) -> callable:
        """Load a metric function by name.

        Resolution order:
        1. If entrypoint provided, load from file
        2. Check database for GLOBAL/REGISTERED metrics
        3. Check built-in metrics

        Returns:
            The metric callable.
        """
        if entrypoint:
            file_path, func_name = _parse_entrypoint(entrypoint, project_dir or Path.cwd())
            return _import_class(file_path, func_name)

        # Check database
        row = self.db.fetch_one(
            "SELECT * FROM metrics WHERE name = ? AND is_obsolete = 0",
            (name,),
        )
        if row and row["metric_path"]:
            file_path = Path(row["metric_path"])
            # Infer function name from metric name
            func_name = name
            return _import_class(file_path, func_name)

        # Check built-in
        if name in BUILTIN_METRICS:
            return BUILTIN_METRICS[name]

        raise ComponentNotFoundError(
            f"Metric '{name}' not found",
            context={"available_builtin": sorted(BUILTIN_METRICS.keys())},
            hint="Check the metric name or add it with 'uesf metric add'.",
        )
