"""UESF Trainer Manager - manages EMBEDDED, REGISTERED, and GLOBAL trainers."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

from uesf.core.config import ConfigManager
from uesf.core.database import DatabaseManager
from uesf.core.exceptions import ComponentNotFoundError
from uesf.core.logging import get_logger
from uesf.managers.model_manager import _import_class, _parse_entrypoint

logger = get_logger("manager.trainer")


class TrainerManager:
    """Manages trainer lifecycle: add, list, remove, load, auto-reregister."""

    def __init__(self, db: DatabaseManager, config: ConfigManager) -> None:
        self.db = db
        self.config = config

    def add_global(
        self,
        source_path: Path,
        name: str,
        description: str | None = None,
    ) -> dict[str, Any]:
        """Add a global trainer from a source file."""
        source_path = Path(source_path).resolve()
        if not source_path.exists():
            raise ComponentNotFoundError(
                f"Trainer source file not found: '{source_path}'",
                hint="Check the file path.",
            )

        source_code = source_path.read_text(encoding="utf-8")

        trainers_dir = self.config.uesf_home / "trainers"
        trainers_dir.mkdir(parents=True, exist_ok=True)
        dest = trainers_dir / f"{name}.py"
        shutil.copy2(source_path, dest)

        with self.db.transaction() as cursor:
            cursor.execute(
                """INSERT INTO trainers (name, description, trainer_path, trainer_type, source_code_snapshot)
                   VALUES (?, ?, ?, ?, ?)""",
                (name, description, str(dest), "GLOBAL", source_code),
            )

        logger.info("Added global trainer '%s' from '%s'", name, source_path)
        return self.get(name)

    def get(self, name: str) -> dict[str, Any]:
        """Get a trainer record by name."""
        row = self.db.fetch_one("SELECT * FROM trainers WHERE name = ?", (name,))
        if row is None:
            raise ComponentNotFoundError(
                f"Trainer '{name}' not found",
                hint="Run 'uesf trainer list' to see available trainers.",
            )
        return row

    def list(self, show_obsolete: bool = False) -> list[dict[str, Any]]:
        """List all trainers."""
        if show_obsolete:
            return self.db.fetch_all("SELECT * FROM trainers ORDER BY name")
        return self.db.fetch_all("SELECT * FROM trainers WHERE is_obsolete = 0 ORDER BY name")

    def remove(self, name: str) -> None:
        """Remove a trainer."""
        record = self.get(name)
        if record["trainer_type"] == "GLOBAL" and record["trainer_path"]:
            path = Path(record["trainer_path"])
            if path.exists():
                path.unlink()

        self.db.execute("DELETE FROM trainers WHERE id = ?", (record["id"],))
        self.db.commit()
        logger.info("Removed trainer '%s'", name)

    def edit(self, name: str, description: str | None = None) -> dict[str, Any]:
        """Edit trainer metadata."""
        self.get(name)
        if description is not None:
            self.db.execute(
                "UPDATE trainers SET description = ?, updated_at = CURRENT_TIMESTAMP WHERE name = ?",
                (description, name),
            )
            self.db.commit()
        return self.get(name)

    def register(self, name: str, entrypoint: str, project_dir: Path) -> dict[str, Any]:
        """Register a project-level trainer (REGISTERED type)."""
        file_path, _ = _parse_entrypoint(entrypoint, project_dir)
        source_code = file_path.read_text(encoding="utf-8")

        with self.db.transaction() as cursor:
            cursor.execute(
                """INSERT INTO trainers (name, description, trainer_path, trainer_type, source_code_snapshot)
                   VALUES (?, ?, ?, ?, ?)""",
                (name, f"Registered from {entrypoint}", str(file_path), "REGISTERED", source_code),
            )

        logger.info("Registered trainer '%s' from '%s'", name, entrypoint)
        return self.get(name)

    def detect_and_reregister(self, name: str, entrypoint: str, project_dir: Path) -> dict[str, Any]:
        """Check if source code has changed and re-register if needed."""
        record = self.get(name)
        if record["trainer_type"] != "REGISTERED":
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
        logger.warning("Trainer '%s' source changed. Archiving as '%s'", name, archive_name)

        with self.db.transaction() as cursor:
            cursor.execute(
                """UPDATE trainers SET name = ?, is_obsolete = 1, updated_at = CURRENT_TIMESTAMP
                   WHERE id = ?""",
                (archive_name, record["id"]),
            )
            cursor.execute(
                """INSERT INTO trainers (name, description, trainer_path, trainer_type, source_code_snapshot)
                   VALUES (?, ?, ?, ?, ?)""",
                (name, f"Re-registered from {entrypoint}", str(file_path), "REGISTERED", current_code),
            )

        return self.get(name)

    def load_class(self, name: str, entrypoint: str | None = None, project_dir: Path | None = None) -> type:
        """Dynamically load a trainer class."""
        if entrypoint:
            file_path, class_name = _parse_entrypoint(entrypoint, project_dir or Path.cwd())
        else:
            record = self.get(name)
            if not record["trainer_path"]:
                raise ComponentNotFoundError(
                    f"Trainer '{name}' has no source path",
                    hint="The trainer may be an embedded trainer.",
                )
            file_path = Path(record["trainer_path"])
            class_name = name.title().replace("_", "")

        return _import_class(file_path, class_name)
