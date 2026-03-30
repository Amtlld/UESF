"""UESF Model Manager - manages EMBEDDED, REGISTERED, and GLOBAL models."""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
from pathlib import Path
from typing import Any

from uesf.core.config import ConfigManager
from uesf.core.database import DatabaseManager
from uesf.core.exceptions import (
    ComponentNotFoundError,
    InterfaceViolationError,
    SnapshotCreationError,
)
from uesf.core.logging import get_logger

logger = get_logger("manager.model")


class ModelManager:
    """Manages model lifecycle: add, list, remove, load, auto-reregister."""

    def __init__(self, db: DatabaseManager, config: ConfigManager) -> None:
        self.db = db
        self.config = config

    def add_global(
        self,
        source_path: Path,
        name: str,
        description: str | None = None,
    ) -> dict[str, Any]:
        """Add a global model from a source file.

        Copies the source file to ~/.uesf/models/ and stores a source code snapshot.

        Args:
            source_path: Path to the Python file containing the model class.
            name: Name for the model.
            description: Optional description.

        Returns:
            The database record.
        """
        source_path = Path(source_path).resolve()
        if not source_path.exists():
            raise ComponentNotFoundError(
                f"Model source file not found: '{source_path}'",
                hint="Check the file path.",
            )

        # Read source code for snapshot
        source_code = source_path.read_text(encoding="utf-8")

        # Copy to global models directory
        models_dir = self.config.uesf_home / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        dest = models_dir / f"{name}.py"
        shutil.copy2(source_path, dest)

        with self.db.transaction() as cursor:
            cursor.execute(
                """INSERT INTO models (name, description, model_path, model_type, source_code_snapshot)
                   VALUES (?, ?, ?, ?, ?)""",
                (name, description, str(dest), "GLOBAL", source_code),
            )

        logger.info("Added global model '%s' from '%s'", name, source_path)
        return self.get(name)

    def get(self, name: str) -> dict[str, Any]:
        """Get a model record by name."""
        row = self.db.fetch_one("SELECT * FROM models WHERE name = ?", (name,))
        if row is None:
            raise ComponentNotFoundError(
                f"Model '{name}' not found",
                hint="Run 'uesf model list' to see available models.",
            )
        return row

    def list(self, show_obsolete: bool = False) -> list[dict[str, Any]]:
        """List all models."""
        if show_obsolete:
            return self.db.fetch_all("SELECT * FROM models ORDER BY name")
        return self.db.fetch_all("SELECT * FROM models WHERE is_obsolete = 0 ORDER BY name")

    def remove(self, name: str) -> None:
        """Remove a model."""
        record = self.get(name)

        # Delete file for GLOBAL models
        if record["model_type"] == "GLOBAL" and record["model_path"]:
            path = Path(record["model_path"])
            if path.exists():
                path.unlink()

        self.db.execute("DELETE FROM models WHERE id = ?", (record["id"],))
        self.db.commit()
        logger.info("Removed model '%s'", name)

    def edit(self, name: str, description: str | None = None) -> dict[str, Any]:
        """Edit model metadata."""
        self.get(name)
        if description is not None:
            self.db.execute(
                "UPDATE models SET description = ?, updated_at = CURRENT_TIMESTAMP WHERE name = ?",
                (description, name),
            )
            self.db.commit()
        return self.get(name)

    def register(
        self, name: str, entrypoint: str, project_dir: Path,
    ) -> dict[str, Any]:
        """Register a project-level model (REGISTERED type).

        Called automatically when an experiment references a model defined
        in project.yml for the first time.

        Args:
            name: Model name.
            entrypoint: "path/to/file.py:ClassName" format.
            project_dir: Project root directory for path resolution.

        Returns:
            The database record.
        """
        file_path, _ = _parse_entrypoint(entrypoint, project_dir)
        source_code = file_path.read_text(encoding="utf-8")

        with self.db.transaction() as cursor:
            cursor.execute(
                """INSERT INTO models (name, description, model_path, model_type, source_code_snapshot)
                   VALUES (?, ?, ?, ?, ?)""",
                (name, f"Registered from {entrypoint}", str(file_path), "REGISTERED", source_code),
            )

        logger.info("Registered model '%s' from '%s'", name, entrypoint)
        return self.get(name)

    def detect_and_reregister(self, name: str, entrypoint: str, project_dir: Path) -> dict[str, Any]:
        """Check if source code has changed and re-register if needed.

        For REGISTERED models: compares current source SHA256 with stored snapshot.
        If changed, archives old record and creates new one.

        Returns:
            The current (possibly new) model record.
        """
        record = self.get(name)
        if record["model_type"] != "REGISTERED":
            return record

        file_path, _ = _parse_entrypoint(entrypoint, project_dir)
        current_code = file_path.read_text(encoding="utf-8")
        current_hash = hashlib.sha256(current_code.encode()).hexdigest()
        stored_hash = hashlib.sha256(
            (record["source_code_snapshot"] or "").encode()
        ).hexdigest()

        if current_hash == stored_hash:
            return record

        # Archive old record
        archive_name = f"{name}_{stored_hash[:8]}"
        logger.warning("Model '%s' source changed. Archiving as '%s'", name, archive_name)

        with self.db.transaction() as cursor:
            cursor.execute(
                """UPDATE models SET name = ?, is_obsolete = 1, updated_at = CURRENT_TIMESTAMP
                   WHERE id = ?""",
                (archive_name, record["id"]),
            )
            cursor.execute(
                """INSERT INTO models (name, description, model_path, model_type, source_code_snapshot)
                   VALUES (?, ?, ?, ?, ?)""",
                (name, f"Re-registered from {entrypoint}", str(file_path), "REGISTERED", current_code),
            )

        return self.get(name)

    def load_class(self, name: str, entrypoint: str | None = None, project_dir: Path | None = None) -> type:
        """Dynamically load a model class.

        Args:
            name: Model name (used to look up record if no entrypoint).
            entrypoint: "path/to/file.py:ClassName" (for project-level models).
            project_dir: Project directory for resolving relative paths.

        Returns:
            The model class.
        """
        if entrypoint:
            file_path, class_name = _parse_entrypoint(entrypoint, project_dir or Path.cwd())
        else:
            record = self.get(name)
            if not record["model_path"]:
                raise ComponentNotFoundError(
                    f"Model '{name}' has no source path",
                    hint="The model may be an embedded model.",
                )
            file_path = Path(record["model_path"])
            # Infer class name from record or use capitalized name
            class_name = name.title().replace("_", "")

        return _import_class(file_path, class_name)


def _parse_entrypoint(entrypoint: str, project_dir: Path) -> tuple[Path, str]:
    """Parse an entrypoint string like './path/to/file.py:ClassName'."""
    if ":" not in entrypoint:
        raise InterfaceViolationError(
            f"Invalid entrypoint format: '{entrypoint}'",
            hint="Use format 'path/to/file.py:ClassName'.",
        )
    file_part, class_name = entrypoint.rsplit(":", 1)
    file_path = (project_dir / file_part).resolve()
    if not file_path.exists():
        raise ComponentNotFoundError(
            f"Entrypoint file not found: '{file_path}'",
            context={"entrypoint": entrypoint, "project_dir": str(project_dir)},
            hint="Check the file path in your entrypoint.",
        )
    return file_path, class_name


def _import_class(file_path: Path, class_name: str) -> type:
    """Dynamically import a class from a Python file."""
    spec = importlib.util.spec_from_file_location(file_path.stem, str(file_path))
    if spec is None or spec.loader is None:
        raise SnapshotCreationError(
            f"Cannot load module from '{file_path}'",
            hint="Ensure the file is a valid Python module.",
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    if not hasattr(module, class_name):
        raise InterfaceViolationError(
            f"Class '{class_name}' not found in '{file_path}'",
            context={"available": [n for n in dir(module) if not n.startswith("_")]},
            hint=f"Ensure '{class_name}' is defined in the file.",
        )
    return getattr(module, class_name)
