"""UESF Project Manager - manages project initialization, loading, and component resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from uesf.core.config import ConfigManager
from uesf.core.database import DatabaseManager
from uesf.core.exceptions import (
    ComponentNotFoundError,
    MissingRequiredKeyError,
    YAMLParseError,
)
from uesf.core.logging import get_logger

logger = get_logger("manager.project")

DEFAULT_PROJECT_YML = """\
project-name: {name}

description: ""

# preprocessed_datasets:
#   - my_preprocessed_dataset

# raw_datasets:
#   - my_raw_dataset
# preprocess_config: ./preprocess.yml

models:
  # my_model:
  #   entrypoint: "./src/models/model.py:MyModelClass"

trainers:
  # my_trainer:
  #   entrypoint: "./src/trainers/trainer.py:MyTrainerClass"

metrics:
  # my_metric:
  #   entrypoint: "./src/metrics/metric.py:my_metric_func"
"""


class ProjectManager:
    """Manages project lifecycle: init, load, resolve components."""

    def __init__(self, db: DatabaseManager, config: ConfigManager) -> None:
        self.db = db
        self.config = config

    def init(self, project_dir: Path) -> Path:
        """Initialize a new project directory with default structure.

        Args:
            project_dir: Path to the project directory.

        Returns:
            Path to the created project.yml.
        """
        project_dir = Path(project_dir).resolve()
        project_dir.mkdir(parents=True, exist_ok=True)

        # Create standard subdirectories
        (project_dir / "experiments").mkdir(exist_ok=True)

        # Create project.yml if it doesn't exist
        yml_path = project_dir / "project.yml"
        if yml_path.exists():
            logger.warning("project.yml already exists in '%s', skipping creation", project_dir)
            return yml_path

        project_name = project_dir.name
        yml_path.write_text(
            DEFAULT_PROJECT_YML.format(name=project_name),
            encoding="utf-8",
        )

        logger.info("Initialized project '%s' at '%s'", project_name, project_dir)
        return yml_path

    def load(self, project_dir: Path) -> dict[str, Any]:
        """Load and parse a project.yml file.

        Args:
            project_dir: Path to the project directory.

        Returns:
            Parsed project configuration dict.

        Raises:
            ComponentNotFoundError: If project.yml not found.
            YAMLParseError: If YAML is invalid.
        """
        project_dir = Path(project_dir).resolve()
        yml_path = project_dir / "project.yml"

        if not yml_path.exists():
            raise ComponentNotFoundError(
                f"project.yml not found in '{project_dir}'",
                hint="Run 'uesf project init' to create a project, or check the directory.",
            )

        try:
            with open(yml_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise YAMLParseError(
                f"Failed to parse project.yml: {exc}",
                hint="Check YAML syntax in project.yml.",
            ) from exc

        if not isinstance(data, dict):
            raise YAMLParseError(
                "project.yml must be a YAML mapping",
                hint="Ensure the file starts with key-value pairs.",
            )

        if "project-name" not in data:
            raise MissingRequiredKeyError(
                "Missing required key 'project-name' in project.yml",
                hint="Add 'project-name: <name>' to your project.yml.",
            )

        return data

    def info(self, project_dir: Path) -> dict[str, Any]:
        """Get project info summary.

        Returns:
            Dict with project name, datasets, models, trainers, metrics.
        """
        data = self.load(project_dir)
        project_dir = Path(project_dir).resolve()

        info = {
            "project_name": data.get("project-name", "unknown"),
            "description": data.get("description", ""),
            "project_dir": str(project_dir),
            "preprocessed_datasets": data.get("preprocessed_datasets", []),
            "raw_datasets": data.get("raw_datasets", []),
            "models": list((data.get("models") or {}).keys()),
            "trainers": list((data.get("trainers") or {}).keys()),
            "metrics": list((data.get("metrics") or {}).keys()),
        }

        return info

    def resolve_component(
        self,
        name: str,
        component_type: str,
        project_config: dict[str, Any],
        project_dir: Path,
    ) -> dict[str, Any]:
        """Resolve a component name using three-level priority.

        Priority (high to low):
        1. Project-level (defined in project.yml)
        2. Global (added via `uesf <type> add`)
        3. Embedded (built-in)

        Args:
            name: Component name to resolve.
            component_type: One of "models", "trainers", "metrics".
            project_config: Parsed project.yml dict.
            project_dir: Project root directory.

        Returns:
            Dict with resolution info: {"source": "PROJECT"|"GLOBAL"|"EMBEDDED",
            "name": str, "entrypoint": str | None}.
        """
        # 1. Project-level
        project_components = project_config.get(component_type) or {}
        if name in project_components:
            comp_config = project_components[name]
            entrypoint = comp_config.get("entrypoint") if isinstance(comp_config, dict) else None

            # Check if name shadows a global or embedded component
            table = _component_type_to_table(component_type)
            existing = self.db.fetch_one(f"SELECT name FROM {table} WHERE name = ?", (name,))
            if existing:
                logger.warning(
                    "Project-level %s '%s' shadows a global/embedded component with the same name",
                    component_type.rstrip("s"),
                    name,
                )

            return {"source": "PROJECT", "name": name, "entrypoint": entrypoint}

        # 2. Global
        table = _component_type_to_table(component_type)
        row = self.db.fetch_one(
            f"SELECT * FROM {table} WHERE name = ? AND is_obsolete = 0",
            (name,),
        )
        if row:
            type_col = _component_type_column(component_type)
            return {
                "source": row[type_col],
                "name": name,
                "entrypoint": None,
                "record": row,
            }

        # 3. Not found
        raise ComponentNotFoundError(
            f"{component_type.rstrip('s').title()} '{name}' not found",
            context={
                "searched": ["project.yml", f"{table} table"],
                "component_type": component_type,
            },
            hint=f"Define '{name}' in project.yml or add it globally with 'uesf {component_type.rstrip('s')} add'.",
        )


def _component_type_to_table(component_type: str) -> str:
    """Map component type to database table name."""
    mapping = {
        "models": "models",
        "trainers": "trainers",
        "metrics": "metrics",
    }
    return mapping[component_type]


def _component_type_column(component_type: str) -> str:
    """Map component type to its type column name."""
    mapping = {
        "models": "model_type",
        "trainers": "trainer_type",
        "metrics": "metric_type",
    }
    return mapping[component_type]
