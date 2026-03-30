"""UESF global configuration manager.

Implements a two-layer config mechanism:
1. Database `configs` table stores read-only defaults
2. ~/.uesf/config.yml file overrides defaults (higher priority)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from uesf.core.database import DatabaseManager
from uesf.core.exceptions import ConfigError
from uesf.core.logging import get_logger

logger = get_logger("manager.config")

VALID_KEYS = {"data_dir", "default_device", "num_workers", "log_level"}


class ConfigManager:
    """Manages UESF global configuration."""

    def __init__(self, db: DatabaseManager, uesf_home: Path | None = None) -> None:
        self.db = db
        self.uesf_home = uesf_home or Path(os.environ.get("UESF_HOME", Path.home() / ".uesf"))
        self._config_file = self.uesf_home / "config.yml"

    def _load_db_defaults(self) -> dict[str, str]:
        """Load default config values from database."""
        rows = self.db.fetch_all("SELECT key, value FROM configs")
        return {row["key"]: row["value"] for row in rows}

    def _load_file_overrides(self) -> dict[str, Any]:
        """Load config overrides from config.yml."""
        if not self._config_file.exists():
            return {}

        with open(self._config_file, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            return {}

        if not isinstance(data, dict):
            logger.warning("config.yml is not a valid YAML mapping, ignoring")
            return {}

        # Warn about unknown keys
        for key in data:
            if key not in VALID_KEYS:
                logger.warning("Unknown config key '%s' in config.yml (ignored)", key)

        return {k: v for k, v in data.items() if k in VALID_KEYS}

    def get_all(self) -> dict[str, Any]:
        """Get merged configuration (DB defaults + file overrides).

        Returns:
            Dict with all config key-value pairs.
        """
        config = self._load_db_defaults()
        overrides = self._load_file_overrides()
        config.update({k: str(v) for k, v in overrides.items()})
        return config

    def get(self, key: str) -> str:
        """Get a single config value.

        Args:
            key: Config key name.

        Returns:
            The effective config value as string.

        Raises:
            ConfigError: If key is not a valid config key.
        """
        if key not in VALID_KEYS:
            raise ConfigError(
                f"Unknown config key: '{key}'",
                context={"valid_keys": sorted(VALID_KEYS)},
                hint=f"Valid config keys are: {', '.join(sorted(VALID_KEYS))}",
            )
        return self.get_all()[key]

    def set(self, key: str, value: str) -> None:
        """Set a config value by writing to config.yml.

        Args:
            key: Config key name.
            value: Config value to set.

        Raises:
            ConfigError: If key is not a valid config key.
        """
        if key not in VALID_KEYS:
            raise ConfigError(
                f"Unknown config key: '{key}'",
                context={"valid_keys": sorted(VALID_KEYS)},
                hint=f"Valid config keys are: {', '.join(sorted(VALID_KEYS))}",
            )

        # Load existing file config or create new
        existing = {}
        if self._config_file.exists():
            with open(self._config_file, encoding="utf-8") as f:
                existing = yaml.safe_load(f) or {}

        existing[key] = value

        self._config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._config_file, "w", encoding="utf-8") as f:
            yaml.dump(existing, f, default_flow_style=False, allow_unicode=True)

        logger.info("Config '%s' set to '%s'", key, value)

    def get_data_dir(self) -> Path:
        """Get the resolved data directory path."""
        raw = self.get("data_dir")
        return Path(os.path.expanduser(raw))
