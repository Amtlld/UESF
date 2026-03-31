"""UESF core utilities."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def get_uesf_home() -> Path:
    """Resolve the UESF home directory.

    Resolution order:
    1. ``UESF_HOME`` environment variable (explicit override, highest priority)
    2. ``$VIRTUAL_ENV/.uesf`` (auto-detected when running inside a venv)
    3. ``~/.uesf`` (fallback default)
    """
    env_home = os.environ.get("UESF_HOME")
    if env_home:
        return Path(env_home)

    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        return Path(venv) / ".uesf"

    return Path.home() / ".uesf"
