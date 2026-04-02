"""UESF core utilities."""

from __future__ import annotations

import os
from pathlib import Path


def get_uesf_home() -> Path:
    """Resolve the UESF home directory.

    Resolution order:
    1. ``$VIRTUAL_ENV/.uesf`` (auto-detected when running inside a venv)
    2. ``~/.uesf`` (fallback default)
    """
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        return Path(venv) / ".uesf"

    return Path.home() / ".uesf"
