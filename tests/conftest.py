"""Shared test fixtures for UESF test suite."""

from __future__ import annotations

import random

import numpy as np
import pytest

from uesf.core.database import DatabaseManager
from uesf.core.logging import reset_logging


@pytest.fixture(autouse=True)
def _reset_logging():
    """Reset logging state between tests."""
    reset_logging()
    yield
    reset_logging()


@pytest.fixture
def uesf_home(tmp_path, monkeypatch):
    """Provide an isolated UESF home directory.

    Sets VIRTUAL_ENV to a temp directory so get_uesf_home() returns
    tmp_path/.uesf, ensuring tests never touch the real ~/.uesf.
    """
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path))
    home = tmp_path / ".uesf"
    home.mkdir()
    return home


@pytest.fixture
def db():
    """Provide an in-memory database, initialized with schema."""
    db = DatabaseManager(":memory:")
    db.initialize()
    yield db
    db.close()


@pytest.fixture
def db_on_disk(uesf_home):
    """Provide a disk-based database in the test UESF home."""
    db = DatabaseManager(uesf_home / "uesf.db")
    db.initialize()
    yield db
    db.close()


@pytest.fixture(autouse=True)
def fixed_seed():
    """Fix random seeds for deterministic tests."""
    random.seed(42)
    np.random.seed(42)
