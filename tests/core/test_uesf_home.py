"""Tests for get_uesf_home() venv-aware resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from uesf.core import get_uesf_home


class TestGetUesfHome:
    """Test UESF home directory resolution priority."""

    def test_env_var_takes_highest_priority(self, tmp_path, monkeypatch):
        explicit = tmp_path / "explicit"
        monkeypatch.setenv("UESF_HOME", str(explicit))
        monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "venv"))
        assert get_uesf_home() == explicit

    def test_virtual_env_used_when_no_env_var(self, tmp_path, monkeypatch):
        monkeypatch.delenv("UESF_HOME", raising=False)
        venv_dir = tmp_path / "my_venv"
        monkeypatch.setenv("VIRTUAL_ENV", str(venv_dir))
        assert get_uesf_home() == venv_dir / ".uesf"

    def test_fallback_to_home_dir(self, monkeypatch):
        monkeypatch.delenv("UESF_HOME", raising=False)
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        assert get_uesf_home() == Path.home() / ".uesf"

    def test_env_var_empty_string_is_falsy(self, monkeypatch):
        """Empty UESF_HOME should fall through to VIRTUAL_ENV."""
        monkeypatch.setenv("UESF_HOME", "")
        venv_dir = "/tmp/test_venv"
        monkeypatch.setenv("VIRTUAL_ENV", venv_dir)
        assert get_uesf_home() == Path(venv_dir) / ".uesf"
