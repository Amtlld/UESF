"""Tests for get_uesf_home() venv-aware resolution."""

from __future__ import annotations

from pathlib import Path

from uesf.core import get_uesf_home


class TestGetUesfHome:
    """Test UESF home directory resolution priority."""

    def test_virtual_env_used(self, tmp_path, monkeypatch):
        venv_dir = tmp_path / "my_venv"
        monkeypatch.setenv("VIRTUAL_ENV", str(venv_dir))
        assert get_uesf_home() == venv_dir / ".uesf"

    def test_fallback_to_home_dir(self, monkeypatch):
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        assert get_uesf_home() == Path.home() / ".uesf"
