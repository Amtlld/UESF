"""CLI E2E tests for uesf project commands."""

from __future__ import annotations

from typer.testing import CliRunner

from uesf.cli.app import app

runner = CliRunner()


class TestProjectInit:
    def test_init_creates_project(self, uesf_home, tmp_path):
        project_dir = tmp_path / "new_project"
        result = runner.invoke(app, ["project", "init", str(project_dir)])
        assert result.exit_code == 0
        assert "initialized" in result.output.lower()
        assert (project_dir / "project.yml").exists()
        assert (project_dir / "experiments").is_dir()

    def test_init_current_dir(self, uesf_home, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["project", "init"])
        assert result.exit_code == 0


class TestProjectInfo:
    def test_info_shows_details(self, uesf_home, tmp_path):
        # Create a project first
        yml = tmp_path / "project.yml"
        yml.write_text(
            "project-name: info_test\ndescription: A test project\n",
            encoding="utf-8",
        )
        result = runner.invoke(app, ["project", "info", str(tmp_path)])
        assert result.exit_code == 0
        assert "info_test" in result.output

    def test_info_missing_project(self, uesf_home, tmp_path):
        result = runner.invoke(app, ["project", "info", str(tmp_path / "nonexistent")])
        assert result.exit_code == 1
