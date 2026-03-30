"""CLI E2E tests for uesf experiment commands."""

from __future__ import annotations

from typer.testing import CliRunner

from uesf.cli.app import app

runner = CliRunner()


class TestExperimentAdd:
    def test_add_experiment(self, uesf_home, tmp_path):
        # Init project first
        project_dir = tmp_path / "proj"
        runner.invoke(app, ["project", "init", str(project_dir)])

        result = runner.invoke(
            app,
            ["experiment", "add", "-p", str(project_dir), "--name", "test_exp"],
        )
        assert result.exit_code == 0
        assert "Created experiment" in result.output
        assert (project_dir / "experiments" / "test_exp.yml").exists()

    def test_add_without_project(self, uesf_home, tmp_path):
        result = runner.invoke(
            app,
            ["experiment", "add", "-p", str(tmp_path), "--name", "bad_exp"],
        )
        assert result.exit_code == 1


class TestExperimentList:
    def test_list_empty(self, uesf_home, tmp_path):
        project_dir = tmp_path / "proj"
        runner.invoke(app, ["project", "init", str(project_dir)])
        result = runner.invoke(app, ["experiment", "list", "-p", str(project_dir)])
        assert result.exit_code == 0
        assert "No experiments" in result.output


class TestExperimentRemove:
    def test_remove_experiment(self, uesf_home, tmp_path):
        project_dir = tmp_path / "proj"
        runner.invoke(app, ["project", "init", str(project_dir)])
        runner.invoke(app, ["experiment", "add", "-p", str(project_dir), "--name", "rm_exp"])

        result = runner.invoke(
            app,
            ["experiment", "remove", "rm_exp", "-p", str(project_dir), "--yes"],
        )
        assert result.exit_code == 0
        assert "Removed" in result.output
