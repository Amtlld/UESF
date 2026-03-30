"""CLI E2E tests for uesf metric commands."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from uesf.cli.app import app

runner = CliRunner()


@pytest.fixture
def metric_file(tmp_path):
    """Create a dummy metric file."""
    src = tmp_path / "cli_metric.py"
    src.write_text(
        'def cli_metric(preds, targets, **kwargs):\n'
        '    return 0.95\n',
        encoding="utf-8",
    )
    return src


class TestMetricAdd:
    def test_add_metric(self, uesf_home, metric_file):
        result = runner.invoke(app, ["metric", "add", str(metric_file), "--name", "test_metric"])
        assert result.exit_code == 0
        assert "Added global metric" in result.output

    def test_add_nonexistent_file(self, uesf_home):
        result = runner.invoke(app, ["metric", "add", "/nonexistent.py", "--name", "bad"])
        assert result.exit_code == 1


class TestMetricList:
    def test_list_empty(self, uesf_home):
        result = runner.invoke(app, ["metric", "list"])
        assert result.exit_code == 0
        assert "No metrics" in result.output

    def test_list_with_metrics(self, uesf_home, metric_file):
        runner.invoke(app, ["metric", "add", str(metric_file), "--name", "m1"])
        result = runner.invoke(app, ["metric", "list"])
        assert result.exit_code == 0
        assert "m1" in result.output


class TestMetricRemove:
    def test_remove_metric(self, uesf_home, metric_file):
        runner.invoke(app, ["metric", "add", str(metric_file), "--name", "rm_metric"])
        result = runner.invoke(app, ["metric", "remove", "rm_metric", "--yes"])
        assert result.exit_code == 0
        assert "Removed metric" in result.output

    def test_remove_nonexistent(self, uesf_home):
        result = runner.invoke(app, ["metric", "remove", "nonexistent", "--yes"])
        assert result.exit_code == 1


class TestMetricEdit:
    def test_edit_description(self, uesf_home, metric_file):
        runner.invoke(app, ["metric", "add", str(metric_file), "--name", "edit_metric"])
        result = runner.invoke(app, ["metric", "edit", "edit_metric", "-d", "Updated"])
        assert result.exit_code == 0
        assert "Updated metric" in result.output

    def test_edit_no_fields(self, uesf_home, metric_file):
        runner.invoke(app, ["metric", "add", str(metric_file), "--name", "noop_metric"])
        result = runner.invoke(app, ["metric", "edit", "noop_metric"])
        assert result.exit_code == 0
        assert "No fields" in result.output
