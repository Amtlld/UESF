"""CLI E2E tests for uesf model commands."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from uesf.cli.app import app

runner = CliRunner()


@pytest.fixture
def model_file(tmp_path):
    """Create a dummy model file."""
    src = tmp_path / "cli_model.py"
    src.write_text(
        'import torch.nn as nn\n'
        'class CliModel(nn.Module):\n'
        '    def __init__(self):\n'
        '        super().__init__()\n'
        '        self.fc = nn.Linear(10, 2)\n'
        '    def forward(self, x):\n'
        '        return self.fc(x)\n',
        encoding="utf-8",
    )
    return src


class TestModelAdd:
    def test_add_model(self, uesf_home, model_file):
        result = runner.invoke(app, ["model", "add", str(model_file), "--name", "test_model"])
        assert result.exit_code == 0
        assert "Added global model" in result.output

    def test_add_model_with_description(self, uesf_home, model_file):
        result = runner.invoke(
            app,
            ["model", "add", str(model_file), "--name", "desc_model", "-d", "A model with desc"],
        )
        assert result.exit_code == 0
        assert "Added global model" in result.output

    def test_add_nonexistent_file(self, uesf_home):
        result = runner.invoke(app, ["model", "add", "/nonexistent.py", "--name", "bad"])
        assert result.exit_code == 1


class TestModelList:
    def test_list_empty(self, uesf_home):
        result = runner.invoke(app, ["model", "list"])
        assert result.exit_code == 0
        assert "No models" in result.output

    def test_list_with_models(self, uesf_home, model_file):
        runner.invoke(app, ["model", "add", str(model_file), "--name", "m1"])
        result = runner.invoke(app, ["model", "list"])
        assert result.exit_code == 0
        assert "m1" in result.output


class TestModelRemove:
    def test_remove_model(self, uesf_home, model_file):
        runner.invoke(app, ["model", "add", str(model_file), "--name", "rm_model"])
        result = runner.invoke(app, ["model", "remove", "rm_model", "--yes"])
        assert result.exit_code == 0
        assert "Removed model" in result.output

    def test_remove_nonexistent(self, uesf_home):
        result = runner.invoke(app, ["model", "remove", "nonexistent", "--yes"])
        assert result.exit_code == 1


class TestModelEdit:
    def test_edit_description(self, uesf_home, model_file):
        runner.invoke(app, ["model", "add", str(model_file), "--name", "edit_model"])
        result = runner.invoke(app, ["model", "edit", "edit_model", "-d", "Updated desc"])
        assert result.exit_code == 0
        assert "Updated model" in result.output

    def test_edit_no_fields(self, uesf_home, model_file):
        runner.invoke(app, ["model", "add", str(model_file), "--name", "noop_model"])
        result = runner.invoke(app, ["model", "edit", "noop_model"])
        assert result.exit_code == 0
        assert "No fields" in result.output
