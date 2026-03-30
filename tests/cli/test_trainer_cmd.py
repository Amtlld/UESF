"""CLI E2E tests for uesf trainer commands."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from uesf.cli.app import app

runner = CliRunner()


@pytest.fixture
def trainer_file(tmp_path):
    """Create a dummy trainer file."""
    src = tmp_path / "cli_trainer.py"
    src.write_text(
        'class CliTrainer:\n'
        '    def __init__(self, model, device, **kwargs):\n'
        '        self.model = model\n'
        '    def training_step(self, batch, batch_idx, optimizer):\n'
        '        return {"loss": 0.0}\n'
        '    def validation_step(self, batch, batch_idx):\n'
        '        return {"preds": None, "targets": None}\n',
        encoding="utf-8",
    )
    return src


class TestTrainerAdd:
    def test_add_trainer(self, uesf_home, trainer_file):
        result = runner.invoke(app, ["trainer", "add", str(trainer_file), "--name", "test_trainer"])
        assert result.exit_code == 0
        assert "Added global trainer" in result.output

    def test_add_trainer_with_description(self, uesf_home, trainer_file):
        result = runner.invoke(
            app,
            ["trainer", "add", str(trainer_file), "--name", "desc_trainer", "-d", "A trainer"],
        )
        assert result.exit_code == 0
        assert "Added global trainer" in result.output

    def test_add_nonexistent_file(self, uesf_home):
        result = runner.invoke(app, ["trainer", "add", "/nonexistent.py", "--name", "bad"])
        assert result.exit_code == 1


class TestTrainerList:
    def test_list_empty(self, uesf_home):
        result = runner.invoke(app, ["trainer", "list"])
        assert result.exit_code == 0
        assert "No trainers" in result.output

    def test_list_with_trainers(self, uesf_home, trainer_file):
        runner.invoke(app, ["trainer", "add", str(trainer_file), "--name", "t1"])
        result = runner.invoke(app, ["trainer", "list"])
        assert result.exit_code == 0
        assert "t1" in result.output


class TestTrainerRemove:
    def test_remove_trainer(self, uesf_home, trainer_file):
        runner.invoke(app, ["trainer", "add", str(trainer_file), "--name", "rm_trainer"])
        result = runner.invoke(app, ["trainer", "remove", "rm_trainer", "--yes"])
        assert result.exit_code == 0
        assert "Removed trainer" in result.output

    def test_remove_nonexistent(self, uesf_home):
        result = runner.invoke(app, ["trainer", "remove", "nonexistent", "--yes"])
        assert result.exit_code == 1


class TestTrainerEdit:
    def test_edit_description(self, uesf_home, trainer_file):
        runner.invoke(app, ["trainer", "add", str(trainer_file), "--name", "edit_trainer"])
        result = runner.invoke(app, ["trainer", "edit", "edit_trainer", "-d", "Updated"])
        assert result.exit_code == 0
        assert "Updated trainer" in result.output

    def test_edit_no_fields(self, uesf_home, trainer_file):
        runner.invoke(app, ["trainer", "add", str(trainer_file), "--name", "noop_trainer"])
        result = runner.invoke(app, ["trainer", "edit", "noop_trainer"])
        assert result.exit_code == 0
        assert "No fields" in result.output
