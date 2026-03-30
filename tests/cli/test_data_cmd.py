"""E2E tests for uesf data CLI commands."""

from __future__ import annotations

from typer.testing import CliRunner

from tests.fixtures.fake_data import create_fake_raw_dataset
from uesf.cli.app import app

runner = CliRunner()


class TestRawRegister:
    def test_register_success(self, uesf_home, tmp_path):
        ds_dir = create_fake_raw_dataset(tmp_path, name="cli_test_ds")
        result = runner.invoke(app, ["data", "raw", "register", str(ds_dir)])
        assert result.exit_code == 0
        assert "cli_test_ds" in result.output

    def test_register_invalid_path(self, uesf_home, tmp_path):
        result = runner.invoke(app, ["data", "raw", "register", str(tmp_path / "nonexistent")])
        assert result.exit_code == 1


class TestRawList:
    def test_list_empty(self, uesf_home):
        result = runner.invoke(app, ["data", "raw", "list"])
        assert result.exit_code == 0
        assert "No raw datasets" in result.output

    def test_list_after_register(self, uesf_home, tmp_path):
        ds_dir = create_fake_raw_dataset(tmp_path, name="listed_ds")
        runner.invoke(app, ["data", "raw", "register", str(ds_dir)])

        result = runner.invoke(app, ["data", "raw", "list"])
        assert result.exit_code == 0
        assert "listed_ds" in result.output


class TestRawRemove:
    def test_remove_with_confirm(self, uesf_home, tmp_path):
        ds_dir = create_fake_raw_dataset(tmp_path, name="to_remove_cli")
        runner.invoke(app, ["data", "raw", "register", str(ds_dir)])

        result = runner.invoke(app, ["data", "raw", "remove", "to_remove_cli", "--yes"])
        assert result.exit_code == 0
        assert "Removed" in result.output

    def test_remove_nonexistent(self, uesf_home):
        result = runner.invoke(app, ["data", "raw", "remove", "nonexistent", "--yes"])
        assert result.exit_code == 1


class TestRawEdit:
    def test_edit_description(self, uesf_home, tmp_path):
        ds_dir = create_fake_raw_dataset(tmp_path, name="edit_cli")
        runner.invoke(app, ["data", "raw", "register", str(ds_dir)])

        result = runner.invoke(app, ["data", "raw", "edit", "edit_cli", "--description", "New desc"])
        assert result.exit_code == 0
        assert "Updated" in result.output


class TestRawInfo:
    def test_info_shows_details(self, uesf_home, tmp_path):
        ds_dir = create_fake_raw_dataset(tmp_path, name="info_ds")
        runner.invoke(app, ["data", "raw", "register", str(ds_dir)])

        result = runner.invoke(app, ["data", "raw", "info", "info_ds"])
        assert result.exit_code == 0
        assert "info_ds" in result.output
        assert "data" in result.output  # eeg_data_key
