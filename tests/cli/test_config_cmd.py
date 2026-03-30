"""E2E tests for uesf config CLI commands."""

import yaml
from typer.testing import CliRunner

from uesf.cli.app import app

runner = CliRunner()


class TestConfigShow:
    def test_show_displays_defaults(self, uesf_home):
        result = runner.invoke(app, ["config", "show"])
        assert result.exit_code == 0
        assert "data_dir" in result.output
        assert "default_device" in result.output
        assert "num_workers" in result.output
        assert "log_level" in result.output

    def test_show_reflects_file_override(self, uesf_home):
        config_file = uesf_home / "config.yml"
        config_file.write_text(yaml.dump({"default_device": "cuda:0"}))

        result = runner.invoke(app, ["config", "show"])
        assert result.exit_code == 0
        assert "cuda:0" in result.output


class TestConfigSet:
    def test_set_valid_key(self, uesf_home):
        result = runner.invoke(app, ["config", "set", "default_device", "cuda:1"])
        assert result.exit_code == 0
        assert "cuda:1" in result.output

        # Verify persisted
        config_file = uesf_home / "config.yml"
        assert config_file.exists()
        with open(config_file) as f:
            data = yaml.safe_load(f)
        assert data["default_device"] == "cuda:1"

    def test_set_invalid_key(self, uesf_home):
        result = runner.invoke(app, ["config", "set", "invalid_key", "value"])
        assert result.exit_code == 1

    def test_set_then_show(self, uesf_home):
        runner.invoke(app, ["config", "set", "log_level", "DEBUG"])
        result = runner.invoke(app, ["config", "show"])
        assert result.exit_code == 0
        assert "DEBUG" in result.output


class TestVersion:
    def test_version_flag(self, uesf_home):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "uesf" in result.output
