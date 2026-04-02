"""Tests for UESF config manager."""

import pytest
import yaml

from uesf.core.config import ConfigManager
from uesf.core.exceptions import ConfigError


class TestConfigDefaults:
    def test_get_all_returns_defaults(self, db, uesf_home):
        mgr = ConfigManager(db, uesf_home)
        config = mgr.get_all()
        assert config["data_dir"] == str(uesf_home / "data")
        assert config["default_device"] == "cpu"
        assert config["num_workers"] == "4"
        assert config["log_level"] == "INFO"

    def test_get_single_key(self, db, uesf_home):
        mgr = ConfigManager(db, uesf_home)
        assert mgr.get("default_device") == "cpu"

    def test_get_invalid_key_raises(self, db, uesf_home):
        mgr = ConfigManager(db, uesf_home)
        with pytest.raises(ConfigError, match="Unknown config key"):
            mgr.get("invalid_key")


class TestConfigFileOverride:
    def test_file_overrides_defaults(self, db, uesf_home):
        config_file = uesf_home / "config.yml"
        config_file.write_text(yaml.dump({"default_device": "cuda:0", "num_workers": 8}))

        mgr = ConfigManager(db, uesf_home)
        config = mgr.get_all()
        assert config["default_device"] == "cuda:0"
        assert config["num_workers"] == "8"
        # Unset keys retain defaults
        assert config["data_dir"] == str(uesf_home / "data")
        assert config["log_level"] == "INFO"

    def test_unknown_key_in_file_ignored(self, db, uesf_home):
        config_file = uesf_home / "config.yml"
        config_file.write_text(yaml.dump({"unknown_key": "value", "log_level": "DEBUG"}))

        mgr = ConfigManager(db, uesf_home)
        config = mgr.get_all()
        assert "unknown_key" not in config
        assert config["log_level"] == "DEBUG"

    def test_empty_config_file(self, db, uesf_home):
        config_file = uesf_home / "config.yml"
        config_file.write_text("")

        mgr = ConfigManager(db, uesf_home)
        config = mgr.get_all()
        assert len(config) == 4

    def test_no_config_file(self, db, uesf_home):
        mgr = ConfigManager(db, uesf_home)
        config = mgr.get_all()
        assert len(config) == 4


class TestConfigSet:
    def test_set_creates_config_file(self, db, uesf_home):
        mgr = ConfigManager(db, uesf_home)
        mgr.set("default_device", "cuda:1")

        config_file = uesf_home / "config.yml"
        assert config_file.exists()
        with open(config_file) as f:
            data = yaml.safe_load(f)
        assert data["default_device"] == "cuda:1"

    def test_set_preserves_existing_keys(self, db, uesf_home):
        mgr = ConfigManager(db, uesf_home)
        mgr.set("default_device", "cuda:0")
        mgr.set("log_level", "DEBUG")

        config_file = uesf_home / "config.yml"
        with open(config_file) as f:
            data = yaml.safe_load(f)
        assert data["default_device"] == "cuda:0"
        assert data["log_level"] == "DEBUG"

    def test_set_roundtrip(self, db, uesf_home):
        mgr = ConfigManager(db, uesf_home)
        mgr.set("num_workers", "16")
        assert mgr.get("num_workers") == "16"

    def test_set_invalid_key_raises(self, db, uesf_home):
        mgr = ConfigManager(db, uesf_home)
        with pytest.raises(ConfigError, match="Unknown config key"):
            mgr.set("bad_key", "value")


class TestGetDataDir:
    def test_default_data_dir(self, db, uesf_home):
        mgr = ConfigManager(db, uesf_home)
        assert mgr.get_data_dir() == uesf_home / "data"

    def test_custom_data_dir(self, db, uesf_home):
        mgr = ConfigManager(db, uesf_home)
        mgr.set("data_dir", "/tmp/my_eeg_data")
        assert mgr.get_data_dir().as_posix() == "/tmp/my_eeg_data"
