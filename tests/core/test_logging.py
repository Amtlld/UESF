"""Tests for UESF logging system."""

import logging

from uesf.core.logging import get_logger, reset_logging, setup_logging


class TestGetLogger:
    def test_returns_namespaced_logger(self):
        logger = get_logger("cli")
        assert logger.name == "uesf.cli"

    def test_nested_namespace(self):
        logger = get_logger("manager.data")
        assert logger.name == "uesf.manager.data"

    def test_root_logger_is_parent(self):
        logger = get_logger("pipeline")
        assert logger.parent.name == "uesf"


class TestSetupLogging:
    def test_setup_adds_handlers(self, uesf_home):
        setup_logging()
        root = logging.getLogger("uesf")
        assert len(root.handlers) >= 2  # Console + File

    def test_setup_is_idempotent(self, uesf_home):
        setup_logging()
        count1 = len(logging.getLogger("uesf").handlers)
        setup_logging()
        count2 = len(logging.getLogger("uesf").handlers)
        assert count1 == count2

    def test_creates_log_directory(self, uesf_home):
        setup_logging()
        assert (uesf_home / "logs").exists()

    def test_log_level_respected(self, uesf_home):
        setup_logging(log_level="WARNING")
        root = logging.getLogger("uesf")
        # Root logger should be DEBUG (catches everything)
        assert root.level == logging.DEBUG
        # Console handler should respect the specified level
        console_handlers = [h for h in root.handlers if not hasattr(h, "baseFilename")]
        assert len(console_handlers) >= 1
        assert console_handlers[0].level == logging.WARNING


class TestResetLogging:
    def test_reset_clears_handlers(self, uesf_home):
        setup_logging()
        root = logging.getLogger("uesf")
        assert len(root.handlers) > 0
        reset_logging()
        assert len(root.handlers) == 0

    def test_reset_allows_reinitialize(self, uesf_home):
        setup_logging()
        reset_logging()
        setup_logging()
        root = logging.getLogger("uesf")
        assert len(root.handlers) >= 2
