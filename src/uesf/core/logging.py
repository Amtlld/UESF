"""UESF logging system.

Provides a dual-channel logging architecture:
- ConsoleHandler: Rich-formatted terminal output, respects user log_level
- GlobalFileHandler: Rotating file at ~/.uesf/logs/uesf.log, captures DEBUG+
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler

_initialized = False


def _get_uesf_home() -> Path:
    from uesf.core import get_uesf_home

    return get_uesf_home()


def setup_logging(log_level: str = "INFO") -> None:
    """Initialize the UESF logging system.

    Should be called once at application startup.
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    root_logger = logging.getLogger("uesf")
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()

    # Console handler (Rich)
    console_handler = RichHandler(
        level=getattr(logging, log_level.upper(), logging.INFO),
        show_path=False,
        show_time=True,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
    )
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(console_handler)

    # Global file handler (rotating)
    uesf_home = _get_uesf_home()
    log_dir = uesf_home / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / "uesf.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root_logger.addHandler(file_handler)


def reset_logging() -> None:
    """Reset logging state. Used by tests."""
    global _initialized
    _initialized = False
    root_logger = logging.getLogger("uesf")
    root_logger.handlers.clear()


def get_logger(name: str) -> logging.Logger:
    """Get a logger within the uesf namespace.

    Args:
        name: Logger name suffix (e.g., "cli", "db", "manager.data").
              The returned logger will be named "uesf.<name>".
    """
    return logging.getLogger(f"uesf.{name}")
