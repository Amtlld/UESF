"""UESF unified exception hierarchy.

All framework exceptions inherit from UESFException and carry three dimensions:
- message: phenomenon description
- context: environment metadata dict (for file logging)
- hint: user-facing remediation advice (for CLI display)
"""

from __future__ import annotations


class UESFException(Exception):
    """Base exception for all UESF framework errors."""

    def __init__(
        self,
        message: str,
        context: dict | None = None,
        hint: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.hint = hint


# --- Config errors ---


class ConfigError(UESFException):
    """Base class for configuration-related errors."""


class YAMLParseError(ConfigError):
    """YAML file has invalid structure or syntax."""


class MissingRequiredKeyError(ConfigError):
    """A required configuration key is missing."""


class TypeMismatchError(ConfigError):
    """Configuration value type or hyperparameter value is invalid."""


# --- Component errors ---


class ComponentError(UESFException):
    """Base class for component registration and integration errors."""


class ComponentNotFoundError(ComponentError):
    """No component found with the given name."""


class InterfaceViolationError(ComponentError):
    """User-defined code does not implement required BaseModel/BaseTrainer interface."""


# --- Data errors ---


class DataError(UESFException):
    """Base class for data processing and I/O errors."""


class DatasetNotFoundError(DataError):
    """Referenced raw/preprocessed dataset is not registered."""


class ShapeMismatchError(DataError):
    """Data dimensions do not match model input or label shape expectations."""


class MemoryOutOfBoundsError(DataError):
    """Lazy loading or data processing exceeded safe memory threshold."""


# --- Experiment errors ---


class ExperimentError(UESFException):
    """Base class for experiment lifecycle errors."""


class InvalidExperimentStateError(ExperimentError):
    """Illegal experiment state transition (e.g., modifying a running experiment)."""


class TrainingDivergenceError(ExperimentError):
    """Training divergence detected (e.g., NaN loss)."""


# --- Storage errors ---


class StorageError(UESFException):
    """Base class for environment and persistence errors."""


class DatabaseLockedError(StorageError):
    """SQLite concurrent operation caused a lock conflict."""


class SnapshotCreationError(StorageError):
    """Failed to create source code snapshot."""
