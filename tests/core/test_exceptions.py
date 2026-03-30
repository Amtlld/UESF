"""Tests for UESF exception hierarchy."""

import pytest

from uesf.core.exceptions import (
    ComponentError,
    ComponentNotFoundError,
    ConfigError,
    DatabaseLockedError,
    DataError,
    DatasetNotFoundError,
    ExperimentError,
    InterfaceViolationError,
    InvalidExperimentStateError,
    MemoryOutOfBoundsError,
    MissingRequiredKeyError,
    ShapeMismatchError,
    SnapshotCreationError,
    StorageError,
    TrainingDivergenceError,
    TypeMismatchError,
    UESFException,
    YAMLParseError,
)


class TestUESFExceptionBase:
    def test_basic_instantiation(self):
        exc = UESFException("test error")
        assert exc.message == "test error"
        assert exc.context == {}
        assert exc.hint is None
        assert str(exc) == "test error"

    def test_full_instantiation(self):
        exc = UESFException(
            "something failed",
            context={"module": "DataManager", "file": "raw.yml"},
            hint="Check your YAML syntax.",
        )
        assert exc.message == "something failed"
        assert exc.context == {"module": "DataManager", "file": "raw.yml"}
        assert exc.hint == "Check your YAML syntax."

    def test_is_exception(self):
        exc = UESFException("err")
        assert isinstance(exc, Exception)


class TestInheritanceChain:
    """Verify the exception class hierarchy matches the design spec."""

    @pytest.mark.parametrize(
        "exc_cls,parent_cls",
        [
            (ConfigError, UESFException),
            (YAMLParseError, ConfigError),
            (MissingRequiredKeyError, ConfigError),
            (TypeMismatchError, ConfigError),
            (ComponentError, UESFException),
            (ComponentNotFoundError, ComponentError),
            (InterfaceViolationError, ComponentError),
            (DataError, UESFException),
            (DatasetNotFoundError, DataError),
            (ShapeMismatchError, DataError),
            (MemoryOutOfBoundsError, DataError),
            (ExperimentError, UESFException),
            (InvalidExperimentStateError, ExperimentError),
            (TrainingDivergenceError, ExperimentError),
            (StorageError, UESFException),
            (DatabaseLockedError, StorageError),
            (SnapshotCreationError, StorageError),
        ],
    )
    def test_inheritance(self, exc_cls, parent_cls):
        assert issubclass(exc_cls, parent_cls)
        # All ultimately inherit from UESFException
        assert issubclass(exc_cls, UESFException)

    @pytest.mark.parametrize(
        "exc_cls",
        [
            YAMLParseError,
            MissingRequiredKeyError,
            TypeMismatchError,
            ComponentNotFoundError,
            InterfaceViolationError,
            DatasetNotFoundError,
            ShapeMismatchError,
            MemoryOutOfBoundsError,
            InvalidExperimentStateError,
            TrainingDivergenceError,
            DatabaseLockedError,
            SnapshotCreationError,
        ],
    )
    def test_leaf_exceptions_carry_all_fields(self, exc_cls):
        exc = exc_cls(
            "msg",
            context={"key": "value"},
            hint="try this",
        )
        assert exc.message == "msg"
        assert exc.context == {"key": "value"}
        assert exc.hint == "try this"


class TestExceptionCatching:
    def test_catch_by_base(self):
        with pytest.raises(UESFException):
            raise YAMLParseError("bad yaml")

    def test_catch_by_category(self):
        with pytest.raises(ConfigError):
            raise MissingRequiredKeyError("missing key")

    def test_catch_by_exact_type(self):
        with pytest.raises(DatasetNotFoundError):
            raise DatasetNotFoundError("no such dataset")
