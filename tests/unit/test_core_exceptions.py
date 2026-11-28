"""Tests for FAMEX core exceptions."""

from __future__ import annotations

import pytest

from famex.core.exceptions import (
    BackendError,
    DependencyError,
    FAMEXError,
    StrategyError,
    StrategyNotFoundError,
)
from famex.utils.validation import (
    BackendError as ReexportedBackendError,
)
from famex.utils.validation import (
    DependencyError as ReexportedDependencyError,
)
from famex.utils.validation import (
    FAMEXError as ReexportedFAMEXError,
)


class TestFAMEXErrorFamily:
    def test_famex_error_suggestion(self):
        error = FAMEXError("boom", "fix it")
        assert "boom" in str(error)
        assert "fix it" in str(error)

    def test_backend_error_is_import_error(self):
        error = BackendError("mock", ["aimnet2"], "test")
        assert isinstance(error, FAMEXError)
        assert isinstance(error, ImportError)
        with pytest.raises(ImportError):
            raise error

    def test_dependency_error_is_import_error(self):
        error = DependencyError("torch", "calc", "pip install torch")
        assert isinstance(error, FAMEXError)
        assert isinstance(error, ImportError)

    def test_reexported_from_validation(self):
        assert ReexportedFAMEXError is FAMEXError
        assert ReexportedBackendError is BackendError
        assert ReexportedDependencyError is DependencyError


class TestStrategyError:
    """Tests for StrategyError base class."""

    def test_strategy_error_is_exception(self):
        """Test that StrategyError is an Exception."""
        error = StrategyError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"


class TestStrategyNotFoundError:
    """Tests for StrategyNotFoundError."""

    def test_strategy_not_found_error_basic(self):
        """Test StrategyNotFoundError with just strategy name."""
        error = StrategyNotFoundError("nonexistent:strategy")
        assert isinstance(error, StrategyError)
        assert error.strategy_name == "nonexistent:strategy"
        assert error.available_strategies == []
        assert "nonexistent:strategy" in str(error)

    def test_strategy_not_found_error_with_available(self):
        """Test StrategyNotFoundError with available strategies."""
        available = ["minima:local", "ts:local", "path:neb"]
        error = StrategyNotFoundError("nonexistent:strategy", available_strategies=available)
        assert error.strategy_name == "nonexistent:strategy"
        assert error.available_strategies == available
        assert "nonexistent:strategy" in str(error)
        assert "minima:local" in str(error)
        assert "ts:local" in str(error)
        assert "path:neb" in str(error)

    def test_strategy_not_found_error_with_none_available(self):
        """Test StrategyNotFoundError with None available strategies."""
        error = StrategyNotFoundError("nonexistent:strategy", available_strategies=None)
        assert error.strategy_name == "nonexistent:strategy"
        assert error.available_strategies == []
        assert "nonexistent:strategy" in str(error)
