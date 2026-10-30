"""Tests for FAMEX core exceptions."""

from __future__ import annotations

from famex.core.exceptions import (
    StrategyError,
    StrategyNotFoundError,
)


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
