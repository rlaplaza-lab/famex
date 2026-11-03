"""Custom exceptions for FAMEX core module.

This module provides specific exception classes for better error handling
and more informative error messages.
"""

from __future__ import annotations


class StrategyError(Exception):
    """Base exception for strategy-related errors."""


class StrategyNotFoundError(StrategyError):
    """Raised when a requested strategy is not found in the registry.

    Attributes
    ----------
    strategy_name : str
        The strategy name that was requested
    available_strategies : list[str]
        List of available strategy names
    """

    def __init__(
        self,
        strategy_name: str,
        available_strategies: list[str] | None = None,
    ) -> None:
        """Initialize StrategyNotFoundError.

        Parameters
        ----------
        strategy_name : str
            The strategy name that was requested
        available_strategies : list[str], optional
            List of available strategy names
        """
        self.strategy_name = strategy_name
        self.available_strategies = available_strategies or []
        msg = f"No strategy found for '{strategy_name}'"
        if self.available_strategies:
            msg += f". Available strategies: {sorted(self.available_strategies)}"
        super().__init__(msg)
