"""Custom exceptions for FAMEX."""

from __future__ import annotations


class FAMEXError(Exception):
    """General FAMEX error with an optional suggestion."""

    def __init__(self, message: str, suggestion: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.suggestion = suggestion

    def __str__(self) -> str:
        if self.suggestion:
            return f"{self.message}\n\n💡 Suggestion: {self.suggestion}"
        return self.message


class BackendError(FAMEXError, ImportError):
    """Raised when a requested backend is unavailable (also an ``ImportError``)."""

    def __init__(self, backend: str, available: list[str], operation: str) -> None:
        if available:
            suggestion = f"Available backends: {', '.join(available)}"
        else:
            suggestion = (
                "No backends are currently available. "
                "Install at least one backend (e.g., pip install torch for aimnet2)."
            )
        message = f"Backend '{backend}' is not available for {operation}. {suggestion}"
        FAMEXError.__init__(self, message)
        ImportError.__init__(self, message)
        self.backend = backend
        self.available_backends = available
        self.operation = operation


class DependencyError(FAMEXError, ImportError):
    """Raised when an optional dependency is missing (also an ``ImportError``)."""

    def __init__(self, dependency: str, purpose: str, install_command: str) -> None:
        message = (
            f"Missing dependency '{dependency}' required for {purpose}. "
            f"Install with: {install_command}"
        )
        FAMEXError.__init__(self, message)
        ImportError.__init__(self, message)
        self.dependency = dependency
        self.purpose = purpose
        self.install_command = install_command


class StrategyError(Exception):
    """Base exception for strategy-related errors."""


class StrategyNotFoundError(StrategyError):
    """Raised when a requested strategy is not found in the registry."""

    def __init__(
        self,
        strategy_name: str,
        available_strategies: list[str] | None = None,
    ) -> None:
        self.strategy_name = strategy_name
        self.available_strategies = available_strategies or []
        msg = f"No strategy found for '{strategy_name}'"
        if self.available_strategies:
            msg += f". Available strategies: {sorted(self.available_strategies)}"
        super().__init__(msg)
