"""Core API exports for FAMEX."""

from __future__ import annotations

from typing import Any

from famex.core.exceptions import (
    BackendError,
    DependencyError,
    FAMEXError,
    StrategyError,
    StrategyNotFoundError,
)

__all__ = [
    "BackendError",
    "DependencyError",
    "Explorer",
    "FAMEXError",
    "StrategyError",
    "StrategyNotFoundError",
]


def __getattr__(name: str) -> Any:
    if name == "Explorer":
        import famex.strategies  # noqa: F401
        from famex.core.explorer import Explorer

        return Explorer
    msg = f"module '{__name__}' has no attribute '{name}'"
    raise AttributeError(msg)
