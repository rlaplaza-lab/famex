"""Core API exports for FAMEX.

This module intentionally provides a compact public surface for the
core package. New exports should be added here so users can import from
``famex.core`` directly.
"""

# Import strategy modules to register them
import famex.strategies  # noqa: F401
from famex.core.exceptions import (
    InvalidInputError,
    InvalidStrategyError,
    StrategyError,
    StrategyNotFoundError,
)
from famex.core.explorer import Explorer

__all__ = [
    "Explorer",
    "StrategyError",
    "StrategyNotFoundError",
    "InvalidStrategyError",
    "InvalidInputError",
]
