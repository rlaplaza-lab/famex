"""Lazy import helpers shared across FAMEX modules."""

from __future__ import annotations

from typing import Any


def get_module_logger(name: str) -> Any:
    """Return a logger for *name*, with a stdlib fallback if FAMEX logging is unavailable."""
    try:
        from famex.utils.logging import get_famex_logger

        return get_famex_logger(name)
    except ImportError:
        import logging

        return logging.getLogger(name)
