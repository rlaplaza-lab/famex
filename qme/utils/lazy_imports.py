"""Lazy import helpers shared across QME modules."""

from __future__ import annotations

from typing import Any


def get_module_logger(name: str) -> Any:
    """Return a logger for *name*, with a stdlib fallback if QME logging is unavailable."""
    try:
        from qme.utils.logging import get_qme_logger

        return get_qme_logger(name)
    except ImportError:
        import logging

        return logging.getLogger(name)
