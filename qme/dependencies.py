"""Facade for backward compatibility: re-export dependency manager from qme.utils.dependencies."""

from .utils.dependencies import *  # noqa: F401,F403

__all__ = [name for name in dir() if not name.startswith("_")]
