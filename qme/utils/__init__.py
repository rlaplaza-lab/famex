"""Utility package facade for qme.

This file provides a minimal facade re-exporting key utilities to allow
incremental refactoring without breaking imports.
"""

from . import settings, validation
from .dependencies import deps

__all__ = ["deps", "settings", "validation"]
