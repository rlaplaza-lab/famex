"""Optimizer-internal exceptions."""

from __future__ import annotations


class ConvergedError(Exception):
    """Exception raised when optimizer has converged."""
