"""Thin facade for qme.core.

This module re-exports the public API from the implementation module
`qme.core.optimizer`. Keep this file minimal so imports remain fast and
to avoid duplicating implementation.
"""

from .optimizer import QMEOptimizer, minimize_structure

__all__ = ["QMEOptimizer", "minimize_structure"]
