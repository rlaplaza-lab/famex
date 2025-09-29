"""qme.core package facade.

Re-export the public API from the optimizer implementation so users can do:

        from qme.core import QMEOptimizer, minimize_structure

Keep this file minimal to avoid heavy imports at package import time.
"""

from .optimizer import QMEOptimizer, minimize_structure

__all__ = ["QMEOptimizer", "minimize_structure"]
