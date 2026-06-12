"""Strategy implementations for FAMEX.

This module contains all optimization strategies including local optimization,
transition state searches, and path optimization methods.
"""

# Import all strategy modules to register them
# Note: cineb is now merged into neb.py
import famex.strategies.growing_string
import famex.strategies.helpers
import famex.strategies.irc
import famex.strategies.minima
import famex.strategies.minima_interpolate  # noqa: F401
import famex.strategies.neb  # Registers both NEB and CI-NEB strategies
import famex.strategies.neb_optimizer
import famex.strategies.path_interpolate
import famex.strategies.ts
import famex.strategies.ts_interpolate  # noqa: F401

__all__ = []
