"""Utility functions for FAMEX.

This module provides various utility functions including backend utilities,
profiling, validation, and device management.
"""

from famex.backends.availability import (
    get_available_backends,
    is_backend_available,
    require_backend,
)
from famex.utils.device import get_device_info, get_optimal_device
from famex.utils.profiler import PerformanceProfiler
from famex.utils.validation import (
    BackendError,
    DependencyError,
    FAMEXError,
    validate_atoms_compatibility,
)

__all__ = [
    "PerformanceProfiler",
    "FAMEXError",
    "BackendError",
    "DependencyError",
    "validate_atoms_compatibility",
    "get_available_backends",
    "is_backend_available",
    "require_backend",
    "get_device_info",
    "get_optimal_device",
]
