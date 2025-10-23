"""Utility functions for QME.

This module provides various utility functions including backend utilities,
profiling, validation, and device management.
"""

from qme.utils.backend_utils import (
    get_available_backends,
    is_backend_available,
    require_backend,
)
from qme.utils.device import get_device_info, get_optimal_device
from qme.utils.profiler import PerformanceProfiler
from qme.utils.validation import (
    BackendError,
    DependencyError,
    QMEError,
    validate_atoms_compatibility,
)

__all__ = [
    "PerformanceProfiler",
    "QMEError",
    "BackendError",
    "DependencyError",
    "validate_atoms_compatibility",
    "get_available_backends",
    "is_backend_available",
    "require_backend",
    "get_device_info",
    "get_optimal_device",
]
