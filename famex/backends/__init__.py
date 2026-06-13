"""Backend management for FAMEX.

This module provides a unified interface for backend availability checking,
dependency management, calculator creation, and caching.
"""

# Core backend functionality
from famex.backends.availability import (
    clear_availability_cache,
    filter_available_backends,
    get_availability_reason,
    get_available_backends,
    get_available_backends_with_logging,
    get_available_ml_backends,
    get_backend_error_message,
    get_backend_pairs,
    is_backend_available,
    print_backend_summary,
    require_any_backend,
    require_backend,
    require_ml_backends,
    validate_backends,
)
from famex.backends.cache import (
    cache_calculator,
    clear_all_caches,
    download_and_cache_model,
    get_cached_calculator,
    get_model_cache,
    get_unified_cache,
)
from famex.backends.constants import (
    ALL_BACKENDS,
    BACKEND_AIMNET2,
    BACKEND_MACE,
    BACKEND_MOCK,
    BACKEND_ORB,
    BACKEND_PET,
    BACKEND_SO3LR,
    BACKEND_TBLITE,
    BACKEND_UMA,
    ML_BACKENDS,
    REGULAR_BACKENDS,
)
from famex.backends.dependencies import deps
from famex.backends.registry import CalculatorRegistry, calculator_registry, create_calculator

__all__ = [
    # Backend availability
    "is_backend_available",
    "get_available_backends",
    "get_available_backends_with_logging",
    "get_available_ml_backends",
    "get_availability_reason",
    "get_backend_error_message",
    "clear_availability_cache",
    "filter_available_backends",
    "validate_backends",
    "print_backend_summary",
    "require_backend",
    "require_any_backend",
    "require_ml_backends",
    "get_backend_pairs",
    # Calculator registry and creation
    "CalculatorRegistry",
    "calculator_registry",
    "create_calculator",
    # Caching
    "get_cached_calculator",
    "cache_calculator",
    "get_model_cache",
    "download_and_cache_model",
    "get_unified_cache",
    "clear_all_caches",
    # Dependencies
    "deps",
    # Constants
    "ALL_BACKENDS",
    "ML_BACKENDS",
    "REGULAR_BACKENDS",
    "BACKEND_MOCK",
    "BACKEND_AIMNET2",
    "BACKEND_UMA",
    "BACKEND_MACE",
    "BACKEND_SO3LR",
    "BACKEND_ORB",
    "BACKEND_TBLITE",
    "BACKEND_PET",
]
