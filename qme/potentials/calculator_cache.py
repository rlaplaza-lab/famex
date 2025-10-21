"""
Calculator caching utilities for QME.

This module provides basic calculator caching to avoid recreating
calculators with the same parameters.

Note: SO3LR calculators are excluded from caching due to internal state
issues that cause "vmap got inconsistent sizes" errors when reused.
This is a known limitation of the SO3LR backend's internal implementation.
"""

import hashlib
from typing import Any
from weakref import WeakValueDictionary


class CalculatorCache:
    """Simple calculator cache using weak references."""

    def __init__(self, max_size: int = 10) -> None:
        """
        Initialize calculator cache.

        Parameters:
        -----------
        max_size : int
            Maximum number of calculators to cache
        """
        self.max_size = max_size
        self._cache: WeakValueDictionary = WeakValueDictionary()
        self._access_order: dict[str, int] = {}
        self._access_counter = 0

    def _generate_key(
        self, backend: str, model_name: str | None, device: str | None, **kwargs: Any
    ) -> str:
        """Generate a cache key for calculator parameters."""
        # Create a sorted dictionary of parameters for consistent hashing
        params = {
            "backend": backend,
            "model_name": model_name,
            "device": device,
            **kwargs,
        }

        # Sort parameters for consistent key generation
        sorted_params = sorted(params.items())
        param_str = str(sorted_params)

        # Generate hash
        return hashlib.md5(param_str.encode()).hexdigest()[:16]

    def get(
        self, backend: str, model_name: str | None, device: str | None, **kwargs: Any
    ) -> Any | None:
        """
        Get cached calculator if available.

        Parameters:
        -----------
        backend : str
            Calculator backend
        model_name : str, optional
            Model name
        device : str, optional
            Device specification
        **kwargs
            Additional calculator parameters

        Returns:
        --------
        Calculator or None
            Cached calculator if available, None otherwise
        """
        key = self._generate_key(backend, model_name, device, **kwargs)

        if key in self._cache:
            # Update access order
            self._access_order[key] = self._access_counter
            self._access_counter += 1
            return self._cache[key]

        return None

    def put(
        self,
        calculator: Any,
        backend: str,
        model_name: str | None,
        device: str | None,
        **kwargs: Any,
    ) -> str:
        """
        Cache a calculator.

        Parameters:
        -----------
        calculator : Any
            Calculator instance to cache
        backend : str
            Calculator backend
        model_name : str, optional
            Model name
        device : str, optional
            Device specification
        **kwargs
            Additional calculator parameters

        Returns:
        --------
        str
            Cache key for the calculator
        """
        key = self._generate_key(backend, model_name, device, **kwargs)

        # Check if we need to evict old entries
        if len(self._cache) >= self.max_size and key not in self._cache:
            self._evict_oldest()

        # Cache the calculator
        self._cache[key] = calculator
        self._access_order[key] = self._access_counter
        self._access_counter += 1

        return key

    def _evict_oldest(self) -> None:
        """Evict the least recently used calculator."""
        if not self._access_order:
            return

        # Find the oldest entry
        oldest_key = min(self._access_order.keys(), key=lambda k: self._access_order[k])

        # Remove from cache (weak reference will handle cleanup)
        if oldest_key in self._cache:
            del self._cache[oldest_key]
        del self._access_order[oldest_key]

    def clear(self) -> None:
        """Clear all cached calculators."""
        self._cache.clear()
        self._access_order.clear()
        self._access_counter = 0



# Global calculator cache instance
_calculator_cache = None


def _get_calculator_cache() -> CalculatorCache:
    """Get the global calculator cache instance."""
    global _calculator_cache
    if _calculator_cache is None:
        _calculator_cache = CalculatorCache()
    return _calculator_cache


def get_cached_calculator(
    backend: str, model_name: str | None, device: str | None, **kwargs: Any
) -> Any | None:
    """
    Get a cached calculator if available.

    Parameters:
    -----------
    backend : str
        Calculator backend
    model_name : str, optional
        Model name
    device : str, optional
        Device specification
    **kwargs
        Additional calculator parameters

    Returns:
    --------
    Calculator or None
        Cached calculator if available, None otherwise
    """
    cache = _get_calculator_cache()
    return cache.get(backend, model_name, device, **kwargs)


def cache_calculator(
    calculator: Any,
    backend: str,
    model_name: str | None,
    device: str | None,
    **kwargs: Any,
) -> str:
    """
    Cache a calculator.

    Parameters:
    -----------
    calculator : Any
        Calculator instance to cache
    backend : str
        Calculator backend
    model_name : str, optional
        Model name
    device : str, optional
        Device specification
    **kwargs
        Additional calculator parameters

    Returns:
    --------
    str
        Cache key for the calculator
    """
    cache = _get_calculator_cache()
    return cache.put(calculator, backend, model_name, device, **kwargs)
