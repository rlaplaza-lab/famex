"""
Calculator factory and registry for QME.

This module provides centralized calculator creation logic to eliminate
code duplication across the codebase.
"""

import importlib
from typing import Callable, Dict, Optional

from .dependencies import deps


class CalculatorRegistry:
    """
    Registry for calculator creation functions with lazy loading.

    This centralizes the mapping between backend names and their
    calculator creation functions, with lazy loading to avoid
    importing heavy ML backends until actually needed.
    """

    def __init__(self):
        self._registry: Dict[str, Callable] = {}
        self._lazy_registry: Dict[str, dict] = {
            "so3lr": {
                "module": "qme.so3lr_potential",
                "function": "get_so3lr_calculator",
            },
            "uma": {
                "module": "qme.uma_potential",
                "function": "get_uma_calculator",
            },
            "aimnet2": {
                "module": "qme.aimnet2_potential",
                "function": "get_aimnet2_calculator",
            },
            "mace": {
                "module": "qme.mace_potential",
                "function": "get_mace_calculator",
            },
            "mock": {
                "module": "qme.mock_calculator",
                "function": "MockCalculator",
                "is_class": True,
            },
        }

    def _load_backend(self, backend_name: str):
        """Lazy load a backend when first accessed."""
        if backend_name in self._registry:
            return  # Already loaded

        if backend_name not in self._lazy_registry:
            return  # Unknown backend

        backend_info = self._lazy_registry[backend_name]
        module_name = backend_info["module"]
        function_name = backend_info["function"]

        try:
            module = importlib.import_module(module_name)
            func_or_class = getattr(module, function_name)

            if backend_info.get("is_class", False):
                # Special case for MockCalculator class
                self._registry[backend_name] = lambda **kwargs: func_or_class(
                    backend=kwargs.get("backend", "generic")
                )
            else:
                # Regular function
                self._registry[backend_name] = func_or_class

        except ImportError as e:
            # Backend not available, this is expected for optional dependencies
            pass
        except Exception as e:
            # Other errors should be logged but not break the system
            import warnings

            warnings.warn(f"Failed to load backend {backend_name}: {e}")
            pass

    def register(self, backend_name: str, factory_func: Callable):
        """Register a new calculator factory function.

        Parameters:
        -----------
        backend_name : str
            Name of the backend
        factory_func : callable
            Function that creates calculator instances
        """
        self._registry[backend_name] = factory_func

    def get_available_backends(self) -> list:
        """Get list of available backend names."""
        # Return all known backends from lazy registry
        # We don't try to load them here to avoid heavy imports
        return list(self._lazy_registry.keys())

    def create_calculator(
        self,
        backend: str,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        """Create a calculator for the specified backend.

        Parameters:
        -----------
        backend : str
            Backend name (e.g., 'so3lr', 'uma', 'aimnet2', 'mock')
        model_name : str, optional
            Name of the model to use
        model_path : str, optional
            Path to model file (SO3LR only)
        device : str, optional
            Device for computations ('cpu', 'cuda')
        **kwargs
            Additional arguments passed to calculator

        Returns:
        --------
        Calculator
            Configured calculator instance

        Raises:
        -------
        ValueError
            If backend is not registered or available
        """
        # Lazy load the backend
        self._load_backend(backend)

        if backend not in self._registry:
            available = self.get_available_backends()
            raise ValueError(
                f"Unknown or unavailable backend: {backend}. Available: {available}"
            )

        factory_func = self._registry[backend]

        # Build arguments based on what the factory function accepts
        factory_kwargs = kwargs.copy()

        if model_name is not None:
            factory_kwargs["model_name"] = model_name
        if model_path is not None:
            factory_kwargs["model_path"] = model_path
        if device is not None:
            factory_kwargs["device"] = device

        # Special handling for mock backend
        if backend == "mock":
            factory_kwargs["backend"] = kwargs.get("mock_backend", "generic")

        return factory_func(**factory_kwargs)

    def is_backend_available(self, backend: str) -> bool:
        """Check if a backend is available (dependencies satisfied).

        Parameters:
        -----------
        backend : str
            Backend name to check

        Returns:
        --------
        bool
            True if backend is available, False otherwise
        """
        if backend == "mock":
            return True
        elif backend == "so3lr":
            return deps.has("so3lr")
        elif backend == "uma":
            return deps.has("fairchem") and deps.has("torch")
        elif backend == "aimnet2":
            return deps.has("torch")
        elif backend == "mace":
            return deps.has("mace") and deps.has("torch")
        else:
            return backend in self._registry


# Global calculator registry instance
calculator_registry = CalculatorRegistry()
