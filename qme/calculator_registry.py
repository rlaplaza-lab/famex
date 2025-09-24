"""
Calculator factory and registry for QME.

This module provides centralized calculator creation logic to eliminate
code duplication across the codebase.
"""

from typing import Callable, Dict, Optional

from .dependencies import deps


class CalculatorRegistry:
    """
    Registry for calculator creation functions.

    This centralizes the mapping between backend names and their
    calculator creation functions, eliminating duplication.
    """

    def __init__(self):
        self._registry: Dict[str, Callable] = {}
        self._register_default_backends()

    def _register_default_backends(self):
        """Register the default backend calculators."""
        # Import here to avoid circular imports
        from .aimnet2_potential import get_aimnet2_calculator
        from .mock_calculator import MockCalculator
        from .so3lr_potential import get_so3lr_calculator
        from .uma_potential import get_uma_calculator

        self._registry = {
            "so3lr": get_so3lr_calculator,
            "uma": get_uma_calculator,
            "aimnet2": get_aimnet2_calculator,
            "mock": lambda **kwargs: MockCalculator(
                backend=kwargs.get("backend", "generic")
            ),
        }

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
        return list(self._registry.keys())

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
            If backend is not registered
        """
        if backend not in self._registry:
            available = list(self._registry.keys())
            raise ValueError(f"Unknown backend: {backend}. Available: {available}")

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
        else:
            return backend in self._registry


# Global calculator registry instance
calculator_registry = CalculatorRegistry()
