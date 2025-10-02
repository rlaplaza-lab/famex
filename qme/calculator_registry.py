"""
Calculator factory and registry for QME.

This module provides centralized calculator creation logic to eliminate
code duplication across the codebase.
"""

import importlib
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from qme.dependencies import deps


class CalculatorRegistry:
    """
    Registry for calculator creation functions with lazy loading.

    This centralizes the mapping between backend names and their
    calculator creation functions, with lazy loading to avoid
    importing heavy ML backends until actually needed.
    """

    def __init__(self):
        self._registry: Dict[str, Callable] = {}

        @dataclass
        class LazyBackend:
            """Typed helper for lazy backend registry entries.

            Attributes:
                module: module path to import (str)
                function: attribute name (factory function or class) to look up
                is_class: whether the attribute is a class that should be
                    instantiated (default: False)
            """

            module: str
            function: str
            is_class: bool = False

        self._lazy_registry: Dict[str, LazyBackend] = {
            "so3lr": LazyBackend(
                module="qme.potentials", function="get_so3lr_calculator"
            ),
            "uma": LazyBackend(module="qme.potentials", function="get_uma_calculator"),
            "aimnet2": LazyBackend(
                module="qme.potentials", function="get_aimnet2_calculator"
            ),
            "mace": LazyBackend(
                module="qme.potentials", function="get_mace_calculator"
            ),
            "torchsim_mace": LazyBackend(
                module="qme.potentials", function="get_torchsim_mace_calculator"
            ),
            "torchsim_uma": LazyBackend(
                module="qme.potentials", function="get_torchsim_uma_calculator"
            ),
            "mock": LazyBackend(
                module="qme.potentials.mock_potential",
                function="MockCalculator",
                is_class=True,
            ),
        }

    def _load_backend(self, backend_name: str):
        """Lazy load a backend when first accessed."""
        if backend_name in self._registry:
            return  # Already loaded

        if backend_name not in self._lazy_registry:
            return  # Unknown backend
        backend_info = self._lazy_registry[backend_name]
        module_name = backend_info.module
        function_name = backend_info.function

        try:
            module = importlib.import_module(module_name)
            func_or_class = getattr(module, function_name)

            if getattr(backend_info, "is_class", False):
                # For class-backed backends, forward all kwargs to the
                # constructor to keep API consistent with function factories.
                self._registry[backend_name] = lambda **kwargs: func_or_class(**kwargs)
            else:
                # Regular function
                self._registry[backend_name] = func_or_class

        except ImportError:
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
            # If the requested backend couldn't be loaded, raise a clear error
            from qme.core.validation import BackendError

            # Get actually available backends (not just registered ones)
            available = [
                b for b in self.get_available_backends() if self.is_backend_available(b)
            ]
            raise BackendError(backend, available, "calculator creation")

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
        """Check if a backend is available and can create real calculators.

        Parameters:
        -----------
        backend : str
            Backend name to check

        Returns:
        --------
        bool
            True if backend is available and can create real calculators,
            False otherwise
        """
        if backend == "mock":
            return True
        elif backend == "so3lr":
            if not deps.has("so3lr"):
                return False
            try:
                from qme.potentials.so3lr_potential import SO3LRPotential

                return True
            except ImportError:
                return False
        elif backend == "uma":
            if not (deps.has("fairchem") and deps.has("torch")):
                return False
            try:
                from qme.potentials.uma_potential import UMAPotential

                return True
            except ImportError:
                return False
        elif backend == "aimnet2":
            if not deps.has("torch"):
                return False
            try:
                from qme.potentials.aimnet2_potential import AIMNet2Potential

                return True
            except ImportError:
                return False
        elif backend == "mace":
            if not (deps.has("mace") and deps.has("torch")):
                return False
            try:
                from qme.potentials.mace_potential import MACEPotential

                # Test if MACE can actually be loaded (check for e3nn compatibility)
                test_calc = MACEPotential(device="cpu")
                test_calc._load_calculator()
                return True
            except (ImportError, ValueError):
                # ImportError: missing dependencies
                # ValueError: e3nn compatibility issue
                return False
        elif backend == "torchsim_mace":
            if not (deps.has("torch_sim") and deps.has("torch")):
                return False
            try:
                from qme.potentials.torchsim_potential import TorchSimPotential
                
                # Test if TorchSim MACE can actually be loaded (check for e3nn compatibility)
                test_calc = TorchSimPotential(backend="mace", device="cpu")
                test_calc._load_calculator()
                return True
            except (ImportError, ValueError):
                # ImportError: missing dependencies
                # ValueError: e3nn compatibility issue
                return False
        elif backend == "torchsim_uma":
            # TorchSim UMA requires torch_sim, torch, AND fairchem
            if not (
                deps.has("torch_sim") and deps.has("torch") and deps.has("fairchem")
            ):
                return False
            try:
                # Additional check for fairchem compatibility
                from fairchem.core.common.utils import setup_imports, setup_logging

                from qme.potentials.torchsim_potential import TorchSimPotential

                return True
            except ImportError:
                return False
        else:
            return backend in self._registry


# Global calculator registry instance
calculator_registry = CalculatorRegistry()
