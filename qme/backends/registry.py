"""Calculator registry and creation for QME.

This module provides centralized calculator creation logic to eliminate
code duplication across the codebase. It combines the registry functionality
with the high-level calculator creation interface.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from qme.backends.availability import is_backend_available
from qme.utils.validation import BackendError

if TYPE_CHECKING:
    from collections.abc import Callable


class CalculatorRegistry:
    """Registry for calculator creation functions with lazy loading.

    This centralizes the mapping between backend names and their
    calculator creation functions, with lazy loading to avoid
    importing heavy ML backends until actually needed.
    """

    def __init__(self) -> None:
        """Initialize the backend registry with empty registry."""
        self._registry: dict[str, Callable[..., Any]] = {}

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

        # Import backend constants for consistent naming
        from qme.backends.constants import (
            BACKEND_AIMNET2,
            BACKEND_MACE,
            BACKEND_MOCK,
            BACKEND_ORB,
            BACKEND_SO3LR,
            BACKEND_TBLITE,
            BACKEND_TORCHSIM_MACE,
            BACKEND_TORCHSIM_UMA,
            BACKEND_UMA,
        )

        self._lazy_registry: dict[str, LazyBackend] = {
            BACKEND_SO3LR: LazyBackend(module="qme.potentials", function="get_so3lr_calculator"),
            BACKEND_UMA: LazyBackend(module="qme.potentials", function="get_uma_calculator"),
            BACKEND_AIMNET2: LazyBackend(
                module="qme.potentials", function="get_aimnet2_calculator"
            ),
            BACKEND_MACE: LazyBackend(module="qme.potentials", function="get_mace_calculator"),
            BACKEND_TORCHSIM_MACE: LazyBackend(
                module="qme.potentials",
                function="get_torchsim_mace_calculator",
            ),
            BACKEND_TORCHSIM_UMA: LazyBackend(
                module="qme.potentials",
                function="get_torchsim_uma_calculator",
            ),
            BACKEND_ORB: LazyBackend(module="qme.potentials", function="get_orb_calculator"),
            BACKEND_TBLITE: LazyBackend(module="qme.potentials", function="get_tblite_calculator"),
            BACKEND_MOCK: LazyBackend(
                module="qme.potentials.mock_potential",
                function="MockCalculator",
                is_class=True,
            ),
        }

    def _load_backend(self, backend_name: str) -> None:
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

            warnings.warn(f"Failed to load backend {backend_name}: {e}", stacklevel=2)

    def register(self, backend_name: str, factory_func: Callable[..., Any]) -> None:
        """Register a new calculator factory function.

        Parameters
        ----------
        backend_name : str
            Name of the backend
        factory_func : callable
            Function that creates calculator instances

        """
        self._registry[backend_name] = factory_func

    def get_registered_backends(self) -> list[str]:
        """Get list of all registered backend names (regardless of availability)."""
        # Return all known backends from lazy registry
        # We don't try to load them here to avoid heavy imports
        return list(self._lazy_registry.keys())

    def create_calculator(
        self,
        backend: str,
        model_name: str | None = None,
        model_path: str | None = None,
        device: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Create a calculator for the specified backend.

        Parameters
        ----------
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
        -------
        Calculator
            Configured calculator instance

        Raises:
        ------
        BackendError
            If backend is not registered or available

        """
        # Lazy load the backend
        self._load_backend(backend)
        if backend not in self._registry:
            # If the requested backend couldn't be loaded, raise a clear error
            # Get actually available backends using the centralized system
            from qme.backends.availability import get_available_backends

            available = get_available_backends(include_mock=False)
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
        from qme.backends.constants import BACKEND_MOCK

        if backend == BACKEND_MOCK:
            factory_kwargs["backend"] = kwargs.get("mock_backend", "generic")

        return factory_func(**factory_kwargs)

    def is_backend_available(self, backend: str) -> bool:
        """Check if a backend is available and can create real calculators.

        Uses fast dependency-based checking to avoid expensive calculator instantiation
        while still catching most compatibility issues through version analysis and
        import checking.

        Parameters
        ----------
        backend : str
            Backend name to check

        Returns:
        -------
        bool
            True if backend is available and can create real calculators,
            False otherwise

        """
        return is_backend_available(backend)


# Global calculator registry instance
calculator_registry = CalculatorRegistry()


def create_calculator(
    backend: str,
    model_name: str | None,
    model_path: str | None,
    device: str | None,
    default_charge: int,
    default_spin: int,
    charge: int | None = None,
    mult: int | None = None,
    use_cache: bool = True,
    verbose: int = 1,
) -> Any:
    """Create calculator based on backend using the registry.

    Parameters
    ----------
    backend : str
        Backend name (e.g., 'uma', 'aimnet2', 'mace', 'so3lr', 'mock')
    model_name : Optional[str]
        Name of the model to use
    model_path : Optional[str]
        Path to model file (for local models)
    device : Optional[str]
        Device for computations ('cpu', 'cuda')
    default_charge : int
        Default charge for the system
    default_spin : int
        Default spin multiplicity for the system
    charge : Optional[int], default None
        Explicit charge (overrides default_charge if provided)
    mult : Optional[int], default None
        Explicit spin multiplicity (overrides default_spin if provided)
    use_cache : bool, default True
        Whether to use cached calculator instances
    verbose : int, default 1
        Verbosity level for calculator creation (0=quiet, 1=normal, 2=verbose)

    Returns:
    -------
    Calculator
        Configured calculator instance

    Raises:
    ------
    BackendError
        If backend is not available or cannot create calculator
    ValueError
        If parameters are invalid

    Notes:
    -----
    New parameters `charge` and `mult` (optional) are forwarded to
    backends that accept explicit molecular charge / multiplicity
    constructor arguments (for example AIMNet2). Older backends that
    expect `default_charge` / `default_spin` will continue to receive
    those values.

    """
    # Use the centralized calculator registry. Forward both naming
    # conventions so different backends can pick what they expect.
    factory_kwargs = {
        "default_charge": default_charge,
        "default_spin": default_spin,
    }

    # If explicit charge/multiplicity were provided (e.g. from a
    # Geometry object), forward them under the common names used by
    # some backends.
    if charge is not None:
        factory_kwargs["charge"] = charge
    if mult is not None:
        factory_kwargs["mult"] = mult

    # Try to get cached calculator first (exclude SO3LR due to state issues)
    backend_lower = backend.lower()
    if use_cache and backend_lower != "so3lr":
        try:
            from qme.backends.cache import get_cached_calculator

            cached_calc = get_cached_calculator(
                backend=backend,
                model_name=model_name,
                model_path=model_path,
                device=device,
                **factory_kwargs,
            )

            if cached_calc is not None:
                return cached_calc
        except ImportError:
            # Calculator cache not available, continue without caching
            pass

    # Create new calculator
    calculator = calculator_registry.create_calculator(
        backend=backend,
        model_name=model_name,
        model_path=model_path,
        device=device,
        verbose=verbose,
        **factory_kwargs,
    )

    # Cache the calculator if caching is enabled (exclude SO3LR due to state issues)
    if use_cache and backend_lower != "so3lr":
        try:
            from qme.backends.cache import cache_calculator

            cache_calculator(
                calculator=calculator,
                backend=backend,
                model_name=model_name,
                model_path=model_path,
                device=device,
                **factory_kwargs,
            )
        except ImportError:
            # Calculator cache not available, continue without caching
            pass

    return calculator
