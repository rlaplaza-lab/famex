"""Calculator registry and creation for QME."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from qme.backends.availability import is_backend_available
from qme.backends.constants import (
    BACKEND_AIMNET2,
    BACKEND_MACE,
    BACKEND_MOCK,
    BACKEND_ORB,
    BACKEND_SO3LR,
    BACKEND_TBLITE,
    BACKEND_UMA,
)
from qme.utils.lazy_imports import get_module_logger
from qme.utils.validation import BackendError

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_module_logger(__name__)

_BACKEND_CLASSES: dict[str, tuple[str, str]] = {
    BACKEND_UMA: ("qme.potentials.uma_potential", "UMAPotential"),
    BACKEND_SO3LR: ("qme.potentials.so3lr_potential", "SO3LRPotential"),
    BACKEND_AIMNET2: ("qme.potentials.aimnet2_potential", "AIMNet2Potential"),
    BACKEND_MACE: ("qme.potentials.mace_potential", "MACEPotential"),
    BACKEND_ORB: ("qme.potentials.orb_potential", "OrbPotential"),
    BACKEND_TBLITE: ("qme.potentials.tblite_potential", "TBLitePotential"),
    BACKEND_MOCK: ("qme.potentials.mock_potential", "MockCalculator"),
}


class CalculatorRegistry:
    """Registry for calculator creation with lazy backend loading."""

    def __init__(self) -> None:
        self._registry: dict[str, Callable[..., Any]] = {}

    def _load_backend(self, backend_name: str) -> None:
        if backend_name in self._registry or backend_name not in _BACKEND_CLASSES:
            return

        module_name, class_name = _BACKEND_CLASSES[backend_name]
        try:
            logger.debug("Loading backend '%s' from %s.%s", backend_name, module_name, class_name)
            module = importlib.import_module(module_name)
            calculator_cls = getattr(module, class_name)
            self._registry[backend_name] = calculator_cls
            logger.debug("Successfully loaded backend '%s'", backend_name)
        except ImportError as exc:
            logger.debug("Backend '%s' not available (ImportError): %s", backend_name, exc)
        except AttributeError as exc:
            import warnings

            error_msg = (
                f"Failed to load backend '{backend_name}': class '{class_name}' "
                f"not found in module '{module_name}'. Error: {exc}."
            )
            logger.error(error_msg)
            warnings.warn(error_msg, stacklevel=2)
        except (RuntimeError, ValueError, TypeError) as exc:
            import warnings

            error_msg = (
                f"Failed to load backend '{backend_name}' from module '{module_name}': {exc}."
            )
            logger.error(error_msg, exc_info=True)
            warnings.warn(error_msg, stacklevel=2)

    def register(self, backend_name: str, factory_func: Callable[..., Any]) -> None:
        self._registry[backend_name] = factory_func

    def get_registered_backends(self) -> list[str]:
        return list(_BACKEND_CLASSES.keys())

    def create_calculator(
        self,
        backend: str,
        model_name: str | None = None,
        model_path: str | None = None,
        device: str | None = None,
        **kwargs: Any,
    ) -> Any:
        self._load_backend(backend)
        if backend not in self._registry:
            from qme.backends.availability import get_available_backends

            available = get_available_backends(include_mock=False)
            logger.error(
                "Backend '%s' is not available for calculator creation. Available: %s",
                backend,
                available,
            )
            raise BackendError(backend, available, "calculator creation")

        calculator_cls = self._registry[backend]
        factory_kwargs = kwargs.copy()

        if backend.lower() == BACKEND_TBLITE and model_name is not None:
            factory_kwargs["method"] = model_name
        elif model_name is not None:
            factory_kwargs["model_name"] = model_name

        if model_path is not None:
            factory_kwargs["model_path"] = model_path
        if device is not None:
            factory_kwargs["device"] = device

        if backend == BACKEND_MOCK:
            factory_kwargs["backend"] = kwargs.get("mock_backend", "generic")

        return calculator_cls(**factory_kwargs)

    def is_backend_available(self, backend: str) -> bool:
        return is_backend_available(backend)


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
    factory_kwargs = {
        "default_charge": default_charge,
        "default_spin": default_spin,
    }
    if charge is not None:
        factory_kwargs["charge"] = charge
    if mult is not None:
        factory_kwargs["mult"] = mult

    backend_lower = backend.lower()
    if use_cache and backend_lower != BACKEND_SO3LR:
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
            pass

    calculator = calculator_registry.create_calculator(
        backend=backend,
        model_name=model_name,
        model_path=model_path,
        device=device,
        verbose=verbose,
        **factory_kwargs,
    )

    if use_cache and backend_lower != BACKEND_SO3LR:
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
            pass

    return calculator
