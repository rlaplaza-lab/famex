"""famex.potentials - lightweight package init.

This module provides small, stable factories and class names for the
potential backends. Heavy backends are imported only when their optional
dependencies are available (via ``famex.backends.dependencies.deps``). When a backend
is unavailable, a clear ImportError is raised with installation instructions.
"""

import importlib
from typing import Any

from famex.backends.availability import get_backend_error_message, is_backend_available
from famex.backends.constants import BACKEND_MOCK
from famex.backends.registry import BACKEND_CLASSES

__all__ = [
    "BasePotential",
    "MockCalculator",
    "get_aimnet2_calculator",
    "get_mace_calculator",
    "get_orb_calculator",
    "get_pet_calculator",
    "get_so3lr_calculator",
    "get_tblite_calculator",
    "get_uma_calculator",
]

# Lightweight core imports (may be None if import fails)
try:
    from famex.potentials.base_potential import BasePotential
except ImportError:  # pragma: no cover - very unlikely
    BasePotential = type(None)  # type: ignore[assignment,misc]

try:
    from famex.potentials.mock_potential import MockCalculator
except ImportError:  # pragma: no cover - tests expect MockCalculator
    # Provide a clear failing type if the mock implementation is missing
    class _MissingMock:
        """Placeholder class when MockCalculator implementation is missing."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            msg = "MockCalculator implementation is missing"
            raise ImportError(msg)

    MockCalculator = _MissingMock  # type: ignore[assignment,misc]


_BACKEND_MODULES = {name: spec for name, spec in BACKEND_CLASSES.items() if name != BACKEND_MOCK}


def _get_calculator_generic(backend: str, **kwargs: Any) -> Any:
    """Create a calculator instance for the given backend."""
    if not is_backend_available(backend):
        raise ImportError(get_backend_error_message(backend))

    if backend not in _BACKEND_MODULES:
        raise ImportError(f"Unknown backend: {backend}")

    module_name, class_or_func_name = _BACKEND_MODULES[backend]

    try:
        module = importlib.import_module(module_name)
        class_or_func = getattr(module, class_or_func_name)

        if callable(class_or_func):
            return class_or_func(**kwargs)
        raise ImportError(f"Expected callable, got {type(class_or_func)}")

    except ImportError as e:
        raise ImportError(get_backend_error_message(backend)) from e


def get_uma_calculator(**kwargs: Any) -> Any:
    """Get UMA (Universal Materials Architecture) calculator."""
    return _get_calculator_generic("uma", **kwargs)


def get_so3lr_calculator(**kwargs: Any) -> Any:
    """Get SO3LR (SO(3) Local Reference) calculator."""
    return _get_calculator_generic("so3lr", **kwargs)


def get_aimnet2_calculator(**kwargs: Any) -> Any:
    """Get AIMNet2 calculator."""
    return _get_calculator_generic("aimnet2", **kwargs)


def get_mace_calculator(**kwargs: Any) -> Any:
    """Get MACE (Multiscale Atomic Cluster Expansion) calculator."""
    return _get_calculator_generic("mace", **kwargs)


def get_orb_calculator(**kwargs: Any) -> Any:
    """Get Orb calculator."""
    return _get_calculator_generic("orb", **kwargs)


def get_tblite_calculator(**kwargs: Any) -> Any:
    """Get TBLite calculator."""
    return _get_calculator_generic("tblite", **kwargs)


def get_pet_calculator(**kwargs: Any) -> Any:
    """Get PET (UPET) calculator."""
    return _get_calculator_generic("pet", **kwargs)
