"""famex.potentials - lightweight package init.

This module provides small, stable factories and class names for the
potential backends. Heavy backends are imported only when their optional
dependencies are available (via ``famex.backends.dependencies.deps``). When a backend
is unavailable, a clear ImportError is raised with installation instructions.
"""

import importlib
from typing import Any

from famex.backends.availability import get_backend_error_message, is_backend_available
from famex.backends.constants import (
    BACKEND_AIMNET2,
    BACKEND_MACE,
    BACKEND_MOCK,
    BACKEND_ORB,
    BACKEND_PET,
    BACKEND_PYSCF,
    BACKEND_SO3LR,
    BACKEND_TBLITE,
    BACKEND_UMA,
)
from famex.backends.registry import BACKEND_CLASSES
from famex.potentials._load_utils import raise_backend_load_error

__all__ = [
    "BasePotential",
    "MockCalculator",
    "get_aimnet2_calculator",
    "get_mace_calculator",
    "get_orb_calculator",
    "get_pet_calculator",
    "get_so3lr_calculator",
    "get_pyscf_calculator",
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
        raise TypeError(f"Expected callable, got {type(class_or_func)}")

    except ImportError as e:
        if not is_backend_available(backend):
            raise ImportError(get_backend_error_message(backend)) from e
        raise_backend_load_error(backend, kwargs.get("model_name"), e)
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
        raise_backend_load_error(backend, kwargs.get("model_name"), exc)


def get_uma_calculator(**kwargs: Any) -> Any:
    """Get UMA (Universal Materials Architecture) calculator."""
    return _get_calculator_generic(BACKEND_UMA, **kwargs)


def get_so3lr_calculator(**kwargs: Any) -> Any:
    """Get SO3LR (SO(3) Local Reference) calculator."""
    return _get_calculator_generic(BACKEND_SO3LR, **kwargs)


def get_aimnet2_calculator(**kwargs: Any) -> Any:
    """Get AIMNet2 calculator."""
    return _get_calculator_generic(BACKEND_AIMNET2, **kwargs)


def get_mace_calculator(**kwargs: Any) -> Any:
    """Get MACE (Multiscale Atomic Cluster Expansion) calculator."""
    return _get_calculator_generic(BACKEND_MACE, **kwargs)


def get_orb_calculator(**kwargs: Any) -> Any:
    """Get Orb calculator."""
    return _get_calculator_generic(BACKEND_ORB, **kwargs)


def get_tblite_calculator(**kwargs: Any) -> Any:
    """Get TBLite calculator."""
    return _get_calculator_generic(BACKEND_TBLITE, **kwargs)


def get_pet_calculator(**kwargs: Any) -> Any:
    """Get PET (UPET) calculator."""
    return _get_calculator_generic(BACKEND_PET, **kwargs)


def get_pyscf_calculator(**kwargs: Any) -> Any:
    """Get PySCF DFT calculator."""
    return _get_calculator_generic(BACKEND_PYSCF, **kwargs)
