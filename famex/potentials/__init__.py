"""famex.potentials - lightweight package init.

This module provides small, stable factories and class names for the
potential backends. Heavy backends are imported only when their optional
dependencies are available (via ``famex.backends.dependencies.deps``). When a backend
is unavailable, a clear ImportError is raised with installation instructions.
"""

import importlib
from typing import Any

from famex.backends.availability import get_backend_error_message, is_backend_available

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


# Backend module mapping for generic factory
# Import constants to avoid hardcoded strings
from famex.backends.constants import (
    BACKEND_AIMNET2,
    BACKEND_MACE,
    BACKEND_ORB,
    BACKEND_PET,
    BACKEND_SO3LR,
    BACKEND_TBLITE,
    BACKEND_UMA,
)

_BACKEND_MODULES = {
    BACKEND_UMA: ("famex.potentials.uma_potential", "UMAPotential"),
    BACKEND_SO3LR: ("famex.potentials.so3lr_potential", "SO3LRPotential"),
    BACKEND_AIMNET2: ("famex.potentials.aimnet2_potential", "AIMNet2Potential"),
    BACKEND_MACE: ("famex.potentials.mace_potential", "MACEPotential"),
    BACKEND_ORB: ("famex.potentials.orb_potential", "OrbPotential"),
    BACKEND_TBLITE: ("famex.potentials.tblite_potential", "TBLitePotential"),
    BACKEND_PET: ("famex.potentials.pet_potential", "PETPotential"),
}


def _get_calculator_generic(backend: str, **kwargs: Any) -> Any:
    """Create a generic calculator factory function.

    Parameters
    ----------
    backend : str
        Backend name
    **kwargs
        Arguments passed to the calculator constructor

    Returns
    -------
    Calculator instance

    Raises
    ------
    ImportError
        If backend is not available or cannot be imported
    """
    if not is_backend_available(backend):
        raise ImportError(get_backend_error_message(backend))

    if backend not in _BACKEND_MODULES:
        raise ImportError(f"Unknown backend: {backend}")

    module_name, class_or_func_name = _BACKEND_MODULES[backend]

    try:
        module = importlib.import_module(module_name)
        class_or_func = getattr(module, class_or_func_name)

        # Handle both classes and functions
        if callable(class_or_func):
            return class_or_func(**kwargs)
        else:
            raise ImportError(f"Expected callable, got {type(class_or_func)}")

    except ImportError as e:
        # Provide helpful error messages for common backends
        error_messages = {
            "uma": f"Failed to import UMA backend: {e}. This may be due to missing FairChem dependencies or version conflicts. Try: pip install fairchem-core",
            "so3lr": f"Failed to import SO3LR backend: {e}. SO3LR requires JAX and must be installed separately from source. See the FAMEX documentation for SO3LR installation instructions.",
            "aimnet2": f"Failed to import AIMNet2 backend: {e}. AIMNet2 requires PyTorch. Try: pip install torch",
            "mace": f"Failed to import MACE backend: {e}. MACE requires PyTorch and mace-torch. Note: MACE cannot be installed with UMA due to e3nn version conflicts. Try: pip install mace-torch",
            "orb": f"Failed to import Orb backend: {e}. Orb requires orb-models and PyTorch. Note: orb-models is a large package and may have compatibility issues. Try: pip install orb-models",
            "tblite": f"Failed to import TBLite backend: {e}. TBLite requires the tblite package. Try: pip install tblite",
            "pet": f"Failed to import PET backend: {e}. PET requires upet and PyTorch (Python 3.11+). Try: pip install upet",
        }

        msg = error_messages.get(backend, f"Failed to import {backend} backend: {e}")
        raise ImportError(msg)


# Individual calculator factory functions (thin wrappers for backward compatibility)
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
