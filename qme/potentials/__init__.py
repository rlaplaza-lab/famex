"""qme.potentials - lightweight package init

This module provides small, stable factories and class names for the
potential backends. Heavy backends are imported only when their optional
dependencies are available (via ``qme.dependencies.deps``). When a backend
is unavailable the factory returns a ``MockCalculator`` and a clear
warning is emitted.
"""

from importlib import import_module
from typing import Callable

from qme.dependencies import deps

__all__ = [
    "BasePotential",
    "MockCalculator",
    "UMAPotential",
    "get_uma_calculator",
    "SO3LRPotential",
    "get_so3lr_calculator",
    "AIMNet2Potential",
    "get_aimnet2_calculator",
    "MACEPotential",
    "get_mace_calculator",
    "TorchSimPotential",
    "get_torchsim_calculator",
    "get_torchsim_mace_calculator",
    "get_torchsim_fairchem_calculator",
]

# Lightweight core imports (may be None if import fails)
try:
    from qme.potentials.base_potential import BasePotential
except ImportError:  # pragma: no cover - very unlikely
    BasePotential = None

try:
    from qme.potentials.mock_potential import MockCalculator
except ImportError:  # pragma: no cover - tests expect MockCalculator
    # Provide a clear failing type if the mock implementation is missing
    class _MissingMock:
        def __init__(self, *args, **kwargs):
            raise ImportError("MockCalculator implementation is missing")

    MockCalculator = _MissingMock


def _make_fallback_factory(backend_name: str) -> Callable[..., object]:
    """Return a factory that warns and returns a configured MockCalculator."""

    def _factory(**kwargs):
        deps.warn_fallback(backend_name, reason="implementation not available")
        return MockCalculator(backend=backend_name, **kwargs)

    return _factory


# Helper to attempt to import a concrete backend module and pull symbols.
# If the concrete module is not importable, return fallback factory/class.
def _import_backend(
    module_name: str, cls_name: str, func_name: str, backend_label: str
):
    try:
        module = import_module(f"qme.potentials.{module_name}")
        cls = getattr(module, cls_name)
        func = getattr(module, func_name)
        return cls, func
    except ImportError as e:
        # If the backend is unavailable, raise a clear error instead of falling back to mock
        raise ImportError(
            f"Failed to import {backend_label} backend: {e}. "
            f"Make sure all required dependencies are installed."
        )


# UMA depends on fairchem-core (deps name 'fairchem') - lazy loading
UMAPotential = None
def get_uma_calculator(**kwargs):
    if not (deps.has("fairchem") or deps.has("uma")):
        raise ImportError(
            "UMA backend requires fairchem-core. Install with: pip install fairchem-core"
        )
    try:
        from qme.potentials.uma_potential import UMAPotential
        return UMAPotential(**kwargs)
    except ImportError as e:
        raise ImportError(f"Failed to import UMA backend: {e}")

# SO3LR backend - lazy loading
SO3LRPotential = None
def get_so3lr_calculator(**kwargs):
    if not deps.has("so3lr"):
        raise ImportError(
            "SO3LR backend requires so3lr. Install with: pip install so3lr"
        )
    try:
        from qme.potentials.so3lr_potential import SO3LRPotential
        return SO3LRPotential(**kwargs)
    except ImportError as e:
        raise ImportError(f"Failed to import SO3LR backend: {e}")

# AIMNet2 backend - lazy loading
AIMNet2Potential = None
def get_aimnet2_calculator(**kwargs):
    if not deps.has("torch"):
        raise ImportError(
            "AIMNet2 backend requires torch. Install with: pip install torch"
        )
    try:
        from qme.potentials.aimnet2_potential import AIMNet2Potential
        return AIMNet2Potential(**kwargs)
    except ImportError as e:
        raise ImportError(f"Failed to import AIMNet2 backend: {e}")

# MACE backend - lazy loading
MACEPotential = None
def get_mace_calculator(**kwargs):
    if not deps.has("mace"):
        raise ImportError(
            "MACE backend requires mace-torch. Install with: pip install mace-torch"
        )
    try:
        from qme.potentials.mace_potential import MACEPotential
        return MACEPotential(**kwargs)
    except ImportError as e:
        raise ImportError(f"Failed to import MACE backend: {e}")

# TorchSim backend - lazy loading
TorchSimPotential = None
def get_torchsim_calculator(**kwargs):
    if not deps.has("torch_sim"):
        raise ImportError(
            "TorchSim backend requires torch-sim-atomistic. Install with: pip install torch-sim-atomistic"
        )
    try:
        from qme.potentials.torchsim_potential import TorchSimPotential
        return TorchSimPotential(**kwargs)
    except ImportError as e:
        raise ImportError(f"Failed to import TorchSim backend: {e}")

def get_torchsim_mace_calculator(**kwargs):
    if not deps.has("torch_sim"):
        raise ImportError(
            "TorchSim MACE calculator requires torch-sim-atomistic. Install with: pip install torch-sim-atomistic"
        )
    try:
        from qme.potentials.torchsim_potential import get_torchsim_mace_calculator as _get_torchsim_mace_calculator
        return _get_torchsim_mace_calculator(**kwargs)
    except ImportError as e:
        raise ImportError(f"Failed to import TorchSim MACE calculator: {e}")

def get_torchsim_fairchem_calculator(**kwargs):
    if not deps.has("torch_sim"):
        raise ImportError(
            "TorchSim Fairchem calculator requires torch-sim-atomistic. Install with: pip install torch-sim-atomistic"
        )
    try:
        from qme.potentials.torchsim_potential import get_torchsim_fairchem_calculator as _get_torchsim_fairchem_calculator
        return _get_torchsim_fairchem_calculator(**kwargs)
    except ImportError as e:
        raise ImportError(f"Failed to import TorchSim Fairchem calculator: {e}")
