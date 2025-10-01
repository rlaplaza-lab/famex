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
    except ImportError:
        # If the backend is unavailable, provide fallbacks
        return MockCalculator, _make_fallback_factory(backend_label)


# UMA depends on fairchem-core (deps name 'fairchem') - special-case availability
if deps.has("fairchem") or deps.has("uma"):
    UMAPotential, get_uma_calculator = _import_backend(
        "uma_potential", "UMAPotential", "get_uma_calculator", "uma"
    )
else:
    # Provide lightweight fallback
    UMAPotential, get_uma_calculator = MockCalculator, _make_fallback_factory("uma")

# SO3LR backend
if deps.has("so3lr"):
    SO3LRPotential, get_so3lr_calculator = _import_backend(
        "so3lr_potential", "SO3LRPotential", "get_so3lr_calculator", "so3lr"
    )
else:
    SO3LRPotential, get_so3lr_calculator = MockCalculator, _make_fallback_factory(
        "so3lr"
    )

# AIMNet2 backend (requires torch)
if deps.has("torch"):
    AIMNet2Potential, get_aimnet2_calculator = _import_backend(
        "aimnet2_potential", "AIMNet2Potential", "get_aimnet2_calculator", "aimnet2"
    )
else:
    AIMNet2Potential, get_aimnet2_calculator = MockCalculator, _make_fallback_factory(
        "aimnet2"
    )

# MACE backend
if deps.has("mace"):
    MACEPotential, get_mace_calculator = _import_backend(
        "mace_potential", "MACEPotential", "get_mace_calculator", "mace"
    )
else:
    MACEPotential, get_mace_calculator = MockCalculator, _make_fallback_factory("mace")

# TorchSim backend
if deps.has("torch_sim"):
    TorchSimPotential, get_torchsim_calculator = _import_backend(
        "torchsim_potential", "TorchSimPotential", "get_torchsim_calculator", "torchsim"
    )
    # Also import convenience functions
    try:
        from qme.potentials.torchsim_potential import (
            get_torchsim_fairchem_calculator,
            get_torchsim_mace_calculator,
        )
    except ImportError:
        get_torchsim_mace_calculator = _make_fallback_factory("torchsim_mace")
        get_torchsim_fairchem_calculator = _make_fallback_factory("torchsim_fairchem")
else:
    TorchSimPotential, get_torchsim_calculator = MockCalculator, _make_fallback_factory(
        "torchsim"
    )
    get_torchsim_mace_calculator = _make_fallback_factory("torchsim_mace")
    get_torchsim_fairchem_calculator = _make_fallback_factory("torchsim_fairchem")
