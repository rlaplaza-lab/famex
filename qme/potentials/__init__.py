"""qme.potentials - lightweight package init

This module provides small, stable factories and class names for the
potential backends. Heavy backends are imported only when their optional
dependencies are available (via ``qme.dependencies.deps``). When a backend
is unavailable, a clear ImportError is raised with installation instructions.
"""

# from collections.abc import Callable  # Unused for now
from typing import Any

from qme.dependencies import deps  # noqa: F401

__all__ = [
    "BasePotential",
    "MockCalculator",
    "get_uma_calculator",
    "get_so3lr_calculator",
    "get_aimnet2_calculator",
    "get_mace_calculator",
    "get_orb_calculator",
    "get_tblite_calculator",
    "get_torchsim_calculator",
    "get_torchsim_mace_calculator",
    "get_torchsim_uma_calculator",
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
        """Placeholder class when MockCalculator implementation is missing."""

        def __init__(self, *args, **kwargs):
            raise ImportError("MockCalculator implementation is missing")

    MockCalculator = _MissingMock




# UMA depends on fairchem-core (deps name 'fairchem') - lazy loading


def get_uma_calculator(**kwargs: Any) -> Any:
    """Get UMA (Universal Materials Architecture) calculator.

    Returns a UMAPotential instance for molecular and materials calculations.
    Requires fairchem-core dependencies.
    """
    from qme.backend_availability import get_backend_error_message, is_backend_available

    if not is_backend_available("uma"):
        raise ImportError(get_backend_error_message("uma"))
    try:
        from qme.potentials.uma_potential import UMAPotential

        return UMAPotential(**kwargs)
    except ImportError as e:
        raise ImportError(
            f"Failed to import UMA backend: {e}. "
            f"This may be due to missing FairChem dependencies or version conflicts. "
            f"Try: pip install qme-ml[uma]"
        )


# SO3LR backend - lazy loading


def get_so3lr_calculator(**kwargs: Any) -> Any:
    """Get SO3LR (SO(3) Local Reference) calculator.

    Returns a SO3LRPotential instance for molecular calculations.
    Requires so3lr dependencies.
    """
    from qme.backend_availability import get_backend_error_message, is_backend_available

    if not is_backend_available("so3lr"):
        raise ImportError(get_backend_error_message("so3lr"))
    try:
        from qme.potentials.so3lr_potential import SO3LRPotential

        return SO3LRPotential(**kwargs)
    except ImportError as e:
        raise ImportError(
            f"Failed to import SO3LR backend: {e}. "
            f"SO3LR requires JAX and must be installed separately from source. "
            f"See the QME documentation for SO3LR installation instructions."
        )


# AIMNet2 backend - lazy loading


def get_aimnet2_calculator(**kwargs: Any) -> Any:
    """Get AIMNet2 calculator.

    Returns an AIMNet2Potential instance for molecular calculations.
    Requires torch and torch_cluster dependencies.
    """
    from qme.backend_availability import get_backend_error_message, is_backend_available

    if not is_backend_available("aimnet2"):
        raise ImportError(get_backend_error_message("aimnet2"))
    try:
        from qme.potentials.aimnet2_potential import AIMNet2Potential

        return AIMNet2Potential(**kwargs)
    except ImportError as e:
        raise ImportError(
            f"Failed to import AIMNet2 backend: {e}. "
            f"AIMNet2 requires PyTorch and torch-cluster. "
            f"Try: pip install qme-ml[aimnet2]"
        )


# MACE backend - lazy loading


def get_mace_calculator(**kwargs: Any) -> Any:
    """Get MACE (Multiscale Atomic Cluster Expansion) calculator.

    Returns a MACEPotential instance for molecular and materials calculations.
    Requires mace-torch dependencies.
    """
    from qme.backend_availability import get_backend_error_message, is_backend_available

    if not is_backend_available("mace"):
        raise ImportError(get_backend_error_message("mace"))
    try:
        from qme.potentials.mace_potential import MACEPotential

        return MACEPotential(**kwargs)
    except ImportError as e:
        raise ImportError(
            f"Failed to import MACE backend: {e}. "
            f"MACE requires PyTorch and mace-torch. "
            f"Note: MACE cannot be installed with UMA due to e3nn version conflicts. "
            f"Try: pip install qme-ml[mace]"
        )


# TorchSim backend - lazy loading


def get_torchsim_calculator(**kwargs: Any) -> Any:
    """Get TorchSim calculator (default MACE backend).

    Returns a TorchSimPotential instance with MACE backend for accelerated calculations.
    Requires torch-sim-atomistic and Python 3.11+.
    """
    from qme.backend_availability import get_backend_error_message, is_backend_available

    if not is_backend_available("torchsim_mace"):
        raise ImportError(get_backend_error_message("torchsim_mace"))
    try:
        from qme.potentials.torchsim_potential import TorchSimPotential

        return TorchSimPotential(**kwargs)
    except ImportError as e:
        raise ImportError(
            f"Failed to import TorchSim backend: {e}. "
            f"TorchSim requires Python 3.11+ and torch-sim-atomistic. "
            f"Try: pip install qme-ml[torchsim]"
        )


def get_torchsim_mace_calculator(**kwargs: Any) -> Any:
    """Get TorchSim calculator with MACE backend.

    Returns a TorchSimPotential instance specifically configured for MACE models.
    Requires both MACE and TorchSim dependencies.
    """
    from qme.backend_availability import get_backend_error_message, is_backend_available

    if not is_backend_available("torchsim_mace"):
        raise ImportError(get_backend_error_message("torchsim_mace"))
    try:
        from qme.potentials.torchsim_potential import (
            get_torchsim_mace_calculator as _get_torchsim_mace_calculator,
        )

        return _get_torchsim_mace_calculator(**kwargs)
    except ImportError as e:
        raise ImportError(
            f"Failed to import TorchSim MACE calculator: {e}. "
            f"TorchSim MACE requires both MACE and TorchSim to be available. "
            f"Try: pip install qme-ml[torchsim] (note: MACE conflicts with UMA)"
        )


def get_torchsim_uma_calculator(**kwargs: Any) -> Any:
    """Get TorchSim calculator with UMA backend.

    Returns a TorchSimPotential instance specifically configured for UMA models.
    Requires both UMA and TorchSim dependencies.
    """
    from qme.backend_availability import get_backend_error_message, is_backend_available

    if not is_backend_available("torchsim_uma"):
        raise ImportError(get_backend_error_message("torchsim_uma"))
    try:
        from qme.potentials.torchsim_potential import (
            get_torchsim_uma_calculator as _get_torchsim_uma_calculator,
        )

        return _get_torchsim_uma_calculator(**kwargs)
    except ImportError as e:
        raise ImportError(
            f"Failed to import TorchSim UMA calculator: {e}. "
            f"TorchSim UMA requires both UMA and TorchSim to be available. "
            f"Try: pip install qme-ml[torchsim,uma]"
        )


# Orb backend - lazy loading


def get_orb_calculator(**kwargs: Any) -> Any:
    """Get Orb calculator.

    Returns an OrbPotential instance for universal forcefield calculations.
    Requires orb-models dependencies.
    """
    from qme.backend_availability import get_backend_error_message, is_backend_available

    if not is_backend_available("orb"):
        raise ImportError(get_backend_error_message("orb"))
    try:
        from qme.potentials.orb_potential import OrbPotential

        return OrbPotential(**kwargs)
    except ImportError as e:
        raise ImportError(
            f"Failed to import Orb backend: {e}. "
            f"Orb requires orb-models and PyTorch. "
            f"Note: orb-models is a large package and may have compatibility issues. "
            f"Try: pip install qme-ml[orb]"
        )


# TBLite backend - lazy loading


def get_tblite_calculator(**kwargs: Any) -> Any:
    """Get TBLite calculator.

    Returns a TBLitePotential instance for semi-empirical quantum chemistry calculations.
    Requires tblite dependencies.
    """
    from qme.backend_availability import get_backend_error_message, is_backend_available

    if not is_backend_available("tblite"):
        raise ImportError(get_backend_error_message("tblite"))
    try:
        from qme.potentials.tblite_potential import TBLitePotential

        return TBLitePotential(**kwargs)
    except ImportError as e:
        raise ImportError(
            f"Failed to import TBLite backend: {e}. "
            f"TBLite requires the tblite package. "
            f"Try: pip install qme-ml[tblite]"
        )
