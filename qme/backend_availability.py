"""
Efficient backend availability checking for QME.

This module provides fast, dependency-based backend availability checking
that avoids expensive calculator instantiation while still catching most
compatibility issues.
"""

import importlib
from typing import Dict, List, Optional

from qme.dependencies import deps

# Backend name constants
BACKEND_MOCK = "mock"
BACKEND_AIMNET2 = "aimnet2"
BACKEND_UMA = "uma"
BACKEND_MACE = "mace"
BACKEND_SO3LR = "so3lr"
BACKEND_ORB = "orb"
BACKEND_TORCHSIM_MACE = "torchsim_mace"
BACKEND_TORCHSIM_UMA = "torchsim_uma"


def _check_package_conflict(
    package1: str, package2: str, conflict_packages: Dict[str, str]
) -> Optional[str]:
    """
    Check if two packages have known version conflicts.

    Args:
        package1: First package name
        package2: Second package name
        conflict_packages: Dict mapping package names to their conflicting versions

    Returns:
        Error message if conflict detected, None otherwise
    """
    # Known conflicts
    conflicts = {
        (
            BACKEND_MACE,
            "fairchem",
        ): "MACE 0.3.14 requires e3nn==0.4.4, but FairChem 2.7.0 requires e3nn>=0.5",
        (
            BACKEND_MACE,
            "torchsim",
        ): "MACE models with TorchSim affected by e3nn version conflicts",
    }

    conflict_key = tuple(sorted([package1, package2]))
    return conflicts.get(conflict_key)


def _get_package_version(package_name: str) -> Optional[str]:
    """Get version of an installed package."""
    try:
        module = importlib.import_module(package_name)
        return getattr(module, "__version__", "unknown")
    except ImportError:
        return None


def _check_e3nn_conflict() -> Optional[str]:
    """
    Check for e3nn version conflicts between MACE and FairChem.

    Returns:
        Error message if conflict detected, None otherwise
    """
    if not (deps.has(BACKEND_MACE) and deps.has("fairchem")):
        return None

    try:
        import e3nn

        e3nn_version = e3nn.__version__

        # MACE 0.3.14 was built with e3nn 0.4.4, FairChem needs e3nn >= 0.5
        if e3nn_version.startswith("0.5") or e3nn_version.startswith("0.6"):
            return (
                f"e3nn version conflict: MACE requires e3nn==0.4.4 but "
                f"e3nn {e3nn_version} is installed (required by FairChem)"
            )
    except ImportError:
        pass

    return None


def _check_torchsim_fairchem_conflict() -> Optional[str]:
    """
    Check for TorchSim-FairChem API compatibility.

    Returns:
        Error message if conflict detected, None otherwise
    """
    if not (deps.has("torch_sim") and deps.has("fairchem")):
        return None

    # Fast check: TorchSim expects load_config/update_config in fairchem.core.common.utils
    # These functions were removed in FairChem 2.x, so if we have FairChem 2.x, there's a conflict
    try:
        import fairchem.core

        version = getattr(fairchem.core, "__version__", "0.0.0")
        if version.startswith("2."):
            return (
                "TorchSim-FairChem API incompatibility: TorchSim expects "
                "load_config/update_config functions not available in FairChem 2.x"
            )
        # If it's not version 2.x, assume it's compatible (avoid expensive checks)
        return None
    except (ImportError, AttributeError):
        # If we can't determine the version, assume there's a conflict
        # (this is safer and avoids expensive imports)
        return (
            "TorchSim-FairChem API incompatibility: Cannot determine FairChem version, "
            "likely missing load_config/update_config functions"
        )


class BackendAvailabilityChecker:
    """Fast, dependency-based backend availability checker."""

    def __init__(self):
        self._cache: Dict[str, bool] = {}
        self._conflict_cache: Dict[str, Optional[str]] = {}

    def _check_basic_dependencies(self, backend: str) -> bool:
        """Check basic package dependencies for a backend."""
        requirements = {
            BACKEND_MOCK: [],
            BACKEND_AIMNET2: ["aimnet2"],  # Use backend name, deps.has() will handle the mapping
            BACKEND_UMA: ["fairchem", "torch"],
            BACKEND_SO3LR: ["so3lr"],
            BACKEND_MACE: ["mace", "torch"],
            BACKEND_ORB: ["orb_models", "torch"],
            BACKEND_TORCHSIM_MACE: ["torch_sim", "torch"],
            BACKEND_TORCHSIM_UMA: ["torch_sim", "torch", "fairchem"],
        }

        required = requirements.get(backend, [])
        return all(deps.has(pkg) for pkg in required)

    def _check_import_compatibility(self, backend: str) -> Optional[str]:
        """Check if backend modules can be imported without errors."""
        # For most backends, if basic dependencies are available and no conflicts
        # are detected, we can assume they'll import successfully.
        # Only do actual imports for backends with complex import chains.

        if backend in [BACKEND_AIMNET2, BACKEND_UMA, BACKEND_SO3LR]:
            # These have simple import chains, skip expensive imports
            return None

        # Only do actual imports for complex backends
        try:
            if backend == BACKEND_MACE:
                # Just check if the main module imports, don't instantiate
                import mace.calculators

                return None
            elif backend in [BACKEND_TORCHSIM_MACE, BACKEND_TORCHSIM_UMA]:
                # Check TorchSim imports
                import torch_sim

                return None
            return None  # No import error
        except ImportError as e:
            return f"Import error: {e}"

    def _check_known_conflicts(self, backend: str) -> Optional[str]:
        """Check for known package version conflicts."""
        if backend in self._conflict_cache:
            return self._conflict_cache[backend]

        conflict = None

        if backend in [BACKEND_MACE, BACKEND_TORCHSIM_MACE]:
            conflict = _check_e3nn_conflict()
        elif backend == BACKEND_TORCHSIM_UMA:
            conflict = _check_torchsim_fairchem_conflict()

        self._conflict_cache[backend] = conflict
        return conflict

    def is_backend_available(self, backend: str) -> bool:
        """
        Check if a backend is available using fast dependency and import checks.

        This avoids expensive calculator instantiation while catching most
        compatibility issues through dependency analysis.
        """
        if backend in self._cache:
            return self._cache[backend]

        # Always available
        if backend == BACKEND_MOCK:
            self._cache[backend] = True
            return True

        # Check basic dependencies first (fastest)
        if not self._check_basic_dependencies(backend):
            self._cache[backend] = False
            return False

        # Check for known conflicts (fast)
        if self._check_known_conflicts(backend):
            self._cache[backend] = False
            return False

        # Check import compatibility (medium speed)
        if self._check_import_compatibility(backend):
            self._cache[backend] = False
            return False

        # If all checks pass, consider it available
        self._cache[backend] = True
        return True

    def get_availability_reason(self, backend: str) -> str:
        """Get detailed reason why a backend is or isn't available."""
        if backend == BACKEND_MOCK:
            return "Always available"

        if not self._check_basic_dependencies(backend):
            requirements = {
                BACKEND_AIMNET2: ["torch", "torch-cluster"],
                BACKEND_UMA: ["fairchem-core", "torch"],
                BACKEND_SO3LR: ["so3lr"],
                BACKEND_MACE: ["mace-torch", "torch"],
                BACKEND_ORB: ["orb-models", "torch"],
                BACKEND_TORCHSIM_MACE: ["torch-sim-atomistic", "torch"],
                BACKEND_TORCHSIM_UMA: ["torch-sim-atomistic", "torch", "fairchem-core"],
            }
            required = requirements.get(backend, [])
            return f"Missing dependencies: {', '.join(required)}"

        conflict = self._check_known_conflicts(backend)
        if conflict:
            return f"Known conflict: {conflict}"

        import_error = self._check_import_compatibility(backend)
        if import_error:
            return f"Import issue: {import_error}"

        return "Available"

    def get_available_backends(self, include_mock: bool = True) -> List[str]:
        """Get list of all available backends."""
        all_backends = [
            BACKEND_MOCK,
            BACKEND_AIMNET2,
            BACKEND_UMA,
            BACKEND_SO3LR,
            BACKEND_MACE,
            BACKEND_ORB,
            BACKEND_TORCHSIM_MACE,
            BACKEND_TORCHSIM_UMA,
        ]
        if not include_mock:
            all_backends = all_backends[1:]  # Remove mock

        return [b for b in all_backends if self.is_backend_available(b)]

    def clear_cache(self):
        """Clear the availability cache (useful for testing)."""
        self._cache.clear()
        self._conflict_cache.clear()


# Global instance
_checker = BackendAvailabilityChecker()


# Convenience functions
def is_backend_available(backend: str) -> bool:
    """Check if a backend is available."""
    return _checker.is_backend_available(backend)


def get_availability_reason(backend: str) -> str:
    """Get reason why a backend is or isn't available."""
    return _checker.get_availability_reason(backend)


def get_available_backends(include_mock: bool = True) -> List[str]:
    """Get list of available backends."""
    return _checker.get_available_backends(include_mock)


def clear_availability_cache():
    """Clear the availability cache."""
    _checker.clear_cache()


def get_backend_error_message(backend: str) -> str:
    """Get a clear error message for why backend is unavailable.

    Args:
        backend: Backend name

    Returns:
        str: Human-readable error message with installation instructions
    """
    reason = get_availability_reason(backend)

    install_commands = {
        BACKEND_AIMNET2: "pip install qme-ml[aimnet2]",
        BACKEND_UMA: "pip install qme-ml[uma]",
        BACKEND_MACE: "pip install qme-ml[mace]",
        BACKEND_SO3LR: "pip install qme-ml[so3lr]",
        BACKEND_ORB: "pip install qme-ml[orb]",
        BACKEND_TORCHSIM_MACE: "pip install qme-ml[torchsim]",
        BACKEND_TORCHSIM_UMA: "pip install qme-ml[torchsim,uma]",
    }

    cmd = install_commands.get(backend, f"pip install qme-ml[{backend}]")

    return f"Backend '{backend}' is not available.\n" f"Reason: {reason}\n" f"Install with: {cmd}"


# Backend categorization constants
ALL_BACKENDS = [
    BACKEND_MOCK,
    BACKEND_AIMNET2,
    BACKEND_MACE,
    BACKEND_UMA,
    BACKEND_SO3LR,
    BACKEND_ORB,
    BACKEND_TORCHSIM_MACE,
    BACKEND_TORCHSIM_UMA,
]

ML_BACKENDS = [
    BACKEND_AIMNET2,
    BACKEND_MACE,
    BACKEND_UMA,
    BACKEND_SO3LR,
    BACKEND_ORB,
    BACKEND_TORCHSIM_MACE,
    BACKEND_TORCHSIM_UMA,
]

TORCHSIM_BACKENDS = [BACKEND_TORCHSIM_MACE, BACKEND_TORCHSIM_UMA]

REGULAR_BACKENDS = [BACKEND_MOCK, BACKEND_AIMNET2, BACKEND_MACE, BACKEND_UMA, BACKEND_SO3LR, BACKEND_ORB]


def get_available_ml_backends(include_torchsim: bool = True, verbose: bool = False) -> List[str]:
    """Get list of ML backends that are available (excludes mock)."""
    available = get_available_backends(include_mock=False)
    if not include_torchsim:
        available = [b for b in available if b not in TORCHSIM_BACKENDS]
    return available


def get_available_torchsim_backends(verbose: bool = False) -> List[str]:
    """Get list of TorchSim backends that are available."""
    available = []
    for backend in TORCHSIM_BACKENDS:
        if is_backend_available(backend):
            available.append(backend)
            if verbose:
                print(f"  ✅ {backend}")
        elif verbose:
            print(f"  ❌ {backend} (dependencies missing or incompatible)")
    return available


def get_backend_pairs() -> List[tuple[str, str]]:
    """Get pairs of (regular_backend, torchsim_backend) for comparison testing."""
    pairs = []

    # Check MACE pair
    if is_backend_available(BACKEND_MACE) and is_backend_available(BACKEND_TORCHSIM_MACE):
        pairs.append((BACKEND_MACE, BACKEND_TORCHSIM_MACE))

    # Check UMA pair
    if is_backend_available(BACKEND_UMA) and is_backend_available(BACKEND_TORCHSIM_UMA):
        pairs.append((BACKEND_UMA, BACKEND_TORCHSIM_UMA))

    return pairs


def require_backend(backend: str):
    """Decorator/function to require a specific backend for a test."""
    try:
        import pytest
    except ImportError:
        # If pytest not available, just check availability
        if not is_backend_available(backend):
            raise ImportError(f"Backend {backend} not available in this environment")
        return

    if not is_backend_available(backend):
        pytest.skip(f"Backend {backend} not available in this environment")


# Pre-computed lists for convenience
AVAILABLE_BACKENDS = get_available_backends()
AVAILABLE_ML_BACKENDS = get_available_ml_backends()
AVAILABLE_TORCHSIM_BACKENDS = get_available_torchsim_backends()
AVAILABLE_BACKEND_PAIRS = get_backend_pairs()
