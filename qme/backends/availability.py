"""Backend availability checking for QME.

This module provides fast, dependency-based backend availability checking
that avoids expensive calculator instantiation while still catching most
compatibility issues. It combines the core availability logic with
convenience functions for logging and user-friendly interfaces.
"""

from __future__ import annotations

import sys
from typing import Any

from qme.backends.constants import (
    ALL_BACKENDS,
    BACKEND_AIMNET2,
    BACKEND_MACE,
    BACKEND_MOCK,
    BACKEND_SO3LR,
    BACKEND_TORCHSIM_MACE,
    BACKEND_TORCHSIM_UMA,
    BACKEND_UMA,
    ML_BACKENDS,
    TORCHSIM_BACKENDS,
)
from qme.backends.dependencies import deps


# Import logger only when needed to avoid circular imports
def _get_logger() -> Any:
    try:
        from qme.utils.logging import get_qme_logger

        return get_qme_logger(__name__)
    except ImportError:
        # Fallback for when logging isn't available yet
        import logging

        return logging.getLogger(__name__)


def _check_e3nn_conflict() -> str | None:
    """Check for e3nn version conflicts between MACE and FairChem.

    Returns:
        Error message if conflict detected, None otherwise

    """
    if not (deps.has(BACKEND_MACE) and deps.has("fairchem")):
        return None

    try:
        import e3nn

        e3nn_version = e3nn.__version__

        # MACE 0.3.14 was built with e3nn 0.4.4, FairChem needs e3nn >= 0.5
        if e3nn_version.startswith(("0.5", "0.6")):
            return (
                f"e3nn version conflict: MACE requires e3nn==0.4.4 but "
                f"e3nn {e3nn_version} is installed (required by FairChem)"
            )
    except ImportError:
        pass

    return None


def _check_torchsim_fairchem_conflict() -> str | None:
    """Check for TorchSim-FairChem API compatibility.

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

    def __init__(self) -> None:
        """Initialize the availability checker with empty caches."""
        self._cache: dict[str, bool] = {}
        self._conflict_cache: dict[str, str | None] = {}

        # Centralized requirements mapping
        self._requirements = {
            BACKEND_MOCK: [],
            BACKEND_AIMNET2: ["aimnet2"],  # Use backend name, deps.has() will handle the mapping
            BACKEND_UMA: ["fairchem", "torch"],
            BACKEND_SO3LR: ["so3lr"],
            BACKEND_MACE: ["mace", "torch"],
            "orb": ["orb_models", "torch"],
            "tblite": ["tblite"],
            BACKEND_TORCHSIM_MACE: ["torch_sim", "torch"],
            BACKEND_TORCHSIM_UMA: ["torch_sim", "torch", "fairchem"],
        }

        # Human-readable package names for error messages
        self._package_names = {
            BACKEND_AIMNET2: ["torch", "torch-cluster"],
            BACKEND_UMA: ["fairchem-core", "torch"],
            BACKEND_SO3LR: ["so3lr"],
            BACKEND_MACE: ["mace-torch", "torch"],
            "orb": ["orb-models", "torch"],
            "tblite": ["tblite"],
            BACKEND_TORCHSIM_MACE: ["torch-sim-atomistic", "torch"],
            BACKEND_TORCHSIM_UMA: ["torch-sim-atomistic", "torch", "fairchem-core"],
        }

    def _check_basic_dependencies(self, backend: str) -> bool:
        """Check basic package dependencies for a backend."""
        required = self._requirements.get(backend, [])
        return all(deps.has(pkg) for pkg in required)

    def _check_import_compatibility(self, backend: str) -> str | None:
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
                import mace.calculators  # noqa: F401

                return None
            if backend in [BACKEND_TORCHSIM_MACE, BACKEND_TORCHSIM_UMA]:
                # Check TorchSim imports
                import torch_sim  # noqa: F401

                return None
            return None  # No import error
        except ImportError as e:
            return f"Import error: {e}"

    def _check_known_conflicts(self, backend: str) -> str | None:
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
        """Check if a backend is available using fast dependency and import checks.

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
            required = self._package_names.get(backend, [])
            return f"Missing dependencies: {', '.join(required)}"

        conflict = self._check_known_conflicts(backend)
        if conflict:
            return f"Known conflict: {conflict}"

        import_error = self._check_import_compatibility(backend)
        if import_error:
            return f"Import issue: {import_error}"

        return "Available"

    def get_available_backends(self, include_mock: bool = True) -> list[str]:
        """Get list of all available backends."""
        all_backends = ALL_BACKENDS.copy()
        if not include_mock:
            all_backends = all_backends[1:]  # Remove mock

        return [b for b in all_backends if self.is_backend_available(b)]

    def clear_cache(self) -> None:
        """Clear the availability cache (useful for testing)."""
        self._cache.clear()
        self._conflict_cache.clear()


# Global instance
_checker = BackendAvailabilityChecker()


# Core convenience functions
def is_backend_available(backend: str) -> bool:
    """Check if a backend is available."""
    return _checker.is_backend_available(backend)


def get_availability_reason(backend: str) -> str:
    """Get reason why a backend is or isn't available."""
    return _checker.get_availability_reason(backend)


def get_available_backends(include_mock: bool = True) -> list[str]:
    """Get list of available backends."""
    return _checker.get_available_backends(include_mock)


def clear_availability_cache() -> None:
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
        BACKEND_AIMNET2: "pip install torch torch-cluster",
        BACKEND_UMA: "pip install fairchem-core",
        BACKEND_MACE: "pip install mace-torch",
        BACKEND_SO3LR: "pip install so3lr",
        "orb": "pip install orb-models",
        "tblite": "pip install tblite",
        BACKEND_TORCHSIM_MACE: "pip install torch-sim-atomistic",
        BACKEND_TORCHSIM_UMA: "pip install torch-sim-atomistic",
    }

    cmd = install_commands.get(backend, f"pip install {backend}")

    return f"Backend '{backend}' is not available.\nReason: {reason}\nInstall with: {cmd}"


# Extended convenience functions with logging (from utils/backend_utils.py)
def get_available_backends_with_logging(
    include_mock: bool = True,
    include_torchsim: bool = True,
    verbose: bool = False,
) -> list[str]:
    """Get list of backends that are actually available in the current environment.

    Args:
        include_mock: Whether to include the mock backend
        include_torchsim: Whether to include TorchSim backends
        verbose: Whether to print availability status for each backend

    Returns:
        List of available backend names

    """
    available = get_available_backends(include_mock=include_mock)

    if not include_torchsim:
        available = [b for b in available if b not in TORCHSIM_BACKENDS]

    if verbose:
        logger = _get_logger()
        backends_to_check = ALL_BACKENDS if include_mock else ML_BACKENDS
        if not include_torchsim:
            backends_to_check = [b for b in backends_to_check if b not in TORCHSIM_BACKENDS]

        for backend in backends_to_check:
            is_available = backend in available
            if is_available:
                logger.info("  ✅ %s", backend)
            else:
                reason = get_availability_reason(backend)
                logger.info("  ❌ %s (%s)", backend, reason)

    return available


def get_available_ml_backends(include_torchsim: bool = True, verbose: bool = False) -> list[str]:
    """Get list of ML backends that are available (excludes mock)."""
    return get_available_backends_with_logging(
        include_mock=False,
        include_torchsim=include_torchsim,
        verbose=verbose,
    )


def get_available_torchsim_backends(verbose: bool = False) -> list[str]:
    """Get list of TorchSim backends that are available."""
    available = []
    logger = _get_logger()

    for backend in TORCHSIM_BACKENDS:
        if is_backend_available(backend):
            available.append(backend)
            if verbose:
                logger.info("  ✅ %s", backend)
        elif verbose:
            logger.info("  ❌ %s (dependencies missing or incompatible)", backend)
    return available


def filter_available_backends(requested_backends: list[str], verbose: bool = False) -> list[str]:
    """Filter a list of requested backends to only include those that are available.

    Args:
        requested_backends: List of backend names to check
        verbose: Whether to print status messages

    Returns:
        List of available backends from the requested list

    """
    available = []
    unavailable = []
    logger = _get_logger()

    for backend in requested_backends:
        if is_backend_available(backend):
            available.append(backend)
        else:
            unavailable.append(backend)

    if verbose and unavailable:
        logger.info("⚠️  Unavailable backends (skipped): %s", unavailable)

    if verbose and available:
        logger.info("✅ Available backends: %s", available)

    return available


def validate_backends(requested_backends: list[str]) -> tuple[list[str], list[str]]:
    """Validate a list of requested backends.

    Args:
        requested_backends: List of backend names to validate

    Returns:
        Tuple of (available_backends, invalid_backends)

    """
    available = []
    invalid = []

    for backend in requested_backends:
        if backend in ALL_BACKENDS:
            if is_backend_available(backend):
                available.append(backend)
            # Note: valid but unavailable backends are not considered "invalid"
        else:
            invalid.append(backend)

    return available, invalid


def require_ml_backends(min_count: int = 1) -> list[str]:
    """Require that at least a minimum number of ML backends are available.

    Args:
        min_count: Minimum number of ML backends required

    Returns:
        List of available ML backends

    Raises:
        SystemExit: If insufficient ML backends are available

    """
    available = get_available_ml_backends()
    logger = _get_logger()

    if len(available) < min_count:
        logger.warning(
            f"❌ Need at least {min_count} ML backend(s), but only {len(available)} available.",
        )
        logger.info("Please install additional ML backends:")
        logger.info("  - UMA: pip install fairchem-core")
        logger.info("  - MACE: pip install mace-torch")
        logger.info("  - AIMNet2: pip install aimnet2")
        logger.info("  - SO3LR: pip install so3lr")
        logger.info("  - TorchSim: pip install torch-sim-atomistic (Python 3.11+)")
        sys.exit(1)

    return available


def print_backend_summary(backends: list[str], title: str = "Backend Summary") -> None:
    """Print a formatted summary of backend availability."""
    logger = _get_logger()
    logger.info("\n%s", title)
    logger.info("=" * len(title))

    if not backends:
        logger.info("No backends available!")
        return

    # Categorize backends
    mock_backends = [b for b in backends if b == "mock"]
    ml_backends = [b for b in backends if b in ML_BACKENDS and b not in TORCHSIM_BACKENDS]
    torchsim_backends = [b for b in backends if b in TORCHSIM_BACKENDS]

    if mock_backends:
        logger.info("Mock: %s", ", ".join(mock_backends))
    if ml_backends:
        logger.info("ML: %s", ", ".join(ml_backends))
    if torchsim_backends:
        logger.info("TorchSim: %s", ", ".join(torchsim_backends))

    logger.info("Total: %d backend(s)", len(backends))


def require_backend(backend: str) -> None:
    """Decorator/function to require a specific backend for a test.

    Usage:
        @require_backend("mace")
        def test_something():
            pass

    Or:
        def test_something():
            require_backend("mace")
            # test code here
    """
    try:
        import pytest
    except ImportError:
        # If pytest not available, just check availability
        if not is_backend_available(backend):
            msg = f"Backend {backend} not available in this environment"
            raise ImportError(msg)
        return

    if not is_backend_available(backend):
        pytest.skip(f"Backend {backend} not available in this environment")


def require_any_backend(backends: list[str]) -> list[str] | None:
    """Require that at least one of the specified backends is available.

    Usage:
        require_any_backend(["mace", "uma"])
    """
    try:
        import pytest
    except ImportError:
        # If pytest not available, just check availability
        available = [b for b in backends if is_backend_available(b)]
        if not available:
            msg = f"None of the required backends are available: {backends}"
            raise ImportError(msg)
        return available

    available = [b for b in backends if is_backend_available(b)]
    if not available:
        pytest.skip(f"None of the required backends are available: {backends}")

    return available


def get_backend_pairs() -> list[tuple[str, str]]:
    """Get pairs of (regular_backend, torchsim_backend) for comparison testing.

    Returns:
        List of tuples like [("mace", "torchsim_mace"), ("uma", "torchsim_uma")]
        Only includes pairs where both backends are available.

    """
    pairs = []

    # Check MACE pair
    if is_backend_available(BACKEND_MACE) and is_backend_available(BACKEND_TORCHSIM_MACE):
        pairs.append((BACKEND_MACE, BACKEND_TORCHSIM_MACE))

    # Check UMA pair
    if is_backend_available(BACKEND_UMA) and is_backend_available(BACKEND_TORCHSIM_UMA):
        pairs.append((BACKEND_UMA, BACKEND_TORCHSIM_UMA))

    return pairs
