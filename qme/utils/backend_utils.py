"""Shared utilities for backend handling in QME.

This module provides convenience wrappers around the core backend availability
system (qme.backend_availability) with additional logging and user-friendly
interfaces. This module is primarily used by examples and tests.

For core backend availability logic, see qme.backend_availability.
"""

import sys

try:
    import qme
except ImportError as e:
    msg = (
        f"Error importing QME: {e}. "
        "Make sure QME is properly installed or you're in the QME package directory."
    )
    raise ImportError(
        msg,
    ) from e

from qme.backend_availability import (
    ALL_BACKENDS,
    ML_BACKENDS,
    TORCHSIM_BACKENDS,
)
from qme.logging_utils import get_qme_logger

logger = get_qme_logger(__name__)


def get_available_backends(
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
    # Use the efficient backend availability checker directly
    from qme.backend_availability import get_availability_reason
    from qme.backend_availability import get_available_backends as _get_available

    available = _get_available(include_mock=include_mock)

    if not include_torchsim:
        available = [b for b in available if b not in TORCHSIM_BACKENDS]

    if verbose:
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
    return get_available_backends(
        include_mock=False,
        include_torchsim=include_torchsim,
        verbose=verbose,
    )


def get_available_torchsim_backends(verbose: bool = False) -> list[str]:
    """Get list of TorchSim backends that are available."""
    available = []
    for backend in TORCHSIM_BACKENDS:
        if qme.calculator_registry.is_backend_available(backend):
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

    for backend in requested_backends:
        if qme.calculator_registry.is_backend_available(backend):
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
            if qme.calculator_registry.is_backend_available(backend):
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


def is_backend_available(backend: str) -> bool:
    """Check if a specific backend is available.

    This is a convenience wrapper around the centralized availability check.
    """
    return qme.calculator_registry.is_backend_available(backend)


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


def require_any_backend(backends: list[str]):
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
    if qme.calculator_registry.is_backend_available(
        "mace",
    ) and qme.calculator_registry.is_backend_available("torchsim_mace"):
        pairs.append(("mace", "torchsim_mace"))

    # Check UMA pair
    if qme.calculator_registry.is_backend_available(
        "uma",
    ) and qme.calculator_registry.is_backend_available("torchsim_uma"):
        pairs.append(("uma", "torchsim_uma"))

    return pairs
