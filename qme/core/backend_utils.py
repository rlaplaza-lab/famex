"""
Shared utilities for backend handling in QME.

This module provides a centralized way to handle backend availability
and ensures consistent behavior across examples, tests, and other modules.
"""

from typing import List, Optional

try:
    import qme
except ImportError as e:
    raise ImportError(
        f"Error importing QME: {e}. "
        "Make sure QME is properly installed or you're in the QME package directory."
    )


# All possible backends that QME supports
ALL_BACKENDS = [
    "mock",
    "aimnet2",
    "mace",
    "uma",
    "so3lr",
    "torchsim_mace",
    "torchsim_uma",
]

# ML backends (excluding mock)
ML_BACKENDS = ["aimnet2", "mace", "uma", "so3lr", "torchsim_mace", "torchsim_uma"]

# TorchSim-specific backends
TORCHSIM_BACKENDS = ["torchsim_mace", "torchsim_uma"]

# Non-TorchSim backends
REGULAR_BACKENDS = ["mock", "aimnet2", "mace", "uma", "so3lr"]


def get_available_backends(
    include_mock: bool = True, include_torchsim: bool = True, verbose: bool = False
) -> List[str]:
    """
    Get list of backends that are actually available in the current environment.

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
            backends_to_check = [
                b for b in backends_to_check if b not in TORCHSIM_BACKENDS
            ]

        for backend in backends_to_check:
            is_available = backend in available
            if is_available:
                print(f"  ✅ {backend}")
            else:
                reason = get_availability_reason(backend)
                print(f"  ❌ {backend} ({reason})")

    return available


def get_available_ml_backends(
    include_torchsim: bool = True, verbose: bool = False
) -> List[str]:
    """Get list of ML backends that are available (excludes mock)."""
    return get_available_backends(
        include_mock=False, include_torchsim=include_torchsim, verbose=verbose
    )


def get_available_torchsim_backends(verbose: bool = False) -> List[str]:
    """Get list of TorchSim backends that are available."""
    available = []
    for backend in TORCHSIM_BACKENDS:
        if qme.calculator_registry.is_backend_available(backend):
            available.append(backend)
            if verbose:
                print(f"  ✅ {backend}")
        elif verbose:
            print(f"  ❌ {backend} (dependencies missing or incompatible)")
    return available


def filter_available_backends(
    requested_backends: List[str], verbose: bool = False
) -> List[str]:
    """
    Filter a list of requested backends to only include those that are available.

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
        print(f"⚠️  Unavailable backends (skipped): {unavailable}")

    if verbose and available:
        print(f"✅ Available backends: {available}")

    return available


def validate_backends(requested_backends: List[str]) -> tuple[List[str], List[str]]:
    """
    Validate a list of requested backends.

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


def require_ml_backends(min_count: int = 1) -> List[str]:
    """
    Require that at least a minimum number of ML backends are available.

    Args:
        min_count: Minimum number of ML backends required

    Returns:
        List of available ML backends

    Raises:
        SystemExit: If insufficient ML backends are available
    """
    available = get_available_ml_backends()

    if len(available) < min_count:
        print(
            f"❌ Need at least {min_count} ML backend(s), but only {len(available)} available."
        )
        print("Please install additional ML backends:")
        print("  - UMA: pip install fairchem-core")
        print("  - MACE: pip install mace-torch")
        print("  - AIMNet2: pip install aimnet2")
        print("  - SO3LR: pip install so3lr")
        print("  - TorchSim: pip install torch-sim-atomistic (Python 3.11+)")
        import sys

        sys.exit(1)

    return available


def print_backend_summary(backends: List[str], title: str = "Backend Summary"):
    """Print a formatted summary of backend availability."""
    print(f"\n{title}")
    print("=" * len(title))

    if not backends:
        print("No backends available!")
        return

    # Categorize backends
    mock_backends = [b for b in backends if b == "mock"]
    ml_backends = [
        b for b in backends if b in ML_BACKENDS and b not in TORCHSIM_BACKENDS
    ]
    torchsim_backends = [b for b in backends if b in TORCHSIM_BACKENDS]

    if mock_backends:
        print(f"Mock: {', '.join(mock_backends)}")
    if ml_backends:
        print(f"ML: {', '.join(ml_backends)}")
    if torchsim_backends:
        print(f"TorchSim: {', '.join(torchsim_backends)}")

    print(f"Total: {len(backends)} backend(s)")


def is_backend_available(backend: str) -> bool:
    """
    Check if a specific backend is available.

    This is a convenience wrapper around the centralized availability check.
    """
    return qme.calculator_registry.is_backend_available(backend)


def require_backend(backend: str):
    """
    Decorator/function to require a specific backend for a test.

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
            raise ImportError(f"Backend {backend} not available in this environment")
        return

    if not is_backend_available(backend):
        pytest.skip(f"Backend {backend} not available in this environment")


def require_any_backend(backends: List[str]):
    """
    Require that at least one of the specified backends is available.

    Usage:
        require_any_backend(["mace", "uma"])
    """
    try:
        import pytest
    except ImportError:
        # If pytest not available, just check availability
        available = [b for b in backends if is_backend_available(b)]
        if not available:
            raise ImportError(
                f"None of the required backends are available: {backends}"
            )
        return available

    available = [b for b in backends if is_backend_available(b)]
    if not available:
        pytest.skip(f"None of the required backends are available: {backends}")

    return available


def get_backend_pairs() -> List[tuple[str, str]]:
    """
    Get pairs of (regular_backend, torchsim_backend) for comparison testing.

    Returns:
        List of tuples like [("mace", "torchsim_mace"), ("uma", "torchsim_uma")]
        Only includes pairs where both backends are available.
    """
    pairs = []

    # Check MACE pair
    if qme.calculator_registry.is_backend_available(
        "mace"
    ) and qme.calculator_registry.is_backend_available("torchsim_mace"):
        pairs.append(("mace", "torchsim_mace"))

    # Check UMA pair
    if qme.calculator_registry.is_backend_available(
        "uma"
    ) and qme.calculator_registry.is_backend_available("torchsim_uma"):
        pairs.append(("uma", "torchsim_uma"))

    return pairs


# Pre-computed lists for convenience (computed at import time)
AVAILABLE_BACKENDS = get_available_backends()
AVAILABLE_ML_BACKENDS = get_available_ml_backends()
AVAILABLE_TORCHSIM_BACKENDS = get_available_torchsim_backends()
AVAILABLE_BACKEND_PAIRS = get_backend_pairs()

