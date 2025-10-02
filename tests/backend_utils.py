"""
Shared utilities for backend testing across all test modules.

This module provides a centralized way to handle backend availability
and ensures consistent behavior across all test files.
"""

import qme

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

# TorchSim-specific backends
TORCHSIM_BACKENDS = ["torchsim_mace", "torchsim_uma"]

# Non-TorchSim backends
REGULAR_BACKENDS = ["mock", "aimnet2", "mace", "uma", "so3lr"]


def get_available_backends(include_torchsim: bool = True) -> list[str]:
    """
    Get list of backends that are actually available in the current environment.

    Args:
        include_torchsim: Whether to include TorchSim backends in the check

    Returns:
        List of available backend names
    """
    available = []
    backends_to_check = ALL_BACKENDS if include_torchsim else REGULAR_BACKENDS

    for backend in backends_to_check:
        if qme.calculator_registry.is_backend_available(backend):
            available.append(backend)

    return available


def get_available_torchsim_backends() -> list[str]:
    """Get list of TorchSim backends that are available."""
    available = []
    for backend in TORCHSIM_BACKENDS:
        if qme.calculator_registry.is_backend_available(backend):
            available.append(backend)
    return available


def get_backend_pairs() -> list[tuple[str, str]]:
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
    import pytest

    if not is_backend_available(backend):
        pytest.skip(f"Backend {backend} not available in this environment")


def require_any_backend(backends: list[str]):
    """
    Require that at least one of the specified backends is available.

    Usage:
        require_any_backend(["mace", "uma"])
    """
    import pytest

    available = [b for b in backends if is_backend_available(b)]
    if not available:
        pytest.skip(f"None of the required backends are available: {backends}")

    return available


# Pre-computed lists for use in test parametrization
# These are computed at import time for efficiency
AVAILABLE_BACKENDS = get_available_backends()
AVAILABLE_TORCHSIM_BACKENDS = get_available_torchsim_backends()
AVAILABLE_BACKEND_PAIRS = get_backend_pairs()

# Backwards compatibility aliases
BACKENDS = AVAILABLE_BACKENDS  # For existing code that uses this name
