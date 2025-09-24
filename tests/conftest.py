"""
Test configuration and fixtures for QME test suite.

This module provides pytest configuration and shared fixtures for testing
QME functionality across different backends.
"""

import pytest

from qme.dependencies import deps


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "backend_uma: mark test as requiring UMA backend",
    )
    config.addinivalue_line(
        "markers",
        "backend_so3lr: mark test as requiring SO3LR backend",
    )
    config.addinivalue_line(
        "markers",
        "backend_aimnet2: mark test as requiring AIMNET2 backend",
    )
    config.addinivalue_line(
        "markers",
        "backend_mock: mark test as using mock backend (always available)",
    )
    config.addinivalue_line(
        "markers",
        "requires_sella: mark test as requiring SELLA for transition states",
    )


def pytest_runtest_setup(item):
    """Skip tests based on backend availability."""
    # Check for backend-specific markers
    for marker in item.iter_markers():
        if marker.name == "backend_uma" and not deps.has("fairchem"):
            pytest.skip("UMA backend not available")
        elif marker.name == "backend_so3lr" and not deps.has("so3lr"):
            pytest.skip("SO3LR backend not available")
        elif marker.name == "backend_aimnet2" and not deps.has("aimnet2"):
            pytest.skip("AIMNET2 backend not available")
        elif marker.name == "requires_sella" and not deps.has("sella"):
            pytest.skip("SELLA not available")


# Define pytest collection hooks for better test organization
def pytest_collection_modifyitems(config, items):
    """Modify collected test items."""
    for item in items:
        # Add backend markers based on test file names
        if "test_backend_uma" in item.nodeid:
            item.add_marker(pytest.mark.backend_uma)
        elif "test_backend_so3lr" in item.nodeid:
            item.add_marker(pytest.mark.backend_so3lr)
        elif "test_backend_aimnet2" in item.nodeid:
            item.add_marker(pytest.mark.backend_aimnet2)
        elif "test_backend_mock" in item.nodeid:
            item.add_marker(pytest.mark.backend_mock)

        # Add SELLA marker for transition state tests
        if "transition" in item.name.lower() or "ts_" in item.name.lower():
            item.add_marker(pytest.mark.requires_sella)


# Shared test fixtures
@pytest.fixture
def h2_molecule():
    """Fixture providing H2 molecule for testing."""
    from ase import Atoms

    return Atoms("H2", positions=[[0, 0, 0], [1.5, 0, 0]])


@pytest.fixture
def water_molecule():
    """Fixture providing water molecule for testing."""
    from ase import Atoms

    return Atoms(
        "H2O",
        positions=[
            [0.0, 0.0, 0.0],  # O
            [1.0, 0.0, 0.0],  # H
            [0.0, 1.0, 0.0],  # H
        ],
    )


@pytest.fixture
def ethane_molecule():
    """Fixture providing ethane molecule for testing."""
    from ase import Atoms

    return Atoms(
        "C2H6",
        positions=[
            [0.0, 0.0, 0.0],  # C
            [1.54, 0.0, 0.0],  # C
            [-0.51, 1.02, 0.0],  # H
            [-0.51, -0.51, 0.88],  # H
            [-0.51, -0.51, -0.88],  # H
            [2.05, 1.02, 0.0],  # H
            [2.05, -0.51, 0.88],  # H
            [2.05, -0.51, -0.88],  # H
        ],
    )


@pytest.fixture(params=["mock"])
def available_backend(request):
    """Parametrized fixture yielding available backends."""
    backend = request.param

    # Always test mock backend
    if backend == "mock":
        return backend

    # Test other backends only if available
    if backend == "uma" and deps.has("fairchem"):
        return backend
    elif backend == "so3lr" and deps.has("so3lr"):
        return backend
    elif backend == "aimnet2" and deps.has("aimnet2"):
        return backend
    else:
        pytest.skip(f"{backend} backend not available")


@pytest.fixture
def available_calculator():
    """Fixture providing an available calculator."""
    import qme

    # Always return mock calculator as fallback
    return qme.get_mock_uma_calculator()
