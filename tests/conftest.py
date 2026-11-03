from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add the parent directory to Python path so tests can import famex modules
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))


# ============================================================================
# Pytest Marker Registration
# ============================================================================


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers",
        "requires_backend(name): mark test as requiring a specific backend (e.g., 'uma', 'mace')",
    )


# Import test utilities after path setup
from tests.test_utils import (  # noqa: E402
    BackendTestRunner,
    BackendTestWarning,
    HarmonicCalculator,
    NoisyCalculator,
    StandardTestAssertions,
    TestMoleculeFactory,
    assert_backend_calculator,
    assert_error_contains,
    handle_backend_errors,
    parametrize_backends,
    requires_backend,
)

# Re-export for easy access in tests
__all__ = [
    "BackendTestRunner",
    "BackendTestWarning",
    "HarmonicCalculator",
    "NoisyCalculator",
    "StandardTestAssertions",
    "TestMoleculeFactory",
    "assert_backend_calculator",
    "assert_error_contains",
    "handle_backend_errors",
    "parametrize_backends",
    "requires_backend",
]

# ============================================================================
# Common Test Fixtures
# ============================================================================


@pytest.fixture
def water_molecule():
    """Water molecule with distorted geometry."""
    return TestMoleculeFactory.get_water_distorted()


@pytest.fixture
def h2_molecule():
    """H2 molecule with stretched bond."""
    return TestMoleculeFactory.get_h2_stretched()


@pytest.fixture
def methane_molecule():
    """Methane molecule with realistic equilibrium tetrahedral geometry (C-H ~1.087 Å)."""
    return TestMoleculeFactory.get_methane_distorted()


@pytest.fixture
def h2_equilibrium_molecule():
    """H2 molecule at equilibrium geometry (bond length ~0.74 Å)."""
    return TestMoleculeFactory.get_h2_equilibrium()


@pytest.fixture
def h2o_molecule():
    """Water molecule at equilibrium geometry."""
    return TestMoleculeFactory.get_h2o_equilibrium()


@pytest.fixture
def harmonic_atoms():
    """Return simple H2 molecule for harmonic oscillator tests."""
    from ase import Atoms

    return Atoms(symbols="HH", positions=[[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]])


@pytest.fixture
def reactant_product_pair():
    """Pair of reactant and product structures for path testing."""
    reactant = TestMoleculeFactory.get_water_distorted()
    product = TestMoleculeFactory.get_water_distorted()
    pos = product.get_positions()
    pos[1, 0] += 0.2
    product.set_positions(pos)
    return reactant, product


@pytest.fixture
def water_dissociation_ts_guess():
    """Water dissociation TS guess (H2O -> H + OH)."""
    return TestMoleculeFactory.get_water_dissociation_ts_guess()


# ============================================================================
# Backend Fixtures
# ============================================================================


@pytest.fixture
def mock_backend():
    """Mock backend calculator fixture."""
    import famex

    return famex.MockCalculator(backend="mock")


# ============================================================================
# Helper Fixtures for Common Test Patterns
# ============================================================================


@pytest.fixture
def h2o_molecule_with_mock(h2o_molecule, mock_backend):
    """H2O molecule at equilibrium geometry with mock calculator."""
    atoms = h2o_molecule.copy()
    atoms.calc = mock_backend
    return atoms


@pytest.fixture
def h2o_molecule_perturbed_with_mock(h2o_molecule, mock_backend):
    """Perturbed H2O molecule with mock calculator (standard test setup)."""
    atoms = TestMoleculeFactory.get_perturbed_molecule(h2o_molecule, seed=42, magnitude=0.05)
    atoms.calc = mock_backend
    return atoms


@pytest.fixture
def water_molecule_with_mock(water_molecule, mock_backend):
    """Water molecule with distorted geometry and mock calculator."""
    atoms = water_molecule.copy()
    atoms.calc = mock_backend
    return atoms


@pytest.fixture
def clear_famex_logger():
    """Clear FAMEX logger handlers before test, restore after."""
    import logging

    import famex.utils.logging as famex_logging

    logger = logging.getLogger("famex")
    original_handlers = logger.handlers[:]
    original_level = logger.level
    original_propagate = logger.propagate
    original_configured = famex_logging._famex_logging_configured
    original_famex_level = famex_logging._famex_log_level

    logger.handlers.clear()
    famex_logging._famex_logging_configured = False

    yield logger

    logger.handlers[:] = original_handlers
    logger.level = original_level
    logger.propagate = original_propagate
    famex_logging._famex_logging_configured = original_configured
    famex_logging._famex_log_level = original_famex_level
