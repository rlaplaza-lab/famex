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
    backend_test_with_warnings,
    create_backend_test_atoms,
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
    "backend_test_with_warnings",
    "create_backend_test_atoms",
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


@pytest.fixture
def ethylene_twisted_ts_guess():
    """Ethylene twisted TS guess (90-degree rotation around C=C bond)."""
    return TestMoleculeFactory.get_ethylene_twisted_ts_guess()


# ============================================================================
# Backend Fixtures
# ============================================================================


@pytest.fixture
def uma_backend():
    """UMA backend calculator fixture."""
    from famex.backends.availability import is_backend_available
    from famex.backends.constants import DEFAULT_UMA_MODEL

    if not is_backend_available("uma"):
        pytest.skip("UMA backend not available")

    import famex

    calc = famex.get_uma_calculator(model_name=DEFAULT_UMA_MODEL)
    calc.ensure_loaded()
    return calc


@pytest.fixture
def mace_backend():
    """MACE backend calculator fixture."""
    from famex.backends.availability import is_backend_available

    if not is_backend_available("mace"):
        pytest.skip("MACE backend not available")

    import famex

    calc = famex.get_mace_calculator(model_name="mace-omol-0")
    calc.ensure_loaded()
    return calc


@pytest.fixture
def mock_backend():
    """Mock backend calculator fixture."""
    import famex

    return famex.MockCalculator(backend="mock")


@pytest.fixture
def any_real_backend_explorer(request):
    """Fixture that provides an Explorer with any available real ML backend.

    Prefers backends that pass ``is_backend_available`` (uma, mace, aimnet2, pet),
    in that order. Skips the test if none can be constructed. Avoids returning a
    lazily-initialized UMA explorer when only another backend is installed.
    """
    from famex.backends.availability import is_backend_available
    from famex.core.explorer import Explorer
    from famex.utils.validation import BackendError

    # Preference order for GPU CI suites and local installs.
    candidates = ("uma", "mace", "aimnet2", "pet")

    def _create_explorer(atoms, skip_message="No real backend available"):
        """Create explorer with any available real backend.

        Args:
            atoms: Single Atoms object or list of Atoms objects
            skip_message: Message to use when skipping test
        """
        available = [name for name in candidates if is_backend_available(name)]
        if not available:
            pytest.skip(skip_message)

        last_error: Exception | None = None
        for backend in available:
            try:
                explorer = Explorer(atoms, backend=backend)
                # Force calculator construction so lazy UMA/fairchem failures
                # do not surface mid-test as unexpected ImportErrors.
                explorer.calculator_manager.create_and_attach_calculator(explorer.atoms_list[0])
                return explorer
            except (ImportError, BackendError, RuntimeError, OSError, ValueError) as exc:
                last_error = exc
                continue

        detail = f" ({last_error})" if last_error is not None else ""
        pytest.skip(f"{skip_message}{detail}")

    return _create_explorer


@pytest.fixture
def atoms_with_mock_calc(water_molecule, mock_backend):
    """Pre-configured atoms with mock calculator."""
    atoms = water_molecule.copy()
    atoms.calc = mock_backend
    return atoms


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
def h2_molecule_with_mock(h2_molecule, mock_backend):
    """H2 molecule with stretched bond and mock calculator."""
    atoms = h2_molecule.copy()
    atoms.calc = mock_backend
    return atoms


@pytest.fixture
def standard_tolerances():
    """Return standard tolerance values for tests."""
    from tests.test_constants import LOOSE_TOL, MODERATE_TOL, TIGHT_TOL

    return {
        "tight": TIGHT_TOL,
        "moderate": MODERATE_TOL,
        "loose": LOOSE_TOL,
    }


@pytest.fixture
def test_config():
    """Standard test configuration dictionary."""
    from tests.test_constants import DEFAULT_DELTA, DEFAULT_FMAX, DEFAULT_STEPS, QUICK_STEPS

    return {
        "delta": DEFAULT_DELTA,
        "fmax": DEFAULT_FMAX,
        "steps": DEFAULT_STEPS,
        "quick_steps": QUICK_STEPS,
    }


@pytest.fixture
def clear_famex_logger():
    """Clear FAMEX logger handlers before test, restore after."""
    import logging

    logger = logging.getLogger("famex")
    original_handlers = logger.handlers[:]
    original_level = logger.level

    # Clear handlers
    logger.handlers.clear()

    yield logger

    # Restore handlers and level
    logger.handlers[:] = original_handlers
    logger.level = original_level
