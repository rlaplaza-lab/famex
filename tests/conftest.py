from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add the parent directory to Python path so tests can import qme modules
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

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
    parametrize_backends,
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
    "parametrize_backends",
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
def reactant_product_pair():
    """Pair of reactant and product structures for path testing."""
    reactant = TestMoleculeFactory.get_water_distorted()
    product = TestMoleculeFactory.get_water_distorted()
    pos = product.get_positions()
    pos[1, 0] += 0.2
    product.set_positions(pos)
    return reactant, product
