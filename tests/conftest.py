import sys
from pathlib import Path

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
    "backend_test_with_warnings",
    "create_backend_test_atoms",
    "parametrize_backends",
]
