"""Pytest configuration and shared fixtures for QME tests.

This file provides:
- Common test fixtures
- Import path setup
- Shared utilities
"""

import sys
from pathlib import Path

# Add the parent directory to Python path so tests can import qme modules
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

# Import test utilities after path setup
from tests.test_utils import (  # noqa: E402
    BackendTestRunner,
    BackendTestWarning,
    StandardTestAssertions,
    TestMoleculeFactory,
    backend_test_with_warnings,
    run_backend_test_with_warnings,
)

# Re-export for easy access in tests
__all__ = [
    "BackendTestRunner",
    "BackendTestWarning",
    "StandardTestAssertions",
    "TestMoleculeFactory",
    "backend_test_with_warnings",
    "run_backend_test_with_warnings",
]
