"""Pytest configuration for QME tests."""

from pathlib import Path

import numpy as np
import pytest

# Configure numpy for consistent test results
np.random.seed(42)


@pytest.fixture
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def tolerance():
    """Default numerical tolerance for comparisons."""
    return 1e-6
