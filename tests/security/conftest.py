"""Pytest configuration for security tests."""

from __future__ import annotations

import pytest
from ase import Atoms


@pytest.fixture
def test_atoms():
    """Standard test atoms fixture for security tests."""
    return Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
