"""Shared utilities for security tests."""

from __future__ import annotations

import pytest
from ase import Atoms

# Standardized list of path traversal attempts for testing
EVIL_PATHS = [
    "../../../etc/passwd",
    "..\\..\\windows\\system32\\evil",
    "../malicious.xyz",
    "subdir/../../etc/passwd",
    "/etc/passwd",
    "~/malicious",
    "$HOME/malicious",
    "model$PATH.jpt",
    "model;rm -rf.txt",
    "model`whoami`.txt",
    "model\x00.jpt",  # Null byte injection
]

# Standardized list of safe paths for testing
SAFE_PATHS = [
    "output.xyz",
    "subdir/output.xyz",
    "path/to/output.xyz",
    "model.jpt",
    "model-v2_final.jpt",
    "subdir/model.jpt",
]

# Windows-specific evil paths
WINDOWS_EVIL_PATHS = [
    "..\\..\\windows\\system32\\evil",
    "C:\\windows\\evil",
    "\\\\?\\C:\\evil",
]

# Additional suspicious patterns
SUSPICIOUS_PATTERNS = [
    "~/malicious",
    "$HOME/malicious",
    "model~backup",
    "model$backup",
]


@pytest.fixture
def test_atoms():
    """Standard test atoms fixture for security tests."""
    return Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])


# Null byte injection test cases
NULL_BYTE_PATHS = [
    "output\x00.xyz",
    "model\x00.jpt",
    "file\x00name.txt",
]
