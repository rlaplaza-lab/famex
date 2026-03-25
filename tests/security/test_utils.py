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


def create_path_traversal_test_pattern(test_func, evil_paths=None, safe_paths=None):
    """Test path traversal patterns.

    Args:
        test_func: Function that takes a path and should raise ValueError for evil paths
        evil_paths: List of evil paths to test (defaults to EVIL_PATHS)
        safe_paths: List of safe paths to test (defaults to SAFE_PATHS)

    Returns
    -------
    pytest.mark.parametrize
        pytest.mark.parametrize marker for testing path traversal patterns.

    """
    import pytest

    if evil_paths is None:
        evil_paths = EVIL_PATHS
    if safe_paths is None:
        safe_paths = SAFE_PATHS

    # This is a helper that can be used to create parametrized tests
    # Example usage:
    # @pytest.mark.parametrize("path", EVIL_PATHS)
    # def test_traversal(path):
    #     with pytest.raises(ValueError):
    #         test_func(path)
    return pytest.mark.parametrize("path", evil_paths + safe_paths)


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
