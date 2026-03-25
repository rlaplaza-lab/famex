"""Path security utilities for QME.

This module provides security utilities to prevent path traversal attacks
and ensure file operations are restricted to safe directories.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


class PathSecurityError(ValueError):
    """Raised when an unsafe path is detected."""


def sanitize_filename(filename: str, allow_path_sep: bool = False) -> str:
    """Sanitize filename by removing all path traversal components.

    This is the most aggressive sanitization - strips ALL directory components
    and returns only the base filename with safe characters.

    Args:
        filename: Input filename (may contain path components)
        allow_path_sep: If False, removes all path separators (default: False)

    Returns
    -------
        Safe filename with only basename and safe characters

    Raises
    ------
        PathSecurityError: If filename is empty after sanitization
    """
    # First, get just the basename (removes all directory components)
    safe = os.path.basename(filename)

    # Remove leading/trailing dots and spaces
    safe = safe.strip(". ")

    # Remove any remaining .. sequences (paranoid)
    safe = safe.replace("..", "_")

    # Remove or replace unsafe characters
    # Allow: alphanumeric, dash, underscore, dot
    if not allow_path_sep:
        safe = re.sub(r"[^\w\-.]", "_", safe)

    # Remove leading dots again after character replacement (hidden files can be dangerous)
    safe = safe.lstrip(".")

    # Ensure we have something left
    if not safe or safe.isspace():
        raise PathSecurityError(f"Filename '{filename}' contains no safe characters")

    return safe


def validate_safe_path(
    path: Path | str,
    base_dir: Path | str | None = None,
    must_exist: bool = False,
    allow_absolute: bool = False,
) -> Path:
    """Validate that a path is safe and within allowed boundaries.

    Performs comprehensive security checks:
    - Resolves symlinks and relative paths
    - Checks for path traversal attempts
    - Ensures path is within base_dir if provided
    - Validates against various bypass techniques

    Args:
        path: Path to validate
        base_dir: Base directory that path must be within (optional)
        must_exist: If True, path must exist (default: False)
        allow_absolute: If True, allows absolute paths (default: False)

    Returns
    -------
        Validated Path object (absolute, resolved)

    Raises
    ------
        PathSecurityError: If path fails security checks
    """
    try:
        path_obj = Path(path)
    except (TypeError, ValueError) as e:
        raise PathSecurityError(f"Invalid path: {e}") from e

    # Convert to string for validation checks
    path_str = str(path)

    # Check for null bytes (can bypass security in C libraries)
    if "\x00" in path_str:
        raise PathSecurityError("Path contains null bytes")

    # If no base_dir is specified and absolute paths not allowed, check for absolute
    # Otherwise, wait until after resolution to check
    if not allow_absolute and not base_dir and path_obj.is_absolute():
        raise PathSecurityError(f"Absolute paths not allowed: {path_str}")

    # Resolve the path (follows symlinks, resolves ..)
    try:
        if base_dir:
            # Resolve relative to base_dir
            base_path = Path(base_dir).resolve()
            if path_obj.is_absolute():
                resolved = path_obj.resolve()
            else:
                resolved = (base_path / path_obj).resolve()
        else:
            resolved = path_obj.resolve()
    except (OSError, RuntimeError) as e:
        raise PathSecurityError(f"Cannot resolve path: {e}") from e

    # Check if path must exist
    if must_exist and not resolved.exists():
        raise PathSecurityError(f"Path does not exist: {path_str}")

    # If base_dir specified, ensure resolved path is within it
    if base_dir:
        base_path = Path(base_dir).resolve()
        try:
            resolved.relative_to(base_path)
        except ValueError as e:
            raise PathSecurityError(
                f"Path '{path_str}' attempts to escape base directory '{base_dir}'",
            ) from e

    return resolved


def is_safe_relative_path(path: str) -> bool:
    """Quick check if a path string looks safe.

    This is a lightweight check for use in validators.
    For actual file operations, use validate_safe_path().

    Args:
        path: Path string to check

    Returns
    -------
        True if path appears safe, False otherwise
    """
    if not path or not isinstance(path, str):
        return False

    # Reject suspicious patterns
    if any(p in path for p in ["..", "~", "$", "\x00", "\\x", "%2e"]):
        return False

    # Reject absolute paths
    if os.path.isabs(path):
        return False

    # Check for unusual characters
    return bool(re.match(r"^[\w\-./]+$", path))


__all__ = [
    "PathSecurityError",
    "sanitize_filename",
    "validate_safe_path",
    "is_safe_relative_path",
]
