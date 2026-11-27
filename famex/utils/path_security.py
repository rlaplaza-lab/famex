"""Path security utilities for FAMEX."""

from __future__ import annotations

import os
import re
from pathlib import Path, PurePosixPath, PureWindowsPath


class PathSecurityError(ValueError):
    """Raised when an unsafe path is detected."""


_ENCODED_TRAVERSAL_MARKERS = ("%2e", "\\x")
_SUSPICIOUS_PATH_SUBSTRINGS = ("~", "$")


def has_unsafe_components(path: Path | str, *, reject_parent_refs: bool = True) -> bool:
    """Return True if ``path`` has ``..``, nulls, or other unsafe markers."""
    try:
        path_obj = Path(path)
    except (TypeError, ValueError):
        return True

    path_str = os.fspath(path)
    if "\x00" in path_str:
        return True
    if reject_parent_refs:
        # Use both POSIX and Windows parsers so ``..\..`` is caught on Linux hosts.
        parts = set(path_obj.parts)
        parts.update(PurePosixPath(path_str).parts)
        parts.update(PureWindowsPath(path_str).parts)
        if ".." in parts:
            return True
    lowered = path_str.lower()
    if any(marker in lowered for marker in _ENCODED_TRAVERSAL_MARKERS):
        return True
    return any(marker in path_str for marker in _SUSPICIOUS_PATH_SUBSTRINGS)


def sanitize_filename(filename: str, allow_path_sep: bool = False) -> str:
    """Return a basename with only safe filename characters."""
    safe = os.path.basename(filename).strip(". ").replace("..", "_")
    if not allow_path_sep:
        safe = re.sub(r"[^\w\-.]", "_", safe)
    safe = safe.lstrip(".")
    if not safe or safe.isspace():
        raise PathSecurityError(f"Filename '{filename}' contains no safe characters")
    return safe


def validate_safe_path(
    path: Path | str,
    base_dir: Path | str | None = None,
    must_exist: bool = False,
    allow_absolute: bool = False,
) -> Path:
    """Validate that a path is safe and within ``base_dir`` when provided."""
    try:
        path_obj = Path(path)
    except (TypeError, ValueError) as e:
        raise PathSecurityError(f"Invalid path: {e}") from e

    path_str = str(path)
    base_path = Path(base_dir).resolve() if base_dir else None
    reject_parent_refs = base_path is None
    if has_unsafe_components(path_obj, reject_parent_refs=reject_parent_refs):
        if "\x00" in path_str:
            raise PathSecurityError("Path contains null bytes")
        if reject_parent_refs and ".." in path_obj.parts:
            raise PathSecurityError("Path contains parent-directory components")
        raise PathSecurityError(f"Unsafe path detected: {path_str}")

    if not allow_absolute and not base_dir and path_obj.is_absolute():
        raise PathSecurityError(f"Absolute paths not allowed: {path_str}")

    try:
        if base_path is not None:
            resolved = (
                path_obj.resolve() if path_obj.is_absolute() else (base_path / path_obj).resolve()
            )
        else:
            resolved = path_obj.resolve()
    except (OSError, RuntimeError) as e:
        raise PathSecurityError(f"Cannot resolve path: {e}") from e

    if must_exist and not resolved.exists():
        raise PathSecurityError(f"Path does not exist: {path_str}")

    if base_path is not None:
        try:
            resolved.relative_to(base_path)
        except ValueError as e:
            raise PathSecurityError(
                f"Path '{path_str}' attempts to escape base directory '{base_dir}'",
            ) from e

    return resolved


def is_safe_relative_path(path: str) -> bool:
    """Lightweight relative-path safety check for validators."""
    if not path or not isinstance(path, str):
        return False
    if has_unsafe_components(path) or os.path.isabs(path):
        return False
    return bool(re.match(r"^[\w\-./]+$", path))


__all__ = [
    "PathSecurityError",
    "has_unsafe_components",
    "sanitize_filename",
    "validate_safe_path",
    "is_safe_relative_path",
]
