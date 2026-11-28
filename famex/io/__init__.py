"""I/O and geometry handling for FAMEX.

This module provides classes and functions for:
- Molecular geometry representation and manipulation
- File I/O for various molecular structure formats
- Reaction pathway management and analysis
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "Geometry",
    "PathManager",
    "modes_output_dir",
    "write_vibrational_modes",
    "default_output_name",
    "history_output_name",
    "read_geometry",
    "select_default_frames",
    "write_geometry",
    "write_run_outputs",
    "format_xyz_comment",
    "parse_xyz_comment",
    "read_xyz_with_metadata",
    "validate_xyz_structure",
    "write_xyz_with_metadata",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "Geometry": ("famex.io.geometry", "Geometry"),
    "read_geometry": ("famex.io.geometry", "read_geometry"),
    "write_geometry": ("famex.io.geometry", "write_geometry"),
    "PathManager": ("famex.io.path_manager", "PathManager"),
    "modes_output_dir": ("famex.io.modes", "modes_output_dir"),
    "write_vibrational_modes": ("famex.io.modes", "write_vibrational_modes"),
    "default_output_name": ("famex.io.run_output", "default_output_name"),
    "history_output_name": ("famex.io.run_output", "history_output_name"),
    "select_default_frames": ("famex.io.run_output", "select_default_frames"),
    "write_run_outputs": ("famex.io.run_output", "write_run_outputs"),
    "format_xyz_comment": ("famex.io.xyz_io", "format_xyz_comment"),
    "parse_xyz_comment": ("famex.io.xyz_io", "parse_xyz_comment"),
    "read_xyz_with_metadata": ("famex.io.xyz_io", "read_xyz_with_metadata"),
    "validate_xyz_structure": ("famex.io.xyz_io", "validate_xyz_structure"),
    "write_xyz_with_metadata": ("famex.io.xyz_io", "write_xyz_with_metadata"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = _EXPORTS[name]
    module = importlib.import_module(module_name)
    value = getattr(module, attr)
    globals()[name] = value
    return value
