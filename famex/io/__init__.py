"""I/O and geometry handling for FAMEX.

This module provides classes and functions for:
- Molecular geometry representation and manipulation
- File I/O for various molecular structure formats
- Reaction pathway management and analysis
"""

from famex.io.geometry import Geometry, read_geometry, write_geometry
from famex.io.path_manager import PathManager
from famex.io.run_output import (
    default_output_name,
    history_output_name,
    select_default_frames,
    write_run_outputs,
)
from famex.io.xyz_io import (
    format_xyz_comment,
    parse_xyz_comment,
    read_xyz_with_metadata,
    validate_xyz_structure,
    write_xyz_with_metadata,
)

__all__ = [
    "Geometry",
    "PathManager",
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
