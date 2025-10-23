"""I/O and geometry handling for QME.

This module provides classes and functions for:
- Molecular geometry representation and manipulation
- File I/O for various molecular structure formats
- Reaction pathway management and analysis
"""

from qme.io.geometry import Geometry, read_geometry, write_geometry
from qme.io.path_manager import PathManager
from qme.io.xyz_io import (
    format_xyz_comment,
    parse_xyz_comment,
    read_xyz_with_metadata,
    validate_xyz_structure,
    write_xyz_with_metadata,
)

__all__ = [
    "Geometry",
    "PathManager",
    "read_geometry",
    "write_geometry",
    "format_xyz_comment",
    "parse_xyz_comment",
    "read_xyz_with_metadata",
    "validate_xyz_structure",
    "write_xyz_with_metadata",
]
