"""Custom XYZ file I/O with metadata preservation and validation.

This module provides enhanced XYZ file reading and writing capabilities that:
- Preserve charge and spin metadata in comment lines
- Validate structure integrity
- Handle multi-frame XYZ files consistently
- Support extended XYZ format with key=value metadata
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.io import read as ase_read
from ase.io import write as ase_write

if TYPE_CHECKING:
    from qme.io.geometry import Geometry


def parse_xyz_comment(comment: str) -> dict[str, Any]:
    """Parse XYZ comment line for metadata.

    Supports key=value pairs in comment lines, e.g.:
    "charge=0 spin=1 energy=-123.45"
    "charge=+1 spin=2"
    "energy=-100.0"

    Parameters
    ----------
    comment : str
        Comment line from XYZ file

    Returns
    -------
    dict[str, Any]
        Parsed metadata with type conversion
    """
    metadata = {}
    if not comment or not comment.strip():
        return metadata

    # Pattern to match key=value pairs
    # Supports: charge=0, spin=1, energy=-123.45, etc.
    pattern = r"(\w+)=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    matches = re.findall(pattern, comment)

    for key, value in matches:
        key = key.lower()
        try:
            # Try integer first, then float
            if "." in value or "e" in value.lower():
                metadata[key] = float(value)
            else:
                metadata[key] = int(value)
        except ValueError:
            # Keep as string if conversion fails
            metadata[key] = value

    return metadata


def format_xyz_comment(atoms: Atoms, energy: float | None = None) -> str:
    """Format comment line for XYZ file with metadata.

    Parameters
    ----------
    atoms : Atoms
        Structure to extract metadata from
    energy : float, optional
        Energy value to include in comment

    Returns
    -------
    str
        Formatted comment line
    """
    parts = []

    # Import Geometry to avoid circular import

    # Extract charge and spin
    charge = None
    spin = None

    # Check Geometry attributes first
    if hasattr(atoms, "charge"):
        try:
            charge = int(atoms.charge)
        except (ValueError, TypeError):
            pass

    if hasattr(atoms, "mult"):
        try:
            spin = int(atoms.mult)
        except (ValueError, TypeError):
            pass

    # Check atoms.info
    if hasattr(atoms, "info") and atoms.info:
        if charge is None and "charge" in atoms.info:
            try:
                charge = int(atoms.info["charge"])
            except (ValueError, TypeError):
                pass
        if spin is None and "spin" in atoms.info:
            try:
                spin = int(atoms.info["spin"])
            except (ValueError, TypeError):
                pass

    # Add charge and spin to comment
    if charge is not None:
        parts.append(f"charge={charge}")
    if spin is not None:
        parts.append(f"spin={spin}")

    # Add energy if provided
    if energy is not None:
        parts.append(f"energy={energy:.6f}")

    # Add default comment if no metadata
    if not parts:
        parts.append("QME structure")

    return " ".join(parts)


def validate_xyz_structure(atoms: Atoms, strict: bool = False) -> list[str]:
    """Validate XYZ structure for common issues.

    Parameters
    ----------
    atoms : Atoms
        Structure to validate
    strict : bool, default False
        If True, perform additional strict checks

    Returns
    -------
    list[str]
        List of validation warnings/errors (empty if valid)
    """
    issues = []

    # Check atom count
    if len(atoms) == 0:
        issues.append("Structure has no atoms")
        return issues

    # Check atomic symbols
    valid_symbols = set(atomic_numbers.keys())
    invalid_symbols = set(atoms.get_chemical_symbols()) - valid_symbols
    if invalid_symbols:
        issues.append(f"Invalid atomic symbols: {sorted(invalid_symbols)}")

    # Check coordinates
    positions = atoms.get_positions()
    if positions.size == 0:
        issues.append("No atomic coordinates found")
        return issues

    # Check for NaN or Inf coordinates
    if np.any(np.isnan(positions)):
        issues.append("NaN coordinates detected")
    if np.any(np.isinf(positions)):
        issues.append("Infinite coordinates detected")

    # Check coordinate ranges
    if not np.any(np.isnan(positions)) and not np.any(np.isinf(positions)):
        max_distance = np.max(np.linalg.norm(positions, axis=1))
        if max_distance > 1000.0:
            issues.append(f"Very large coordinates detected (max distance: {max_distance:.1f} Å)")

        # Check for atoms too close together
        if len(atoms) > 1:
            distances = atoms.get_all_distances()
            # Remove diagonal (self-distances)
            np.fill_diagonal(distances, np.inf)
            min_distance = np.min(distances)
            if min_distance < 0.1:
                issues.append(f"Atoms very close together (min distance: {min_distance:.3f} Å)")

    # Strict validation
    if strict:
        # Check charge/spin consistency if available
        if hasattr(atoms, "info") and atoms.info:
            charge = atoms.info.get("charge")
            spin = atoms.info.get("spin")
            if charge is not None and spin is not None:
                # Basic spin-charge consistency check
                if spin < 1:
                    issues.append(f"Invalid spin multiplicity: {spin} (must be >= 1)")
                elif charge is not None and abs(charge) > 100:
                    issues.append(f"Unusual charge: {charge}")

    return issues


def read_xyz_with_metadata(
    filename: str | Path, frame: str | int = "last", validate: bool = True, **kwargs: Any
) -> Geometry | list[Geometry]:
    """Read XYZ file with metadata parsing and validation.

    Parameters
    ----------
    filename : str or Path
        Path to XYZ file
    frame : str or int, default "last"
        Frame selection for multi-frame files:
        - "first": Take first frame
        - "last": Take last frame
        - "all": Return all frames as list
        - int: Take specific frame index (0-based)
    validate : bool, default True
        Whether to validate structure integrity
    **kwargs
        Additional arguments passed to ASE read

    Returns
    -------
    Geometry or list[Geometry]
        Loaded structure(s) with metadata preserved

    Raises
    ------
    FileNotFoundError
        If file doesn't exist
    ValueError
        If file format is invalid or validation fails
    """
    filename = Path(filename)
    if not filename.exists():
        raise FileNotFoundError(f"XYZ file not found: {filename}")

    # Read with ASE - always read all frames for multi-frame support
    try:
        atoms_list = ase_read(str(filename), ":", **kwargs)
    except Exception as e:
        raise ValueError(f"Failed to read XYZ file {filename}: {e}") from e

    # Ensure we have a list
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]

    if not atoms_list:
        raise ValueError(f"XYZ file {filename} contains no structures")

    # Select frame(s)
    if frame == "all":
        selected_frames = atoms_list
    elif frame == "first":
        selected_frames = [atoms_list[0]]
    elif frame == "last":
        selected_frames = [atoms_list[-1]]
    elif isinstance(frame, int):
        if frame < 0 or frame >= len(atoms_list):
            raise ValueError(f"Frame index {frame} out of range (0-{len(atoms_list) - 1})")
        selected_frames = [atoms_list[frame]]
    else:
        raise ValueError(f"Invalid frame selection: {frame}")

    # Convert to Geometry objects with metadata
    geometries = []
    for atoms in selected_frames:
        # Import here to avoid circular import
        from qme.io.geometry import Geometry

        # Create Geometry object
        geom = Geometry(ase_atoms=atoms)

        # The Geometry constructor already extracts charge/spin from atoms.info
        # No need to override it here

        # Validate structure
        if validate:
            issues = validate_xyz_structure(geom)
            if issues:
                # For now, just warn - could be made stricter
                import warnings

                warnings.warn(
                    f"XYZ validation issues in {filename}: {'; '.join(issues)}", stacklevel=2
                )

        geometries.append(geom)

    # Return single geometry or list
    if len(geometries) == 1 and frame != "all":
        return geometries[0]
    return geometries


def write_xyz_with_metadata(
    atoms: Atoms | Geometry | list[Atoms | Geometry],
    filename: str | Path,
    energy: float | None = None,
    **kwargs: Any,
) -> None:
    """Write XYZ file with metadata in comment line.

    Parameters
    ----------
    atoms : Atoms or list[Atoms]
        Structure(s) to write
    filename : str or Path
        Output filename
    energy : float, optional
        Energy value to include in comment
    **kwargs
        Additional arguments passed to ASE write

    Raises
    ------
    OSError
        If file cannot be written
    """
    filename = Path(filename)

    # Ensure output directory exists
    filename.parent.mkdir(parents=True, exist_ok=True)

    # Handle single structure
    if not isinstance(atoms, list):
        atoms_list = [atoms]
    else:
        atoms_list = atoms

    # Prepare structures with metadata
    prepared_atoms = []
    for atom in atoms_list:
        # Create copy to avoid modifying original
        atom_copy = atom.copy()

        # Format comment line with metadata
        comment = format_xyz_comment(atom_copy, energy)

        # Set comment in atoms.info
        if atom_copy.info is None:
            atom_copy.info = {}
        atom_copy.info["comment"] = comment

        prepared_atoms.append(atom_copy)

    # Write with ASE
    try:
        if len(prepared_atoms) == 1:
            ase_write(str(filename), prepared_atoms[0], **kwargs)
        else:
            ase_write(str(filename), prepared_atoms, **kwargs)
    except Exception as e:
        raise OSError(f"Failed to write XYZ file {filename}: {e}") from e


__all__ = [
    "parse_xyz_comment",
    "format_xyz_comment",
    "validate_xyz_structure",
    "read_xyz_with_metadata",
    "write_xyz_with_metadata",
]
