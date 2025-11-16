"""File I/O helper functions for QME Explorer.

This module provides consolidated file I/O operations to reduce code duplication
and improve error handling.
"""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.io import write

from qme.io.xyz_io import write_xyz_with_metadata


def validate_output_path(output_file: str | Path) -> Path:
    """Validate output file path for security.

    Parameters
    ----------
    output_file : str or Path
        Output file path to validate

    Returns
    -------
    Path
        Validated Path object

    Raises
    ------
    ValueError
        If path contains unsafe patterns

    """
    output_file = Path(output_file)

    # SECURITY: Validate path doesn't contain traversal patterns
    if ".." in str(output_file) or "\x00" in str(output_file):
        msg = f"Unsafe output path detected: {output_file}"
        raise ValueError(msg)

    return output_file


def _create_clean_atoms(atoms: Atoms) -> Atoms:
    """Create a clean atoms object with only essential data.

    Parameters
    ----------
    atoms : Atoms
        Original atoms object

    Returns
    -------
    Atoms
        Clean atoms object with essential data only

    """
    clean_atoms = Atoms(
        symbols=atoms.symbols,
        positions=atoms.positions,
        cell=atoms.cell,
        pbc=atoms.pbc,
    )
    # Copy over essential info
    if hasattr(atoms, "info") and atoms.info:
        for key in ["charge", "spin"]:
            if key in atoms.info:
                clean_atoms.info[key] = atoms.info[key]
    return clean_atoms


def write_atoms_safely(
    atoms: Atoms,
    output_file: str | Path,
    format: str | None = None,
) -> None:
    """Write atoms to file with comprehensive error handling.

    Parameters
    ----------
    atoms : Atoms
        Structure to save
    output_file : str or Path
        Output file path
    format : str, optional
        File format (inferred from extension if None)

    Raises
    ------
    ValueError
        If path is unsafe
    RuntimeError
        If file write fails

    """
    output_file = validate_output_path(output_file)

    # Use custom XYZ writer for .xyz files to preserve metadata
    if str(output_file).lower().endswith(".xyz"):
        try:
            write_xyz_with_metadata(atoms, str(output_file))
            return
        except OSError as e:
            # File system errors (permissions, disk full, etc.)
            msg = (
                f"Failed to save XYZ structure to {output_file}: {e}. "
                f"This may be due to file system permissions, insufficient disk space, "
                f"or an invalid file path."
            )
            raise RuntimeError(msg) from e
        except (ValueError, TypeError) as e:
            # Data format errors (invalid structure data)
            msg = (
                f"Failed to save XYZ structure to {output_file}: {e}. "
                f"This may indicate invalid or corrupted structure data."
            )
            raise RuntimeError(msg) from e

    # Use ASE for other formats
    try:
        if format is not None:
            write(output_file, atoms, format=format)
        else:
            write(output_file, atoms)
    except OSError as e:
        # File system errors - try with cleaned atoms object
        try:
            clean_atoms = _create_clean_atoms(atoms)
            if format is not None:
                write(output_file, clean_atoms, format=format)
            else:
                write(output_file, clean_atoms)
        except OSError as e2:
            msg = (
                f"Failed to save structure to {output_file}: {e}. "
                f"Clean attempt also failed: {e2}. "
                f"This may be due to file system permissions, insufficient disk space, "
                f"or an invalid file path."
            )
            raise RuntimeError(msg) from e2
        except (ValueError, TypeError, KeyError) as e2:
            # Data or format errors even with cleaned atoms
            msg = (
                f"Failed to save structure to {output_file}: {e}. "
                f"Clean attempt also failed: {e2}. "
                f"This may indicate an unsupported format or corrupted structure data."
            )
            raise RuntimeError(msg) from e2
    except (ValueError, TypeError, KeyError) as e:
        # Data format errors (invalid structure data or unsupported format)
        msg = (
            f"Failed to save structure to {output_file}: {e}. "
            f"This may indicate an unsupported format, invalid structure data, "
            f"or missing format-specific requirements."
        )
        raise RuntimeError(msg) from e


def write_trajectory_safely(
    atoms_list: list[Atoms],
    output_file: str | Path,
    format: str | None = None,
) -> None:
    """Write trajectory (multiple structures) to file with comprehensive error handling.

    Parameters
    ----------
    atoms_list : list[Atoms]
        List of structures to save as trajectory
    output_file : str or Path
        Output file path
    format : str, optional
        File format (inferred from extension if None)

    Raises
    ------
    ValueError
        If path is unsafe
    RuntimeError
        If file write fails

    """
    output_file = validate_output_path(output_file)

    # Use custom XYZ writer for .xyz files to preserve metadata
    if str(output_file).lower().endswith(".xyz"):
        try:
            write_xyz_with_metadata(atoms_list, str(output_file))
            return
        except OSError as e:
            # File system errors (permissions, disk full, etc.)
            msg = (
                f"Failed to save XYZ trajectory to {output_file}: {e}. "
                f"This may be due to file system permissions, insufficient disk space, "
                f"or an invalid file path."
            )
            raise RuntimeError(msg) from e
        except (ValueError, TypeError) as e:
            # Data format errors (invalid structure data)
            msg = (
                f"Failed to save XYZ trajectory to {output_file}: {e}. "
                f"This may indicate invalid or corrupted structure data in the trajectory."
            )
            raise RuntimeError(msg) from e

    # Use ASE for other formats
    try:
        if format is not None:
            write(output_file, atoms_list, format=format)
        else:
            write(output_file, atoms_list)
    except OSError as e:
        # File system errors - try with cleaned atoms objects
        try:
            clean_atoms_list = [_create_clean_atoms(atoms) for atoms in atoms_list]
            if format is not None:
                write(output_file, clean_atoms_list, format=format)
            else:
                write(output_file, clean_atoms_list)
        except OSError as e2:
            msg = (
                f"Failed to save trajectory to {output_file}: {e}. "
                f"Clean attempt also failed: {e2}. "
                f"This may be due to file system permissions, insufficient disk space, "
                f"or an invalid file path."
            )
            raise RuntimeError(msg) from e2
        except (ValueError, TypeError, KeyError) as e2:
            # Data or format errors even with cleaned atoms
            msg = (
                f"Failed to save trajectory to {output_file}: {e}. "
                f"Clean attempt also failed: {e2}. "
                f"This may indicate an unsupported format or corrupted structure data."
            )
            raise RuntimeError(msg) from e2
    except (ValueError, TypeError, KeyError) as e:
        # Data format errors (invalid structure data or unsupported format)
        msg = (
            f"Failed to save trajectory to {output_file}: {e}. "
            f"This may indicate an unsupported format, invalid structure data, "
            f"or missing format-specific requirements."
        )
        raise RuntimeError(msg) from e
