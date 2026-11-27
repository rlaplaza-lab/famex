"""Custom XYZ file I/O with metadata preservation and validation."""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.io import read as ase_read
from ase.io import write as ase_write

from famex.utils.hessian_ops import optional_int
from famex.utils.logging import get_famex_logger

if TYPE_CHECKING:
    from famex.io.geometry import Geometry

logger = get_famex_logger(__name__)


def _raise_xyz_write_error(filename: Path | str, error: Exception, *, data_error: bool) -> None:
    if data_error:
        logger.exception("Failed to write XYZ file %s: data format error", filename)
        detail = "This may indicate invalid or corrupted structure data."
    else:
        logger.exception("Failed to write XYZ file %s: file system error", filename)
        detail = (
            "This may be due to file system permissions, insufficient disk space, "
            "or an invalid file path."
        )
    msg = f"Failed to write XYZ file {filename}: {error}. {detail}"
    raise OSError(msg) from error


def parse_xyz_comment(comment: str) -> dict[str, Any]:
    """Parse XYZ comment line for metadata (supports key=value pairs)."""
    metadata: dict[str, Any] = {}
    if not comment or not comment.strip():
        return metadata

    pattern = r"(\w+)=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    matches = re.findall(pattern, comment)

    for key, value in matches:
        key = key.lower()
        if "." in value or "e" in value.lower():
            metadata[key] = float(value)
        else:
            metadata[key] = int(value)

    return metadata


def format_xyz_comment(atoms: Atoms, energy: float | None = None) -> str:
    """Build the XYZ comment line for a structure.

    Parameters
    ----------
    atoms : Atoms
        Structure whose charge, spin and ``info`` entries (``iteration``, ``image``,
        ``step``, ``mode``, ``frequency``) are written.
    energy : float, optional
        Energy in eV to include.

    Returns
    -------
    str
        Space-separated ``key=value`` pairs, or ``"FAMEX structure"`` when there is
        no metadata.

    """
    parts = []

    charge = None
    spin = None

    if hasattr(atoms, "charge"):
        charge = optional_int(atoms.charge)

    if hasattr(atoms, "mult"):
        spin = optional_int(atoms.mult)

    info = atoms.info if hasattr(atoms, "info") and atoms.info else {}
    if charge is None and "charge" in info:
        charge = optional_int(info["charge"])
    if spin is None and "spin" in info:
        spin = optional_int(info["spin"])

    if charge is not None:
        parts.append(f"charge={charge}")
    if spin is not None:
        parts.append(f"spin={spin}")

    if energy is not None:
        parts.append(f"energy={energy:.6f}")

    for key in ("iteration", "image", "step", "mode"):
        if key not in info:
            continue
        coerced = optional_int(info[key])
        if coerced is not None:
            parts.append(f"{key}={coerced}")

    if "frequency" in info:
        try:
            parts.append(f"frequency={float(info['frequency']):.6f}")
        except (TypeError, ValueError):
            pass

    if not parts:
        parts.append("FAMEX structure")

    return " ".join(parts)


def validate_xyz_structure(atoms: Atoms, strict: bool = False) -> list[str]:
    """Check a structure for common problems.

    Parameters
    ----------
    atoms : Atoms
        Structure to check.
    strict : bool, default False
        Also check that the charge and spin multiplicity in ``atoms.info`` are
        sensible.

    Returns
    -------
    list[str]
        Description of each problem found (empty, unknown elements, NaN or
        infinite coordinates, atoms closer than 0.1 Å, coordinates beyond 1000 Å).
        An empty list means no problems.

    """
    issues = []

    if len(atoms) == 0:
        issues.append("Structure has no atoms")
        return issues

    valid_symbols = set(atomic_numbers.keys())
    invalid_symbols = set(atoms.get_chemical_symbols()) - valid_symbols
    if invalid_symbols:
        issues.append(f"Invalid atomic symbols: {sorted(invalid_symbols)}")

    positions = atoms.get_positions()
    if positions.size == 0:
        issues.append("No atomic coordinates found")
        return issues

    if not np.all(np.isfinite(positions)):
        if np.any(np.isnan(positions)):
            issues.append("NaN coordinates detected")
        if np.any(np.isinf(positions)):
            issues.append("Infinite coordinates detected")
    else:
        max_distance = np.max(np.linalg.norm(positions, axis=1))
        if max_distance > 1000.0:
            issues.append(f"Very large coordinates detected (max distance: {max_distance:.1f} Å)")

        if len(atoms) > 1:
            distances = atoms.get_all_distances()
            np.fill_diagonal(distances, np.inf)
            min_distance = np.min(distances)
            if min_distance < 0.1:
                issues.append(f"Atoms very close together (min distance: {min_distance:.3f} Å)")

    if strict:
        if hasattr(atoms, "info") and atoms.info:
            charge = atoms.info.get("charge")
            spin = atoms.info.get("spin")
            if charge is not None and spin is not None:
                if spin < 1:
                    issues.append(f"Invalid spin multiplicity: {spin} (must be >= 1)")
                elif abs(charge) > 100:
                    issues.append(f"Unusual charge: {charge}")

    return issues


def read_xyz_with_metadata(
    filename: str | Path,
    frame: str | int = "last",
    validate: bool = True,
    **kwargs: Any,
) -> Geometry | list[Geometry]:
    """Read an XYZ file, including charge, spin and energy from the comment line.

    Parameters
    ----------
    filename : str or Path
        Path to the XYZ file.
    frame : str or int, default "last"
        Frame to read: ``"first"``, ``"last"``, ``"all"`` or an integer index.
    validate : bool, default True
        Log a warning for structural problems found by
        :func:`validate_xyz_structure`.
    **kwargs
        Passed to :func:`ase.io.read`.

    Returns
    -------
    Geometry or list[Geometry]
        One geometry, or a list when ``frame="all"``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file cannot be read, is empty, or ``frame`` is invalid.

    """
    filename = Path(filename)
    if not filename.exists():
        logger.error("XYZ file not found: %s", filename)
        raise FileNotFoundError(f"XYZ file not found: {filename}")

    try:
        atoms_list = ase_read(str(filename), ":", **kwargs)
    except Exception as e:
        logger.exception("Failed to read XYZ file %s", filename)
        raise ValueError(f"Failed to read XYZ file {filename}: {e}") from e

    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]

    if not atoms_list:
        logger.error("XYZ file %s contains no structures", filename)
        raise ValueError(f"XYZ file {filename} contains no structures")

    if frame == "all":
        selected_frames = atoms_list
    elif frame == "first":
        selected_frames = [atoms_list[0]]
    elif frame == "last":
        selected_frames = [atoms_list[-1]]
    elif isinstance(frame, int):
        if frame < 0 or frame >= len(atoms_list):
            logger.error(
                "Frame index %d out of range (0-%d) for file %s",
                frame,
                len(atoms_list) - 1,
                filename,
            )
            raise ValueError(f"Frame index {frame} out of range (0-{len(atoms_list) - 1})")
        selected_frames = [atoms_list[frame]]
    else:
        logger.error("Invalid frame selection '%s' for file %s", frame, filename)
        raise ValueError(f"Invalid frame selection: {frame}")

    from famex.io.geometry import Geometry

    geometries = []
    for atoms in selected_frames:
        geom = Geometry(ase_atoms=atoms)

        if validate:
            issues = validate_xyz_structure(geom)
            if issues:
                warnings.warn(
                    f"XYZ validation issues in {filename}: {'; '.join(issues)}",
                    UserWarning,
                    stacklevel=2,
                )

        geometries.append(geom)

    if len(geometries) == 1 and frame != "all":
        return geometries[0]
    return geometries


def write_xyz_with_metadata(
    atoms: Atoms | Geometry | list[Atoms | Geometry],
    filename: str | Path,
    energy: float | None = None,
    **kwargs: Any,
) -> None:
    """Write one or more structures to an XYZ file with metadata in the comment line.

    Parameters
    ----------
    atoms : Atoms, Geometry or list
        Structure or list of structures; a list is written as a multi-frame file.
    filename : str or Path
        Output path. Parent directories are created if needed.
    energy : float, optional
        Energy in eV written to every frame's comment line.
    **kwargs
        Passed to :func:`ase.io.write`.

    Raises
    ------
    OSError
        If the file cannot be written.

    """
    filename = Path(filename)

    filename.parent.mkdir(parents=True, exist_ok=True)

    if not isinstance(atoms, list):
        atoms_list = [atoms]
    else:
        atoms_list = atoms

    prepared_atoms = []
    for atom in atoms_list:
        atom_copy = atom.copy()

        comment = format_xyz_comment(atom_copy, energy)

        if atom_copy.info is None:
            atom_copy.info = {}
        atom_copy.info["comment"] = comment

        prepared_atoms.append(atom_copy)

    try:
        if len(prepared_atoms) == 1:
            ase_write(str(filename), prepared_atoms[0], **kwargs)
        else:
            ase_write(str(filename), prepared_atoms, **kwargs)
    except OSError as e:
        _raise_xyz_write_error(filename, e, data_error=False)
    except (ValueError, TypeError, KeyError) as e:
        _raise_xyz_write_error(filename, e, data_error=True)


__all__ = [
    "parse_xyz_comment",
    "format_xyz_comment",
    "validate_xyz_structure",
    "read_xyz_with_metadata",
    "write_xyz_with_metadata",
]
