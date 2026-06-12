"""Custom XYZ file I/O with metadata preservation and validation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.io import read as ase_read
from ase.io import write as ase_write

from famex.utils.logging import get_famex_logger

if TYPE_CHECKING:
    from famex.io.geometry import Geometry

logger = get_famex_logger(__name__)


def parse_xyz_comment(comment: str) -> dict[str, Any]:
    """Parse XYZ comment line for metadata (supports key=value pairs)."""
    metadata: dict[str, Any] = {}
    if not comment or not comment.strip():
        return metadata

    pattern = r"(\w+)=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    matches = re.findall(pattern, comment)

    for key, value in matches:
        key = key.lower()
        try:
            if "." in value or "e" in value.lower():
                metadata[key] = float(value)
            else:
                metadata[key] = int(value)
        except ValueError:
            metadata[key] = value

    return metadata


def format_xyz_comment(atoms: Atoms, energy: float | None = None) -> str:
    parts = []

    charge = None
    spin = None

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

    if charge is not None:
        parts.append(f"charge={charge}")
    if spin is not None:
        parts.append(f"spin={spin}")

    if energy is not None:
        parts.append(f"energy={energy:.6f}")

    if not parts:
        parts.append("FAMEX structure")

    return " ".join(parts)


def validate_xyz_structure(atoms: Atoms, strict: bool = False) -> list[str]:
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

    if np.any(np.isnan(positions)):
        issues.append("NaN coordinates detected")
    if np.any(np.isinf(positions)):
        issues.append("Infinite coordinates detected")

    if not np.any(np.isnan(positions)) and not np.any(np.isinf(positions)):
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
                elif charge is not None and abs(charge) > 100:
                    issues.append(f"Unusual charge: {charge}")

    return issues


def read_xyz_with_metadata(
    filename: str | Path,
    frame: str | int = "last",
    validate: bool = True,
    **kwargs: Any,
) -> Geometry | list[Geometry]:
    filename = Path(filename)
    if not filename.exists():
        logger.error("XYZ file not found: %s", filename)
        raise FileNotFoundError(f"XYZ file not found: {filename}")

    try:
        atoms_list = ase_read(str(filename), ":", **kwargs)
    except Exception as e:
        logger.exception("Failed to read XYZ file %s: %s", filename, e)
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

    geometries = []
    for atoms in selected_frames:
        from famex.io.geometry import Geometry

        geom = Geometry(ase_atoms=atoms)

        if validate:
            issues = validate_xyz_structure(geom)
            if issues:
                import warnings

                warnings.warn(
                    f"XYZ validation issues in {filename}: {'; '.join(issues)}",
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
        logger.exception("Failed to write XYZ file %s: file system error", filename)
        msg = (
            f"Failed to write XYZ file {filename}: {e}. "
            f"This may be due to file system permissions, insufficient disk space, "
            f"or an invalid file path."
        )
        raise OSError(msg) from e
    except (ValueError, TypeError, KeyError) as e:
        logger.exception("Failed to write XYZ file %s: data format error", filename)
        msg = (
            f"Failed to write XYZ file {filename}: {e}. "
            f"This may indicate invalid or corrupted structure data."
        )
        raise OSError(msg) from e


__all__ = [
    "parse_xyz_comment",
    "format_xyz_comment",
    "validate_xyz_structure",
    "read_xyz_with_metadata",
    "write_xyz_with_metadata",
]
