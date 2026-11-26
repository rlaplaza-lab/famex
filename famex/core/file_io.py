"""File I/O helper functions for FAMEX Explorer."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.io import write

from famex.io.xyz_io import write_xyz_with_metadata
from famex.utils.path_security import PathSecurityError


def validate_output_path(output_file: str | Path) -> Path:
    """Validate output file path for security."""
    output_file = Path(output_file)

    if ".." in str(output_file) or "\x00" in str(output_file):
        raise PathSecurityError(f"Unsafe output path detected: {output_file}")

    return output_file


def _create_clean_atoms(atoms: Atoms) -> Atoms:
    clean_atoms = Atoms(
        symbols=atoms.symbols,
        positions=atoms.positions,
        cell=atoms.cell,
        pbc=atoms.pbc,
    )
    if hasattr(atoms, "info") and atoms.info:
        for key in ["charge", "spin"]:
            if key in atoms.info:
                clean_atoms.info[key] = atoms.info[key]
    return clean_atoms


def _write_with_clean_retry(
    payload: Atoms | list[Atoms],
    output_file: Path,
    format: str | None,
    *,
    entity_label: str,
    clean_payload: Atoms | list[Atoms],
) -> None:
    try:
        if format is not None:
            write(output_file, payload, format=format)
        else:
            write(output_file, payload)
    except OSError as e:
        try:
            if format is not None:
                write(output_file, clean_payload, format=format)
            else:
                write(output_file, clean_payload)
        except OSError as e2:
            raise RuntimeError(
                f"Failed to save {entity_label} to {output_file}: {e}. "
                f"Clean attempt also failed: {e2}"
            ) from e2
        except (ValueError, TypeError, KeyError) as e2:
            raise RuntimeError(
                f"Failed to save {entity_label} to {output_file}: {e}. "
                f"Clean attempt also failed: {e2}"
            ) from e2
    except (ValueError, TypeError, KeyError) as e:
        raise RuntimeError(f"Failed to save {entity_label} to {output_file}: {e}") from e


def write_atoms_safely(
    atoms: Atoms,
    output_file: str | Path,
    format: str | None = None,
) -> None:
    output_file = validate_output_path(output_file)

    if str(output_file).lower().endswith(".xyz"):
        try:
            write_xyz_with_metadata(atoms, str(output_file))
            return
        except (OSError, ValueError, TypeError) as e:
            raise RuntimeError(f"Failed to save XYZ structure to {output_file}: {e}") from e

    _write_with_clean_retry(
        atoms,
        output_file,
        format,
        entity_label="structure",
        clean_payload=_create_clean_atoms(atoms),
    )


def write_trajectory_safely(
    atoms_list: list[Atoms],
    output_file: str | Path,
    format: str | None = None,
) -> None:
    output_file = validate_output_path(output_file)

    if str(output_file).lower().endswith(".xyz"):
        try:
            write_xyz_with_metadata(atoms_list, str(output_file))
            return
        except (OSError, ValueError, TypeError) as e:
            raise RuntimeError(f"Failed to save XYZ trajectory to {output_file}: {e}") from e

    _write_with_clean_retry(
        atoms_list,
        output_file,
        format,
        entity_label="trajectory",
        clean_payload=[_create_clean_atoms(atoms) for atoms in atoms_list],
    )
