"""File I/O helper functions for FAMEX Explorer."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.io import write

from famex.io.xyz_io import write_xyz_with_metadata
from famex.utils.path_security import validate_output_path

__all__ = [
    "validate_output_path",
    "write_atoms_safely",
    "write_trajectory_safely",
]


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
    path_str = str(output_file)

    def do_write(data: Atoms | list[Atoms]) -> None:
        if format is not None:
            write(path_str, data, format=format)
        else:
            write(path_str, data)

    try:
        do_write(payload)
    except OSError as e:
        try:
            do_write(clean_payload)
        except (OSError, ValueError, TypeError, KeyError) as e2:
            raise RuntimeError(
                f"Failed to save {entity_label} to {output_file}: {e}. "
                f"Clean attempt also failed: {e2}"
            ) from e2
    except (ValueError, TypeError, KeyError) as e:
        raise RuntimeError(f"Failed to save {entity_label} to {output_file}: {e}") from e


def _write_xyz_safely(
    payload: Atoms | list[Atoms],
    output_file: Path,
    *,
    entity_label: str,
) -> bool:
    if not str(output_file).lower().endswith(".xyz"):
        return False
    try:
        write_xyz_with_metadata(payload, output_file)
    except (OSError, ValueError, TypeError) as e:
        raise RuntimeError(f"Failed to save XYZ {entity_label} to {output_file}: {e}") from e
    return True


def write_atoms_safely(
    atoms: Atoms,
    output_file: str | Path,
    format: str | None = None,
) -> None:
    output_file = validate_output_path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if _write_xyz_safely(atoms, output_file, entity_label="structure"):
        return

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
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if _write_xyz_safely(atoms_list, output_file, entity_label="trajectory"):
        return

    _write_with_clean_retry(
        atoms_list,
        output_file,
        format,
        entity_label="trajectory",
        clean_payload=[_create_clean_atoms(atoms) for atoms in atoms_list],
    )
