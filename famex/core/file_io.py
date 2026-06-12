"""File I/O helper functions for FAMEX Explorer."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.io import write

from famex.io.xyz_io import write_xyz_with_metadata


def validate_output_path(output_file: str | Path) -> Path:
    """Validate output file path for security."""
    output_file = Path(output_file)

    if ".." in str(output_file) or "\x00" in str(output_file):
        raise ValueError(f"Unsafe output path detected: {output_file}")

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
        except OSError as e:
            raise RuntimeError(f"Failed to save XYZ structure to {output_file}: {e}") from e
        except (ValueError, TypeError) as e:
            raise RuntimeError(f"Failed to save XYZ structure to {output_file}: {e}") from e

    try:
        if format is not None:
            write(output_file, atoms, format=format)
        else:
            write(output_file, atoms)
    except OSError as e:
        try:
            clean_atoms = _create_clean_atoms(atoms)
            if format is not None:
                write(output_file, clean_atoms, format=format)
            else:
                write(output_file, clean_atoms)
        except OSError as e2:
            raise RuntimeError(
                f"Failed to save structure to {output_file}: {e}. Clean attempt also failed: {e2}"
            ) from e2
        except (ValueError, TypeError, KeyError) as e2:
            raise RuntimeError(
                f"Failed to save structure to {output_file}: {e}. Clean attempt also failed: {e2}"
            ) from e2
    except (ValueError, TypeError, KeyError) as e:
        raise RuntimeError(f"Failed to save structure to {output_file}: {e}") from e


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
        except OSError as e:
            raise RuntimeError(f"Failed to save XYZ trajectory to {output_file}: {e}") from e
        except (ValueError, TypeError) as e:
            raise RuntimeError(f"Failed to save XYZ trajectory to {output_file}: {e}") from e

    try:
        if format is not None:
            write(output_file, atoms_list, format=format)
        else:
            write(output_file, atoms_list)
    except OSError as e:
        try:
            clean_atoms_list = [_create_clean_atoms(atoms) for atoms in atoms_list]
            if format is not None:
                write(output_file, clean_atoms_list, format=format)
            else:
                write(output_file, clean_atoms_list)
        except OSError as e2:
            raise RuntimeError(
                f"Failed to save trajectory to {output_file}: {e}. Clean attempt also failed: {e2}"
            ) from e2
        except (ValueError, TypeError, KeyError) as e2:
            raise RuntimeError(
                f"Failed to save trajectory to {output_file}: {e}. Clean attempt also failed: {e2}"
            ) from e2
    except (ValueError, TypeError, KeyError) as e:
        raise RuntimeError(f"Failed to save trajectory to {output_file}: {e}") from e
