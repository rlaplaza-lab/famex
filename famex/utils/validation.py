"""Validation functions for FAMEX Explorer runs.

Exception classes are defined in ``famex.core.exceptions`` and re-exported here
for backward compatibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from famex.core.exceptions import BackendError, DependencyError, FAMEXError

if TYPE_CHECKING:
    from ase import Atoms

__all__ = [
    "BackendError",
    "DependencyError",
    "FAMEXError",
    "format_atom_count_mismatch",
    "validate_atoms_compatibility",
]


def format_atom_count_mismatch(labels_and_counts: list[tuple[str, int]]) -> str:
    detail = ", ".join(f"{label} ({count} atoms)" for label, count in labels_and_counts)
    return f"Atom counts do not match: {detail}"


def validate_atoms_compatibility(atoms1: Atoms, atoms2: Atoms, context: str = "operation") -> None:
    if len(atoms1) != len(atoms2):
        raise ValueError(
            format_atom_count_mismatch(
                [(f"{context} structure 0", len(atoms1)), (f"{context} structure 1", len(atoms2))]
            )
        )
    if list(atoms1.symbols) != list(atoms2.symbols):
        raise ValueError(
            f"Incompatible atoms for {context}: different atomic symbols "
            f"({list(atoms1.symbols)} vs {list(atoms2.symbols)})"
        )
