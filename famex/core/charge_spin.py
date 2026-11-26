"""Charge and spin extraction utilities for FAMEX Explorer."""

from __future__ import annotations

from typing import Any

from ase import Atoms

from famex.io.xyz_io import parse_xyz_comment


def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def extract_charge_spin_from_atoms(
    atoms: Atoms,
    default_charge: int = 0,
    default_spin: int = 1,
) -> tuple[int, int]:
    charge = None
    spin = None

    if hasattr(atoms, "info") and atoms.info is not None:
        comment = atoms.info.get("comment", "")
        if comment:
            metadata = parse_xyz_comment(comment)
            if "charge" in metadata:
                charge = _as_optional_int(metadata["charge"])
            if "spin" in metadata:
                spin = _as_optional_int(metadata["spin"])

    if hasattr(atoms, "charge") and charge is None:
        charge = _as_optional_int(atoms.charge)

    if hasattr(atoms, "mult") and spin is None:
        spin = _as_optional_int(atoms.mult)

    if hasattr(atoms, "info") and atoms.info is not None:
        if charge is None and "charge" in atoms.info:
            charge = _as_optional_int(atoms.info.get("charge"))
        if spin is None and "spin" in atoms.info:
            spin = _as_optional_int(atoms.info.get("spin"))

    if charge is None:
        charge = default_charge
    if spin is None:
        spin = default_spin

    return charge, spin


def check_missing_charge_spin(atoms: Atoms) -> tuple[bool, bool]:
    charge_missing = True
    spin_missing = True

    if hasattr(atoms, "charge") and _as_optional_int(atoms.charge) is not None:
        charge_missing = False

    if hasattr(atoms, "mult") and _as_optional_int(atoms.mult) is not None:
        spin_missing = False

    if hasattr(atoms, "info") and atoms.info is not None:
        if "charge" in atoms.info and _as_optional_int(atoms.info.get("charge")) is not None:
            charge_missing = False
        if "spin" in atoms.info and _as_optional_int(atoms.info.get("spin")) is not None:
            spin_missing = False

    return charge_missing, spin_missing
