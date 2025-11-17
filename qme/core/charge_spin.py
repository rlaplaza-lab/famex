"""Charge and spin extraction utilities for QME Explorer."""

from __future__ import annotations

from ase import Atoms

from qme.io.xyz_io import parse_xyz_comment


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
                try:
                    charge = int(metadata["charge"])
                except (ValueError, TypeError):
                    pass
            if "spin" in metadata:
                try:
                    spin = int(metadata["spin"])
                except (ValueError, TypeError):
                    pass

    if hasattr(atoms, "charge") and charge is None:
        try:
            charge = int(atoms.charge)
        except (ValueError, TypeError):
            pass

    if hasattr(atoms, "mult") and spin is None:
        try:
            spin = int(atoms.mult)
        except (ValueError, TypeError):
            pass

    if hasattr(atoms, "info") and atoms.info is not None:
        if charge is None and "charge" in atoms.info:
            try:
                charge_val = atoms.info.get("charge")
                if charge_val is not None:
                    charge = int(charge_val)
            except (ValueError, TypeError):
                pass

        if spin is None and "spin" in atoms.info:
            try:
                spin_val = atoms.info.get("spin")
                if spin_val is not None:
                    spin = int(spin_val)
            except (ValueError, TypeError):
                pass

    if charge is None:
        charge = default_charge
    if spin is None:
        spin = default_spin

    return charge, spin


def check_missing_charge_spin(atoms: Atoms) -> tuple[bool, bool]:
    charge_missing = True
    spin_missing = True

    if hasattr(atoms, "charge"):
        try:
            if atoms.charge is not None:
                int(atoms.charge)
                charge_missing = False
        except (ValueError, TypeError):
            pass

    if hasattr(atoms, "mult"):
        try:
            if atoms.mult is not None:
                int(atoms.mult)
                spin_missing = False
        except (ValueError, TypeError):
            pass

    if hasattr(atoms, "info") and atoms.info is not None:
        if "charge" in atoms.info:
            try:
                charge_val = atoms.info.get("charge")
                if charge_val is not None:
                    charge_missing = False
            except (ValueError, TypeError):
                pass

        if "spin" in atoms.info:
            try:
                spin_val = atoms.info.get("spin")
                if spin_val is not None:
                    spin_missing = False
            except (ValueError, TypeError):
                pass

    return charge_missing, spin_missing
