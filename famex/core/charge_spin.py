"""Charge and spin extraction utilities for FAMEX Explorer."""

from __future__ import annotations

from typing import Any, Literal, overload

from ase import Atoms

from famex.io.xyz_io import parse_xyz_comment
from famex.utils.hessian_ops import optional_int


def _apply_source(
    current: int | None,
    present: bool,
    raw: Any,
    *,
    prefer_existing: bool,
) -> tuple[int | None, bool]:
    parsed = optional_int(raw)
    if parsed is None:
        return current, present
    if prefer_existing and current is not None:
        return current, True
    return parsed, True


def _read_charge_spin_sources(
    atoms: Atoms,
) -> tuple[int | None, int | None, bool, bool]:
    """Values and presence flags from XYZ comment, attrs, then info."""
    charge: int | None = None
    spin: int | None = None
    charge_present = False
    spin_present = False

    info = getattr(atoms, "info", None)
    if info is not None:
        comment = info.get("comment", "")
        if comment:
            metadata = parse_xyz_comment(comment)
            if "charge" in metadata:
                charge, charge_present = _apply_source(
                    charge, charge_present, metadata["charge"], prefer_existing=False
                )
            if "spin" in metadata:
                spin, spin_present = _apply_source(
                    spin, spin_present, metadata["spin"], prefer_existing=False
                )

    if hasattr(atoms, "charge"):
        charge, charge_present = _apply_source(
            charge, charge_present, atoms.charge, prefer_existing=True
        )
    if hasattr(atoms, "mult"):
        spin, spin_present = _apply_source(spin, spin_present, atoms.mult, prefer_existing=True)

    if info is not None:
        if "charge" in info:
            charge, charge_present = _apply_source(
                charge, charge_present, info.get("charge"), prefer_existing=True
            )
        if "spin" in info:
            spin, spin_present = _apply_source(
                spin, spin_present, info.get("spin"), prefer_existing=True
            )

    return charge, spin, charge_present, spin_present


@overload
def extract_charge_spin_from_atoms(
    atoms: Atoms,
    default_charge: int = 0,
    default_spin: int = 1,
    *,
    return_presence: Literal[False] = False,
) -> tuple[int, int]: ...


@overload
def extract_charge_spin_from_atoms(
    atoms: Atoms,
    default_charge: int = 0,
    default_spin: int = 1,
    *,
    return_presence: Literal[True],
) -> tuple[int, int, bool, bool]: ...


def extract_charge_spin_from_atoms(
    atoms: Atoms,
    default_charge: int = 0,
    default_spin: int = 1,
    *,
    return_presence: bool = False,
) -> tuple[int, int] | tuple[int, int, bool, bool]:
    """Extract charge/spin. Precedence: XYZ comment → attrs → info."""
    charge, spin, charge_present, spin_present = _read_charge_spin_sources(atoms)
    if charge is None:
        charge = default_charge
    if spin is None:
        spin = default_spin
    if return_presence:
        return charge, spin, charge_present, spin_present
    return charge, spin


def check_missing_charge_spin(atoms: Atoms) -> tuple[bool, bool]:
    """Return ``(charge_missing, spin_missing)``, including XYZ comments."""
    _charge, _spin, charge_present, spin_present = _read_charge_spin_sources(atoms)
    return (not charge_present, not spin_present)


def set_charge_spin_defaults(
    atoms: Atoms,
    charge: int = 0,
    spin: int = 1,
    *,
    overwrite: bool = False,
    coerce: bool = True,
) -> None:
    """Ensure ``atoms.info`` has integer charge and spin."""
    if getattr(atoms, "info", None) is None:
        atoms.info = {}
    info: dict[str, Any] = atoms.info

    if overwrite or "charge" not in info:
        info["charge"] = int(charge)
    elif coerce:
        parsed = optional_int(info.get("charge"))
        info["charge"] = int(charge) if parsed is None else parsed

    if overwrite or "spin" not in info:
        info["spin"] = int(spin)
    elif coerce:
        parsed = optional_int(info.get("spin"))
        info["spin"] = int(spin) if parsed is None else parsed
