"""Shared defaults and supported CLI constraint types."""

from __future__ import annotations

from ase.units import Ang, eV

DEFAULT_HARMONIC_FORCE_CONSTANT = 10.0
DEFAULT_HARMONIC_ANGLE_FORCE_CONSTANT = 5.0
DEFAULT_CLI_HARMONIC_FORCE_CONSTANT = 10.0 * eV / Ang**2

SUPPORTED_CLI_CONSTRAINT_TYPES = (
    "fix",
    "harmonic_position",
    "harmonic_bond",
    "harmonic_angle",
    "fixinternals_bond",
    "fixinternals_angle",
    "fixinternals_dihedral",
)

SUPPORTED_CLI_CONSTRAINT_TYPES_MSG = ", ".join(SUPPORTED_CLI_CONSTRAINT_TYPES)
