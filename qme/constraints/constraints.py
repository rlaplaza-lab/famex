"""Enhanced constraint handling for QME optimizations.

This module provides simplified constraint management with three core types:
1. Fixed Atoms: Exactly fix atom positions (enhanced version of FixAtoms)
2. Harmonic Constraints: Soft constraints based on initial geometry (bonds, angles, positions)
3. FixInternals: Select target values for bonds, angles, dihedrals, and bond combinations

The design is deliberately simplified to cover 90% of practical constraint needs
while maintaining ease of use and integration with existing QME workflows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from ase.constraints import FixAtoms, FixInternals, Hookean
from ase.units import Ang, eV

from qme.utils.logging import get_qme_logger

if TYPE_CHECKING:
    from ase import Atoms

logger = get_qme_logger(__name__)


class QMEConstraintManager:
    def __init__(self, reference_atoms: Atoms) -> None:
        self.reference_atoms = reference_atoms.copy()
        self.constraints: list[Any] = []

    def add_fixed_atoms(self, atom_indices: list[int]) -> None:
        logger.debug("Adding fixed atoms constraint for indices: %s", atom_indices)
        constraint = FixedAtomsConstraint(atom_indices, self.reference_atoms)
        self.constraints.append(constraint)

    def add_harmonic_constraint(
        self,
        constraint_type: str,
        atom_indices: list[int],
        force_constant: float = 10.0,
    ) -> None:
        if constraint_type == "position":
            constraint: Any = HarmonicPositionConstraint(
                atom_indices,
                self.reference_atoms,
                force_constant,
            )
        elif constraint_type == "bond":
            constraint = HarmonicBondConstraint(atom_indices, self.reference_atoms, force_constant)
        elif constraint_type == "angle":
            constraint = HarmonicAngleConstraint(atom_indices, self.reference_atoms, force_constant)
        else:
            msg = f"Unknown constraint type: {constraint_type}"
            logger.error(msg)
            raise ValueError(msg)

        logger.debug(
            "Adding harmonic constraint: type=%s, atoms=%s, k=%.2f",
            constraint_type,
            atom_indices,
            force_constant,
        )
        self.constraints.append(constraint)

    def add_fixinternals_constraint(
        self,
        constraint_type: str,
        atom_indices: list[int],
        target_value: float | None = None,
    ) -> None:
        constraint = FixInternalsConstraint(constraint_type, atom_indices, target_value)
        logger.debug(
            "Adding FixInternals constraint: type=%s, atoms=%s, value=%s",
            constraint_type,
            atom_indices,
            target_value,
        )
        self.constraints.append(constraint)

    def apply_constraints(self, atoms: Atoms) -> list[Any]:
        logger.debug("Applying %d constraints to atoms object", len(self.constraints))
        ase_constraints = []
        for constraint in self.constraints:
            ase_constraints.extend(constraint.to_ase_constraints())

        # Combine with any existing constraints
        existing_constraints = getattr(atoms, "constraints", [])
        all_constraints = existing_constraints + ase_constraints
        atoms.set_constraint(all_constraints)
        logger.debug(
            "Applied %d total constraints (%d existing, %d new)",
            len(all_constraints),
            len(existing_constraints),
            len(ase_constraints),
        )
        return all_constraints

    def get_constraint_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "fixed_atoms": [],
            "harmonic_constraints": [],
            "fixinternals_constraints": [],
        }

        for constraint in self.constraints:
            if isinstance(constraint, FixedAtomsConstraint):
                info["fixed_atoms"].extend(constraint.atom_indices)
            elif isinstance(constraint, FixInternalsConstraint):
                info["fixinternals_constraints"].append(
                    {
                        "type": constraint.constraint_type,
                        "atoms": constraint.atom_indices,
                        "target_value": constraint.target_value,
                    },
                )
            else:
                info["harmonic_constraints"].append(
                    {
                        "type": constraint.constraint_type,
                        "atoms": constraint.atom_indices,
                        "force_constant": constraint.force_constant,
                        "reference_value": constraint.reference_value,
                    },
                )

        return info


class FixedAtomsConstraint:
    def __init__(self, atom_indices: list[int], reference_atoms: Atoms) -> None:
        self.atom_indices = atom_indices
        self.reference_positions = reference_atoms.positions[atom_indices].copy()

    def to_ase_constraints(self) -> list:
        return [FixAtoms(indices=self.atom_indices)]


class HarmonicPositionConstraint:
    def __init__(
        self,
        atom_indices: list[int],
        reference_atoms: Atoms,
        force_constant: float = 10.0,
    ) -> None:
        self.atom_indices = atom_indices
        self.reference_positions = reference_atoms.positions[atom_indices].copy()
        self.force_constant = force_constant
        self.constraint_type = "position"
        self.reference_value = self.reference_positions.copy()

    def to_ase_constraints(self) -> list:
        return []


class HarmonicBondConstraint:
    def __init__(
        self,
        atom_indices: list[int],
        reference_atoms: Atoms,
        force_constant: float = 10.0,
    ) -> None:
        if len(atom_indices) != 2:
            msg = "Bond constraint requires exactly 2 atoms"
            logger.error("%s, got %d atoms", msg, len(atom_indices))
            raise ValueError(msg)

        self.atom_indices = atom_indices
        self.force_constant = force_constant
        self.constraint_type = "bond"

        pos1 = reference_atoms.positions[atom_indices[0]]
        pos2 = reference_atoms.positions[atom_indices[1]]
        self.reference_value = np.linalg.norm(pos2 - pos1)

    def to_ase_constraints(self) -> list:
        return [
            Hookean(
                a1=self.atom_indices[0],
                a2=self.atom_indices[1],
                rt=self.reference_value,
                k=self.force_constant,
            ),
        ]


class HarmonicAngleConstraint:
    def __init__(
        self,
        atom_indices: list[int],
        reference_atoms: Atoms,
        force_constant: float = 5.0,
    ) -> None:
        if len(atom_indices) != 3:
            msg = "Angle constraint requires exactly 3 atoms"
            logger.error("%s, got %d atoms", msg, len(atom_indices))
            raise ValueError(msg)

        self.atom_indices = atom_indices
        self.force_constant = force_constant
        self.constraint_type = "angle"

        pos1 = reference_atoms.positions[atom_indices[0]]
        pos2 = reference_atoms.positions[atom_indices[1]]
        pos3 = reference_atoms.positions[atom_indices[2]]

        vec1 = pos1 - pos2
        vec2 = pos3 - pos2

        cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        self.reference_value = np.arccos(cos_angle)

    def to_ase_constraints(self) -> list:
        return []


class FixInternalsConstraint:
    def __init__(
        self,
        constraint_type: str,
        atom_indices: list[int],
        target_value: float | None = None,
    ) -> None:
        expected_atoms = {"bond": 2, "angle": 3, "dihedral": 4}
        if constraint_type not in expected_atoms:
            msg = f"Unknown FixInternals constraint type: {constraint_type}"
            logger.error(msg)
            raise ValueError(msg)

        n_expected = expected_atoms[constraint_type]
        if len(atom_indices) != n_expected:
            msg = f"FixInternals {constraint_type} constraint requires exactly {n_expected} atoms"
            logger.error("%s, got %d atoms", msg, len(atom_indices))
            raise ValueError(msg)

        self.constraint_type = constraint_type
        self.atom_indices = atom_indices
        self.target_value = target_value

    def to_ase_constraints(self) -> list:
        internal_coordinate = [self.target_value, self.atom_indices]

        if self.constraint_type == "bond":
            return [FixInternals(bonds=[internal_coordinate])]
        if self.constraint_type == "angle":
            return [FixInternals(angles_deg=[internal_coordinate])]
        if self.constraint_type == "dihedral":
            return [FixInternals(dihedrals_deg=[internal_coordinate])]

        msg = f"Unknown FixInternals constraint type: {self.constraint_type}"
        logger.error(msg)
        raise ValueError(msg)


def parse_constraint_string(constraint_str: str, reference_atoms: Atoms) -> QMEConstraintManager:
    constraint_manager = QMEConstraintManager(reference_atoms)
    constraint_specs = [spec.strip() for spec in constraint_str.split(";") if spec.strip()]

    for spec in constraint_specs:
        parts = spec.strip().split()
        if len(parts) < 2:
            msg = f"Invalid constraint specification: {spec}"
            logger.error("%s (expected format: 'type indices [k=value]')", msg)
            raise ValueError(msg)

        constraint_type = parts[0]

        if constraint_type == "fix":
            atom_indices = [int(x.strip()) for x in parts[1].split(",")]
            constraint_manager.add_fixed_atoms(atom_indices)

        elif constraint_type.startswith("harmonic_"):
            harmonic_type = constraint_type.replace("harmonic_", "")
            atom_indices = [int(x.strip()) for x in parts[1].split(",")]
            force_constant = 10.0 * eV / Ang**2
            for part in parts[2:]:
                if part.startswith("k="):
                    force_constant = float(part.replace("k=", ""))

            constraint_manager.add_harmonic_constraint(harmonic_type, atom_indices, force_constant)

        elif constraint_type.startswith("fixinternals_"):
            fixinternals_type = constraint_type.replace("fixinternals_", "")
            atom_indices = [int(x.strip()) for x in parts[1].split(",")]
            target_value = None
            for part in parts[2:]:
                if part.startswith("value="):
                    target_value = float(part.replace("value=", ""))

            constraint_manager.add_fixinternals_constraint(
                fixinternals_type,
                atom_indices,
                target_value,
            )

        else:
            msg = f"Unknown constraint type: {constraint_type}"
            logger.error(
                "%s (supported types: fix, harmonic_position, harmonic_bond, harmonic_angle, "
                "fixinternals_bond, fixinternals_angle, fixinternals_dihedral)",
                msg,
            )
            raise ValueError(msg)

    logger.debug("Parsed %d constraint specifications", len(constraint_specs))
    return constraint_manager


def validate_atom_indices(atom_indices: list[int], atoms: Atoms) -> bool:
    n_atoms = len(atoms)
    for idx in atom_indices:
        assert isinstance(idx, int), f"Atom index must be integer, got {type(idx)}"
        if idx < 0 or idx >= n_atoms:
            msg = f"Atom index {idx} out of range [0, {n_atoms - 1}]"
            logger.error("%s", msg)
            raise ValueError(msg)

    return True


def get_constraint_summary(atoms: Atoms) -> dict[str, Any]:
    """Get summary of applied constraints on an atoms object.

    Parameters
    ----------
    - atoms: Atoms object with constraints

    Returns
    -------
        Dictionary summarizing constraints

    """
    summary: dict[str, Any] = {
        "fixed_atoms": [],
        "hookean_constraints": [],
        "fixinternals_constraints": [],
        "other_constraints": [],
    }

    constraints = getattr(atoms, "constraints", [])

    for constraint in constraints:
        if isinstance(constraint, FixAtoms):
            # Use hasattr to safely access constraint attributes
            if hasattr(constraint, "index"):
                # FixAtoms has an 'index' attribute that contains the indices
                indices = getattr(constraint, "index", [])
                if isinstance(indices, list | tuple | np.ndarray):
                    summary["fixed_atoms"].extend(indices)
                else:
                    summary["fixed_atoms"].append(indices)

        elif isinstance(constraint, Hookean):
            # Use hasattr to safely access Hookean constraint attributes
            constraint_info: dict[str, Any] = {}

            # Get atom indices
            atoms_list = []
            if hasattr(constraint, "a1"):
                atoms_list.append(constraint.a1)
            if hasattr(constraint, "a2"):
                a2 = getattr(constraint, "a2", None)
                if a2 is not None:
                    atoms_list.append(a2)
                    constraint_info["type"] = "bond"
                else:
                    constraint_info["type"] = "position"

            constraint_info["atoms"] = atoms_list

            # Get reference value and force constant
            if hasattr(constraint, "rt"):
                constraint_info["reference"] = constraint.rt
            if hasattr(constraint, "k"):
                constraint_info["force_constant"] = constraint.k

            summary["hookean_constraints"].append(constraint_info)

        elif isinstance(constraint, FixInternals):
            summary["fixinternals_constraints"].append(
                {"type": type(constraint).__name__, "constraint": constraint},
            )

        else:
            summary["other_constraints"].append(
                {"type": type(constraint).__name__, "constraint": constraint},
            )

    return summary
