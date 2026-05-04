"""Constraint parsing and handling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from qme.constraints.constraints import parse_constraint_string
from qme.utils.logging import get_qme_logger

if TYPE_CHECKING:
    from ase import Atoms
    from ase.constraints import FixConstraint
else:
    from ase.constraints import FixConstraint  # noqa: PLC0415

logger = get_qme_logger(__name__)


def parse_constraints(
    constraint_specs: str | list[str] | list[FixConstraint] | dict[str, Any],
    atoms: Atoms,
    verbose: int = 0,
) -> list[FixConstraint]:
    """Parse constraint specifications and return ASE-compatible constraints.

    Supports simplified constraint parsing with three core types:
    1. Fixed Atoms: Exactly fix atom positions
    2. Harmonic Constraints: Soft constraints based on initial geometry
    3. FixInternals: Select target values for bonds, angles, and dihedrals

    Parameters
    ----------
    constraint_specs : str, list of str, list of FixConstraint, or dict
        Constraint specifications in various formats
    atoms : ase.Atoms
        Atoms object (used as reference geometry)
    verbose : int, default 0
        Verbosity level (0=quiet, 1=normal, 2=verbose)

    Returns
    -------
    List[FixConstraint]
        List of ASE-compatible constraint objects

    Notes
    -----
    Supported string formats:
    - "fix 0,1,2,3": Fix atoms at indices 0,1,2,3
    - "harmonic_position 5,6 k=10.0": Harmonic position constraint for atoms 5,6
    - "harmonic_bond 0,1 k=5.0": Harmonic bond constraint between atoms 0,1
    - "harmonic_angle 0,1,2 k=2.0": Harmonic angle constraint for atoms 0,1,2
    - "fixinternals_bond 0,1 value=1.25": Fix bond length with ASE FixInternals
    - "fixinternals_angle 0,1,2 value=109.5": Fix angle in degrees
    - "fixinternals_dihedral 0,1,2,3 value=180.0": Fix dihedral in degrees

    Multiple constraints can be separated by semicolons:
    - "fix 0,1; harmonic_bond 2,3 k=5.0; fixinternals_bond 4,5"

    """
    # Handle different input formats
    if isinstance(constraint_specs, str):
        constraint_manager = parse_constraint_string(constraint_specs, atoms)
    elif isinstance(constraint_specs, list):
        # Handle list of pre-made ASE constraints
        if all(callable(c) or hasattr(c, "adjust_positions") for c in constraint_specs):
            # Already ASE constraints - cast to ensure type safety
            constraints = cast(list[FixConstraint], constraint_specs)
            atoms.set_constraint(constraints)
            return constraints
        # List of constraint specifications to parse
        constraint_strings: list[str] = []
        for spec in constraint_specs:
            if isinstance(spec, str):
                constraint_strings.append(spec)
            else:
                msg = f"Unsupported constraint specification in list: {spec}"
                raise ValueError(msg)

        constraint_str = "; ".join(constraint_strings)
        constraint_manager = parse_constraint_string(constraint_str, atoms)
    elif isinstance(constraint_specs, dict):
        msg = "Dict constraint specifications not yet implemented"
        raise NotImplementedError(msg)
    # All cases are covered by the type signature: str | list[str] | list[FixConstraint] | dict[str, Any]

    # Apply constraints to atoms and get ASE constraint list
    constraint_manager.apply_constraints(atoms)

    if verbose >= 1:
        info = constraint_manager.get_constraint_info()
        logger.info("Applied constraints:")
        if info["fixed_atoms"]:
            logger.info("  Fixed atoms: %s", info["fixed_atoms"])
        for hc in info["harmonic_constraints"]:
            logger.info(
                "  Harmonic %s: atoms %s, k=%s, ref=%.3f",
                hc["type"],
                hc["atoms"],
                hc["force_constant"],
                hc["reference_value"],
            )
        for fic in info["fixinternals_constraints"]:
            logger.info(
                "  FixInternals %s: atoms %s, value=%s",
                fic["type"],
                fic["atoms"],
                fic["target_value"],
            )

    # atoms.constraints is a list of FixConstraint objects
    return cast(list[FixConstraint], list(atoms.constraints))
