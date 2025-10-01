"""Constraint parsing and handling."""

from typing import Any, Dict, List, Optional, Union

from ase import Atoms

from qme.core.constraints import get_constraint_summary, parse_constraint_string


def parse_constraints(
    constraint_specs: Union[str, List, Dict],
    atoms: Atoms,
    verbose: bool = False,
) -> List:
    """
    Parse constraint specifications and return ASE-compatible constraints.

    Supports simplified constraint parsing with two core types:
    1. Fixed Atoms: Exactly fix atom positions
    2. Harmonic Constraints: Soft constraints based on initial geometry

    Parameters:
    - constraint_specs: Constraint specifications in various formats
    - atoms: Atoms object (used as reference geometry)
    - verbose: Print constraint information

    Supported string formats:
    - "fix 0,1,2,3": Fix atoms at indices 0,1,2,3
    - "harmonic_position 5,6 k=10.0": Harmonic position constraint for atoms 5,6
    - "harmonic_bond 0,1 k=5.0": Harmonic bond constraint between atoms 0,1
    - "harmonic_angle 0,1,2 k=2.0": Harmonic angle constraint for atoms 0,1,2

    Multiple constraints can be separated by semicolons:
    - "fix 0,1; harmonic_bond 2,3 k=5.0"

    Returns:
        List of ASE-compatible constraint objects
    """

    if constraint_specs is None:
        return []

    # Handle different input formats
    if isinstance(constraint_specs, str):
        constraint_manager = parse_constraint_string(constraint_specs, atoms)
    elif isinstance(constraint_specs, list):
        # Handle list of pre-made ASE constraints
        if all(
            hasattr(c, "__call__") or hasattr(c, "adjust_positions")
            for c in constraint_specs
        ):
            # Already ASE constraints
            return constraint_specs
        else:
            # List of constraint specifications to parse
            constraint_strings = []
            for spec in constraint_specs:
                if isinstance(spec, str):
                    constraint_strings.append(spec)
                else:
                    raise ValueError(
                        f"Unsupported constraint specification in list: {spec}"
                    )

            constraint_str = "; ".join(constraint_strings)
            constraint_manager = parse_constraint_string(constraint_str, atoms)
    else:
        raise ValueError(
            f"Unsupported constraint specification type: {type(constraint_specs)}"
        )

    # Apply constraints to atoms and get ASE constraint list
    constraint_manager.apply_constraints(atoms)

    if verbose:
        info = constraint_manager.get_constraint_info()
        print("Applied constraints:")
        if info["fixed_atoms"]:
            print(f"  Fixed atoms: {info['fixed_atoms']}")
        for hc in info["harmonic_constraints"]:
            print(
                f"  Harmonic {hc['type']}: atoms {hc['atoms']}, "
                f"k={hc['force_constant']}, ref={hc['reference_value']:.3f}"
            )

    return atoms.constraints
