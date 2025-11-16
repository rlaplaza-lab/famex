"""Constraint management for QME Explorer.

This module provides the ConstraintManager class for handling constraint parsing
and application to atoms objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ase import Atoms

from qme.constraints.parser import parse_constraints

if TYPE_CHECKING:
    pass


class ConstraintManager:
    """Manages constraint parsing and application for Explorer.

    Handles constraint specification parsing and application to atoms objects.
    Provides caching of parsed constraints for performance.
    """

    def __init__(
        self,
        constraints_spec: str | list | dict | None = None,
        cache_parsed: bool = True,
    ) -> None:
        """Initialize ConstraintManager.

        Parameters
        ----------
        constraints_spec : str, list, dict, optional
            Constraint specification. Can be:
            - String: "fix 0 1 2" (fix atoms 0, 1, 2)
            - List: [FixAtoms(indices=[0, 1, 2])]
            - Dict: {"fix": [0, 1, 2]}
        cache_parsed : bool, default True
            Whether to cache parsed constraints for reuse
        """
        self.constraints_spec = constraints_spec
        self.cache_parsed = cache_parsed
        self._cached_constraints: list[Any] | None = None
        self._cached_atoms_hash: int | None = None

    def apply_constraints(self, atoms: Atoms) -> list[Any]:
        """Parse and apply constraints to atoms if specified.

        Returns the ASE constraints list after application.

        Parameters
        ----------
        atoms : Atoms
            Atoms object to apply constraints to

        Returns
        -------
        list[Any]
            List of ASE constraint objects
        """
        if self.constraints_spec is None:
            return []

        # Use cached constraints if available and caching is enabled
        if self.cache_parsed and self._cached_constraints is not None:
            # Check if atoms structure matches cached constraints
            # (simple hash check - could be improved)
            atoms_hash = hash((len(atoms), tuple(atoms.get_chemical_symbols())))
            if atoms_hash == self._cached_atoms_hash:
                return self._cached_constraints.copy()

        # Parse constraints
        constraints = parse_constraints(self.constraints_spec, atoms, verbose=False)

        # Cache if enabled
        if self.cache_parsed:
            self._cached_constraints = constraints.copy()
            atoms_hash = hash((len(atoms), tuple(atoms.get_chemical_symbols())))
            self._cached_atoms_hash = atoms_hash

        return constraints

    def clear_cache(self) -> None:
        """Clear the cached constraints."""
        self._cached_constraints = None
        self._cached_atoms_hash = None

    def update_constraints(self, constraints_spec: str | list | dict | None) -> None:
        """Update the constraint specification and clear cache.

        Parameters
        ----------
        constraints_spec : str, list, dict, optional
            New constraint specification
        """
        self.constraints_spec = constraints_spec
        self.clear_cache()
