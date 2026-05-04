"""Constraint management for QME Explorer."""

from __future__ import annotations

from typing import Any

from ase import Atoms

from qme.constraints.parser import parse_constraints


class ConstraintManager:
    """Manages constraint parsing and application for Explorer."""

    def __init__(
        self,
        constraints_spec: str | list | dict | None = None,
        cache_parsed: bool = True,
    ) -> None:
        self.constraints_spec = constraints_spec
        self.cache_parsed = cache_parsed
        self._cached_constraints: list[Any] | None = None
        self._cached_atoms_hash: int | None = None

    def _can_cache_constraints(self) -> bool:
        if not self.cache_parsed or not isinstance(self.constraints_spec, list):
            return False

        return all(
            callable(constraint) or hasattr(constraint, "adjust_positions")
            for constraint in self.constraints_spec
        )

    def apply_constraints(self, atoms: Atoms) -> list[Any]:
        if self.constraints_spec is None:
            return []

        can_cache = self._can_cache_constraints()

        if can_cache and self._cached_constraints is not None:
            atoms_hash = hash((len(atoms), tuple(atoms.get_chemical_symbols())))
            if atoms_hash == self._cached_atoms_hash:
                constraints = self._cached_constraints.copy()
                atoms.set_constraint(constraints)
                return constraints

        constraints = parse_constraints(self.constraints_spec, atoms, verbose=False)

        if can_cache:
            self._cached_constraints = constraints.copy()
            atoms_hash = hash((len(atoms), tuple(atoms.get_chemical_symbols())))
            self._cached_atoms_hash = atoms_hash

        return constraints

    def clear_cache(self) -> None:
        self._cached_constraints = None
        self._cached_atoms_hash = None

    def update_constraints(self, constraints_spec: str | list | dict | None) -> None:
        self.constraints_spec = constraints_spec
        self.clear_cache()
