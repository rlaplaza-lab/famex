"""Geometry class for representing molecular structures in QME."""

from __future__ import annotations

from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read, write

from qme.io.xyz_io import read_xyz_with_metadata, write_xyz_with_metadata


class Geometry(Atoms):
    """Molecular geometry with atomic positions, types, and properties."""

    def __init__(
        self,
        atoms: str | list[str] | None = None,
        coords: np.ndarray | None = None,
        positions: np.ndarray | None = None,
        charge: int = 0,
        mult: int = 1,
        ase_atoms: Atoms | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a Geometry object."""
        if ase_atoms is not None:
            super().__init__(
                symbols=ase_atoms.get_chemical_symbols(),
                positions=ase_atoms.get_positions(),
                cell=ase_atoms.get_cell(),
                pbc=ase_atoms.get_pbc(),
                **kwargs,
            )
            if ase_atoms.calc is not None:
                self.calc = ase_atoms.calc
            if hasattr(ase_atoms, "info") and ase_atoms.info:
                self.info = dict(ase_atoms.info)

            if hasattr(ase_atoms, "info") and ase_atoms.info:
                if "charge" in ase_atoms.info:
                    charge = ase_atoms.info["charge"]
                if "spin" in ase_atoms.info:
                    mult = ase_atoms.info["spin"]
        elif atoms is not None:
            symbols = list(atoms) if isinstance(atoms, str) else atoms

            if coords is not None:
                coords = np.array(coords)
                positions = coords.reshape(-1, 3) if coords.ndim == 1 else coords
            elif positions is not None:
                positions = np.array(positions)
            else:
                raise ValueError("Must provide either 'coords' or 'positions'")

            super().__init__(symbols=symbols, positions=positions, **kwargs)
        else:
            super().__init__(**kwargs)

        self.charge = charge
        self.mult = mult
        self._energy: float | None = None
        self._forces: np.ndarray | None = None

    @property
    def coords3d(self) -> np.ndarray:
        """Get coordinates as (n_atoms, 3) array."""
        return np.array(self.get_positions())

    @coords3d.setter
    def coords3d(self, positions: np.ndarray) -> None:
        """Set coordinates from (n_atoms, 3) array."""
        self.set_positions(positions)

    @property
    def coords(self) -> np.ndarray:
        """Get coordinates as flat array [x1, y1, z1, x2, y2, z2, ...]."""
        return np.array(self.coords3d).flatten()

    @coords.setter
    def coords(self, coords: np.ndarray) -> None:
        """Set coordinates from flat array."""
        coords = np.array(coords)
        self.coords3d = coords.reshape(-1, 3)

    def get_symbols(self) -> list[str]:
        """Get atomic symbols as list."""
        return list(super().get_chemical_symbols())

    @property
    def energy(self) -> float | None:
        """Get energy if calculated."""
        if self.calc is not None:
            try:
                energy_val = self.get_potential_energy()
                return float(energy_val) if energy_val is not None else None
            except Exception:
                return self._energy
        return self._energy

    @energy.setter
    def energy(self, value: float | None) -> None:
        """Set energy value."""
        self._energy = value

    def get_forces(self, apply_constraint: bool = True, md: bool = False) -> np.ndarray | None:
        """Get forces if calculated."""
        if self.calc is not None:
            try:
                forces = super().get_forces(apply_constraint, md)
                return np.array(forces) if forces is not None else None
            except Exception:
                return getattr(self, "_forces", None)
        return getattr(self, "_forces", None)

    def copy(self) -> Geometry:
        """Create a copy of the geometry."""
        atoms_copy = super().copy()
        new_geom = Geometry(ase_atoms=atoms_copy, charge=self.charge, mult=self.mult)
        new_geom._energy = self._energy
        new_geom._forces = self._forces
        return new_geom

    def get_distance_between(self, atom1: int, atom2: int) -> float:
        """Get distance between two atoms."""
        return float(self.get_distance(atom1, atom2))

    def get_all_pairwise_distances(self) -> np.ndarray:
        """Get all pairwise distances."""
        return np.array(super().get_all_distances())

    def get_angle_degrees(self, atom1: int, atom2: int, atom3: int) -> float:
        """Get angle between three atoms in degrees (atom2 is the center atom)."""
        return float(self.get_angle(atom1, atom2, atom3) * 180.0 / np.pi)

    def get_dihedral_degrees(self, atom1: int, atom2: int, atom3: int, atom4: int) -> float:
        """Get dihedral angle between four atoms in degrees."""
        return float(self.get_dihedral(atom1, atom2, atom3, atom4) * 180.0 / np.pi)

    def center_of_mass(self) -> np.ndarray:
        """Get center of mass coordinates."""
        return np.array(self.get_center_of_mass())

    def __str__(self) -> str:
        """Return string representation."""
        return f"Geometry({len(self)} atoms, charge={self.charge}, mult={self.mult})"

    def __repr__(self) -> str:
        return self.__str__()


def read_geometry(filename: str, **kwargs: Any) -> Geometry | list[Geometry]:
    """Read geometry from file using ASE or custom XYZ reader."""
    filename_str = str(filename)

    if filename_str.lower().endswith(".xyz"):
        frame = kwargs.pop("frame", "first")
        return read_xyz_with_metadata(filename_str, frame=frame, **kwargs)

    atoms = read(filename, **kwargs)
    if isinstance(atoms, list):
        return [Geometry(ase_atoms=atom) for atom in atoms]
    return Geometry(ase_atoms=atoms)


def write_geometry(geometry: Geometry | Atoms, filename: str, **kwargs: Any) -> None:
    """Write geometry to file using ASE or custom XYZ writer."""
    filename_str = str(filename)

    if filename_str.lower().endswith(".xyz"):
        energy = kwargs.pop("energy", None)
        write_xyz_with_metadata(geometry, filename_str, energy=energy, **kwargs)
        return

    write(filename, geometry, **kwargs)


def read_gaussian_input(filename: str) -> tuple[Geometry, str]:
    """Read a Gaussian input file and determine calculation type."""
    with open(filename) as f:
        lines = f.readlines()

    route_line = ""
    job_type = None
    charge_mult_line_index = -1

    for i, line in enumerate(lines):
        line_lower = line.strip().lower()
        if line_lower.startswith("#"):
            route_line = line_lower
            charge_mult_line_index = i + 3
            break

    if not route_line:
        msg = "Route section (starting with #) not found in Gaussian input."
        raise ValueError(msg)

    if "opt=ts" in route_line or "opt=saddle" in route_line:
        job_type = "transition_state"
    elif "opt" in route_line:
        job_type = "minimize"
    else:
        msg = "Could not determine job type. Route line must contain 'opt' or 'opt=ts'."
        raise ValueError(msg)

    if charge_mult_line_index >= len(lines) or not lines[charge_mult_line_index].strip():
        for i in range(charge_mult_line_index - 2, len(lines)):
            if len(lines[i].strip().split()) == 2 and all(
                c.isdigit() or c == "-" for c in lines[i].strip().split()[0]
            ):
                charge_mult_line_index = i
                break
        else:
            msg = "Could not find charge and multiplicity line."
            raise ValueError(msg)

    try:
        parts = lines[charge_mult_line_index].strip().split()
        charge = int(parts[0])
        multiplicity = int(parts[1])
    except (ValueError, IndexError):
        msg = f"Invalid charge/multiplicity line: '{lines[charge_mult_line_index].strip()}'"
        raise ValueError(
            msg,
        )

    coord_lines = lines[charge_mult_line_index + 1 :]
    symbols = []
    positions = []
    for line in coord_lines:
        parts = line.strip().split()
        if len(parts) == 4:
            try:
                symbols.append(parts[0])
                positions.append([float(p) for p in parts[1:]])
            except ValueError:
                break
        elif len(parts) == 0:
            break

    if not symbols:
        msg = "No atomic coordinates found in the input file."
        raise ValueError(msg)

    atoms = Atoms(symbols=symbols, positions=positions)
    return Geometry(ase_atoms=atoms, charge=charge, mult=multiplicity), job_type
