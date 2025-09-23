"""Molecular geometry handling for QME."""

from typing import List, Optional

import numpy as np


class Geometry:
    """Class for handling molecular geometries and properties.

    Inspired by pysisyphus geometry handling but simplified for MLP/NNP usage.
    """

    def __init__(
        self,
        atoms: List[str],
        coords: np.ndarray,
        energy: Optional[float] = None,
        forces: Optional[np.ndarray] = None,
        charge: int = 0,
        mult: int = 1,
    ):
        """Initialize a molecular geometry.

        Args:
            atoms: List of atomic symbols
            coords: 3N array of Cartesian coordinates in Angstroms
            energy: Energy in Hartree (optional)
            forces: 3N array of forces in Hartree/Bohr (optional)
            charge: Molecular charge
            mult: Spin multiplicity
        """
        self.atoms = atoms
        self.coords = np.array(coords).flatten()
        self.energy = energy
        self.forces = forces
        self.charge = charge
        self.mult = mult

        # Validate dimensions
        if len(self.coords) != 3 * len(atoms):
            raise ValueError("Coordinates array must have 3*N elements")

        if forces is not None and len(forces) != len(self.coords):
            raise ValueError("Forces array must match coordinates dimensions")

    @property
    def coords3d(self) -> np.ndarray:
        """Return coordinates as Nx3 array."""
        return self.coords.reshape(-1, 3)

    @coords3d.setter
    def coords3d(self, coords: np.ndarray):
        """Set coordinates from Nx3 array."""
        self.coords = coords.flatten()

    @property
    def natoms(self) -> int:
        """Number of atoms."""
        return len(self.atoms)

    def copy(self) -> "Geometry":
        """Create a copy of this geometry."""
        return Geometry(
            atoms=self.atoms.copy(),
            coords=self.coords.copy(),
            energy=self.energy,
            forces=self.forces.copy() if self.forces is not None else None,
            charge=self.charge,
            mult=self.mult,
        )

    def center_of_mass(self) -> np.ndarray:
        """Calculate center of mass."""
        masses = np.array([self._atomic_mass(atom) for atom in self.atoms])
        total_mass = masses.sum()
        com = np.zeros(3)

        for i, mass in enumerate(masses):
            com += mass * self.coords3d[i]

        return com / total_mass

    def rmsd(self, other: "Geometry") -> float:
        """Calculate RMSD with another geometry."""
        if len(self.atoms) != len(other.atoms):
            raise ValueError("Geometries must have same number of atoms")

        coords_diff = self.coords3d - other.coords3d
        return np.sqrt(np.sum(coords_diff**2) / self.natoms)

    def as_xyz(self) -> str:
        """Return geometry in XYZ format."""
        xyz_lines = [str(self.natoms)]

        comment = ""
        if self.energy is not None:
            comment = f"Energy: {self.energy:.8f} Hartree"
        xyz_lines.append(comment)

        for atom, coord in zip(self.atoms, self.coords3d):
            xyz_lines.append(
                f"{atom:2s} {coord[0]:12.8f} {coord[1]:12.8f} {coord[2]:12.8f}"
            )

        return "\n".join(xyz_lines)

    @classmethod
    def from_xyz(cls, xyz_str: str, charge: int = 0, mult: int = 1) -> "Geometry":
        """Create geometry from XYZ string."""
        lines = xyz_str.strip().split("\n")
        natoms = int(lines[0])

        atoms = []
        coords = []

        for i in range(2, 2 + natoms):
            parts = lines[i].split()
            atoms.append(parts[0])
            coords.extend([float(x) for x in parts[1:4]])

        return cls(atoms=atoms, coords=np.array(coords), charge=charge, mult=mult)

    def _atomic_mass(self, symbol: str) -> float:
        """Get atomic mass for common elements (simplified)."""
        masses = {
            "H": 1.008,
            "C": 12.011,
            "N": 14.007,
            "O": 15.999,
            "F": 18.998,
            "P": 30.974,
            "S": 32.06,
            "Cl": 35.45,
            "Br": 79.904,
            "I": 126.904,
        }
        return masses.get(symbol, 1.0)  # Default to 1 for unknown elements

    def __repr__(self) -> str:
        """String representation."""
        return f"Geometry(natoms={self.natoms}, energy={self.energy})"
