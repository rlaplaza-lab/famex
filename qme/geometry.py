"""
Geometry class for representing molecular structures in QME.
"""

from typing import List, Optional, Union

import numpy as np
from ase import Atoms
from ase.io import read, write


class Geometry:
    """
    Represents a molecular geometry with atomic positions, types, and properties.

    This class provides a high-level interface for molecular structures,
    wrapping ASE Atoms objects with additional convenience methods for
    reaction pathway analysis and geometry manipulation.
    """

    def __init__(
        self,
        atoms: Optional[Union[str, List[str]]] = None,
        coords: Optional[np.ndarray] = None,
        positions: Optional[np.ndarray] = None,
        charge: int = 0,
        mult: int = 1,
        ase_atoms: Optional[Atoms] = None,
    ):
        """
        Initialize a Geometry object.

        Parameters
        ----------
        atoms : str, list of str, optional
            Atomic symbols (e.g., ['H', 'H'] or 'HH')
        coords : np.ndarray, optional
            Flat array of coordinates [x1, y1, z1, x2, y2, z2, ...]
        positions : np.ndarray, optional
            Array of shape (n_atoms, 3) with atomic positions
        charge : int, default 0
            Total charge of the system
        mult : int, default 1
            Spin multiplicity (2S + 1)
        ase_atoms : Atoms, optional
            Pre-constructed ASE Atoms object
        """

        if ase_atoms is not None:
            self.atoms = ase_atoms.copy()
        else:
            if atoms is None:
                raise ValueError("Must provide either 'atoms' or 'ase_atoms'")

            # Convert atomic symbols to list if string
            if isinstance(atoms, str):
                symbols = list(atoms)
            else:
                symbols = atoms

            # Handle coordinates
            if coords is not None:
                coords = np.array(coords)
                if coords.ndim == 1:
                    positions = coords.reshape(-1, 3)
                else:
                    positions = coords
            elif positions is not None:
                positions = np.array(positions)
            else:
                raise ValueError("Must provide either 'coords' or 'positions'")

            # Create ASE Atoms object
            self.atoms = Atoms(symbols=symbols, positions=positions)

        # Store additional properties
        self.charge = charge
        self.mult = mult
        self._energy = None
        self._forces = None

    @property
    def coords3d(self) -> np.ndarray:
        """Get coordinates as (n_atoms, 3) array."""
        return self.atoms.get_positions()

    @coords3d.setter
    def coords3d(self, positions: np.ndarray):
        """Set coordinates from (n_atoms, 3) array."""
        self.atoms.set_positions(positions)

    @property
    def coords(self) -> np.ndarray:
        """Get coordinates as flat array [x1, y1, z1, x2, y2, z2, ...]."""
        return self.coords3d.flatten()

    @coords.setter
    def coords(self, coords: np.ndarray):
        """Set coordinates from flat array."""
        coords = np.array(coords)
        self.coords3d = coords.reshape(-1, 3)

    @property
    def symbols(self) -> List[str]:
        """Get atomic symbols."""
        return self.atoms.get_chemical_symbols()

    @property
    def energy(self) -> Optional[float]:
        """Get energy if calculated."""
        if self.atoms.calc is not None:
            try:
                return self.atoms.get_potential_energy()
            except Exception:
                return self._energy
        return self._energy

    @energy.setter
    def energy(self, value: float):
        """Set energy value."""
        self._energy = value

    @property
    def forces(self) -> Optional[np.ndarray]:
        """Get forces if calculated."""
        if self.atoms.calc is not None:
            try:
                return self.atoms.get_forces()
            except Exception:
                return self._forces
        return self._forces

    @forces.setter
    def forces(self, value: np.ndarray):
        """Set forces."""
        self._forces = value

    def copy(self) -> "Geometry":
        """Create a copy of the geometry."""
        return Geometry(ase_atoms=self.atoms.copy(), charge=self.charge, mult=self.mult)

    def get_distance(self, atom1: int, atom2: int) -> float:
        """Get distance between two atoms."""
        return self.atoms.get_distance(atom1, atom2)

    def get_distances(self) -> np.ndarray:
        """Get all pairwise distances."""
        return self.atoms.get_all_distances()

    def get_angle(self, atom1: int, atom2: int, atom3: int) -> float:
        """Get angle between three atoms in degrees.

        Parameters
        ----------
        atom1, atom2, atom3 : int
            Atom indices (atom2 is the center atom)

        Returns
        -------
        float
            Angle in degrees
        """
        return self.atoms.get_angle(atom1, atom2, atom3) * 180.0 / np.pi

    def get_dihedral(self, atom1: int, atom2: int, atom3: int, atom4: int) -> float:
        """Get dihedral angle between four atoms in degrees.

        Parameters
        ----------
        atom1, atom2, atom3, atom4 : int
            Atom indices

        Returns
        -------
        float
            Dihedral angle in degrees
        """
        return self.atoms.get_dihedral(atom1, atom2, atom3, atom4) * 180.0 / np.pi

    def center_of_mass(self) -> np.ndarray:
        """Get center of mass coordinates.

        Returns
        -------
        np.ndarray
            Center of mass coordinates (x, y, z)
        """
        return self.atoms.get_center_of_mass()

    def __len__(self) -> int:
        """Number of atoms."""
        return len(self.atoms)

    def __str__(self) -> str:
        """String representation."""
        return f"Geometry({len(self)} atoms, charge={self.charge}, mult={self.mult})"

    def __repr__(self) -> str:
        return self.__str__()


def read_geometry(filename: str, **kwargs) -> Geometry:
    """
    Read geometry from file using ASE.

    Parameters
    ----------
    filename : str
        Path to geometry file
    **kwargs
        Additional arguments passed to ase.io.read

    Returns
    -------
    Geometry
        Loaded geometry
    """
    atoms = read(filename, **kwargs)
    return Geometry(ase_atoms=atoms)


def write_geometry(geometry: Geometry, filename: str, **kwargs):
    """
    Write geometry to file using ASE.

    Parameters
    ----------
    geometry : Geometry
        Geometry to write
    filename : str
        Output filename
    **kwargs
        Additional arguments passed to ase.io.write
    """
    write(filename, geometry.atoms, **kwargs)
