"""Utility functions for TRIC implementation."""

import numpy as np
from typing import List, Tuple, Union


# Atomic masses (in atomic units)
ATOMIC_MASSES = {
    'H': 1.007825,
    'C': 12.000000,
    'N': 14.003074,
    'O': 15.994915,
    'F': 18.998403,
    'P': 30.973762,
    'S': 31.972071,
    'Cl': 34.968853,
    'Br': 78.918338,
    'I': 126.904473,
}


def atomic_masses(symbols: List[str]) -> np.ndarray:
    """Get atomic masses for a list of atomic symbols.
    
    Parameters
    ----------
    symbols : List[str]
        List of atomic symbols
        
    Returns
    -------
    np.ndarray
        Array of atomic masses
    """
    masses = []
    for symbol in symbols:
        mass = ATOMIC_MASSES.get(symbol, 12.0)  # Default to carbon mass
        masses.append(mass)
    return np.array(masses)


class Geometry:
    """Simple geometry class for internal coordinate calculations."""
    
    def __init__(self, symbols: List[str], positions: np.ndarray):
        """Initialize geometry.
        
        Parameters
        ----------
        symbols : List[str]
            Atomic symbols
        positions : np.ndarray
            Atomic positions (N, 3) in Angstroms
        """
        self.symbols = symbols
        self.positions = np.array(positions, dtype=float)
        self.n_atoms = len(symbols)
        
        if self.positions.shape != (self.n_atoms, 3):
            raise ValueError(f"Positions shape {self.positions.shape} != ({self.n_atoms}, 3)")
    
    @classmethod
    def from_atoms(cls, atoms):
        """Create Geometry from ASE Atoms object."""
        return cls(atoms.get_chemical_symbols(), atoms.get_positions())
    
    def __len__(self):
        return self.n_atoms
    
    def __getitem__(self, idx):
        return self.symbols[idx], self.positions[idx]
    
    def copy(self):
        """Create a copy of the geometry."""
        return Geometry(self.symbols.copy(), self.positions.copy())
    
    def get_masses(self) -> np.ndarray:
        """Get atomic masses."""
        return atomic_masses(self.symbols)
    
    def center_of_mass(self) -> np.ndarray:
        """Calculate center of mass."""
        masses = self.get_masses()
        return np.average(self.positions, weights=masses, axis=0)
    
    def moment_of_inertia_tensor(self) -> np.ndarray:
        """Calculate moment of inertia tensor."""
        masses = self.get_masses()
        com = self.center_of_mass()
        
        # Translate to center of mass
        positions = self.positions - com
        
        # Calculate inertia tensor
        I = np.zeros((3, 3))
        for i in range(self.n_atoms):
            r = positions[i]
            m = masses[i]
            
            # I_xx = sum(m * (y^2 + z^2))
            I[0, 0] += m * (r[1]**2 + r[2]**2)
            # I_yy = sum(m * (x^2 + z^2))
            I[1, 1] += m * (r[0]**2 + r[2]**2)
            # I_zz = sum(m * (x^2 + y^2))
            I[2, 2] += m * (r[0]**2 + r[1]**2)
            
            # Off-diagonal elements
            I[0, 1] -= m * r[0] * r[1]
            I[0, 2] -= m * r[0] * r[2]
            I[1, 2] -= m * r[1] * r[2]
        
        # Make symmetric
        I[1, 0] = I[0, 1]
        I[2, 0] = I[0, 2]
        I[2, 1] = I[1, 2]
        
        return I


def find_bonds(geometry: Geometry, factor: float = 1.3) -> List[Tuple[int, int]]:
    """Find bonds between atoms based on covalent radii.
    
    Parameters
    ----------
    geometry : Geometry
        Molecular geometry
    factor : float, default 1.3
        Factor to multiply covalent radii
        
    Returns
    -------
    List[Tuple[int, int]]
        List of (i, j) bond pairs
    """
    # Covalent radii (in Angstroms)
    covalent_radii = {
        'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
        'P': 1.07, 'S': 1.05, 'Cl': 0.99, 'Br': 1.20, 'I': 1.39
    }
    
    bonds = []
    positions = geometry.positions
    
    for i in range(len(geometry)):
        for j in range(i + 1, len(geometry)):
            # Get covalent radii
            r_i = covalent_radii.get(geometry.symbols[i], 1.0)
            r_j = covalent_radii.get(geometry.symbols[j], 1.0)
            
            # Calculate distance
            distance = np.linalg.norm(positions[i] - positions[j])
            
            # Check if bonded
            if distance <= factor * (r_i + r_j):
                bonds.append((i, j))
    
    return bonds


def find_angles(bonds: List[Tuple[int, int]]) -> List[Tuple[int, int, int]]:
    """Find angles from bond connectivity.
    
    Parameters
    ----------
    bonds : List[Tuple[int, int]]
        List of bonds
        
    Returns
    -------
    List[Tuple[int, int, int]]
        List of (i, j, k) angle triplets where j is the central atom
    """
    # Build adjacency list
    adjacency = {}
    for i, j in bonds:
        if i not in adjacency:
            adjacency[i] = []
        if j not in adjacency:
            adjacency[j] = []
        adjacency[i].append(j)
        adjacency[j].append(i)
    
    angles = []
    for j in adjacency:
        neighbors = adjacency[j]
        if len(neighbors) >= 2:
            # Find all angle triplets with j as central atom
            for i in range(len(neighbors)):
                for k in range(i + 1, len(neighbors)):
                    angles.append((neighbors[i], j, neighbors[k]))
    
    return angles


def find_dihedrals(bonds: List[Tuple[int, int]]) -> List[Tuple[int, int, int, int]]:
    """Find dihedral angles from bond connectivity.
    
    Parameters
    ----------
    bonds : List[Tuple[int, int]]
        List of bonds
        
    Returns
    -------
    List[Tuple[int, int, int, int]]
        List of (i, j, k, l) dihedral quadruplets
    """
    # Build adjacency list
    adjacency = {}
    for i, j in bonds:
        if i not in adjacency:
            adjacency[i] = []
        if j not in adjacency:
            adjacency[j] = []
        adjacency[i].append(j)
        adjacency[j].append(i)
    
    dihedrals = []
    for j in adjacency:
        for k in adjacency[j]:
            if k != j:
                # Find dihedrals j-k-l-i where j-k and k-l are bonds
                for l in adjacency[k]:
                    if l != j and l != k:
                        for i in adjacency[j]:
                            if i != k and i != l:
                                dihedrals.append((i, j, k, l))
    
    return dihedrals


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize a vector, handling zero vectors."""
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return np.zeros_like(v)
    return v / norm


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate angle between two vectors in radians."""
    v1_norm = normalize_vector(v1)
    v2_norm = normalize_vector(v2)
    
    if np.allclose(v1_norm, 0) or np.allclose(v2_norm, 0):
        return 0.0
    
    # Ensure dot product is in valid range for arccos
    dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    return np.arccos(dot_product)


def dihedral_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> float:
    """Calculate dihedral angle between four points.
    
    Parameters
    ----------
    p1, p2, p3, p4 : np.ndarray
        Four points defining the dihedral angle
        
    Returns
    -------
    float
        Dihedral angle in radians
    """
    # Vectors along the bonds
    v1 = p2 - p1  # p1 -> p2
    v2 = p3 - p2  # p2 -> p3  
    v3 = p4 - p3  # p3 -> p4
    
    # Normal vectors to the planes
    n1 = np.cross(v1, v2)
    n2 = np.cross(v2, v3)
    
    # Normalize
    n1 = normalize_vector(n1)
    n2 = normalize_vector(n2)
    
    if np.allclose(n1, 0) or np.allclose(n2, 0):
        return 0.0
    
    # Calculate angle between normal vectors
    dot_product = np.clip(np.dot(n1, n2), -1.0, 1.0)
    angle = np.arccos(dot_product)
    
    # Determine sign using cross product
    sign = np.sign(np.dot(np.cross(n1, n2), v2))
    
    return sign * angle
