"""Translation-Rotation (TR) coordinate system for TRIC optimizer.

This module implements the TR coordinates needed for proper TRIC optimization,
including translation coordinates (center of mass) and rotation coordinates
(using exponential map approach inspired by geomeTRIC).

The TR projection removes global translation and rotation from optimization,
ensuring that only internal molecular motions are optimized.
"""

import numpy as np
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod

from .utils import Geometry


class TRCoordinate(ABC):
    """Abstract base class for TR coordinates."""
    
    def __init__(self, atoms: List[int]):
        """Initialize TR coordinate.
        
        Parameters
        ----------
        atoms : List[int]
            Atomic indices involved in this coordinate
        """
        self.atoms = atoms
    
    @abstractmethod
    def value(self, positions: np.ndarray) -> float:
        """Calculate coordinate value."""
        pass
    
    @abstractmethod
    def derivative(self, positions: np.ndarray) -> np.ndarray:
        """Calculate coordinate derivatives w.r.t. atomic positions."""
        pass


class TranslationCoordinate(TRCoordinate):
    """Translation coordinate (center of mass motion)."""
    
    def __init__(self, axis: int, atoms: List[int], masses: np.ndarray):
        """Initialize translation coordinate.
        
        Parameters
        ----------
        axis : int
            Cartesian axis (0=x, 1=y, 2=z)
        atoms : List[int]
            All atomic indices
        masses : np.ndarray
            Atomic masses
        """
        super().__init__(atoms)
        self.axis = axis
        self.masses = masses
        self.total_mass = np.sum(masses)
    
    def value(self, positions: np.ndarray) -> float:
        """Calculate center of mass along specified axis."""
        return np.sum(self.masses * positions[:, self.axis]) / self.total_mass
    
    def derivative(self, positions: np.ndarray) -> np.ndarray:
        """Calculate derivatives of COM w.r.t. atomic positions."""
        derivatives = np.zeros_like(positions)
        for i, atom_idx in enumerate(self.atoms):
            derivatives[atom_idx, self.axis] = self.masses[i] / self.total_mass
        return derivatives


class RotationCoordinate(TRCoordinate):
    """Rotation coordinate using exponential map approach (inspired by geomeTRIC)."""
    
    def __init__(self, axis: int, atoms: List[int], masses: np.ndarray, 
                 reference_positions: np.ndarray):
        """Initialize rotation coordinate.
        
        Parameters
        ----------
        axis : int
            Rotation axis (0=x, 1=y, 2=z)
        atoms : List[int]
            All atomic indices
        masses : np.ndarray
            Atomic masses
        reference_positions : np.ndarray
            Reference geometry for rotation calculation
        """
        super().__init__(atoms)
        self.axis = axis
        self.masses = masses
        self.reference_positions = reference_positions.copy()
        self.center_of_mass_ref = self._calculate_com(reference_positions)
        
        # Translate reference to COM
        self.ref_positions_com = reference_positions - self.center_of_mass_ref
        
        # Calculate reference rotation vector (exponential map)
        self.ref_rotation_vector = self._calculate_rotation_vector(
            self.ref_positions_com, self.ref_positions_com
        )
    
    def _calculate_com(self, positions: np.ndarray) -> np.ndarray:
        """Calculate center of mass."""
        return np.sum(self.masses[:, np.newaxis] * positions, axis=0) / np.sum(self.masses)
    
    def _calculate_rotation_vector(self, positions1: np.ndarray, 
                                  positions2: np.ndarray) -> np.ndarray:
        """Calculate rotation vector between two sets of positions.
        
        This implements a simplified exponential map approach for small rotations.
        """
        # For small rotations, we can use the cross product approach
        # This is a simplified version of the full exponential map
        
        rotation_vector = np.zeros(3)
        
        # Calculate rotation around each axis
        for atom_idx, atom in enumerate(self.atoms):
            r1 = positions1[atom_idx]
            r2 = positions2[atom_idx]
            
            # Cross product gives rotation direction
            cross = np.cross(r1, r2)
            rotation_vector += self.masses[atom_idx] * cross
        
        # Normalize by total mass
        rotation_vector /= np.sum(self.masses)
        
        return rotation_vector
    
    def value(self, positions: np.ndarray) -> float:
        """Calculate rotation coordinate value."""
        com_current = self._calculate_com(positions)
        positions_com = positions - com_current
        
        rotation_vector = self._calculate_rotation_vector(
            self.ref_positions_com, positions_com
        )
        
        return rotation_vector[self.axis]
    
    def derivative(self, positions: np.ndarray) -> np.ndarray:
        """Calculate rotation coordinate derivatives.
        
        This is a simplified implementation for the exponential map derivatives.
        """
        derivatives = np.zeros_like(positions)
        
        # Simplified derivative calculation
        # For a more accurate implementation, we would need the full
        # exponential map derivative formulas
        
        com_current = self._calculate_com(positions)
        positions_com = positions - com_current
        
        for atom_idx, atom in enumerate(self.atoms):
            # Derivative of rotation vector component
            r_ref = self.ref_positions_com[atom_idx]
            r_curr = positions_com[atom_idx]
            
            # Simplified derivative for rotation coordinate
            if self.axis == 0:  # x-rotation
                derivatives[atom_idx, 1] = -r_ref[2] * self.masses[atom_idx] / np.sum(self.masses)
                derivatives[atom_idx, 2] = r_ref[1] * self.masses[atom_idx] / np.sum(self.masses)
            elif self.axis == 1:  # y-rotation
                derivatives[atom_idx, 0] = r_ref[2] * self.masses[atom_idx] / np.sum(self.masses)
                derivatives[atom_idx, 2] = -r_ref[0] * self.masses[atom_idx] / np.sum(self.masses)
            else:  # z-rotation
                derivatives[atom_idx, 0] = -r_ref[1] * self.masses[atom_idx] / np.sum(self.masses)
                derivatives[atom_idx, 1] = r_ref[0] * self.masses[atom_idx] / np.sum(self.masses)
        
        return derivatives


class TRProjector:
    """Translation-Rotation projector for removing global motions."""
    
    def __init__(self, geometry: Geometry):
        """Initialize TR projector.
        
        Parameters
        ----------
        geometry : Geometry
            Molecular geometry
        """
        self.geometry = geometry
        self.n_atoms = len(geometry)
        self.masses = geometry.get_masses()
        
        # Create TR coordinates
        self.tr_coords = self._create_tr_coordinates()
        
        # Build projection matrix
        self.projection_matrix = self._build_projection_matrix()
    
    def _create_tr_coordinates(self) -> List[TRCoordinate]:
        """Create all TR coordinates."""
        tr_coords = []
        
        # Translation coordinates (3 DOF)
        for axis in range(3):
            tr_coords.append(
                TranslationCoordinate(axis, list(range(self.n_atoms)), self.masses)
            )
        
        # Rotation coordinates (3 DOF)
        reference_positions = self.geometry.positions
        for axis in range(3):
            tr_coords.append(
                RotationCoordinate(axis, list(range(self.n_atoms)), self.masses, 
                                 reference_positions)
            )
        
        return tr_coords
    
    def _build_projection_matrix(self) -> np.ndarray:
        """Build TR projection matrix.
        
        The projection matrix P removes translation and rotation:
        P = I - G(G^T G)^(-1) G^T
        
        where G is the matrix of TR coordinate gradients.
        """
        n_cartesian = 3 * self.n_atoms
        n_tr = len(self.tr_coords)  # Should be 6 (3 trans + 3 rot)
        
        # Build G matrix (n_cartesian × n_tr)
        G = np.zeros((n_cartesian, n_tr))
        positions = self.geometry.positions
        
        for i, tr_coord in enumerate(self.tr_coords):
            derivatives = tr_coord.derivative(positions)
            G[:, i] = derivatives.flatten()
        
        # Build projection matrix P = I - G(G^T G)^(-1) G^T
        try:
            # Check if G^T G is invertible
            GTG = G.T @ G
            if np.linalg.det(GTG) < 1e-12:
                # Use pseudo-inverse if singular
                G_pinv = np.linalg.pinv(G)
                P = np.eye(n_cartesian) - G @ G_pinv
            else:
                # Use regular inverse
                GTG_inv = np.linalg.inv(GTG)
                P = np.eye(n_cartesian) - G @ GTG_inv @ G.T
        except np.linalg.LinAlgError:
            # Fallback to identity if numerical issues
            print("Warning: Could not build TR projection matrix, using identity")
            P = np.eye(n_cartesian)
        
        return P
    
    def project_vector(self, vector: np.ndarray) -> np.ndarray:
        """Project a vector to remove TR components.
        
        Parameters
        ----------
        vector : np.ndarray
            Vector to project (should be flattened Cartesian coordinates)
            
        Returns
        -------
        np.ndarray
            Projected vector with TR components removed
        """
        if vector.ndim == 2:
            # Reshape if needed
            vector = vector.flatten()
        
        return self.projection_matrix @ vector
    
    def project_gradient(self, gradient: np.ndarray) -> np.ndarray:
        """Project gradient to remove TR components.
        
        Parameters
        ----------
        gradient : np.ndarray
            Gradient array (N_atoms, 3)
            
        Returns
        -------
        np.ndarray
            Projected gradient with TR components removed
        """
        gradient_flat = gradient.flatten()
        projected_flat = self.project_vector(gradient_flat)
        return projected_flat.reshape(-1, 3)
    
    def update_reference(self, positions: np.ndarray):
        """Update reference positions for rotation coordinates.
        
        This should be called when the geometry changes significantly.
        """
        self.geometry.positions = positions.copy()
        
        # Recreate rotation coordinates with new reference
        for i, tr_coord in enumerate(self.tr_coords):
            if isinstance(tr_coord, RotationCoordinate):
                tr_coord.ref_positions_com = positions - tr_coord._calculate_com(positions)
        
        # Rebuild projection matrix
        self.projection_matrix = self._build_projection_matrix()


def create_tr_projector(geometry: Geometry) -> TRProjector:
    """Factory function to create TR projector.
    
    Parameters
    ----------
    geometry : Geometry
        Molecular geometry
        
    Returns
    -------
    TRProjector
        Initialized TR projector
    """
    return TRProjector(geometry)
