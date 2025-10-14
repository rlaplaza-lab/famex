"""B-matrix calculation and manipulation for TRIC implementation."""

import numpy as np
from typing import Tuple, Optional
from .utils import Geometry
from .internal_coords import InternalCoords


class BMatrixCalculator:
    """Calculator for B-matrix and related operations."""
    
    def __init__(self, internal_coords: InternalCoords):
        """Initialize B-matrix calculator.
        
        Parameters
        ----------
        internal_coords : InternalCoords
            Internal coordinate system
        """
        self.internal_coords = internal_coords
    
    def calculate_B_matrix(self, geometry: Geometry) -> np.ndarray:
        """Calculate B-matrix (Jacobian).
        
        Parameters
        ----------
        geometry : Geometry
            Molecular geometry
            
        Returns
        -------
        np.ndarray
            B-matrix of shape (n_internal, 3*n_atoms)
        """
        return self.internal_coords.B_matrix(geometry)
    
    def calculate_B_inverse(self, geometry: Geometry, 
                           method: str = 'pinv') -> np.ndarray:
        """Calculate pseudo-inverse of B-matrix.
        
        Parameters
        ----------
        geometry : Geometry
            Molecular geometry
        method : str, default 'pinv'
            Method for pseudo-inverse calculation ('pinv' or 'svd')
            
        Returns
        -------
        np.ndarray
            Pseudo-inverse B-matrix
        """
        B = self.calculate_B_matrix(geometry)
        
        if method == 'pinv':
            return np.linalg.pinv(B)
        elif method == 'svd':
            return self._svd_pseudo_inverse(B)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _svd_pseudo_inverse(self, B: np.ndarray, 
                           threshold: float = 1e-12) -> np.ndarray:
        """Calculate pseudo-inverse using SVD.
        
        Parameters
        ----------
        B : np.ndarray
            B-matrix
        threshold : float, default 1e-12
            Threshold for singular values
            
        Returns
        -------
        np.ndarray
            Pseudo-inverse B-matrix
        """
        U, s, Vt = np.linalg.svd(B, full_matrices=False)
        
        # Invert singular values above threshold
        s_inv = np.zeros_like(s)
        s_inv[s > threshold] = 1.0 / s[s > threshold]
        
        # Construct pseudo-inverse
        B_inv = Vt.T @ np.diag(s_inv) @ U.T
        
        return B_inv
    
    def check_rank(self, geometry: Geometry) -> Tuple[int, float]:
        """Check rank and condition number of B-matrix.
        
        Parameters
        ----------
        geometry : Geometry
            Molecular geometry
            
        Returns
        -------
        Tuple[int, float]
            (rank, condition_number)
        """
        B = self.calculate_B_matrix(geometry)
        
        # Calculate rank using SVD
        s = np.linalg.svd(B, compute_uv=False)
        rank = np.sum(s > 1e-12)
        
        # Condition number
        if s[0] > 1e-12:
            condition_number = s[0] / s[-1]
        else:
            condition_number = np.inf
        
        return rank, condition_number
    
    def project_cartesian_forces(self, geometry: Geometry, 
                                cartesian_forces: np.ndarray) -> np.ndarray:
        """Project Cartesian forces to internal coordinate space.
        
        Parameters
        ----------
        geometry : Geometry
            Molecular geometry
        cartesian_forces : np.ndarray
            Cartesian forces (N, 3)
            
        Returns
        -------
        np.ndarray
            Internal coordinate forces
        """
        B = self.calculate_B_matrix(geometry)
        
        # Flatten Cartesian forces
        forces_flat = cartesian_forces.flatten()
        
        # Calculate Wilson G-matrix: G = B @ B.T
        G = B @ B.T
        
        # Calculate internal forces: f_internal = G^(-1) @ B @ f_cartesian
        # This is the correct TRIC transformation following geomeTRIC/pysisyphus
        try:
            G_inv = np.linalg.inv(G)
            internal_forces = G_inv @ B @ forces_flat
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if G is singular
            G_inv = np.linalg.pinv(G)
            internal_forces = G_inv @ B @ forces_flat
        
        return internal_forces
    
    def project_internal_step(self, geometry: Geometry, 
                             internal_step: np.ndarray) -> np.ndarray:
        """Project internal coordinate step to Cartesian space.
        
        Parameters
        ----------
        geometry : Geometry
            Molecular geometry
        internal_step : np.ndarray
            Internal coordinate step
            
        Returns
        -------
        np.ndarray
            Cartesian step (N, 3)
        """
        B = self.calculate_B_matrix(geometry)
        
        # Calculate Wilson G-matrix: G = B @ B.T
        G = B @ B.T
        
        # Project to Cartesian: dq_cartesian = B.T @ G^(-1) @ dq_internal
        # This is the correct TRIC transformation following geomeTRIC/pysisyphus
        try:
            G_inv = np.linalg.inv(G)
            cartesian_step = B.T @ G_inv @ internal_step
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if G is singular
            G_inv = np.linalg.pinv(G)
            cartesian_step = B.T @ G_inv @ internal_step
        
        return cartesian_step.reshape(-1, 3)
    
    def add_translation_rotation_coords(self, geometry: Geometry) -> 'InternalCoords':
        """Add translation and rotation coordinates for TRIC.
        
        Parameters
        ----------
        geometry : Geometry
            Molecular geometry
            
        Returns
        -------
        InternalCoords
            Extended internal coordinate system with TRIC coordinates
        """
        from .tr_coordinates import create_tr_projector
        
        # Create TR projector
        self.tr_projector = create_tr_projector(geometry)
        
        # Apply TR projection to B-matrix
        if hasattr(self, 'B_matrix'):
            # Project the B-matrix to remove TR components
            B_flat = self.B_matrix.flatten()
            projected_B_flat = self.tr_projector.project_vector(B_flat)
            self.B_matrix = projected_B_flat.reshape(self.B_matrix.shape)
        
        return self.internal_coords
    
    def project_gradient(self, gradient: np.ndarray) -> np.ndarray:
        """Project gradient to remove translation/rotation components.
        
        Parameters
        ----------
        gradient : np.ndarray
            Gradient array (N_atoms, 3)
            
        Returns
        -------
        np.ndarray
            Projected gradient with TR components removed
        """
        if hasattr(self, 'tr_projector'):
            return self.tr_projector.project_gradient(gradient)
        else:
            return gradient
    
    def calculate_hessian_guess(self, geometry: Geometry) -> np.ndarray:
        """Calculate initial Hessian guess in internal coordinates.
        
        Parameters
        ----------
        geometry : Geometry
            Molecular geometry
            
        Returns
        -------
        np.ndarray
            Initial Hessian matrix
        """
        n_coords = len(self.internal_coords)
        hessian = np.eye(n_coords)
        
        # Improved Hessian guess based on coordinate types
        # Typical force constants: bonds ~0.5-1.0, angles ~0.1-0.3, dihedrals ~0.01-0.1
        for i, coord in enumerate(self.internal_coords):
            coord_class = coord.__class__.__name__
            if coord_class == 'Bond':
                # Bond stretching - higher force constant
                hessian[i, i] = 0.8
            elif coord_class == 'Angle':
                # Angle bending - medium force constant
                hessian[i, i] = 0.2
            elif coord_class == 'Dihedral':
                # Dihedral rotation - lower force constant
                hessian[i, i] = 0.05
            else:
                # Default for unknown coordinate types
                hessian[i, i] = 0.1
        
        return hessian
