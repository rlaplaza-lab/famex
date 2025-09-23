"""Reaction pathway handling for QME."""

from typing import List, Optional, Tuple

import numpy as np

from .geometry import Geometry


class Reaction:
    """Class for handling chemical reaction pathways.

    Inspired by pysisyphus chain-of-states methods but simplified for MLP/NNP usage.
    """

    def __init__(
        self,
        reactant: Geometry,
        product: Geometry,
        ts: Optional[Geometry] = None,
        name: str = "reaction",
    ):
        """Initialize a reaction pathway.

        Args:
            reactant: Starting geometry
            product: Final geometry
            ts: Transition state geometry (optional)
            name: Descriptive name for the reaction
        """
        self.reactant = reactant
        self.product = product
        self.ts = ts
        self.name = name

        # Validate that reactant and product have same atoms
        if len(reactant.atoms) != len(product.atoms):
            raise ValueError("Reactant and product must have same number of atoms")

        if not all(a1 == a2 for a1, a2 in zip(reactant.atoms, product.atoms)):
            raise ValueError("Reactant and product must have same atomic composition")

    @property
    def has_ts(self) -> bool:
        """Whether transition state is available."""
        return self.ts is not None

    @property
    def reaction_energy(self) -> Optional[float]:
        """Reaction energy (product - reactant)."""
        if self.reactant.energy is None or self.product.energy is None:
            return None
        return self.product.energy - self.reactant.energy

    @property
    def activation_energy(self) -> Optional[float]:
        """Activation energy (TS - reactant)."""
        if not self.has_ts or self.ts.energy is None or self.reactant.energy is None:
            return None
        return self.ts.energy - self.reactant.energy
    
    def interpolate(self, npoints: int = 10, method: str = "linear", 
                    optimize_path: bool = False, calculator=None) -> List[Geometry]:
        """Create interpolation between reactant and product.
        
        Args:
            npoints: Number of interpolated points including endpoints
            method: Interpolation method - "linear" or "geodesic"
            optimize_path: Whether to optimize the interpolated path
            calculator: Calculator for optimization (required if optimize_path=True)
            
        Returns:
            List of interpolated geometries
        """
        if npoints < 2:
            raise ValueError("Need at least 2 points for interpolation")
        
        if optimize_path and calculator is None:
            raise ValueError("Calculator required for path optimization")
        
        if method == "linear":
            path = self._linear_interpolation(npoints)
        elif method == "geodesic":
            path = self._geodesic_interpolation(npoints)
        else:
            raise ValueError(f"Unknown interpolation method: {method}. "
                           "Choose 'linear' or 'geodesic'.")
        
        if optimize_path:
            path = self._optimize_path(path, calculator)
            
        return path
    
    def _linear_interpolation(self, npoints: int) -> List[Geometry]:
        """Create linear interpolation between reactant and product."""

        geoms = []
        coords_start = self.reactant.coords
        coords_end = self.product.coords

        for i in range(npoints):
            alpha = i / (npoints - 1)  # 0 to 1
            coords_interp = (1 - alpha) * coords_start + alpha * coords_end

            geom = Geometry(
                atoms=self.reactant.atoms.copy(),
                coords=coords_interp,
                charge=self.reactant.charge,
                mult=self.reactant.mult,
            )
            geoms.append(geom)
        return geoms
    
    def _geodesic_interpolation(self, npoints: int) -> List[Geometry]:
        """Create geodesic interpolation using distance-based internal coordinates.
        
        This method interpolates in pairwise distance space, which approximates
        geodesic paths better than Cartesian interpolation for molecular systems.
        """
        # Convert to distance matrices
        dist_start = self._coords_to_distance_matrix(self.reactant.coords3d)
        dist_end = self._coords_to_distance_matrix(self.product.coords3d)
        
        geoms = []
        for i in range(npoints):
            alpha = i / (npoints - 1)  # 0 to 1
            
            if i == 0:
                # Start point
                geom = self.reactant.copy()
            elif i == npoints - 1:
                # End point  
                geom = self.product.copy()
            else:
                # Interpolate distance matrix
                dist_interp = (1 - alpha) * dist_start + alpha * dist_end
                
                # Convert back to Cartesian coordinates
                coords_interp = self._distance_matrix_to_coords(
                    dist_interp, self.reactant.coords3d
                )
                
                geom = Geometry(
                    atoms=self.reactant.atoms.copy(),
                    coords=coords_interp.flatten(),
                    charge=self.reactant.charge,
                    mult=self.reactant.mult
                )
            
            geoms.append(geom)
        
        return geoms
    
    def _coords_to_distance_matrix(self, coords3d: np.ndarray) -> np.ndarray:
        """Convert 3D coordinates to distance matrix."""
        natoms = coords3d.shape[0]
        dist_matrix = np.zeros((natoms, natoms))
        
        for i in range(natoms):
            for j in range(i + 1, natoms):
                dist = np.linalg.norm(coords3d[i] - coords3d[j])
                dist_matrix[i, j] = dist_matrix[j, i] = dist
                
        return dist_matrix
    
    def _distance_matrix_to_coords(self, dist_matrix: np.ndarray, 
                                  reference_coords: np.ndarray) -> np.ndarray:
        """Convert distance matrix back to Cartesian coordinates using MDS.
        
        This uses classical multidimensional scaling to embed the distance
        matrix into 3D space, using the reference coordinates for alignment.
        """
        natoms = dist_matrix.shape[0]
        
        # Classical MDS algorithm
        # Center the squared distance matrix
        D_sq = dist_matrix ** 2
        H = np.eye(natoms) - np.ones((natoms, natoms)) / natoms
        B = -0.5 * H @ D_sq @ H
        
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(B)
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Take only positive eigenvalues and their corresponding eigenvectors
        # Use the top 3 dimensions for 3D embedding
        n_dims = min(3, sum(eigenvalues > 1e-10))
        if n_dims < 3:
            n_dims = 3  # Force 3D even with small negative eigenvalues
            
        coords_mds = eigenvectors[:, :n_dims] @ np.diag(np.sqrt(np.abs(eigenvalues[:n_dims])))
        
        # Pad with zeros if we have fewer than 3 dimensions
        if coords_mds.shape[1] < 3:
            padding = np.zeros((natoms, 3 - coords_mds.shape[1]))
            coords_mds = np.hstack([coords_mds, padding])
        
        # Align with reference coordinates using Procrustes analysis
        coords_aligned = self._procrustes_alignment(coords_mds, reference_coords)
        
        return coords_aligned
    
    def _procrustes_alignment(self, coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
        """Align coords1 to coords2 using Procrustes analysis."""
        # Center both coordinate sets
        centroid1 = np.mean(coords1, axis=0)
        centroid2 = np.mean(coords2, axis=0)
        
        coords1_centered = coords1 - centroid1
        coords2_centered = coords2 - centroid2
        
        # Compute optimal rotation using SVD
        H = coords1_centered.T @ coords2_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Ensure proper rotation (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Apply transformation
        coords1_aligned = coords1_centered @ R + centroid2
        
        return coords1_aligned

    def _optimize_path(self, path: List[Geometry], calculator) -> List[Geometry]:
        """Optimize reaction path using NEB-like spring forces.
        
        This is a simplified NEB implementation for demonstration.
        A full implementation would use proper NEB algorithms.
        """
        from .calculators import Calculator
        
        if not isinstance(calculator, Calculator):
            raise ValueError("Calculator must be an instance of Calculator")
        
        optimized_path = []
        
        for i, geom in enumerate(path):
            # Skip endpoints (reactant and product)
            if i == 0 or i == len(path) - 1:
                optimized_path.append(geom.copy())
                continue
            
            # For intermediate images, calculate energy and forces
            calculator.calculate(geom)
            
            # Simple optimization: move slightly downhill in energy
            # This is a very simplified approach
            if geom.forces is not None:
                # Take a small step opposite to forces
                step_size = 0.01  # Small step size
                new_coords = geom.coords - step_size * geom.forces.flatten()
                
                optimized_geom = Geometry(
                    atoms=geom.atoms.copy(),
                    coords=new_coords,
                    charge=geom.charge,
                    mult=geom.mult
                )
                optimized_path.append(optimized_geom)
            else:
                # If no forces available, keep original
                optimized_path.append(geom.copy())
        
        return optimized_path

    def find_transition_state_guess(self, npoints: int = 20, method: str = "geodesic") -> Optional[Geometry]:
        """Find transition state guess using geodesic interpolation.
        
        This method creates an interpolated path and returns the geometry
        with the highest energy as a TS guess.
        
        Args:
            npoints: Number of points to interpolate
            method: Interpolation method to use
            
        Returns:
            Geometry with highest energy along the path, or None if no energies
        """
        # Import here to avoid circular imports
        from .calculators import MLPCalculator, HarmonicCalculator
        
        # Create path
        path = self.interpolate(npoints=npoints, method=method)
        
        # We need a calculator to evaluate energies
        # For demonstration, use a mock calculator
        calculator = MLPCalculator(model_type="mock")
        
        energies = []
        for geom in path:
            calculator.calculate(geom)
            energies.append(geom.energy if geom.energy is not None else 0.0)
        
        if all(e == 0.0 for e in energies):
            return None  # No meaningful energies
        
        # Find highest energy geometry
        max_energy_idx = np.argmax(energies)
        ts_guess = path[max_energy_idx]
        
        return ts_guess

    def get_rmsd_profile(
        self, geometries: List[Geometry]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate RMSD profile along a reaction path.

        Args:
            geometries: List of geometries along the path

        Returns:
            Tuple of (RMSD from reactant, RMSD from product)
        """
        rmsd_from_reactant = np.array([self.reactant.rmsd(geom) for geom in geometries])
        rmsd_from_product = np.array([self.product.rmsd(geom) for geom in geometries])

        return rmsd_from_reactant, rmsd_from_product

    def to_xyz_trajectory(self, geometries: Optional[List[Geometry]] = None) -> str:
        """Export reaction pathway as XYZ trajectory file.

        Args:
            geometries: Optional list of geometries, defaults to [reactant, ts, product]

        Returns:
            XYZ trajectory string
        """
        if geometries is None:
            geometries = [self.reactant]
            if self.has_ts:
                geometries.append(self.ts)
            geometries.append(self.product)

        xyz_parts = []
        for i, geom in enumerate(geometries):
            xyz_parts.append(geom.as_xyz())

        return "\n".join(xyz_parts)

    def __repr__(self) -> str:
        """String representation."""
        ts_info = f", TS={self.has_ts}" if self.has_ts else ""
        energy_info = ""
        if self.reaction_energy is not None:
            energy_info = f", ΔE={self.reaction_energy:.6f}"

        return f"Reaction(name='{self.name}'{ts_info}{energy_info})"
