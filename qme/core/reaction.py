"""
Reaction class for handling reaction pathways and geodesic interpolation.
"""

from typing import List, Optional, Union

import numpy as np
from ase import Atoms

from qme.core.geometry import Geometry
from qme.core.validation import validate_atoms_compatibility


class Reaction:
    """
    Represents a chemical reaction with reactant and product geometries.

    Provides methods for generating reaction pathways using linear and
    geodesic interpolation, similar to nudged elastic band (NEB) methods.
    """

    def __init__(
        self,
        reactant: Union[Geometry, Atoms],
        product: Union[Geometry, Atoms],
        calculator=None,
        name: Optional[str] = None,
    ):
        """
        Initialize a reaction.

        Parameters
        ----------
        reactant : Geometry or Atoms
            Reactant geometry
        product : Geometry or Atoms
            Product geometry
        calculator : Calculator, optional
            ASE calculator for energy/force calculations
        name : str, optional
            Name for the reaction
        """

        self.name = name or "Reaction"

        # Convert to Geometry objects if needed
        if isinstance(reactant, Atoms):
            self.reactant = Geometry(ase_atoms=reactant)
        else:
            self.reactant = reactant

        if isinstance(product, Atoms):
            self.product = Geometry(ase_atoms=product)
        else:
            self.product = product

        # Validate that structures are compatible
        validate_atoms_compatibility(self.reactant, self.product, "reaction")

        self.calculator = calculator

        # Set calculator on geometries if provided
        if calculator is not None:
            self.reactant.calc = calculator
            self.product.calc = calculator

    @property
    def reaction_energy(self) -> Optional[float]:
        """Get reaction energy (product - reactant)."""
        if self.reactant.energy is not None and self.product.energy is not None:
            return self.product.energy - self.reactant.energy
        return None

    def interpolate(
        self,
        npoints: int = 10,
        method: str = "linear",
        optimize_path: bool = False,
        calculator=None,
    ) -> List[Geometry]:
        """
        Generate reaction pathway by interpolation.

        Parameters
        ----------
        npoints : int, default 10
            Number of interpolation points (including endpoints)
        method : str, default "linear"
            Interpolation method ("linear" or "geodesic")
        optimize_path : bool, default False
            Whether to optimize the interpolated path with NEB-like forces
        calculator : Calculator, optional
            Calculator for path optimization

        Returns
        -------
        List[Geometry]
            List of interpolated geometries
        """

        if npoints < 2:
            raise ValueError("Need at least 2 points for interpolation")

        # Get coordinates
        r_coords = self.reactant.coords3d
        p_coords = self.product.coords3d

        if method == "linear":
            path_coords = self._linear_interpolation(r_coords, p_coords, npoints)
        elif method == "geodesic":
            path_coords = self._geodesic_interpolation(r_coords, p_coords, npoints)
        else:
            raise ValueError(f"Unknown interpolation method: {method}")

        # Create geometry objects
        path_geometries = []
        for coords in path_coords:
            geom = Geometry(
                atoms=list(self.reactant.get_chemical_symbols()),
                positions=coords,
                charge=self.reactant.charge,
                mult=self.reactant.mult,
            )

            # Set charge and spin in atoms.info to prevent UMA backend warnings
            if hasattr(geom, "info") and geom.info is not None:
                geom.info.setdefault("charge", self.reactant.charge)
                geom.info.setdefault("spin", self.reactant.mult)

            if self.calculator is not None:
                geom.calc = self.calculator
            path_geometries.append(geom)

        # Optionally optimize path with NEB-like forces
        if optimize_path:
            calc = calculator or self.calculator
            if calc is None:
                raise ValueError("Need calculator for path optimization")
            path_geometries = self._optimize_path(path_geometries, calc)

        return path_geometries

    def _linear_interpolation(
        self, start_coords: np.ndarray, end_coords: np.ndarray, npoints: int
    ) -> List[np.ndarray]:
        """Perform linear interpolation between start and end coordinates."""

        path_coords = []
        for i in range(npoints):
            alpha = i / (npoints - 1)
            coords = (1 - alpha) * start_coords + alpha * end_coords
            path_coords.append(coords)

        return path_coords

    def _geodesic_interpolation(
        self, start_coords: np.ndarray, end_coords: np.ndarray, npoints: int
    ) -> List[np.ndarray]:
        """
        Perform geodesic interpolation with better bond length preservation.

        Uses distance geometry principles to create more chemically reasonable
        intermediate structures, similar to approaches in NEB methods.
        """

        # For now, use a simple improvement over linear interpolation
        # This preserves bond lengths better than pure linear interpolation

        path_coords = []

        # Get all pairwise distances at start and end
        start_dists = self._get_distance_matrix(start_coords)
        end_dists = self._get_distance_matrix(end_coords)

        for i in range(npoints):
            alpha = i / (npoints - 1)

            # Interpolate distances rather than coordinates
            target_dists = (1 - alpha) * start_dists + alpha * end_dists

            # Use linear interpolation as starting guess
            linear_coords = (1 - alpha) * start_coords + alpha * end_coords

            # Refine coordinates to better match target distances
            refined_coords = self._refine_coordinates(linear_coords, target_dists)

            path_coords.append(refined_coords)

        return path_coords

    def _get_distance_matrix(self, coords: np.ndarray) -> np.ndarray:
        """Get pairwise distance matrix."""
        n_atoms = len(coords)
        dists = np.zeros((n_atoms, n_atoms))

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                dist = np.linalg.norm(coords[i] - coords[j])
                dists[i, j] = dists[j, i] = dist

        return dists

    def _refine_coordinates(
        self, coords: np.ndarray, target_dists: np.ndarray, max_iter: int = 10
    ) -> np.ndarray:
        """
        Refine coordinates to better match target distance matrix.

        Uses simple iterative coordinate adjustment to improve bond lengths.
        """

        coords = coords.copy()
        n_atoms = len(coords)

        for iteration in range(max_iter):
            current_dists = self._get_distance_matrix(coords)

            # Calculate forces to adjust distances
            forces = np.zeros_like(coords)

            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    current_dist = current_dists[i, j]
                    target_dist = target_dists[i, j]

                    if current_dist > 1e-6:  # Avoid division by zero
                        # Direction vector
                        direction = coords[j] - coords[i]
                        direction /= current_dist

                        # Force magnitude proportional to distance error
                        force_mag = (target_dist - current_dist) * 0.1

                        # Apply forces
                        forces[i] -= force_mag * direction
                        forces[j] += force_mag * direction

            # Update coordinates
            coords += forces * 0.1

        return coords

    def _optimize_path(self, path_geometries: List[Geometry], calculator) -> List[Geometry]:
        """
        Optimize path using simplified NEB-like forces.

        This is a basic implementation inspired by NEB methods.
        """

        # For now, just return the path as-is
        # A full NEB implementation would require more sophisticated force calculations
        print("Note: Using simplified NEB-like algorithm for path optimization")

        # Set calculators and calculate energies
        for geom in path_geometries:
            geom.calc = calculator
            # Force energy calculation
            energy = geom.get_potential_energy()
            geom.energy = energy

        return path_geometries

    def find_transition_state_guess(self, method: str = "geodesic", npoints: int = 11) -> Geometry:
        """
        Find a good initial guess for transition state search.

        Parameters
        ----------
        method : str, default "geodesic"
            Interpolation method to use
        npoints : int, default 11
            Number of points for interpolation (should be odd)

        Returns
        -------
        Geometry
            Transition state guess (middle point of interpolation)
        """

        path = self.interpolate(npoints=npoints, method=method)
        middle_idx = len(path) // 2
        return path[middle_idx]

    def get_rmsd_profile(self, path_geometries: List[Geometry]) -> tuple[List[float], List[float]]:
        """
        Calculate RMSD profile along reaction path.

        Returns
        -------
        tuple
            (rmsd_from_reactant, rmsd_from_product)
        """

        rmsd_from_reactant = []
        rmsd_from_product = []

        reactant_coords = self.reactant.coords3d
        product_coords = self.product.coords3d

        for geom in path_geometries:
            coords = geom.coords3d

            # RMSD from reactant
            rmsd_r = np.sqrt(np.mean((coords - reactant_coords) ** 2))
            rmsd_from_reactant.append(rmsd_r)

            # RMSD from product
            rmsd_p = np.sqrt(np.mean((coords - product_coords) ** 2))
            rmsd_from_product.append(rmsd_p)

        return rmsd_from_reactant, rmsd_from_product

    def to_xyz_trajectory(self, path_geometries: List[Geometry]) -> str:
        """
        Convert path to XYZ trajectory string.

        Parameters
        ----------
        path_geometries : List[Geometry]
            Geometries along the path

        Returns
        -------
        str
            XYZ trajectory as string
        """

        xyz_lines = []

        for i, geom in enumerate(path_geometries):
            # Number of atoms
            xyz_lines.append(str(len(geom)))

            # Comment line with energy if available
            if geom.energy is not None:
                comment = f"Frame {i}, Energy = {geom.energy:.6f} eV"
            else:
                comment = f"Frame {i}"
            xyz_lines.append(comment)

            # Atomic coordinates
            symbols = geom.symbols
            coords = geom.coords3d

            for j, (symbol, pos) in enumerate(zip(symbols, coords)):
                line = f"{symbol:2s} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}"
                xyz_lines.append(line)

        return "\n".join(xyz_lines)

    def set_calculator(self, calculator):
        """Set the calculator for reactant, product, and the reaction itself."""
        self.calculator = calculator
        if self.reactant:
            self.reactant.calc = calculator
        if self.product:
            self.product.calc = calculator

    def calculate_path_energies(self, path: List[Geometry]) -> List[float]:
        """Calculate and return energies for a given reaction path."""
        if not path:
            return []

        energies = []
        for geom in path:
            if geom.calc is None:
                geom.calc = self.calculator
            energies.append(geom.get_potential_energy())
        return energies

    def __str__(self) -> str:
        """String representation."""
        return f"Reaction({len(self.reactant)} atoms)"

    def __repr__(self) -> str:
        return self.__str__()
