"""Interpolation strategies for reaction pathway generation.

This module provides pluggable interpolation strategies for generating
reaction pathways between molecular structures. Each strategy implements
a different approach to interpolating between start and end coordinates.

Available strategies:
- linear: Simple linear interpolation between coordinates
- geodesic: Distance-preserving interpolation with bond length refinement
- idpp: Image-Dependent Pair Potential interpolation
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial.distance import cdist

from famex.utils.logging import get_famex_logger

logger = get_famex_logger(__name__)


class InterpolationStrategy(ABC):
    """Base class for interpolation strategies."""

    @abstractmethod
    def interpolate(
        self,
        start_coords: np.ndarray,
        end_coords: np.ndarray,
        npoints: int,
    ) -> list[np.ndarray]:
        """Interpolate between start and end coordinates.

        Parameters
        ----------
        start_coords : np.ndarray
            Starting coordinates (N, 3)
        end_coords : np.ndarray
            Ending coordinates (N, 3)
        npoints : int
            Number of interpolation points (including endpoints)

        Returns
        -------
        list[np.ndarray]
            List of interpolated coordinate arrays

        """


class LinearInterpolation(InterpolationStrategy):
    """Simple linear interpolation between coordinates."""

    def interpolate(
        self,
        start_coords: np.ndarray,
        end_coords: np.ndarray,
        npoints: int,
    ) -> list[np.ndarray]:
        """Perform linear interpolation between start and end coordinates."""
        n_atoms, n_dims = start_coords.shape
        path_coords_array = np.empty((npoints, n_atoms, n_dims))

        for i in range(npoints):
            alpha = i / (npoints - 1)
            path_coords_array[i] = (1 - alpha) * start_coords + alpha * end_coords

        # Convert to list of arrays for API compatibility
        return [path_coords_array[i] for i in range(npoints)]


class GeodesicInterpolation(InterpolationStrategy):
    """Geodesic interpolation with better bond length preservation.

    Uses distance geometry principles to create more chemically reasonable
    intermediate structures, similar to approaches in NEB methods.
    """

    def interpolate(
        self,
        start_coords: np.ndarray,
        end_coords: np.ndarray,
        npoints: int,
    ) -> list[np.ndarray]:
        """Perform geodesic interpolation with bond length preservation."""
        n_atoms, n_dims = start_coords.shape
        path_coords_array = np.empty((npoints, n_atoms, n_dims))

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
            path_coords_array[i] = refined_coords

        # Convert to list of arrays for API compatibility
        return [path_coords_array[i] for i in range(npoints)]

    def _get_distance_matrix(self, coords: np.ndarray) -> np.ndarray:
        """Get pairwise distance matrix using vectorized operations."""
        return cdist(coords, coords, metric="euclidean")

    def _refine_coordinates(
        self,
        coords: np.ndarray,
        target_dists: np.ndarray,
        max_iter: int = 10,
    ) -> np.ndarray:
        """Refine coordinates to better match target distance matrix.

        Uses simple iterative coordinate adjustment to improve bond lengths.
        """
        coords = coords.copy()
        n_atoms = len(coords)

        for _iteration in range(max_iter):
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


def idpp_energy_and_forces(
    positions: np.ndarray,
    target: np.ndarray,
) -> tuple[float, np.ndarray]:
    """IDPP energy and forces matching ASE ``ase.mep.neb.IDPP``."""
    positions = np.asarray(positions, dtype=float)
    target = np.asarray(target, dtype=float)

    diffs = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
    dists = np.linalg.norm(diffs, axis=2)
    dd = dists - target
    dists_safe = dists.copy()
    np.fill_diagonal(dists_safe, 1.0)
    energy = 0.5 * float((dd**2 / dists_safe**4).sum())
    # ASE sums over the first pair index: force_j ∝ Σ_i coeff[i,j] * (r_j - r_i)
    coeff = dd * (1.0 - 2.0 * dd / dists_safe) / dists_safe**5
    forces = -2.0 * (coeff[..., np.newaxis] * diffs).sum(axis=0)
    return energy, np.asarray(forces, dtype=float)


class IDPPInterpolation(InterpolationStrategy):
    """Image-Dependent Pair Potential (Smidstrup / ASE IDPP)."""

    def __init__(
        self,
        *,
        fmax: float = 0.1,
        steps: int = 100,
        spring_constant: float = 0.1,
    ) -> None:
        self.fmax = fmax
        self.steps = steps
        self.spring_constant = spring_constant

    def interpolate(
        self,
        start_coords: np.ndarray,
        end_coords: np.ndarray,
        npoints: int,
    ) -> list[np.ndarray]:
        if npoints < 2:
            msg = "IDPP interpolation requires at least 2 points"
            raise ValueError(msg)

        path = [
            np.asarray(c, dtype=float).copy()
            for c in LinearInterpolation().interpolate(start_coords, end_coords, npoints)
        ]
        if npoints == 2:
            return path

        d1 = cdist(start_coords, start_coords, metric="euclidean")
        d2 = cdist(end_coords, end_coords, metric="euclidean")
        d_step = (d2 - d1) / (npoints - 1)
        targets = [d1 + i * d_step for i in range(npoints)]
        self._relax_path(path, targets)
        return path

    def _relax_path(self, path: list[np.ndarray], targets: list[np.ndarray]) -> None:
        # Deferred import: PathManager (used by strategies) imports this module.
        from famex.strategies.neb_optimizer import _FIREBandStepper, compute_neb_forces

        fire = _FIREBandStepper()
        npoints = len(path)
        natoms = path[0].shape[0]

        for _step in range(self.steps):
            energies = []
            true_forces = []
            for coords, target in zip(path, targets, strict=True):
                energy, forces = idpp_energy_and_forces(coords, target)
                energies.append(energy)
                true_forces.append(forces)

            neb_forces, _ = compute_neb_forces(
                path,
                energies,
                true_forces,
                k=self.spring_constant,
                climb=False,
            )
            interior = neb_forces[1:-1]
            if max(float(np.max(np.abs(f))) for f in interior) < self.fmax:
                break

            interior_pos = np.concatenate([c.ravel() for c in path[1:-1]])
            interior_f = np.concatenate([f.ravel() for f in interior])
            new_pos = fire.step(interior_pos, interior_f)
            ncoords = 3 * natoms
            for idx in range(npoints - 2):
                start = idx * ncoords
                path[idx + 1] = new_pos[start : start + ncoords].reshape(natoms, 3)


# Registry of available interpolation strategies
INTERPOLATION_REGISTRY: dict[str, type[InterpolationStrategy]] = {
    "linear": LinearInterpolation,
    "geodesic": GeodesicInterpolation,
    "idpp": IDPPInterpolation,
}


def get_interpolation_strategy(method: str) -> InterpolationStrategy:
    """Get interpolation strategy by name.

    Parameters
    ----------
    method : str
        Name of the interpolation method

    Returns
    -------
    InterpolationStrategy
        Interpolation strategy instance

    Raises
    ------
    ValueError
        If method is not recognized

    """
    method_lower = method.lower().strip()

    if method_lower not in INTERPOLATION_REGISTRY:
        available = ", ".join(INTERPOLATION_REGISTRY.keys())
        msg = f"Unknown interpolation method: '{method}'. Available methods: {available}"
        raise ValueError(
            msg,
        )

    # All classes in registry are concrete implementations
    strategy_class = INTERPOLATION_REGISTRY[method_lower]
    return strategy_class()


def list_interpolation_methods() -> dict[str, str]:
    """List available interpolation methods and their descriptions.

    Returns
    -------
    dict[str, str]
        Dictionary mapping method names to descriptions

    """
    descriptions = {
        "linear": "Simple linear interpolation between coordinates",
        "geodesic": "Distance-preserving interpolation with bond length refinement",
        "idpp": "Image-Dependent Pair Potential interpolation (robust for large changes)",
    }

    return {
        method: descriptions.get(method, "No description available")
        for method in INTERPOLATION_REGISTRY
    }
