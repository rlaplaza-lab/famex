"""Interpolation strategies for reaction pathway generation.

This module provides pluggable interpolation strategies for generating
reaction pathways between molecular structures. Each strategy implements
a different approach to interpolating between start and end coordinates.

Available strategies:
- linear: Simple linear interpolation between coordinates
- geodesic: Distance-preserving interpolation with bond length refinement
- idpp: Image-Dependent Pair Potential interpolation
- quadratic: Quadratic curve fitting through start, midpoint, and end
- spline: Cubic spline interpolation for smooth pathways
"""

from abc import ABC, abstractmethod

import numpy as np

from qme.logging_utils import get_qme_logger

logger = get_qme_logger(__name__)


class InterpolationStrategy(ABC):
    """Base class for interpolation strategies."""

    @abstractmethod
    def interpolate(
        self, start_coords: np.ndarray, end_coords: np.ndarray, npoints: int
    ) -> list[np.ndarray]:
        """
        Interpolate between start and end coordinates.

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
        self, start_coords: np.ndarray, end_coords: np.ndarray, npoints: int
    ) -> list[np.ndarray]:
        """Perform linear interpolation between start and end coordinates."""
        path_coords = []
        for i in range(npoints):
            alpha = i / (npoints - 1)
            coords = (1 - alpha) * start_coords + alpha * end_coords
            path_coords.append(coords)
        return path_coords


class GeodesicInterpolation(InterpolationStrategy):
    """
    Geodesic interpolation with better bond length preservation.

    Uses distance geometry principles to create more chemically reasonable
    intermediate structures, similar to approaches in NEB methods.
    """

    def interpolate(
        self, start_coords: np.ndarray, end_coords: np.ndarray, npoints: int
    ) -> list[np.ndarray]:
        """Perform geodesic interpolation with bond length preservation."""
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


class IDPPInterpolation(InterpolationStrategy):
    """
    Image-Dependent Pair Potential interpolation.

    Iteratively refines linear interpolation using pairwise distance constraints
    to create chemically reasonable intermediate structures. More robust for
    large geometry changes than geodesic interpolation.

    Reference: Smidstrup et al., J. Chem. Phys. 140, 214106 (2014)
    """

    def interpolate(
        self, start_coords: np.ndarray, end_coords: np.ndarray, npoints: int
    ) -> list[np.ndarray]:
        """Perform IDPP interpolation."""
        path_coords = []

        # Start with linear interpolation
        linear_interp = LinearInterpolation()
        initial_path = linear_interp.interpolate(start_coords, end_coords, npoints)

        # Refine each image using IDPP
        for i, coords in enumerate(initial_path):
            if i in (0, npoints - 1):
                # Keep endpoints unchanged
                path_coords.append(coords)
            else:
                # Refine intermediate images
                refined_coords = self._idpp_refine(coords, start_coords, end_coords, i, npoints)
                path_coords.append(refined_coords)

        return path_coords

    def _idpp_refine(
        self,
        coords: np.ndarray,
        start_coords: np.ndarray,
        end_coords: np.ndarray,
        image_idx: int,
        npoints: int,
    ) -> np.ndarray:
        """Refine coordinates using IDPP potential."""
        coords = coords.copy()
        n_atoms = len(coords)

        # Calculate target distances for this image
        alpha = image_idx / (npoints - 1)
        target_dists = (1 - alpha) * self._get_distance_matrix(
            start_coords
        ) + alpha * self._get_distance_matrix(end_coords)

        # IDPP refinement iterations
        for _iteration in range(20):  # Max iterations
            current_dists = self._get_distance_matrix(coords)
            forces = np.zeros_like(coords)

            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    current_dist = current_dists[i, j]
                    target_dist = target_dists[i, j]

                    if current_dist > 1e-6:
                        # IDPP force calculation
                        direction = coords[j] - coords[i]
                        direction /= current_dist

                        # Force magnitude based on distance difference
                        force_mag = (target_dist - current_dist) * 0.2

                        forces[i] -= force_mag * direction
                        forces[j] += force_mag * direction

            # Update coordinates
            coords += forces * 0.1

            # Check convergence
            max_force = np.max(np.linalg.norm(forces, axis=1))
            if max_force < 1e-4:
                break

        return coords

    def _get_distance_matrix(self, coords: np.ndarray) -> np.ndarray:
        """Get pairwise distance matrix."""
        n_atoms = len(coords)
        dists = np.zeros((n_atoms, n_atoms))

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                dist = np.linalg.norm(coords[i] - coords[j])
                dists[i, j] = dists[j, i] = dist

        return dists


class QuadraticInterpolation(InterpolationStrategy):
    """
    Quadratic interpolation through start, midpoint, and end.

    Fits a quadratic curve through the start coordinates, a midpoint guess,
    and end coordinates. Useful when approximate transition region is known.
    """

    def interpolate(
        self, start_coords: np.ndarray, end_coords: np.ndarray, npoints: int
    ) -> list[np.ndarray]:
        """Perform quadratic interpolation."""
        path_coords = []

        # Create midpoint guess (average of start and end)
        midpoint_coords = 0.5 * (start_coords + end_coords)

        # Fit quadratic curve through three points
        for i in range(npoints):
            alpha = i / (npoints - 1)

            if alpha <= 0.5:
                # First half: quadratic from start to midpoint
                t = 2 * alpha
                coords = self._quadratic_interpolate(start_coords, midpoint_coords, t)
            else:
                # Second half: quadratic from midpoint to end
                t = 2 * (alpha - 0.5)
                coords = self._quadratic_interpolate(midpoint_coords, end_coords, t)

            path_coords.append(coords)

        return path_coords

    def _quadratic_interpolate(self, start: np.ndarray, end: np.ndarray, t: float) -> np.ndarray:
        """Quadratic interpolation between two points."""
        # Simple quadratic: start + t * (end - start) + t * (1 - t) * offset
        # where offset creates a smooth curve
        linear = start + t * (end - start)
        offset = 0.1 * (end - start)  # Small offset for curvature
        return linear + t * (1 - t) * offset


class CubicSplineInterpolation(InterpolationStrategy):
    """
    Cubic spline interpolation for smooth pathways.

    Uses cubic splines to create smooth interpolation with better continuity
    properties than linear interpolation.
    """

    def interpolate(
        self, start_coords: np.ndarray, end_coords: np.ndarray, npoints: int
    ) -> list[np.ndarray]:
        """Perform cubic spline interpolation."""
        path_coords = []

        # Create control points for spline
        # Use start, two intermediate points, and end
        control_points = [
            start_coords,
            start_coords + 0.33 * (end_coords - start_coords),
            start_coords + 0.67 * (end_coords - start_coords),
            end_coords,
        ]

        # Generate spline points
        for i in range(npoints):
            t = i / (npoints - 1)
            coords = self._cubic_spline_interpolate(control_points, t)
            path_coords.append(coords)

        return path_coords

    def _cubic_spline_interpolate(self, control_points: list[np.ndarray], t: float) -> np.ndarray:
        """Cubic spline interpolation through control points."""
        if len(control_points) != 4:
            raise ValueError("Cubic spline requires exactly 4 control points")

        # De Casteljau's algorithm for cubic Bezier curve
        p0, p1, p2, p3 = control_points

        # First level
        q0 = (1 - t) * p0 + t * p1
        q1 = (1 - t) * p1 + t * p2
        q2 = (1 - t) * p2 + t * p3

        # Second level
        r0 = (1 - t) * q0 + t * q1
        r1 = (1 - t) * q1 + t * q2

        # Final level
        return (1 - t) * r0 + t * r1


# Registry of available interpolation strategies
INTERPOLATION_REGISTRY = {
    "linear": LinearInterpolation,
    "geodesic": GeodesicInterpolation,
    "idpp": IDPPInterpolation,
    "quadratic": QuadraticInterpolation,
    "spline": CubicSplineInterpolation,
}


def get_interpolation_strategy(method: str) -> InterpolationStrategy:
    """
    Get interpolation strategy by name.

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
        raise ValueError(
            f"Unknown interpolation method: '{method}'. " f"Available methods: {available}"
        )

    strategy_class = INTERPOLATION_REGISTRY[method_lower]
    return strategy_class()


def list_interpolation_methods() -> dict[str, str]:
    """
    List available interpolation methods and their descriptions.

    Returns
    -------
    dict[str, str]
        Dictionary mapping method names to descriptions
    """
    descriptions = {
        "linear": "Simple linear interpolation between coordinates",
        "geodesic": "Distance-preserving interpolation with bond length refinement",
        "idpp": "Image-Dependent Pair Potential interpolation (robust for large changes)",
        "quadratic": "Quadratic curve fitting through start, midpoint, and end",
        "spline": "Cubic spline interpolation for smooth pathways",
    }

    return {
        method: descriptions.get(method, "No description available")
        for method in INTERPOLATION_REGISTRY
    }
