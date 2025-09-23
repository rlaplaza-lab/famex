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

    def interpolate(self, npoints: int = 10) -> List[Geometry]:
        """Create linear interpolation between reactant and product.

        Args:
            npoints: Number of interpolated points including endpoints

        Returns:
            List of interpolated geometries
        """
        if npoints < 2:
            raise ValueError("Need at least 2 points for interpolation")

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
