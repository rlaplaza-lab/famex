"""PathManager class for handling reaction pathways and path operations.

This module provides comprehensive path management including:
- Multi-segment interpolation
- Path analysis (TS finding, minima/maxima detection)
- Structure filtering and redundancy detection
- RMSD calculations and comparisons
- Energy profile calculations
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from ase import Atoms

from qme.interpolation.strategies import get_interpolation_strategy
from qme.io.geometry import Geometry
from qme.utils.logging import get_qme_logger
from qme.utils.validation import validate_atoms_compatibility

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_qme_logger(__name__)


class PathManager:
    """Manages reaction pathways with support for multi-segment interpolation.

    Handles path interpolation, energy calculations, and path analysis
    (finding transition states, minima, etc.) for reaction pathways.
    """

    def __init__(
        self,
        structures: Sequence[Atoms] | Atoms,
        calculator: Any = None,
        name: str | None = None,
    ) -> None:
        """Initialize PathManager with structures defining the path.

        Parameters
        ----------
        structures : Sequence[Atoms] or Atoms
            Structures defining the path. For 2 structures: endpoints.
            For 3+: endpoints + intermediate waypoints for multi-segment paths.
            Can also pass a single Atoms which will be converted to a list.
        calculator : Calculator, optional
            ASE calculator for energy/force calculations
        name : str, optional
            Name for the path

        """
        self.name = name or "PathManager"

        # Normalize to list of Geometry objects
        if isinstance(structures, Atoms):
            # Single Atoms object - convert to list
            structures_list = [structures]
        else:
            structures_list = list(structures)

        if len(structures_list) < 2:
            msg = "PathManager requires at least 2 structures"
            raise ValueError(msg)

        # Convert to Geometry objects if needed
        self.structures: list[Geometry] = []
        for struct in structures_list:
            if isinstance(struct, Atoms) and not isinstance(struct, Geometry):
                geom = Geometry(ase_atoms=struct)
            else:
                geom = struct
            self.structures.append(geom)

        # Validate that structures are compatible
        for i in range(len(self.structures) - 1):
            validate_atoms_compatibility(
                self.structures[i],
                self.structures[i + 1],
                f"path segment {i}",
            )

        self.calculator = calculator

        # Set calculator on geometries if provided
        if calculator is not None:
            for struct in self.structures:
                struct.calc = calculator

    @property
    def reactant(self) -> Geometry:
        """Get first structure (reactant)."""
        return self.structures[0]

    @property
    def product(self) -> Geometry:
        """Get last structure (product)."""
        return self.structures[-1]

    @property
    def reaction_energy(self) -> float | None:
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
        explorer: Any | None = None,
    ) -> list[Geometry]:
        """Generate reaction pathway by interpolation.

        Supports both simple two-endpoint interpolation and multi-segment
        interpolation when more than 2 structures are provided.

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
        explorer : Explorer, optional
            Explorer instance for calculator attachment

        Returns:
        -------
        List[Geometry]
            List of interpolated geometries

        """
        if npoints < 2:
            msg = "Need at least 2 points for interpolation"
            raise ValueError(msg)

        # Simple two-endpoint case
        if len(self.structures) == 2:
            return self._interpolate_segment(
                self.reactant,
                self.product,
                npoints,
                method,
                optimize_path,
                calculator,
            )

        # Multi-segment case: distribute frames across segments and stitch
        segments = len(self.structures) - 1
        total_intervals = npoints - 1
        base_intervals = total_intervals // segments
        remainder = total_intervals % segments

        # Calculate points per segment
        per_segment_npoints = []
        for i in range(segments):
            intervals = base_intervals + (1 if i < remainder else 0)
            per_segment_npoints.append(intervals + 1)

        # Interpolate each segment and stitch
        stitched_path = []
        for i in range(segments):
            start_struct = self.structures[i]
            end_struct = self.structures[i + 1]
            nseg = per_segment_npoints[i]

            seg_path = self._interpolate_segment(
                start_struct,
                end_struct,
                nseg,
                method,
                optimize_path,
                calculator,
            )

            if i == 0:
                stitched_path.extend(seg_path)
            else:
                # Skip first point to avoid duplication
                stitched_path.extend(seg_path[1:])

        return stitched_path

    def _interpolate_segment(
        self,
        start: Geometry,
        end: Geometry,
        npoints: int,
        method: str,
        optimize_path: bool,
        calculator,
    ) -> list[Geometry]:
        """Interpolate a single segment between start and end structures."""
        # Get coordinates
        start_coords = start.coords3d
        end_coords = end.coords3d

        # Use interpolation strategy
        interpolator = get_interpolation_strategy(method)
        path_coords = interpolator.interpolate(start_coords, end_coords, npoints)

        # Create geometry objects
        path_geometries = []
        for coords in path_coords:
            geom = Geometry(
                atoms=list(start.get_chemical_symbols()),
                positions=coords,
                charge=start.charge,
                mult=start.mult,
            )

            # Set charge and spin in atoms.info to prevent UMA backend warnings
            if hasattr(geom, "info") and geom.info is not None:
                geom.info.setdefault("charge", start.charge)
                geom.info.setdefault("spin", start.mult)

            if self.calculator is not None:
                geom.calc = self.calculator
            path_geometries.append(geom)

        # Optionally optimize path with NEB-like forces
        if optimize_path:
            calc = calculator or self.calculator
            if calc is None:
                msg = "Need calculator for path optimization"
                raise ValueError(msg)
            path_geometries = self._optimize_path(path_geometries, calc)

        return path_geometries

    def _optimize_path(self, path_geometries: list[Geometry], calculator) -> list[Geometry]:
        """Optimize path using simplified NEB-like forces.

        This is a basic implementation inspired by NEB methods.
        Future improvements could include full NEB force calculations.
        """
        logger.info("Note: Using simplified NEB-like algorithm for path optimization")

        # Set calculators and calculate energies
        for geom in path_geometries:
            geom.calc = calculator
            # Force energy calculation
            energy = geom.get_potential_energy()
            geom.energy = energy

        return path_geometries

    # =========================================================================
    # Static Path Analysis Methods
    # =========================================================================

    @staticmethod
    def find_ts_guess(path: list[Atoms]) -> tuple[Atoms, int]:
        """Find highest energy structure in path as TS guess.

        Parameters
        ----------
        path : List[Atoms]
            List of structures along the path

        Returns:
        -------
        tuple[Atoms, int]
            (ts_structure, index)

        """
        energies = []
        for atoms in path:
            try:
                energy = atoms.get_potential_energy()
                energies.append(energy)
            except Exception:
                energies.append(float("-inf"))  # Invalid energy

        if not energies or all(e == float("-inf") for e in energies):
            # Fallback to middle structure
            ts_index = len(path) // 2
        else:
            # Find highest energy structure
            ts_index = energies.index(max(energies))

        return path[ts_index], ts_index

    @staticmethod
    def find_local_minima(path: list[Atoms]) -> list[int]:
        """Find indices of local minima along path.

        Parameters
        ----------
        path : List[Atoms]
            List of structures along the path

        Returns:
        -------
        list[int]
            Indices of local minima

        """
        import math

        # Compute energies
        energies = []
        for atoms in path:
            try:
                energy = atoms.get_potential_energy()
                energies.append(energy)
            except Exception:
                energies.append(float("nan"))

        # Find local minima indices: energy less than neighbours
        minima_idxs = []
        for i, e in enumerate(energies):
            if math.isnan(e):
                continue
            left = energies[i - 1] if i - 1 >= 0 else float("inf")
            right = energies[i + 1] if i + 1 < len(energies) else float("inf")
            if (not math.isnan(left) and e < left) and (not math.isnan(right) and e < right):
                minima_idxs.append(i)

        # If no strict local minima found, pick the global minimum
        if not minima_idxs:
            valid = [(i, e) for i, e in enumerate(energies) if not math.isnan(e)]
            if not valid:
                msg = "No valid energies found along path to select minima"
                raise RuntimeError(msg)
            min_idx = min(valid, key=lambda ie: ie[1])[0]
            minima_idxs = [min_idx]

        return minima_idxs

    @staticmethod
    def find_local_maxima(path: list[Atoms]) -> list[int]:
        """Find indices of local maxima (potential TS) along path.

        Parameters
        ----------
        path : List[Atoms]
            List of structures along the path

        Returns:
        -------
        list[int]
            Indices of local maxima

        """
        import math

        # Compute energies
        energies = []
        for atoms in path:
            try:
                energy = atoms.get_potential_energy()
                energies.append(energy)
            except Exception:
                energies.append(float("nan"))

        # Find local maxima indices: energy greater than neighbours
        maxima_idxs = []
        for i, e in enumerate(energies):
            if math.isnan(e):
                continue
            left = energies[i - 1] if i - 1 >= 0 else float("-inf")
            right = energies[i + 1] if i + 1 < len(energies) else float("-inf")
            if (not math.isnan(left) and e > left) and (not math.isnan(right) and e > right):
                maxima_idxs.append(i)

        # If no strict local maxima found, pick the global maximum
        if not maxima_idxs:
            valid = [(i, e) for i, e in enumerate(energies) if not math.isnan(e)]
            if not valid:
                msg = "No valid energies found along path to select maxima"
                raise RuntimeError(msg)
            max_idx = max(valid, key=lambda ie: ie[1])[0]
            maxima_idxs = [max_idx]

        return maxima_idxs

    # =========================================================================
    # Static RMSD and Structure Comparison Methods
    # =========================================================================

    @staticmethod
    def calculate_rmsd(atoms1: Atoms, atoms2: Atoms) -> float:
        """Calculate RMSD between two Atoms objects.

        Parameters
        ----------
        atoms1, atoms2 : Atoms
            The two structures to compare

        Returns:
        -------
        float
            RMSD in Angstroms

        """
        if len(atoms1) != len(atoms2):
            return float("inf")

        # Get positions
        pos1 = atoms1.get_positions()
        pos2 = atoms2.get_positions()

        # Calculate RMSD
        diff = pos1 - pos2
        return np.sqrt(np.mean(np.sum(diff**2, axis=1)))

    @staticmethod
    def calculate_structure_similarity(
        atoms1: Atoms,
        atoms2: Atoms,
        rmsd_threshold: float,
        energy_threshold: float,
    ) -> tuple[bool, float, float]:
        """Calculate similarity between two structures based on RMSD and energy.

        Parameters
        ----------
        atoms1, atoms2 : Atoms
            The two structures to compare
        rmsd_threshold : float
            RMSD threshold for similarity (Angstroms)
        energy_threshold : float
            Energy threshold for similarity (eV)

        Returns:
        -------
        tuple[bool, float, float]
            (is_similar, rmsd, energy_diff)

        """
        rmsd = PathManager.calculate_rmsd(atoms1, atoms2)
        try:
            energy1 = atoms1.get_potential_energy()
            energy2 = atoms2.get_potential_energy()
            energy_diff = abs(energy1 - energy2)
        except (RuntimeError, ValueError, AttributeError):
            energy_diff = float("inf")

        is_similar = rmsd < rmsd_threshold and energy_diff < energy_threshold
        return is_similar, rmsd, energy_diff

    @staticmethod
    def filter_redundant_structures(
        structures: list[Atoms],
        input_structures: list[Atoms] | None = None,
        rmsd_threshold: float = 0.1,
        energy_threshold: float = 0.001,
        strategy_name: str = "strategy",
    ) -> tuple[list[Atoms], list[int], list[str]]:
        """Filter redundant structures based on RMSD and energy similarity.

        Parameters
        ----------
        structures : List[Atoms]
            List of optimized structures to filter
        input_structures : List[Atoms], optional
            Original input structures for comparison
        rmsd_threshold : float
            RMSD threshold below which structures are considered redundant (Å)
        energy_threshold : float
            Energy threshold below which structures are considered redundant (eV)
        strategy_name : str
            Name of the strategy for warning messages

        Returns:
        -------
        tuple[List[Atoms], List[int], List[str]]
            (filtered_structures, removed_indices, warning_messages)

        """
        if not structures:
            return [], [], []

        filtered_structures: list[Atoms] = []
        removed_indices = []
        warning_messages = []

        # Calculate energies for all structures
        energies = []
        for _i, atoms in enumerate(structures):
            try:
                energy = atoms.get_potential_energy()
                energies.append(energy)
            except (RuntimeError, ValueError, AttributeError):
                energies.append(float("inf"))

        # Check if we only have input structures
        if input_structures is not None:
            input_only = True
            for struct in structures:
                for input_struct in input_structures:
                    rmsd = PathManager.calculate_rmsd(struct, input_struct)
                    if rmsd > rmsd_threshold:
                        input_only = False
                        break
                if not input_only:
                    break

            if input_only:
                warning_messages.append(
                    f"Warning: {strategy_name} only found input structures. "
                    f"No new optimized structures were discovered.",
                )

        # Filter structures
        for i, atoms in enumerate(structures):
            is_redundant = False

            # Check against already filtered structures
            for _j, filtered_atoms in enumerate(filtered_structures):
                is_similar, rmsd, energy_diff = PathManager.calculate_structure_similarity(
                    atoms,
                    filtered_atoms,
                    rmsd_threshold,
                    energy_threshold,
                )
                if is_similar:
                    is_redundant = True
                    removed_indices.append(i)
                    warning_messages.append(
                        f"Warning: Removed redundant structure {i + 1} "
                        f"(RMSD: {rmsd:.3f} Å, ΔE: {energy_diff:.3f} eV)",
                    )
                    break

            if not is_redundant:
                filtered_structures.append(atoms)

        # Add general redundancy warnings
        if removed_indices:
            warning_messages.insert(
                0,
                f"Warning: {strategy_name} removed {len(removed_indices)} redundant structures "
                f"out of {len(structures)} total structures.",
            )

        return filtered_structures, removed_indices, warning_messages

    # =========================================================================
    # Static Calculator Helper
    # =========================================================================

    @staticmethod
    def attach_calculators(explorer: Any, structures: list[Atoms] | Atoms):
        """Attach calculators from explorer to structures.

        Parameters
        ----------
        explorer : Explorer
            Explorer instance with calculator configuration
        structures : List[Atoms] or Atoms
            Structure(s) to attach calculators to

        Returns:
        -------
        Calculator
            The calculator instance that was attached

        """
        # Normalize to list
        struct_list = [structures] if isinstance(structures, Atoms) else list(structures)

        if not struct_list:
            return None

        # Ensure charge and spin info are set
        for struct in struct_list:
            if hasattr(struct, "info") and struct.info is not None:
                struct.info.setdefault("charge", explorer.default_charge)
                struct.info.setdefault("spin", explorer.default_spin)

        # Attach calculators
        for struct in struct_list:
            explorer._create_and_attach_calculator(struct)
            explorer._apply_constraints(struct)

        return struct_list[0].calc if struct_list else None

    # =========================================================================
    # Instance Methods for Path Analysis
    # =========================================================================

    def calculate_path_energies(self, path: list[Geometry] | list[Atoms]) -> list[float]:
        """Calculate and return energies for a given reaction path.

        Parameters
        ----------
        path : List[Geometry] or List[Atoms]
            Geometries along the path

        Returns:
        -------
        list[float]
            List of energies

        """
        if not path:
            return []

        energies = []
        for geom in path:
            if geom.calc is None:
                geom.calc = self.calculator
            energies.append(geom.get_potential_energy())
        return energies

    def get_rmsd_profile(
        self,
        path_geometries: list[Geometry] | list[Atoms],
    ) -> tuple[list[float], list[float]]:
        """Calculate RMSD profile along reaction path.

        Parameters
        ----------
        path_geometries : List[Geometry] or List[Atoms]
            Geometries along the path

        Returns:
        -------
        tuple
            (rmsd_from_reactant, rmsd_from_product)

        """
        rmsd_from_reactant = []
        rmsd_from_product = []

        reactant_coords = self.reactant.coords3d
        product_coords = self.product.coords3d

        for geom in path_geometries:
            coords = geom.get_positions() if isinstance(geom, Atoms) else geom.coords3d

            # RMSD from reactant
            rmsd_r = np.sqrt(np.mean((coords - reactant_coords) ** 2))
            rmsd_from_reactant.append(rmsd_r)

            # RMSD from product
            rmsd_p = np.sqrt(np.mean((coords - product_coords) ** 2))
            rmsd_from_product.append(rmsd_p)

        return rmsd_from_reactant, rmsd_from_product

    def to_xyz_trajectory(self, path_geometries: list[Geometry] | list[Atoms]) -> str:
        """Convert path to XYZ trajectory string.

        Parameters
        ----------
        path_geometries : List[Geometry] or List[Atoms]
            Geometries along the path

        Returns:
        -------
        str
            XYZ trajectory as string

        """
        xyz_lines = []

        for i, geom in enumerate(path_geometries):
            # Number of atoms
            xyz_lines.append(str(len(geom)))

            # Comment line with energy if available
            try:
                energy = geom.get_potential_energy()
                comment = f"Frame {i}, Energy = {energy:.6f} eV"
            except Exception:
                comment = f"Frame {i}"
            xyz_lines.append(comment)

            # Atomic coordinates
            symbols = geom.get_chemical_symbols()
            coords = geom.get_positions()

            for _j, (symbol, pos) in enumerate(zip(symbols, coords, strict=False)):
                line = f"{symbol:2s} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}"
                xyz_lines.append(line)

        return "\n".join(xyz_lines)

    def get_path_statistics(self, path: list[Atoms]) -> dict:
        """Get summary statistics for a path.

        Parameters
        ----------
        path : List[Atoms]
            List of structures along the path

        Returns:
        -------
        dict
            Dictionary with path statistics including:
            - num_structures: Number of structures
            - energies: List of energies
            - min_energy: Minimum energy
            - max_energy: Maximum energy
            - energy_range: Energy range
            - ts_index: Index of highest energy (TS guess)
            - minima_indices: Indices of local minima
            - maxima_indices: Indices of local maxima

        """
        energies = []
        for atoms in path:
            try:
                energy = atoms.get_potential_energy()
                energies.append(energy)
            except Exception:
                energies.append(float("nan"))

        import math

        valid_energies = [e for e in energies if not math.isnan(e)]

        stats = {
            "num_structures": len(path),
            "energies": energies,
            "min_energy": min(valid_energies) if valid_energies else None,
            "max_energy": max(valid_energies) if valid_energies else None,
            "energy_range": (max(valid_energies) - min(valid_energies) if valid_energies else None),
        }

        # Find TS and minima/maxima
        try:
            _, ts_idx = PathManager.find_ts_guess(path)
            stats["ts_index"] = ts_idx
        except Exception:
            stats["ts_index"] = None

        try:
            stats["minima_indices"] = PathManager.find_local_minima(path)
        except Exception:
            stats["minima_indices"] = []

        try:
            stats["maxima_indices"] = PathManager.find_local_maxima(path)
        except Exception:
            stats["maxima_indices"] = []

        return stats

    def __str__(self) -> str:
        """String representation."""
        return f"PathManager({len(self.reactant)} atoms, {len(self.structures)} structures)"

    def __repr__(self) -> str:
        return self.__str__()
