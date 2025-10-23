"""Multi-structure minima optimization strategy via interpolation."""

from typing import Any

from ase import Atoms

from qme.core.base_strategy import BaseStrategy, StrategyMetadata
from qme.core.registry import REGISTRY
from qme.io.path_manager import PathManager
from qme.strategies.minima import LocalMinimaStrategy


class MultiStructureMinimaInterpolateStrategy(BaseStrategy):
    """Multi-structure minima optimization strategy via interpolation."""

    metadata = StrategyMetadata(
        name="minima:interpolate",
        target="minima",
        strategy="interpolate",
        description="Minima optimization via interpolated path",
        aliases=[],
        requires_multiple_structures=True,
    )

    def run(
        self,
        atoms_list: list[Atoms],
        npoints: int = 11,
        method: str = "geodesic",
        fmax: float = 0.05,
        steps: int = 1000,
        calculate_frequencies: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """Run multi-structure minima optimization via interpolation.

        Parameters
        ----------
        atoms_list : list[Atoms]
            List of structures defining the path endpoints
        npoints : int, default=11
            Number of images in the interpolated path
        method : str, default="geodesic"
            Interpolation method for initial path generation
        fmax : float, default=0.05
            Force convergence threshold
        steps : int, default=1000
            Maximum optimization steps
        calculate_frequencies : bool, default=False
            Whether to perform frequency analysis on final optimized structures
        **kwargs
            Additional keyword arguments

        Returns
        -------
        dict[str, Any]
            Results dictionary containing optimized structures and metadata

        """
        self.validate_inputs(atoms_list)

        # Use PathManager to interpolate initial path
        path_mgr = PathManager(atoms_list)
        # Only forward kwargs accepted by PathManager.interpolate
        allowed_keys = {"rmsd_threshold", "energy_threshold", "calculator"}
        interpolate_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}
        initial_path = path_mgr.interpolate(
            npoints=npoints,
            method=method,
            optimize_path=False,
            explorer=self.explorer,
            **interpolate_kwargs,
        )

        # Now optimize each structure in the path to minima
        local_minima_strategy = LocalMinimaStrategy(explorer=self.explorer, profiler=self.profiler)

        optimized_structures = []
        converged_flags = []
        steps_taken = []
        frequency_results = []

        for atoms in initial_path:
            try:
                result = local_minima_strategy.run(
                    atoms_list=[atoms],
                    fmax=fmax,
                    steps=steps,
                    calculate_frequencies=calculate_frequencies,
                    **kwargs,
                )
                optimized_structures.append(result["optimized_atoms"])
                converged_flags.append(result.get("converged", True))
                steps_taken.append(result.get("steps_taken", 0))
                # Collect frequency analysis results if available
                if calculate_frequencies and "frequency_analysis" in result:
                    frequency_results.append(result["frequency_analysis"])
                else:
                    frequency_results.append(None)
            except Exception:
                # If optimization fails for one structure, use the original
                optimized_structures.append(atoms)
                converged_flags.append(False)
                steps_taken.append(0)
                frequency_results.append(None)

        result = self.prepare_result(
            optimized_atoms=optimized_structures,
            trajectory=optimized_structures,
            converged=converged_flags,
            steps_taken=steps_taken,
            strategy="minima:interpolate",
        )

        # Add frequency analysis results if available
        if calculate_frequencies and any(f is not None for f in frequency_results):
            result["frequency_analysis"] = frequency_results
            # For multistructure, we'll use the first valid frequency analysis for summary
            first_valid_freq = next((f for f in frequency_results if f is not None), None)
            if first_valid_freq:
                result["is_minimum"] = first_valid_freq.get("is_minimum")
                # Calculate free energy correction from thermodynamic properties
                thermo = first_valid_freq.get("thermodynamic_properties", {})
                temperature = thermo.get("temperature", 298.15)
                entropy = thermo.get("entropy", 0.0)
                result["free_energy_correction"] = -entropy * temperature / 1000.0

        return result


# Register the strategy
REGISTRY.register(MultiStructureMinimaInterpolateStrategy)
