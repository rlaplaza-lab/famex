"""Multi-structure minima optimization strategy via interpolation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from ase import Atoms

from famex.core.base_strategy import BaseStrategy, StrategyMetadata
from famex.core.registry import REGISTRY
from famex.io.path_manager import PathManager
from famex.strategies.helpers import filter_interpolation_kwargs
from famex.strategies.minima import LocalMinimaStrategy
from famex.utils.logging import get_famex_logger

logger = get_famex_logger(__name__)


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
        **kwargs: Any,
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
        # Filter kwargs to only include parameters accepted by PathManager.interpolate
        interpolate_kwargs = filter_interpolation_kwargs(kwargs)
        initial_path = path_mgr.interpolate(
            npoints=npoints,
            method=method,
            optimize_path=False,
            explorer=self.explorer,
            **interpolate_kwargs,
        )

        # Now optimize each structure in the path to minima
        local_minima_strategy = LocalMinimaStrategy(explorer=self.explorer, profiler=self.profiler)

        optimized_structures: list[Atoms | list[Atoms]] = []
        converged_flags: list[bool] = []
        steps_taken: list[int] = []
        frequency_results: list[dict[str, Any] | None] = []

        for atoms in initial_path:
            try:
                run_result = local_minima_strategy.run(
                    atoms_list=[atoms],
                    fmax=fmax,
                    steps=steps,
                    calculate_frequencies=calculate_frequencies,
                    **kwargs,
                )
                optimized_atoms = run_result["optimized_atoms"]
                # Ensure optimized_atoms is Atoms or list[Atoms]
                if isinstance(optimized_atoms, Atoms | list):
                    optimized_structures.append(optimized_atoms)
                else:
                    # Fallback to original atoms if type is unexpected
                    optimized_structures.append(atoms)
                # Type narrowing for result dict values
                converged = run_result.get("converged", True)
                steps_val = run_result.get("steps_taken", 0)
                converged_flags.append(
                    bool(converged) if isinstance(converged, bool | int) else True
                )
                # Type narrowing: ensure steps is an int
                if isinstance(steps_val, int | float):
                    steps_taken.append(int(steps_val))
                else:
                    steps_taken.append(0)
                # Collect frequency analysis results if available
                if calculate_frequencies and "frequency_analysis" in run_result:
                    freq_result: Any = run_result["frequency_analysis"]
                    if isinstance(freq_result, dict):
                        frequency_results.append(freq_result)
                    else:
                        frequency_results.append(None)
                else:
                    frequency_results.append(None)
            except Exception:
                # If optimization fails for one structure, use the original
                optimized_structures.append(atoms)
                converged_flags.append(False)
                steps_taken.append(0)
                frequency_results.append(None)

        # prepare_result expects Atoms | Sequence[Atoms]
        # optimized_structures is list[Atoms | list[Atoms]], which should be compatible
        # Use type narrowing to help mypy understand the type
        if len(optimized_structures) > 1:
            atoms_for_result: Atoms | Sequence[Atoms] = cast(Sequence[Atoms], optimized_structures)
        elif len(optimized_structures) == 1:
            first_item = optimized_structures[0]
            # Ensure we return Atoms or list[Atoms], not list[Atoms | list[Atoms]]
            atoms_for_result = first_item
        else:
            atoms_for_result = self.explorer.atoms_list[0]
        result: dict[str, Any] = self.prepare_result(
            optimized_atoms=atoms_for_result,
            trajectory=optimized_structures,
            converged=converged_flags,
            steps_taken=steps_taken,
            strategy="minima:interpolate",
        )

        # Add frequency analysis results if available
        if calculate_frequencies and any(f is not None for f in frequency_results):
            # prepare_result converts non-standard types to strings, but we want to keep the list
            # So we'll add it after prepare_result
            result["frequency_analysis"] = frequency_results
            # For multistructure, we'll use the first valid frequency analysis for summary
            first_valid_freq = next((f for f in frequency_results if f is not None), None)
            if isinstance(first_valid_freq, dict):
                is_minimum = first_valid_freq.get("is_minimum")
                if isinstance(is_minimum, bool):
                    result["is_minimum"] = is_minimum
                # Calculate free energy correction from thermodynamic properties
                thermo = first_valid_freq.get("thermodynamic_properties", {})
                if isinstance(thermo, dict):
                    temperature = thermo.get("temperature", 298.15)
                    entropy = thermo.get("entropy", 0.0)
                    if isinstance(temperature, int | float) and isinstance(entropy, int | float):
                        result["free_energy_correction"] = -entropy * temperature / 1000.0

        return result


# Register the strategy
REGISTRY.register(MultiStructureMinimaInterpolateStrategy)
