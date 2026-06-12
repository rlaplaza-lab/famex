"""Core Explorer class for FAMEX molecular geometry optimization.

This module provides the main Explorer class that serves as the primary
interface for molecular geometry optimization using ASE and SELLA optimizers
combined with machine learning potentials.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ase import Atoms

from famex.core.calculator_manager import CalculatorManager
from famex.core.constraint_manager import ConstraintManager
from famex.core.exceptions import StrategyNotFoundError
from famex.core.file_io import write_atoms_safely, write_trajectory_safely
from famex.core.registry import REGISTRY
from famex.io.geometry import read_geometry
from famex.utils.profiler import PerformanceProfiler

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy as np


class Explorer:
    """Explorer runs optimizations/TS searches on one or more Atoms.

    Uses a target/strategy paradigm:
    - **target**: What you want (minima, ts, path)
    - **strategy**: How to get there (local, neb, cineb, interpolate, growing_string, irc)

    Parameters
    ----------
    atoms : Atoms or Sequence[Atoms]
        Single ASE Atoms object or a sequence of Atoms to operate on.
        For multi-structure strategies (NEB, CI-NEB), provide multiple structures.
    backend : str, default "uma"
        Calculator backend: uma, aimnet2, mace, orb, so3lr, tblite, mock
    model_name : str, optional
        Name of the specific model to use. If None, uses backend defaults.
    model_path : str, optional
        Path to local model file (required for SO3LR, optional for others).
    device : str, optional
        Device for computations ("cpu" or "cuda"). Auto-detected if None.
    default_charge : int, default 0
        Default total charge used when per-structure metadata is not available.
    default_spin : int, default 1
        Default spin multiplicity used when per-structure metadata is not available.
    local_optimizer : str, default "default"
        Local optimizer for geometry optimization. If "default", auto-selects based
        on target: "sella" for TS searches, "lbfgs" for minima/path optimizations.
        Options:
        - "default": Auto-select based on target (default)
        - "sella": SELLA optimizer
        - "lbfgs": L-BFGS optimizer
        - "bfgs": BFGS optimizer
        - "fire": FIRE optimizer
        - "trust-krylov": Trust-region with Krylov subspace
        - "trust-ncg": Trust-region with nonlinear CG
        - "trust-exact": Trust-region with exact Hessian
        - "newton-cg": Newton-CG method
        - "rfo": Rational Function Optimization for TS
    optimizer_kwargs : dict[str, Any], optional
        Keyword arguments forwarded to the local optimizer.
    strategy : str, optional, default "local"
        Strategy type for optimization. Options:
        - "local": Direct local optimization
        - "neb": Nudged Elastic Band
        - "cineb": Climbing Image NEB
        - "interpolate": Path interpolation only
        - "growing_string": Growing string method
        - "irc": Intrinsic Reaction Coordinate
    target : str, optional, default "minima"
        Target type for optimization. Options:
        - "minima": Find local minimum
        - "ts": Find transition state
        - "path": Find reaction pathway
    ts_kwargs : dict[str, Any], optional
        Keyword arguments forwarded to transition state optimizers.
    constraints : str or list or dict, optional
        Constraint specification. Can be:
        - String: "fix 0 1 2" (fix atoms 0, 1, 2)
        - List: [FixAtoms(indices=[0, 1, 2])]
        - Dict: {"fix": [0, 1, 2]}
    initial_hessian : np.ndarray, optional
        Initial Hessian matrix for optimization (3N x 3N).
    auto_register : bool, default True
        If True, registers package default strategies automatically.
    verbose : int, default 1
        Verbosity level (0=quiet, 1=normal, 2=verbose).

    Notes
    -----
    - Structure ``charge``/``spin`` attributes override defaults.
    - Use :meth:`list_strategies` to discover available strategies.
    - Calculator creation and caching handled automatically.

    Examples
    --------
    >>> # Minima optimization (local is default)
    >>> explorer = Explorer(atoms, target="minima")
    >>> result = explorer.run()

    >>> # TS from local search
    >>> explorer = Explorer(atoms, target="ts", strategy="local")
    >>> result = explorer.run()

    >>> # TS from interpolated guess between reactant/product
    >>> explorer = Explorer(atoms=[reactant, product], target="ts", strategy="interpolate")
    >>> result = explorer.run()

    >>> # Reaction path with NEB
    >>> explorer = Explorer(atoms=[reactant, product], target="path", strategy="neb")
    >>> result = explorer.run()

    >>> # Reaction path with CI-NEB
    >>> explorer = Explorer(atoms=[reactant, product], target="path", strategy="cineb")
    >>> result = explorer.run()

    >>> # IRC path from transition state
    >>> explorer = Explorer(atoms=ts_structure, target="path", strategy="irc")
    >>> result = explorer.run()

    >>> # Generate interpolated path only (no optimization)
    >>> explorer = Explorer(atoms=[reactant, product], target="path", strategy="interpolate")
    >>> result = explorer.run(npoints=10)

    Target/Strategy Matrix:
    ┌──────────┬──────────────────┬─────────────────────────────────┐
    │ target   │ strategy         │ Description                     │
    ├──────────┼──────────────────┼─────────────────────────────────┤
    │ minima   │ local            │ Direct local optimization       │
    │ minima   │ interpolate      │ Minima from interpolated path   │
    │ ts       │ local            │ Local TS search                 │
    │ ts       │ interpolate      │ TS guess from interpolation     │
    │ ts       │ growing_string   │ Growing string method (DE-GSM)  │
    │ path     │ neb              │ NEB path optimization           │
    │ path     │ cineb            │ CI-NEB path optimization        │
    │ path     │ irc              │ IRC path from transition state  │
    │ path     │ interpolate      │ Generate path only (no opt)     │
    └──────────┴──────────────────┴─────────────────────────────────┘

    """

    # --- Initialization ---

    def __init__(
        self,
        atoms: Atoms | Sequence[Atoms],
        backend: str = "uma",
        model_name: str | None = None,
        model_path: str | None = None,
        device: str | None = None,
        default_charge: int = 0,
        default_spin: int = 1,
        local_optimizer: str = "default",
        optimizer_kwargs: dict[str, Any] | None = None,
        strategy: str | None = "local",
        target: str | None = "minima",
        ts_kwargs: dict[str, Any] | None = None,
        constraints: str | list | dict | None = None,
        initial_hessian: np.ndarray | None = None,
        auto_register: bool = True,
        verbose: int = 1,
        profile: bool = False,
        force_finite_diff_hessian: bool = False,
    ) -> None:
        """Initialize the Explorer with molecular structure and configuration.

        Parameters
        ----------
        atoms : Atoms | Sequence[Atoms]
            Molecular structure(s) to optimize
        backend : str, default "uma"
            ML potential backend to use
        model_name : str | None, default None
            Specific model name for the backend
        model_path : str | None, default None
            Path to model file
        device : str | None, default None
            Device for calculations (cpu, cuda, etc.)
        default_charge : int, default 0
            Default molecular charge
        default_spin : int, default 1
            Default spin multiplicity
        local_optimizer : str, default "default"
            Local optimizer to use
        optimizer_kwargs : dict[str, Any] | None, default None
            Additional optimizer parameters
        strategy : str | None, default "local"
            Optimization strategy
        target : str | None, default "minima"
            Optimization target
        ts_kwargs : dict[str, Any] | None, default None
            Transition state specific parameters
        constraints : str | list | dict | None, default None
            Geometric constraints
        initial_hessian : np.ndarray | None, default None
            Initial Hessian matrix
        auto_register : bool, default True
            Whether to auto-register strategies
        verbose : int, default 1
            Verbosity level
        profile : bool, default False
            Whether to enable profiling
        force_finite_diff_hessian : bool, default False
            If True, forces use of finite difference hessians for TS optimizers
            and frequency calculations. When True and target is "ts", automatically
            sets hessian_method="finite_differences" in ts_kwargs if not already specified.
        """
        if isinstance(atoms, Atoms):
            self.atoms_list: list[Atoms] = [atoms]
        else:
            self.atoms_list = list(atoms)

        self.backend = backend
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.default_charge = default_charge
        self.default_spin = default_spin
        self.verbose = verbose

        # Initialize performance profiler if requested
        self.profiler = PerformanceProfiler() if profile else None

        from famex.utils.logging import setup_famex_logging

        setup_famex_logging(verbosity=verbose)

        self.local_optimizer_name = local_optimizer
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.strategy = (strategy or "").strip().lower()
        self.target = (target or "").strip().lower()
        self.ts_kwargs = ts_kwargs or {}
        self.force_finite_diff_hessian = force_finite_diff_hessian
        if self.force_finite_diff_hessian and self.target == "ts":
            if "hessian_method" not in self.ts_kwargs:
                self.ts_kwargs["hessian_method"] = "finite_differences"

        self.initial_hessian = initial_hessian

        self.calculator_manager = CalculatorManager(
            backend=backend,
            model_name=model_name,
            model_path=model_path,
            device=device,
            default_charge=default_charge,
            default_spin=default_spin,
            verbose=verbose,
        )
        self.constraint_manager = ConstraintManager(
            constraints_spec=constraints,
            cache_parsed=True,
        )

        if auto_register:
            pass

    # --- Calculator & Constraints Management ---

    def _get_effective_model_name(self) -> str:
        """Get effective model name after applying backend defaults."""
        return self.calculator_manager.get_effective_model_name()

    def _get_effective_optimizer(self) -> str:
        """Get effective optimizer name after applying defaults."""
        if self.local_optimizer_name != "default":
            return self.local_optimizer_name
        if self.target == "ts":
            return "sella"
        return "lbfgs"

    def _create_and_attach_calculator(self, atoms: Atoms) -> Any:
        """Create and attach calculator to atoms."""
        return self.calculator_manager.create_and_attach_calculator(atoms)

    def _apply_constraints(self, atoms: Atoms) -> list[Any]:
        """Apply constraints to atoms."""
        return self.constraint_manager.apply_constraints(atoms)

    # --- Strategy Management ---

    def list_strategies(self, kind: str | None = None) -> dict[str, dict[str, str]]:
        """List available strategies; optionally filter by kind."""
        out: dict[str, dict[str, str]] = {}
        for name, metadata in REGISTRY.list_strategies().items():
            if kind is None or metadata.target == kind:
                out[name] = {
                    "type": (
                        "multi-structure" if metadata.requires_multiple_structures else "local"
                    ),
                    "description": metadata.description,
                }
                for alias in metadata.aliases:
                    out[alias] = {
                        "type": (
                            "multi-structure" if metadata.requires_multiple_structures else "local"
                        ),
                        "description": metadata.description,
                    }
        return out

    # --- Execution ---

    def explain_run(self) -> dict[str, str | bool | int | None]:
        """Explain what strategy would be selected without running."""
        try:
            if self.target and self.strategy:
                strategy_name = f"{self.target}:{self.strategy}"
            else:
                strategy_name = "minima:local"

            strategy_class = REGISTRY.get(strategy_name)
            metadata = strategy_class.metadata

            effective_optimizer = self._get_effective_optimizer()

            return {
                "target": metadata.target,
                "strategy": metadata.strategy,
                "strategy_key": strategy_name,
                "runner": strategy_class.__name__,
                "valid": True,
                "strategy_type": (
                    "multi-structure" if metadata.requires_multiple_structures else "local"
                ),
                "description": metadata.description,
                "notes": f"Will use {effective_optimizer} optimizer",
            }
        except StrategyNotFoundError as e:
            available = sorted(REGISTRY.list_strategies().keys())
            return {
                "target": "unknown",
                "strategy": "unknown",
                "strategy_key": "unknown",
                "runner": None,
                "valid": False,
                "error": f"No strategy found for '{strategy_name}'. Available: {available}",
                "notes": f"Error resolving strategy: {e}",
            }
        except Exception as e:
            return {
                "target": "unknown",
                "strategy": "unknown",
                "strategy_key": "unknown",
                "runner": None,
                "valid": False,
                "error": str(e),
                "notes": f"Error resolving strategy: {e}",
            }

    def run(
        self,
        runner: Callable[..., Any] | None = None,
        calculate_frequencies: bool = False,
        **kwargs: Any,
    ) -> dict[str, Atoms | list[Atoms] | bool | int | float | str]:
        """Execute a registered or user-supplied runner.

        This method is the main entry point for running optimization tasks. It uses
        the new strategy registry to find and execute the appropriate strategy.

        Parameters
        ----------
        runner : callable, optional
            Optional callable to execute directly, bypassing strategy selection.
        calculate_frequencies : bool, default=False
            Whether to perform frequency analysis after optimization.
        **kwargs
            Additional keyword arguments forwarded to the strategy runner.

        Returns
        -------
        dict[str, Atoms | list[Atoms] | bool | int | float | str]
            Standardized result dictionary containing:
            - optimized_atoms: Optimized structure(s) (Atoms or list[Atoms])
            - strategy: Strategy name used (str)
            - converged: Whether optimization converged (bool)
            - steps_taken: Number of optimization steps (int)
            - frequency_analysis: Frequency analysis results (dict, optional)
            - is_minimum/is_ts: Validation results (bool, optional)
            - free_energy_correction: Free energy correction in eV (float, optional)
            - Additional strategy-specific metadata

        Examples
        --------
        >>> # Minima optimization
        >>> explorer = Explorer(atoms, target="minima", strategy="local")
        >>> result = explorer.run()

        >>> # Path optimization with NEB
        >>> explorer = Explorer(atoms, target="path", strategy="neb")
        >>> result = explorer.run(npoints=7, fmax=0.05)

        >>> # Transition state optimization
        >>> explorer = Explorer(atoms, target="ts", strategy="local")
        >>> result = explorer.run(fmax=0.01)

        """
        if runner is not None:
            call_kwargs = dict(kwargs)
            call_kwargs.setdefault("explorer", self)
            call_kwargs.setdefault("local_optimizer_name", self._get_effective_optimizer())
            call_kwargs.setdefault("verbose", self.verbose)
            call_kwargs.setdefault("calculate_frequencies", calculate_frequencies)
            result = runner(self.atoms_list, **call_kwargs)
            if isinstance(result, dict):
                return result
            return {"optimized_atoms": result, "strategy": "custom"}

        if self.target and self.strategy:
            strategy_name = f"{self.target}:{self.strategy}"
        else:
            strategy_name = "minima:local"

        try:
            strategy_class = REGISTRY.get(strategy_name)
        except StrategyNotFoundError as e:
            raise NotImplementedError(
                f"No strategy found for '{strategy_name}'. "
                f"Available strategies: {sorted(REGISTRY.list_strategies().keys())}"
            ) from e

        strategy_instance = strategy_class(explorer=self, profiler=self.profiler)
        kwargs.setdefault("local_optimizer_name", self._get_effective_optimizer())
        kwargs.setdefault("verbose", self.verbose)
        kwargs.setdefault("calculate_frequencies", calculate_frequencies)
        return strategy_instance.run(self.atoms_list, **kwargs)

    # --- File I/O Methods ---

    @classmethod
    def from_file(
        cls,
        filename: str | Path,
        backend: str = "uma",
        model_name: str | None = None,
        model_path: str | None = None,
        device: str | None = None,
        default_charge: int = 0,
        default_spin: int = 1,
        verbose: int = 1,
        profile: bool = False,
        **kwargs: Any,
    ) -> Explorer:
        """Create Explorer instance from a geometry file.

        This is a convenience method that loads a molecular structure from a file
        and creates an Explorer instance ready for optimization.

        Parameters
        ----------
        filename : str or Path
            Path to geometry file. Supported formats:
            - .xyz: Extended XYZ format
            - .cif: Crystallographic Information File
            - .pdb: Protein Data Bank format
            - .vasp: VASP POSCAR format
            - .json: ASE JSON format
        backend : str, default "uma"
            Calculator backend to use (see Explorer.__init__ for options).
        model_name : str, optional
            Specific model name to use.
        model_path : str, optional
            Path to local model file.
        device : str, optional
            Device for computations ("cpu" or "cuda").
        default_charge : int, default 0
            Default charge for the system.
        default_spin : int, default 1
            Default spin multiplicity for the system.
        verbose : int, default 1
            Verbosity level for the Explorer.
        **kwargs : Any
            Additional arguments passed to Explorer constructor.

        Returns
        -------
        Explorer
            New Explorer instance with loaded geometry ready for optimization.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        ValueError
            If the file format is not supported or file is corrupted.

        Examples
        --------
        >>> # Load from XYZ file
        >>> explorer = Explorer.from_file("molecule.xyz", backend="aimnet2")
        >>> result = explorer.run(target="minima")

        >>> # Load with custom settings
        >>> explorer = Explorer.from_file(
        ...     "structure.cif",
        ...     backend="mace",
        ...     device="cuda",
        ...     default_charge=1
        ... )

        """
        geom = read_geometry(str(filename))
        if isinstance(geom, list):
            geom = geom[0]
        return cls(
            atoms=geom,
            backend=backend,
            model_name=model_name,
            model_path=model_path,
            device=device,
            default_charge=default_charge,
            default_spin=default_spin,
            verbose=verbose,
            profile=profile,
            **kwargs,
        )

    def load_structure(self, filename_or_geom: str | Path | Atoms) -> Atoms:
        """Load structure from file or geometry object and update atoms_list."""
        if isinstance(filename_or_geom, str | Path):
            file_path = Path(filename_or_geom)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            try:
                geom: Atoms | list[Atoms] = read_geometry(str(filename_or_geom))  # type: ignore[assignment]
            except Exception as e:
                raise ValueError(f"Failed to load geometry from {file_path}: {e}") from e
        elif isinstance(filename_or_geom, Atoms):
            geom = filename_or_geom
        else:
            raise TypeError(f"Expected str, Path, or Atoms, got {type(filename_or_geom).__name__}")

        if isinstance(geom, list):
            if not geom:
                raise ValueError("Loaded geometry list is empty")
            geom = geom[0]

        atoms_result: Atoms = geom

        if len(atoms_result) == 0:
            raise ValueError("Loaded structure has no atoms")
        self.atoms_list = [atoms_result]
        return atoms_result

    def save_structure(
        self,
        atoms: Atoms,
        output_file: str | Path,
        format: str | None = None,
    ) -> None:
        """Save structure to file.

        Parameters
        ----------
        atoms : Atoms
            Structure to save
        output_file : str or Path
            Output file path
        format : str, optional
            File format (inferred from extension if None)

        """
        write_atoms_safely(atoms, output_file, format)

    def save_trajectory(
        self,
        atoms_list: list[Atoms],
        output_file: str | Path,
        format: str | None = None,
    ) -> None:
        """Save multiple structures (trajectory) to file.

        This method is particularly useful for saving complete reaction pathways
        from NEB or CI-NEB calculations, which return multiple images along
        the reaction coordinate.

        Parameters
        ----------
        atoms_list : List[Atoms]
            List of structures to save as trajectory
        output_file : str or Path
            Output file path
        format : str, optional
            File format (inferred from extension if None)

        Examples
        --------
        >>> explorer = Explorer(atoms=[reactant, product], target="path")
        >>> result = explorer.run(npoints=7)
        >>> explorer.save_trajectory(result, "reaction_path.xyz")

        """
        write_trajectory_safely(atoms_list, output_file, format)

    # --- Analysis Methods ---

    def calculate_frequencies(
        self,
        atoms: Atoms | None = None,
        delta: float = 0.01,
        method: str = "auto",
        temperature: float = 298.15,
        save_hessian: bool = True,
        indices: list[int] | None = None,
        **kwargs: Any,
    ) -> dict[str, list[float] | float | dict[str, Any] | str | int | np.ndarray]:
        """Calculate vibrational frequencies and thermodynamic properties.

        Parameters
        ----------
        atoms : Atoms, optional
            Structure to analyze (uses first in atoms_list if None)
        delta : float
            Finite difference step size for Hessian calculation
        method : str
            Method for Hessian calculation ("auto", "numerical")
        temperature : float
            Temperature for thermodynamic properties (K)
        save_hessian : bool
            Whether to include Hessian matrix in results
        indices : list of int, optional
            Atom indices to include in calculation (all if None)
        **kwargs
            Additional arguments passed to FrequencyAnalysis

        Returns
        -------
        dict[str, list[float] | float | dict[str, Any] | str | int | np.ndarray]
            Dictionary containing:
            - frequencies: Vibrational frequencies in cm⁻¹ (list[float])
            - all_frequencies: All frequencies including trans/rot modes (list[float])
            - normal_modes: Normal mode vectors (list[float])
            - zero_point_energy: Zero-point energy in eV (float)
            - thermodynamic_properties: Thermodynamic data (dict[str, Any])
            - ts_analysis: Transition state analysis (dict[str, Any])
            - minima_analysis: Minima analysis (dict[str, Any])
            - is_ts: Whether structure is a transition state (bool)
            - is_minimum: Whether structure is a minimum (bool)
            - method_used: Hessian calculation method (str)
            - delta: Finite difference step size (float)
            - temperature: Temperature for thermodynamic properties (float)
            - n_atoms: Number of atoms (int)
            - indices: Atom indices included (list[int])
            - hessian: Hessian matrix (np.ndarray, optional)

        """
        from famex.analysis.frequency import FrequencyAnalysis

        if atoms is None:
            if not self.atoms_list:
                msg = "No structure available for frequency calculation"
                raise RuntimeError(msg)
            atoms = self.atoms_list[0]

        if getattr(atoms, "calc", None) is None:
            self._create_and_attach_calculator(atoms)

        freq_analysis = FrequencyAnalysis(
            atoms=atoms,
            calculator=atoms.calc,
            delta=delta,
            indices=indices,
        )
        if self.force_finite_diff_hessian and method == "auto":
            method = "finite_differences"
        hessian = freq_analysis.calculate_hessian(method=method)
        frequencies, _normal_modes = freq_analysis.diagonalize_hessian()
        vib_frequencies = freq_analysis.get_frequencies()
        vib_normal_modes = freq_analysis.get_normal_modes()
        thermo = freq_analysis.get_thermodynamic_properties(temperature)
        ts_analysis = freq_analysis.is_transition_state()
        minima_analysis = freq_analysis.is_minima()

        results = {
            "frequencies": vib_frequencies.tolist(),
            "all_frequencies": frequencies.tolist(),
            "normal_modes": vib_normal_modes.tolist(),
            "zero_point_energy": freq_analysis.get_zero_point_energy(),
            "thermodynamic_properties": thermo,
            "ts_analysis": ts_analysis,
            "minima_analysis": minima_analysis,
            "is_ts": ts_analysis["is_transition_state"],
            "is_minimum": minima_analysis["is_minimum"],
            "method_used": method,
            "delta": delta,
            "temperature": temperature,
            "n_atoms": len(atoms),
            "indices": indices if indices is not None else list(range(len(atoms))),
        }
        if save_hessian:
            results["hessian"] = hessian.tolist()

        return results


__all__ = ["Explorer"]
