"""Core Explorer class for QME molecular geometry optimization.

This module provides the main Explorer class that serves as the primary
interface for molecular geometry optimization using ASE and SELLA optimizers
combined with machine learning potentials.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ase import Atoms
from ase.io import write

from qme.backends.registry import create_calculator
from qme.constraints.parser import parse_constraints
from qme.core.registry import REGISTRY
from qme.io.geometry import read_geometry
from qme.io.xyz_io import parse_xyz_comment, write_xyz_with_metadata
from qme.utils.profiler import PerformanceProfiler

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy as np


def _extract_charge_spin(
    atoms: Atoms,
    default_charge: int = 0,
    default_spin: int = 1,
) -> tuple[int, int]:
    """Extract charge and spin from atoms object.

    Precedence: XYZ comment metadata > atoms.charge/atoms.mult > atoms.info['charge']/atoms.info['spin'] > defaults

    Parameters
    ----------
    atoms : Atoms
        The atoms object to extract charge/spin from
    default_charge : int
        Default charge if not found
    default_spin : int
        Default spin if not found

    Returns:
    -------
    tuple[int, int]
        (charge, spin) tuple

    """
    charge = None
    spin = None

    # Check XYZ comment line metadata first (highest priority)
    if hasattr(atoms, "info") and atoms.info is not None:
        comment = atoms.info.get("comment", "")
        if comment:
            metadata = parse_xyz_comment(comment)
            if "charge" in metadata:
                try:
                    charge = int(metadata["charge"])
                except (ValueError, TypeError):
                    pass
            if "spin" in metadata:
                try:
                    spin = int(metadata["spin"])
                except (ValueError, TypeError):
                    pass

    # Check for attributes (high priority)
    if hasattr(atoms, "charge") and charge is None:
        try:
            charge = int(atoms.charge)
        except (ValueError, TypeError):
            pass

    if hasattr(atoms, "mult") and spin is None:
        try:
            spin = int(atoms.mult)
        except (ValueError, TypeError):
            pass

    # Check atoms.info (medium priority)
    if hasattr(atoms, "info") and atoms.info is not None:
        if charge is None and "charge" in atoms.info:
            try:
                charge_val = atoms.info.get("charge")
                if charge_val is not None:
                    charge = int(charge_val)
            except (ValueError, TypeError):
                pass

        if spin is None and "spin" in atoms.info:
            try:
                spin_val = atoms.info.get("spin")
                if spin_val is not None:
                    spin = int(spin_val)
            except (ValueError, TypeError):
                pass

    # Use defaults if still None
    if charge is None:
        charge = default_charge
    if spin is None:
        spin = default_spin

    return charge, spin


class Explorer:
    """Explorer runs optimizations/TS searches on one or more Atoms.

    The Explorer provides a clear semantic interface for quantum chemistry calculations
    using machine learning potentials. It supports various optimization strategies
    for finding minima, transition states, and reaction pathways.

    The Explorer uses a target/strategy paradigm:
    - **target**: What you want to obtain (minima, ts, path)
    - **strategy**: How to get there (local, neb, cineb, interpolate)

    Parameters
    ----------
    atoms : Atoms or Sequence[Atoms]
        Single ASE Atoms object or a sequence of Atoms to operate on.
        For multi-structure strategies (NEB, CI-NEB), provide multiple structures.
    backend : str, default "uma"
        Calculator backend key. Available options:
        - "uma": Universal Model for Atoms (recommended)
        - "aimnet2": AIMNet2 neural network potential
        - "mace": MACE (Message Passing Neural Network)
        - "so3lr": SO3LR equivariant neural network
        - "mock": Mock calculator for testing
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
    local_optimizer : str, default "sella"
        Local optimizer for geometry optimization. Options:
        - "sella": SELLA optimizer (recommended for TS searches)
        - "lbfgs": L-BFGS optimizer
        - "bfgs": BFGS optimizer
        - "fire": FIRE optimizer
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

    Notes:
    -----
    - If a provided structure exposes ``charge``/``mult`` attributes or
      ``atoms.info`` includes ``charge``/``spin``, those values override the
      defaults when creating calculators.
    - Use :meth:`list_strategies` to discover available strategies and their
      descriptions.
    - The Explorer automatically handles calculator creation and caching.

    Examples:
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

    # =============================================================================
    # Initialization
    # =============================================================================

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

        # Setup QME logging with specified verbosity
        from qme.utils.logging import setup_qme_logging

        setup_qme_logging(verbosity=verbose)

        self.local_optimizer_name = local_optimizer
        self.optimizer_kwargs = optimizer_kwargs or {}

        # Strategy/target control how run() selects a strategy.
        self.strategy = (strategy or "").strip().lower()
        self.target = (target or "").strip().lower()
        self.ts_kwargs = ts_kwargs or {}

        self.constraints_spec = constraints
        self.initial_hessian = initial_hessian

        # Auto-register default strategies unless caller opts out
        if auto_register:
            # Strategies are auto-registered via imports in __init__.py
            pass

    # =============================================================================
    # Calculator & Constraints Management
    # =============================================================================

    def _get_effective_model_name(self) -> str:
        """Get the effective model name that will actually be used by the backend.

        Returns the model name that will be used after applying backend-specific defaults.
        """
        if self.model_name is not None:
            return self.model_name

        # Apply backend-specific defaults
        backend_lower = self.backend.lower()
        if backend_lower == "uma":
            return "uma-s-1p1"
        if backend_lower == "aimnet2":
            return "aimnet2"
        if backend_lower in ("mace", "torchsim_mace"):
            return "mace-omol-0"
        if backend_lower == "torchsim_uma":
            return "uma-s-1p1"
        if backend_lower == "so3lr":
            # SO3LR requires a model_path, not model_name
            return self.model_path or "so3lr-model"
        if backend_lower == "mock":
            return "mock-model"
        return "default-model"

    def _get_effective_optimizer(self) -> str:
        """Get the effective optimizer that will actually be used.

        Returns the optimizer name after applying context-aware defaults.
        """
        if self.local_optimizer_name != "default":
            return self.local_optimizer_name

        # Context-aware selection based on target
        if self.target == "ts":
            return "sella"
        # minima or path
        return "lbfgs"

    def _check_missing_charge_spin(self, atoms: Atoms) -> tuple[bool, bool]:
        """Check if charge and/or spin are missing from atoms.

        Parameters
        ----------
        atoms : Atoms
            The atoms object to check

        Returns:
        -------
        tuple[bool, bool]
            (charge_missing, spin_missing) indicating if each is missing

        """
        charge_missing = True
        spin_missing = True

        # Check for attributes first (highest priority)
        if hasattr(atoms, "charge"):
            try:
                if atoms.charge is not None:
                    int(atoms.charge)  # Test conversion
                    charge_missing = False
            except (ValueError, TypeError):
                pass

        if hasattr(atoms, "mult"):
            try:
                if atoms.mult is not None:
                    int(atoms.mult)  # Test conversion
                    spin_missing = False
            except (ValueError, TypeError):
                pass

        # Check atoms.info (medium priority)
        if hasattr(atoms, "info") and atoms.info is not None:
            if "charge" in atoms.info:
                try:
                    charge_val = atoms.info.get("charge")
                    if charge_val is not None:
                        charge_missing = False
                except (ValueError, TypeError):
                    pass

            if "spin" in atoms.info:
                try:
                    spin_val = atoms.info.get("spin")
                    if spin_val is not None:
                        spin_missing = False
                except (ValueError, TypeError):
                    pass

        return charge_missing, spin_missing

    def _create_and_attach_calculator(self, atoms: Atoms) -> Any:
        """Create and attach an ASE calculator to ``atoms``.

        Prefers explicit ``charge``/``mult`` found on Geometry-like objects or
        in ``atoms.info``. Falls back to the Explorer defaults otherwise.
        """
        # Extract charge and spin using helper function
        charge, spin = _extract_charge_spin(atoms, self.default_charge, self.default_spin)

        # Check if we're using defaults and warn once
        charge_missing, spin_missing = self._check_missing_charge_spin(atoms)

        if (charge_missing or spin_missing) and not getattr(self, "_warned_about_defaults", False):
            from qme.utils.logging import get_qme_logger

            logger = get_qme_logger(__name__)

            missing_parts = []
            if charge_missing:
                missing_parts.append(f"charge={charge}")
            if spin_missing:
                missing_parts.append(f"spin={spin}")

            logger.warning(
                f"Charge and/or spin not specified in atoms. Using defaults: {', '.join(missing_parts)}",
            )
            self._warned_about_defaults = True

        # Ensure atoms.info contains values so calculators that read
        # atoms.info (UMA, SO3LR, etc.) see the intended settings.
        if getattr(atoms, "info", None) is not None:
            # Coerce to built-in int types to satisfy backends that enforce strict typing
            try:
                atoms.info["charge"] = int(atoms.info.get("charge", charge))
            except (ValueError, TypeError):
                atoms.info["charge"] = int(charge)
            try:
                atoms.info["spin"] = int(atoms.info.get("spin", spin))
            except (ValueError, TypeError):
                atoms.info["spin"] = int(spin)

        # Show model initialization info when creating the first calculator
        if not hasattr(self, "_calculator_created"):
            from qme.utils.logging import print_model_info

            # Get the effective model name that will actually be used
            effective_model_name = self._get_effective_model_name()
            print_model_info(self.backend, effective_model_name, self.model_path, self.device)
            self._calculator_created = True

        calc = create_calculator(
            backend=self.backend,
            model_name=self.model_name,
            model_path=self.model_path,
            device=self.device,
            default_charge=self.default_charge,
            default_spin=self.default_spin,
            charge=charge,
            mult=spin,
            verbose=self.verbose,
        )
        atoms.calc = calc
        return calc

    def _apply_constraints(self, atoms: Atoms) -> list[Any]:
        """Parse and apply constraints to ``atoms`` if specified.

        Returns the ASE constraints list after application.
        """
        if self.constraints_spec is None:
            return []
        return parse_constraints(self.constraints_spec, atoms, verbose=False)

    # =============================================================================
    # Strategy Management
    # =============================================================================

    def list_strategies(self, kind: str | None = None) -> dict[str, dict[str, str]]:
        """List available strategies; optionally filter by kind."""
        out: dict[str, dict[str, str]] = {}
        for name, metadata in REGISTRY.list_strategies().items():
            if kind is None or metadata.target == kind:
                # Add main strategy name
                out[name] = {
                    "type": "multi-structure" if metadata.requires_multiple_structures else "local",
                    "description": metadata.description,
                }
                # Add aliases
                for alias in metadata.aliases:
                    out[alias] = {
                        "type": (
                            "multi-structure" if metadata.requires_multiple_structures else "local"
                        ),
                        "description": metadata.description,
                    }
        return out

    # =============================================================================
    # Execution
    # =============================================================================

    def explain_run(self) -> dict[str, str | bool | int | None]:
        """Explain what strategy would be selected without running.

        Returns:
        -------
        dict[str, Union[str, bool, int, None]]
            Dictionary with strategy selection details containing:
            - target: Target type (str)
            - strategy: Strategy type (str)
            - strategy_key: Full strategy name (str)
            - runner: Strategy class name (str or None)
            - valid: Whether strategy is valid (bool)
            - strategy_type: Type of strategy (str)
            - description: Strategy description (str)
            - notes: Additional notes (str)
            - error: Error message if invalid (str, optional)

        """
        try:
            # Determine strategy name from constructor target/strategy
            if self.target and self.strategy:
                strategy_name = f"{self.target}:{self.strategy}"
            else:
                strategy_name = "minima:local"

            # Get strategy class from registry
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
        except KeyError as e:
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

        Returns:
        -------
        dict[str, Union[Atoms, list[Atoms], bool, int, float, str]]
            Standardized result dictionary containing:
            - optimized_atoms: Optimized structure(s) (Atoms or list[Atoms])
            - strategy: Strategy name used (str)
            - converged: Whether optimization converged (bool)
            - steps_taken: Number of optimization steps (int)
            - frequency_analysis: Frequency analysis results (dict, optional)
            - is_minimum/is_ts: Validation results (bool, optional)
            - free_energy_correction: Free energy correction in eV (float, optional)
            - Additional strategy-specific metadata

        Examples:
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
        # If the caller passed an explicit runner function, use it directly
        if runner is not None:
            call_kwargs = dict(kwargs)
            call_kwargs.setdefault("explorer", self)
            call_kwargs.setdefault("local_optimizer_name", self._get_effective_optimizer())
            call_kwargs.setdefault("verbose", self.verbose)
            call_kwargs.setdefault("calculate_frequencies", calculate_frequencies)
            result = runner(self.atoms_list, **call_kwargs)
            # Ensure result is properly typed
            if isinstance(result, dict):
                return result
            # Convert to expected format if needed
            return {"optimized_atoms": result, "strategy": "custom"}

        # Determine strategy name from constructor target/strategy
        if self.target and self.strategy:
            strategy_name = f"{self.target}:{self.strategy}"
        else:
            # Default to minima:local
            strategy_name = "minima:local"

        # Get strategy class from registry
        try:
            strategy_class = REGISTRY.get(strategy_name)
        except KeyError as e:
            available = sorted(REGISTRY.list_strategies().keys())
            msg = f"No strategy found for '{strategy_name}'. Available strategies: {available}"
            raise NotImplementedError(
                msg,
            ) from e

        # Instantiate and run strategy
        strategy_instance = strategy_class(explorer=self, profiler=self.profiler)
        # Pass the effective optimizer name to the strategy
        kwargs.setdefault("local_optimizer_name", self._get_effective_optimizer())
        kwargs.setdefault("verbose", self.verbose)
        kwargs.setdefault("calculate_frequencies", calculate_frequencies)
        return strategy_instance.run(self.atoms_list, **kwargs)

    # =============================================================================
    # File I/O Methods
    # =============================================================================

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

        Returns:
        -------
        Explorer
            New Explorer instance with loaded geometry ready for optimization.

        Raises:
        ------
        FileNotFoundError
            If the specified file does not exist.
        ValueError
            If the file format is not supported or file is corrupted.

        Examples:
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
        """Load structure from file or geometry object and update atoms_list.

        Parameters
        ----------
        filename_or_geom : str, Path, or Atoms
            File path or geometry object to load

        Returns:
        -------
        Atoms
            Loaded geometry

        """
        if isinstance(filename_or_geom, (str, Path)):
            geom = read_geometry(str(filename_or_geom))
        else:
            geom = filename_or_geom  # type: ignore[assignment]

        if isinstance(geom, list):
            geom = geom[0]

        # Update the atoms_list to contain this new geometry
        self.atoms_list = [geom]
        return geom

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
        output_file = Path(output_file)

        # Use custom XYZ writer for .xyz files to preserve metadata
        if str(output_file).lower().endswith(".xyz"):
            try:
                write_xyz_with_metadata(atoms, str(output_file))
                return
            except Exception as e:
                msg = f"Failed to save XYZ structure to {output_file}: {e}"
                raise RuntimeError(msg) from e

        # Use ASE for other formats
        try:
            if format is not None:
                write(output_file, atoms, format=format)
            else:
                write(output_file, atoms)
        except Exception as e:
            # If writing fails, try with a cleaned atoms object to avoid
            # issues with contaminated global state from test isolation
            try:
                # Create a clean atoms object with only essential data
                clean_atoms = Atoms(
                    symbols=atoms.symbols,
                    positions=atoms.positions,
                    cell=atoms.cell,
                    pbc=atoms.pbc,
                )
                # Copy over essential info
                if hasattr(atoms, "info") and atoms.info:
                    for key in ["charge", "spin"]:
                        if key in atoms.info:
                            clean_atoms.info[key] = atoms.info[key]

                if format is not None:
                    write(output_file, clean_atoms, format=format)
                else:
                    write(output_file, clean_atoms)
            except Exception as e2:
                msg = (
                    f"Failed to save structure to {output_file}: {e}. "
                    f"Clean attempt also failed: {e2}"
                )
                raise RuntimeError(
                    msg,
                )

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

        Examples:
        --------
        >>> explorer = Explorer(atoms=[reactant, product], target="path")
        >>> result = explorer.run(npoints=7)
        >>> explorer.save_trajectory(result, "reaction_path.xyz")

        """
        output_file = Path(output_file)

        # Use custom XYZ writer for .xyz files to preserve metadata
        if str(output_file).lower().endswith(".xyz"):
            try:
                write_xyz_with_metadata(atoms_list, str(output_file))
                return
            except Exception as e:
                msg = f"Failed to save XYZ trajectory to {output_file}: {e}"
                raise RuntimeError(msg) from e

        # Use ASE for other formats
        try:
            if format is not None:
                write(output_file, atoms_list, format=format)
            else:
                write(output_file, atoms_list)
        except Exception as e:
            # If writing fails, try with cleaned atoms objects to avoid
            # issues with contaminated global state from test isolation
            try:
                # Create clean atoms objects with only essential data
                clean_atoms_list = []
                for atoms in atoms_list:
                    clean_atoms = Atoms(
                        symbols=atoms.symbols,
                        positions=atoms.positions,
                        cell=atoms.cell,
                        pbc=atoms.pbc,
                    )
                    # Copy over essential info
                    if hasattr(atoms, "info") and atoms.info:
                        for key in ["charge", "spin"]:
                            if key in atoms.info:
                                clean_atoms.info[key] = atoms.info[key]
                    clean_atoms_list.append(clean_atoms)

                if format is not None:
                    write(output_file, clean_atoms_list, format=format)
                else:
                    write(output_file, clean_atoms_list)
            except Exception as e2:
                msg = (
                    f"Failed to save trajectory to {output_file}: {e}. "
                    f"Clean attempt also failed: {e2}"
                )
                raise RuntimeError(
                    msg,
                )

    # =============================================================================
    # Analysis Methods
    # =============================================================================

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

        Returns:
        -------
        dict[str, Union[list[float], float, dict[str, Any], str, int, np.ndarray]]
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
        from qme.analysis.frequency import FrequencyAnalysis

        if atoms is None:
            if not self.atoms_list:
                msg = "No structure available for frequency calculation"
                raise RuntimeError(msg)
            atoms = self.atoms_list[0]

        if getattr(atoms, "calc", None) is None:
            # Ensure atoms.info has charge/spin info before creating calculator
            # This prevents repeated warnings during frequency calculations
            if getattr(atoms, "info", None) is not None:
                charge, spin = _extract_charge_spin(atoms, self.default_charge, self.default_spin)
                try:
                    atoms.info["charge"] = int(atoms.info.get("charge", charge))
                except Exception:
                    atoms.info["charge"] = int(charge)
                try:
                    atoms.info["spin"] = int(atoms.info.get("spin", spin))
                except Exception:
                    atoms.info["spin"] = int(spin)
            self._create_and_attach_calculator(atoms)

        freq_analysis = FrequencyAnalysis(
            atoms=atoms,
            calculator=atoms.calc,
            delta=delta,
            indices=indices,
        )
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
