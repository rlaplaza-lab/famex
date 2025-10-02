"""Explorer: ASE-like multi-structure optimizer and TS search wrapper.

The :class:`Explorer` provides an ASE-inspired façade for common tasks:
- Local minima optimization on one or more structures
- Local transition-state (TS) searches
- Two-ended, interpolation-based path workflows for minima or TS guesses

Design goals:
- Keep an ASE-optimizer-like surface (e.g., ``optimize_minima``, ``optimize_ts``)
- Provide robust defaults and graceful fallbacks when optional deps are missing
- Offer a pluggable strategy system to register new runners without core edits

Implementation notes:
- Calculator creation is delegated to ``qme.core.calculator_setup``
- Constraints are parsed/applied via ``qme.core.constraint_parser``
- Default local runners live in ``qme.core.local_strategies``
- Default two-ended runners live in ``qme.core.twoended_strategies``
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from ase import Atoms
from ase.io import write

from qme.core.calculator_setup import create_calculator
from qme.core.constraint_parser import parse_constraints
from qme.core.geometry import read_geometry
from qme.core.local_strategies import local_minima_runner, local_ts_runner
from qme.core.twoended_strategies import (
    twoended_minima_runner,
    twoended_neb_runner,
    twoended_ts_guess_runner,
)
from qme.dependencies import deps


class Explorer:
    """Explorer runs optimizations/TS searches on one or more Atoms.

    Parameters
    ----------
    atoms
        Single :class:`ase.Atoms` or a sequence of Atoms to operate on.
    backend
        Calculator backend key (e.g., ``uma``, ``aimnet2``, ``mace``, ``so3lr``, ``mock``).
    model_name, model_path, device
        Forwarded to calculator factory when applicable.
    default_charge, default_spin
        Default total charge and spin multiplicity used when per-structure
        metadata is not available.
    local_optimizer
        Local optimizer short name, e.g. ``sella``, ``lbfgs``, ``bfgs``, ``fire``.
    optimizer_kwargs
        Keyword arguments forwarded to local minima optimizer.
    strategy, target
        High-level run selection. ``strategy`` is typically ``local`` or
        ``two-ended``; ``target`` is ``minima`` or ``ts``.
    ts_kwargs
        Keyword arguments forwarded to the local TS optimizer.
    constraints
        Constraint specification as string or pre-built ASE constraints.
    auto_register
        If True, registers package default strategies; set False for fully
        custom registries.

    Notes
    -----
    - If a provided structure exposes ``charge``/``mult`` attributes or
      ``atoms.info`` includes ``charge``/``spin``, those values override the
      defaults when creating calculators.
    - Use :meth:`list_strategies` to discover available runners and their
      descriptions.
    """

    def __init__(
        self,
        atoms: Union[Atoms, Sequence[Atoms]],
        backend: str = "uma",
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        default_charge: int = 0,
        default_spin: int = 1,
        local_optimizer: str = "sella",
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        strategy: Optional[str] = "local",
        target: Optional[str] = "minima",
        mode: Optional[str] = None,
        ts_method: Optional[str] = None,
        ts_kwargs: Optional[Dict[str, Any]] = None,
        constraints: Optional[Union[str, List, Dict]] = None,
        auto_register: bool = True,
    ):
        if isinstance(atoms, Atoms):
            self.atoms_list: List[Atoms] = [atoms]
        else:
            self.atoms_list = list(atoms)

        self.backend = backend
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.default_charge = default_charge
        self.default_spin = default_spin

        self.local_optimizer_name = local_optimizer
        self.optimizer_kwargs = optimizer_kwargs or {}

        # Strategy/target/mode control how run() selects a runner.
        self.strategy = (strategy or "").strip().lower()
        self.target = (target or "").strip().lower()
        # Mode is a secondary selector; for two-ended it should be 'interpolate'
        self.mode = (mode or "").strip().lower() if mode is not None else None

        self.ts_method = ts_method
        self.ts_kwargs = ts_kwargs or {}

        self.constraints_spec = constraints

        # Auto-register default strategies unless caller opts out
        if getattr(self, "_strategies", None) is None:
            self._strategies = {}

        # Register default strategies. Local minima is always available.
        self.register_strategy(
            "minima",
            local_minima_runner,
            strategy_type="local",
            description="Local minima optimization (ASE/LBFGS or SELLA)",
            aliases=["local:minima"],
        )

        # Register local TS runner
        self.register_strategy(
            "ts",
            local_ts_runner,
            strategy_type="local",
            description="Local transition-state optimization (SELLA preferred)",
            aliases=["local:ts"],
        )

        # Two-ended runners are available regardless; callers will be
        # validated at run time (requires two or more Atoms). We expose
        # both TS-guess and minima variants.
        self.register_strategy(
            "twoended:ts",
            twoended_ts_guess_runner,
            strategy_type="two-ended",
            description="Two-ended TS guess via interpolation with local TS refinement",
            aliases=["twoended-ts"],
        )
        self.register_strategy(
            "twoended:minima",
            twoended_minima_runner,
            strategy_type="two-ended",
            description="Two-ended minima optimization on low-energy frames",
            aliases=["twoended-minima"],
        )
        self.register_strategy(
            "twoended:neb",
            twoended_neb_runner,
            strategy_type="two-ended",
            description="Nudged Elastic Band (NEB) path optimization with geodesic interpolation",
            aliases=["neb", "twoended-neb"],
        )

    # --- Backend and constraints helpers
    def _create_and_attach_calculator(self, atoms: Atoms):
        """Create and attach an ASE calculator to ``atoms``.

        Prefers explicit ``charge``/``mult`` found on Geometry-like objects or
        in ``atoms.info``. Falls back to the Explorer defaults otherwise.
        """
        geom_charge = None
        geom_mult = None

        # Some Geometry subclasses store as attributes `charge` and `mult`.
        if hasattr(atoms, "charge"):
            try:
                geom_charge = int(getattr(atoms, "charge"))
            except Exception:
                geom_charge = None
        if hasattr(atoms, "mult"):
            try:
                geom_mult = int(getattr(atoms, "mult"))
            except Exception:
                geom_mult = None

        # ASE Atoms may store info in atoms.info (e.g. calculators expect
        # 'charge' and 'spin' keys). Use those if present unless Geometry
        # attributes override them.
        if getattr(atoms, "info", None) is not None:
            if "charge" in atoms.info and geom_charge is None:
                try:
                    geom_charge = int(atoms.info.get("charge"))
                except Exception:
                    geom_charge = geom_charge
            if "spin" in atoms.info and geom_mult is None:
                try:
                    geom_mult = int(atoms.info.get("spin"))
                except Exception:
                    geom_mult = geom_mult

        # Ensure atoms.info contains values so calculators that read
        # atoms.info (UMA, SO3LR, etc.) see the intended settings.
        if getattr(atoms, "info", None) is not None:
            if "charge" not in atoms.info:
                atoms.info["charge"] = (
                    geom_charge if geom_charge is not None else self.default_charge
                )
            if "spin" not in atoms.info:
                atoms.info["spin"] = (
                    geom_mult if geom_mult is not None else self.default_spin
                )

        calc = create_calculator(
            backend=self.backend,
            model_name=self.model_name,
            model_path=self.model_path,
            device=self.device,
            default_charge=self.default_charge,
            default_spin=self.default_spin,
            charge=geom_charge,
            mult=geom_mult,
        )
        atoms.calc = calc
        return calc

    def _apply_constraints(self, atoms: Atoms):
        """Parse and apply constraints to ``atoms`` if specified.

        Returns the ASE constraints list after application.
        """
        if self.constraints_spec is None:
            return []
        return parse_constraints(self.constraints_spec, atoms, verbose=False)

    def register_strategy(
        self,
        name: str,
        func,
        strategy_type: str = "global",
        description: Optional[str] = None,
        aliases: Optional[List[str]] = None,
    ):
        """Register a strategy/runner callable under a name.

        The callable must accept the atoms list as the first positional
        argument. Explorer injects ``explorer=self`` and a few standardized
        keywords (e.g., ``local_optimizer_name``) unless explicitly overridden.

        Parameters
        ----------
        name
            Unique key under which the strategy is registered.
        func
            Callable implementing the strategy.
        strategy_type
            One of ``local``, ``two-ended`` or ``global`` (dispatch shape).
        description
            Short human-readable description for discovery/CLI help.
        aliases
            Optional list of alternative keys mapping to the same entry.

        Returns
        -------
        None
        """
        if not hasattr(self, "_strategies"):
            self._strategies = {}
        # Normalize strategy_type to one of the supported kinds
        stype = (strategy_type or "").strip().lower()
        if stype in ("local", "one-ended", "oneended", "one_ended"):
            stype = "local"
        elif stype in ("two-ended", "two_ended", "twoended", "two"):
            stype = "two-ended"
        else:
            stype = "global"

        # Store a small descriptor so run() can apply simple arity/shape
        # checks and adapt calling convention for local (single-Atoms)
        # strategies.
        entry = {"func": func, "type": stype, "description": description or ""}
        self._strategies[name] = entry
        if aliases:
            for alias in aliases:
                self._strategies[alias] = entry

    def list_strategies(self, kind: Optional[str] = None) -> Dict[str, Dict[str, str]]:
        """List available strategies; optionally filter by kind (local/two-ended/global)."""
        out: Dict[str, Dict[str, str]] = {}
        for key, meta in getattr(self, "_strategies", {}).items():
            if kind is None or meta.get("type") == kind:
                out[key] = {
                    "type": meta.get("type", ""),
                    "description": meta.get("description", ""),
                }
        return out

    def run(self, mode: Optional[str] = None, runner=None, **kwargs):
        """Execute a registered or user-supplied runner.

        Parameters
        ----------
        mode
            Short selector used to choose among registered strategies when
            ``runner`` is not provided. Typically ``minima`` or ``ts`` for
            local strategies, or ``interpolate`` for two-ended.
        runner
            Optional callable to execute directly (signature: ``runner(atoms_list, **kwargs)``).
        **kwargs
            Forwarded to the runner; Explorer injects ``explorer=self`` and
            ``local_optimizer_name`` defaults.

        Returns
        -------
        Any
            Whatever the runner returns.
        """
        # Decide effective mode: explicit call-time mode overrides instance target
        effective_mode = (mode or self.target or "minima").strip().lower()

        # Normalize strategy name
        st = (self.strategy or "").strip().lower()
        if st in ("two-ended", "two_ended", "twoended", "two"):
            effective_strategy = "two-ended"
        else:
            effective_strategy = "local"

        # Two-ended strategies: select runner based on requested target
        if effective_strategy == "two-ended":
            # Only interpolation-driven path construction is supported for two-ended
            if effective_mode not in (
                "interpolate",
                "ts",
                "minima",
                "neb",
                "twoended",
                "two-ended",
            ):
                warnings.warn(
                    "Two-ended strategy expects interpolation; forcing mode to 'interpolate'"
                )
                effective_mode = "interpolate"

            # Choose between TS-guess path refinement, minima refinement, or NEB along the path
            if self.target in (
                "ts",
                "transition",
                "transition-state",
                "transition_state",
            ):
                preferred = ["twoended:ts", "twoended-ts"]
            elif self.target in ("neb", "nudged-elastic-band"):
                preferred = ["twoended:neb", "neb", "twoended-neb"]
            else:
                preferred = ["twoended:minima", "twoended-minima"]

            strategies = getattr(self, "_strategies", {})
            strategy_key = next(
                (k for k in preferred if k in strategies), preferred[-1]
            )

        else:
            # Local strategies support minima and ts (if registered)
            if effective_mode in (
                "ts",
                "transition",
                "transition-state",
                "transition_state",
            ):
                preferred = ["local:ts", "ts"]
            else:
                preferred = ["local:minima", "minima"]

            strategies = getattr(self, "_strategies", {})
            strategy_key = next(
                (k for k in preferred if k in strategies), preferred[-1]
            )

        # If the caller passed an explicit runner, use it; otherwise look up
        # the registered strategy entry.
        # Allow an explicit runner name override via 'runner_name' kwarg
        runner_name = kwargs.pop("runner_name", None)
        if runner is None:
            if isinstance(runner_name, str):
                strategy_key = runner_name
            strategies = getattr(self, "_strategies", {})
            entry = strategies.get(strategy_key)
            if entry is None:
                raise NotImplementedError(
                    f"No registered strategy for '{strategy_key}'. "
                    f"Requested strategy: {self.strategy}, mode: {effective_mode}"
                )
            runner = entry.get("func") if isinstance(entry, dict) else entry
            strategy_type = (
                entry.get("type", "global") if isinstance(entry, dict) else "global"
            )
        else:
            # If runner provided directly, we assume caller knows what they're doing
            strategy_type = "global"

        # Prepare kwargs for runner
        call_kwargs = dict(kwargs)
        call_kwargs.setdefault("explorer", self)
        call_kwargs.setdefault("local_optimizer_name", self.local_optimizer_name)

        # Dispatch according to strategy type
        if strategy_type == "local":
            results = []
            for atoms in self.atoms_list:
                out = runner(atoms, **call_kwargs)
                results.append(out)
            return results

        if strategy_type == "two-ended":
            if len(self.atoms_list) < 2:
                raise ValueError(
                    "Two-ended strategies require two or more Atoms objects"
                )
            return runner(self.atoms_list, **call_kwargs)

        # global/default: pass the full list as-is
        return runner(self.atoms_list, **call_kwargs)

    # --- Convenience wrappers mirroring ASE Optimizer ergonomics ---
    def optimize_minima(self, fmax: float = 0.05, steps: int = 1000, **kwargs):
        """Run a local minima optimization (ASE-like convenience method)."""
        return self.run(mode="minima", fmax=fmax, steps=steps, **kwargs)

    def optimize_ts(self, fmax: float = 0.05, steps: int = 1000, **kwargs):
        """Run a local transition-state optimization (ASE-like convenience method)."""
        return self.run(mode="ts", fmax=fmax, steps=steps, **kwargs)

    # --- Additional convenience methods for compatibility ---
    @classmethod
    def from_file(
        cls,
        filename: Union[str, Path],
        backend: str = "uma",
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        default_charge: int = 0,
        default_spin: int = 1,
        **kwargs,
    ) -> "Explorer":
        """Create Explorer instance from a geometry file.

        Parameters
        ----------
        filename : str or Path
            Path to geometry file (xyz, cif, pdb, etc.)
        backend, model_name, model_path, device, default_charge, default_spin
            Same as Explorer constructor
        **kwargs
            Additional arguments passed to Explorer constructor

        Returns
        -------
        Explorer
            New Explorer instance with loaded geometry
        """
        geom = read_geometry(filename)
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
            **kwargs,
        )

    def load_structure(self, filename_or_geom):
        """Load structure from file or geometry object and update atoms_list.

        Parameters
        ----------
        filename_or_geom : str, Path, or Atoms
            File path or geometry object to load

        Returns
        -------
        Atoms
            Loaded geometry
        """
        if isinstance(filename_or_geom, (str, Path)):
            geom = read_geometry(filename_or_geom)
        else:
            geom = filename_or_geom

        if isinstance(geom, list):
            geom = geom[0]

        # Update the atoms_list to contain this new geometry
        self.atoms_list = [geom]
        return geom

    def save_structure(
        self, atoms: Atoms, output_file: Union[str, Path], format: Optional[str] = None
    ):
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

        try:
            if format is not None:
                write(output_file, atoms, format=format)
            else:
                write(output_file, atoms)
        except Exception as e:
            raise RuntimeError(f"Failed to save structure to {output_file}: {e}")

    def calculate_frequencies(
        self,
        atoms: Optional[Atoms] = None,
        delta: float = 0.01,
        method: str = "auto",
        temperature: float = 298.15,
        save_hessian: bool = True,
        indices: Optional[List[int]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
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
        dict
            Dictionary containing frequencies, normal modes, thermodynamic properties, etc.
        """
        from qme.analysis.frequency import FrequencyAnalysis

        if atoms is None:
            if not self.atoms_list:
                raise RuntimeError("No structure available for frequency calculation")
            atoms = self.atoms_list[0]

        if getattr(atoms, "calc", None) is None:
            self._create_and_attach_calculator(atoms)

        freq_analysis = FrequencyAnalysis(
            atoms=atoms, calculator=atoms.calc, delta=delta, indices=indices
        )
        hessian = freq_analysis.calculate_hessian(method=method)
        frequencies, normal_modes = freq_analysis.diagonalize_hessian()
        vib_frequencies = freq_analysis.get_frequencies()
        vib_normal_modes = freq_analysis.get_normal_modes()
        thermo = freq_analysis.get_thermodynamic_properties(temperature)
        ts_analysis = freq_analysis.is_transition_state()

        results = {
            "frequencies": vib_frequencies.tolist(),
            "all_frequencies": frequencies.tolist(),
            "normal_modes": vib_normal_modes.tolist(),
            "zero_point_energy": freq_analysis.get_zero_point_energy(),
            "thermodynamic_properties": thermo,
            "ts_analysis": ts_analysis,
            "is_ts": ts_analysis["is_transition_state"],
            "method_used": method,
            "delta": delta,
            "temperature": temperature,
            "n_atoms": len(atoms),
            "indices": indices if indices is not None else list(range(len(atoms))),
        }
        if save_hessian:
            results["hessian"] = hessian.tolist()

        return results

    # --- Backward compatibility methods ---
    def optimize_minimum(
        self, atoms=None, optimizer=None, fmax=0.05, steps=1000, **kwargs
    ):
        """Backward compatibility method for optimize_minimum.

        Parameters
        ----------
        atoms : Atoms, optional
            Structure to optimize (loads if provided, uses existing if None)
        optimizer : str, optional
            Optimizer name (overrides local_optimizer_name)
        fmax : float
            Force convergence threshold
        steps : int
            Maximum optimization steps
        **kwargs
            Additional arguments

        Returns
        -------
        dict
            Results dictionary with converged, final_energy, optimized_atoms, etc.
        """
        from qme.core.validation import (
            validate_atoms_structure,
            validate_optimization_parameters,
        )

        # Handle backward compatibility: if atoms is passed, load it
        if atoms is not None:
            self.load_structure(atoms)

        if not self.atoms_list:
            raise RuntimeError(
                "No structure loaded. Call load_structure() first or pass atoms parameter."
            )

        # Use first atoms in list for single optimization
        target_atoms = self.atoms_list[0]

        # Validate input parameters
        local_opt_name = (optimizer or self.local_optimizer_name).lower()
        validate_optimization_parameters(fmax, steps, local_opt_name)
        validate_atoms_structure(target_atoms, "optimization")

        # Allow callers to pass optimizer_kwargs inside kwargs
        old_optimizer_kwargs = self.optimizer_kwargs.copy()
        self.optimizer_kwargs.update(kwargs.get("optimizer_kwargs", {}))

        try:
            res = self.run(
                mode="minima",
                fmax=fmax,
                steps=steps,
                local_optimizer_name=local_opt_name,
            )
            optimized = res[0] if isinstance(res, list) and len(res) == 1 else res

            energy = None
            try:
                energy = float(optimized.get_potential_energy())
            except Exception:
                energy = None
            try:
                forces = optimized.get_forces()
                max_force = float(abs(forces).max())
            except Exception:
                max_force = None

            return {
                "converged": True,
                "final_energy": energy,
                "steps_taken": None,
                "optimized_atoms": optimized,
                "max_force": max_force,
            }
        finally:
            # Restore original optimizer_kwargs
            self.optimizer_kwargs = old_optimizer_kwargs

    def find_transition_state(
        self, atoms=None, fmax=0.05, steps=1000, **kwargs
    ) -> Dict[str, Any]:
        """Backward compatibility method for transition state optimization.

        Parameters
        ----------
        atoms : Atoms, optional
            Structure to optimize (loads if provided, uses existing if None)
        fmax : float
            Force convergence threshold
        steps : int
            Maximum optimization steps
        **kwargs
            Additional arguments including ts_method, ts_kwargs, etc.

        Returns
        -------
        dict
            Results dictionary with converged, final_energy, ts_atoms, etc.
        """
        from qme.core.validation import (
            validate_atoms_structure,
            validate_optimization_parameters,
        )

        # Handle backward compatibility: if atoms is passed, load it
        if atoms is not None:
            self.load_structure(atoms)

        if not self.atoms_list:
            raise RuntimeError(
                "No structure loaded. Call load_structure() first or pass atoms parameter."
            )

        # Use first atoms in list for single TS optimization
        target_atoms = self.atoms_list[0]

        # Validate input parameters
        local_opt_name = kwargs.get("local_optimizer_name", self.local_optimizer_name)
        validate_optimization_parameters(fmax, steps, local_opt_name)
        validate_atoms_structure(target_atoms, "transition state search")

        # Update TS settings from kwargs
        old_ts_method = getattr(self, "ts_method", None)
        old_ts_kwargs = self.ts_kwargs.copy()

        if "ts_method" in kwargs:
            self.ts_method = kwargs["ts_method"]
        if "ts_kwargs" in kwargs:
            self.ts_kwargs.update(kwargs["ts_kwargs"])

        try:
            res = self.run(
                mode="ts",
                fmax=fmax,
                steps=steps,
                local_optimizer_name=local_opt_name,
            )
            optimized = res[0] if isinstance(res, list) and len(res) == 1 else res

            energy = None
            try:
                energy = float(optimized.get_potential_energy())
            except Exception:
                energy = None
            try:
                forces = optimized.get_forces()
                max_force = float(abs(forces).max())
            except Exception:
                max_force = None

            return {
                "converged": True,
                "final_energy": energy,
                "steps_taken": None,
                "optimized_atoms": optimized,
                "ts_atoms": optimized,  # Alias for backward compatibility
                "max_force": max_force,
            }
        finally:
            # Restore original settings
            if old_ts_method is not None:
                self.ts_method = old_ts_method
            self.ts_kwargs = old_ts_kwargs

    def ts_opt(self, atoms=None, fmax=0.05, steps=1000, **kwargs) -> Dict[str, Any]:
        """Alias for find_transition_state for backward compatibility."""
        if atoms is not None:
            self.load_structure(atoms)
        return self.find_transition_state(fmax=fmax, steps=steps, **kwargs)


__all__ = ["Explorer"]
