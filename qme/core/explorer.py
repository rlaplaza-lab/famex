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

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import write

from qme.core.calculator_setup import create_calculator
from qme.core.constraint_parser import parse_constraints
from qme.core.geometry import read_geometry
from qme.core.strategy import REGISTRY
from qme.core.validation import (
    STRATEGY_CINEB,
    STRATEGY_INTERPOLATE,
    STRATEGY_IRC,
    STRATEGY_LOCAL,
    STRATEGY_NEB,
    TARGET_MINIMA,
    TARGET_PATH,
    TARGET_TS,
    normalize_strategy,
    normalize_target,
)


def _extract_charge_spin(
    atoms: Atoms, default_charge: int = 0, default_spin: int = 1
) -> tuple[int, int]:
    """Extract charge and spin from atoms object.

    Precedence: atoms.charge/atoms.mult > atoms.info['charge']/atoms.info['spin'] > defaults

    Parameters
    ----------
    atoms : Atoms
        The atoms object to extract charge/spin from
    default_charge : int
        Default charge if not found
    default_spin : int
        Default spin if not found

    Returns
    -------
    tuple[int, int]
        (charge, spin) tuple
    """
    charge = None
    spin = None

    # Check for attributes first (highest priority)
    if hasattr(atoms, "charge"):
        try:
            charge = int(atoms.charge)
        except (ValueError, TypeError):
            charge = None

    if hasattr(atoms, "mult"):
        try:
            spin = int(atoms.mult)
        except (ValueError, TypeError):
            spin = None

    # Check atoms.info (medium priority)
    if hasattr(atoms, "info") and atoms.info is not None:
        if charge is None and "charge" in atoms.info:
            try:
                charge = int(atoms.info.get("charge"))
            except (ValueError, TypeError):
                pass

        if spin is None and "spin" in atoms.info:
            try:
                spin = int(atoms.info.get("spin"))
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

    The Explorer provides a clear semantic interface for quantum chemistry calculations:

    - **target**: What you want to obtain (minima, ts, path)
    - **strategy**: How to get there (local, neb, cineb, interpolate)

    Parameters
    ----------
    atoms
        Single :class:`ase.Atoms` or a sequence of Atoms to operate on.
    backend
        Calculator backend key (e.g., ``uma``, ``aimnet2``, ``mace``, ``so3lr``,
        ``mock``).
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
        High-level run selection. ``target`` specifies what you want (minima, ts, path).
        ``strategy`` specifies how to get there (local, neb, cineb, interpolate).
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

    Migration Guide
    ---------------
    The Explorer API has been updated with clearer target/strategy semantics:

    **Old API (still supported via aliases):**
    ```python
    # Old two-ended approach
    explorer = Explorer(atoms, strategy="two-ended", target="path")
    result = explorer.run(mode="neb")

    # Old local approach
    explorer = Explorer(atoms, strategy="local", target="minima")
    result = explorer.run(mode="minima")
    ```

    **New API (recommended):**
    ```python
    # Clear target/strategy semantics
    explorer = Explorer(atoms, target="path", strategy="neb")
    result = explorer.run(mode="neb")

    explorer = Explorer(atoms, target="minima", strategy="local")
    result = explorer.run(mode="minima")
    ```

    **Key Changes:**
    - ``target``: What you want (minima, ts, path)
    - ``strategy``: How to get there (local, neb, cineb, interpolate)
    - Old ``strategy="two-ended"`` → New ``strategy="interpolate"`` or ``"neb"`` or ``"cineb"``
    - Backward compatibility maintained through aliases

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
        local_optimizer: str = "sella",
        optimizer_kwargs: dict[str, Any] | None = None,
        strategy: str | None = "local",
        target: str | None = "minima",
        mode: str | None = None,
        ts_method: str | None = None,
        ts_kwargs: dict[str, Any] | None = None,
        constraints: str | list | dict | None = None,
        initial_hessian: np.ndarray | None = None,
        auto_register: bool = True,
        verbose: int = 1,
    ) -> None:
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

        # Setup QME logging with specified verbosity
        from qme.logging_utils import setup_qme_logging

        setup_qme_logging(verbosity=verbose)

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
        self.initial_hessian = initial_hessian

        # Auto-register default strategies unless caller opts out
        if auto_register:
            self._register_default_strategies()

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
        elif backend_lower == "aimnet2":
            return "aimnet2"
        elif backend_lower in ("mace", "torchsim_mace"):
            return "mace-omol-0"
        elif backend_lower == "torchsim_uma":
            return "uma-s-1p1"
        elif backend_lower == "so3lr":
            # SO3LR requires a model_path, not model_name
            return self.model_path or "so3lr-model"
        elif backend_lower == "mock":
            return "mock-model"
        else:
            return "default-model"

    def _create_and_attach_calculator(self, atoms: Atoms):
        """Create and attach an ASE calculator to ``atoms``.

        Prefers explicit ``charge``/``mult`` found on Geometry-like objects or
        in ``atoms.info``. Falls back to the Explorer defaults otherwise.
        """
        # Extract charge and spin using helper function
        charge, spin = _extract_charge_spin(atoms, self.default_charge, self.default_spin)

        # Ensure atoms.info contains values so calculators that read
        # atoms.info (UMA, SO3LR, etc.) see the intended settings.
        if getattr(atoms, "info", None) is not None:
            # Coerce to built-in int types to satisfy backends that enforce strict typing
            try:
                atoms.info["charge"] = int(atoms.info.get("charge", charge))
            except Exception:
                atoms.info["charge"] = int(charge)
            try:
                atoms.info["spin"] = int(atoms.info.get("spin", spin))
            except Exception:
                atoms.info["spin"] = int(spin)

        # Show model initialization info when creating the first calculator
        if not hasattr(self, "_calculator_created"):
            from qme.logging_utils import print_model_info

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

    def _apply_constraints(self, atoms: Atoms):
        """Parse and apply constraints to ``atoms`` if specified.

        Returns the ASE constraints list after application.
        """
        if self.constraints_spec is None:
            return []
        return parse_constraints(self.constraints_spec, atoms, verbose=False)

    # =============================================================================
    # Strategy Management
    # =============================================================================

    def _register_default_strategies(self) -> None:
        """Register default strategies using the new class-based registry.

        This method imports the strategy modules, which automatically register
        their strategy classes with the global REGISTRY.
        """
        # Import strategy modules to trigger registration

    def register_strategy(
        self,
        name: str,
        func: Callable[..., Any],
        strategy_type: str = "global",
        description: str | None = None,
        aliases: list[str] | None = None,
    ) -> None:
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
            self._strategies: dict[str, Any] = {}
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

    def list_strategies(self, kind: str | None = None) -> dict[str, dict[str, str]]:
        """List available strategies; optionally filter by kind."""
        out: dict[str, dict[str, str]] = {}
        for name, metadata in REGISTRY.list_strategies().items():
            if kind is None or metadata.target == kind:
                # Add main strategy name
                out[name] = {
                    "type": "two-ended" if metadata.requires_multiple_structures else "local",
                    "description": metadata.description,
                }
                # Add aliases
                for alias in metadata.aliases:
                    out[alias] = {
                        "type": "two-ended" if metadata.requires_multiple_structures else "local",
                        "description": metadata.description,
                    }
        return out

    # =============================================================================
    # Target/Strategy Resolution and Selection
    # =============================================================================

    def _resolve_target_and_strategy(self, mode: str | None = None) -> tuple[str, str]:
        """Resolve target and strategy with clear precedence rules.

        This method determines the final target and strategy to use based on:
        1. Runtime mode parameter (highest priority)
        2. Instance target and strategy parameters
        3. Smart defaults based on context

        Parameters
        ----------
        mode : str, optional
            Runtime mode parameter that can override instance settings.
            Can be a target (minima, ts, path) or strategy (local, neb, cineb, interpolate).

        Returns
        -------
        tuple[str, str]
            (target, strategy) tuple with normalized values

        Examples
        --------
        >>> explorer = Explorer(atoms, target="path", strategy="neb")
        >>> target, strategy = explorer._resolve_target_and_strategy("cineb")
        >>> # Returns: ("path", "cineb") - strategy from mode, target from instance

        >>> explorer = Explorer(atoms, target="minima")
        >>> target, strategy = explorer._resolve_target_and_strategy("ts")
        >>> # Returns: ("ts", "local") - target from mode, strategy defaults to local
        """
        if mode:
            # Runtime mode overrides everything
            # Check if mode is a strategy first (prioritize strategy over target)
            normalized_strategy = normalize_strategy(mode)

            # If mode is a known strategy, treat it as a strategy
            if normalized_strategy in [
                STRATEGY_NEB,
                STRATEGY_CINEB,
                STRATEGY_INTERPOLATE,
                STRATEGY_LOCAL,
                STRATEGY_IRC,
            ]:
                # Strategy specified as mode - infer target from instance or default to path
                target = normalize_target(self.target or TARGET_PATH)
                strategy = normalized_strategy
            else:
                # Mode is a target
                target = normalize_target(mode)
                # Use instance strategy or smart default
                if self.strategy:
                    strategy = normalize_strategy(self.strategy)
                else:
                    # Smart defaults based on target
                    if target == TARGET_PATH:
                        strategy = STRATEGY_NEB  # Default to NEB for path
                    elif target == TARGET_TS:
                        strategy = STRATEGY_LOCAL  # Default to local for TS
                    else:
                        strategy = STRATEGY_LOCAL  # Default to local for minima
        else:
            # Use instance parameters
            target = normalize_target(self.target or TARGET_MINIMA)

            if self.strategy:
                strategy = normalize_strategy(self.strategy)
            else:
                # Smart defaults based on target
                if target == TARGET_PATH:
                    strategy = STRATEGY_NEB  # Default to NEB for path
                elif target == TARGET_TS:
                    strategy = STRATEGY_LOCAL  # Default to local for TS
                else:
                    strategy = STRATEGY_LOCAL  # Default to local for minima

        # Auto-infer strategy from target if ambiguous
        # Only override strategy if it wasn't explicitly set via mode parameter
        if mode and normalize_strategy(mode) in [
            STRATEGY_NEB,
            STRATEGY_CINEB,
            STRATEGY_INTERPOLATE,
            STRATEGY_LOCAL,
            STRATEGY_IRC,
        ]:
            # Strategy was explicitly set via mode, don't override it
            pass
        elif target == TARGET_PATH and strategy == STRATEGY_LOCAL:
            # Path target with local strategy doesn't make sense, default to NEB
            # UNLESS it's IRC which is a local strategy for path
            strategy = STRATEGY_NEB
        return target, strategy

    def _select_strategy_runner(self, target: str, strategy: str) -> tuple[str, str]:
        """Select the appropriate strategy runner based on target and strategy.

        This method maps the resolved target/strategy combination to the actual
        registered strategy runner and its type.

        Parameters
        ----------
        target : str
            Normalized target (minima, ts, path)
        strategy : str
            Normalized strategy (local, neb, cineb, interpolate)

        Returns
        -------
        tuple[str, str]
            (strategy_key, strategy_type) where:
            - strategy_key: The registered strategy name (e.g., "minima:local", "path:neb")
            - strategy_type: The strategy type ("local", "two-ended", "global")

        Raises
        ------
        NotImplementedError
            If no matching strategy is found for the target/strategy combination

        Examples
        --------
        >>> explorer = Explorer(atoms)
        >>> key, type_ = explorer._select_strategy_runner("minima", "local")
        >>> # Returns: ("minima:local", "local")

        >>> key, type_ = explorer._select_strategy_runner("path", "neb")
        >>> # Returns: ("path:neb", "two-ended")
        """
        strategies = getattr(self, "_strategies", {})

        # Build the primary strategy key
        primary_key = f"{target}:{strategy}"
        # Check if primary key exists
        if primary_key in strategies:
            entry = strategies[primary_key]
            strategy_type = entry.get("type", "global")
            return primary_key, strategy_type

        # Fallback to aliases
        for key, entry in strategies.items():
            if isinstance(entry, dict) and "aliases" in entry:
                aliases = entry.get("aliases", [])
                if strategy in aliases or target in aliases:
                    strategy_type = entry.get("type", "global")
                    return key, strategy_type

        # Final fallback - try to find any matching strategy
        if target == TARGET_PATH and strategy in [STRATEGY_NEB, STRATEGY_CINEB]:
            # Look for path strategies
            for key in ["path:neb", "path:cineb", "twoended:neb", "twoended:cineb"]:
                if key in strategies:
                    entry = strategies[key]
                    strategy_type = entry.get("type", "global")
                    return key, strategy_type
        elif target == TARGET_TS and strategy == STRATEGY_LOCAL:
            # Look for local TS strategies
            for key in ["ts:local", "local:ts", "ts"]:
                if key in strategies:
                    entry = strategies[key]
                    strategy_type = entry.get("type", "global")
                    return key, strategy_type
        elif target == TARGET_MINIMA and strategy == STRATEGY_LOCAL:
            # Look for local minima strategies
            for key in ["minima:local", "local:minima", "minima"]:
                if key in strategies:
                    entry = strategies[key]
                    strategy_type = entry.get("type", "global")
                    return key, strategy_type

        # If nothing found, raise error
        raise NotImplementedError(
            f"No strategy found for target='{target}', strategy='{strategy}'. "
            f"Available strategies: {list(strategies.keys())}"
        )

    # =============================================================================
    # Execution
    # =============================================================================

    def explain_run(self, mode: str | None = None) -> dict[str, Any]:
        """Explain what strategy would be selected without running.

        Parameters
        ----------
        mode : str, optional
            Mode to explain (uses instance target if None)

        Returns
        -------
        dict
            Dictionary with strategy selection details
        """
        try:
            # Determine strategy name
            if mode:
                strategy_name = mode
            elif self.target and self.strategy:
                strategy_name = f"{self.target}:{self.strategy}"
            else:
                strategy_name = "minima:local"

            # Get strategy class from registry
            strategy_class = REGISTRY.get(strategy_name)
            metadata = strategy_class.metadata

            return {
                "target": metadata.target,
                "strategy": metadata.strategy,
                "strategy_key": strategy_name,
                "runner": strategy_class.__name__,
                "valid": True,
                "strategy_type": "two-ended" if metadata.requires_multiple_structures else "local",
                "description": metadata.description,
                "notes": f"Will use {self.local_optimizer_name} optimizer",
            }
        except KeyError as e:
            available = sorted(REGISTRY.list_strategies().keys())
            return {
                "target": "unknown",
                "strategy": "unknown",
                "strategy_key": mode or "unknown",
                "runner": None,
                "valid": False,
                "error": f"No strategy found for '{mode or 'default'}'. Available: {available}",
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
        self, mode: str | None = None, runner: Callable[..., Any] | None = None, **kwargs: Any
    ) -> Any:
        """Execute a registered or user-supplied runner.

        This method is the main entry point for running optimization tasks. It uses
        the new strategy registry to find and execute the appropriate strategy.

        Parameters
        ----------
        mode : str, optional
            Strategy name in format "target:strategy" (e.g., "minima:local", "path:neb")
            or short alias (e.g., "neb", "cineb"). If not provided, uses instance
            target and strategy parameters.
        runner : callable, optional
            Optional callable to execute directly, bypassing strategy selection.
        **kwargs
            Additional keyword arguments forwarded to the strategy runner.

        Returns
        -------
        Any
            Whatever the strategy returns. Typically a dictionary with optimization results.

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

        >>> # Using explicit mode
        >>> explorer = Explorer(atoms)
        >>> result = explorer.run(mode="minima:local")
        >>> result = explorer.run(mode="path:neb", npoints=7)
        """
        # If the caller passed an explicit runner, use it directly
        if runner is not None:
            call_kwargs = dict(kwargs)
            call_kwargs.setdefault("explorer", self)
            call_kwargs.setdefault("local_optimizer_name", self.local_optimizer_name)
            call_kwargs.setdefault("verbose", self.verbose)
            return runner(self.atoms_list, **call_kwargs)

        # Determine strategy name
        if mode:
            strategy_name = mode  # e.g., "minima:local", "path:neb", "neb"
        elif self.target and self.strategy:
            strategy_name = f"{self.target}:{self.strategy}"
        else:
            # Only minimal fallback - default to minima:local
            strategy_name = "minima:local"

        # Get strategy class from registry
        try:
            strategy_class = REGISTRY.get(strategy_name)
        except KeyError as e:
            available = sorted(REGISTRY.list_strategies().keys())
            raise NotImplementedError(
                f"No strategy found for '{strategy_name}'. "
                f"Available strategies: {available}"
            ) from e

        # Instantiate and run strategy
        strategy_instance = strategy_class(explorer=self)
        return strategy_instance.run(self.atoms_list, **kwargs)

    # =============================================================================
    # Convenience Methods (ASE-like interface)
    # =============================================================================

    def optimize_minima(self, fmax: float = 0.05, steps: int = 1000, **kwargs: Any) -> Any:
        """Run a local minima optimization (ASE-like convenience method)."""
        return self.run(mode="minima", fmax=fmax, steps=steps, **kwargs)

    def optimize_ts(self, fmax: float = 0.05, steps: int = 1000, **kwargs: Any) -> Any:
        """Run a local transition-state optimization (ASE-like convenience method)."""
        return self.run(mode="ts", fmax=fmax, steps=steps, **kwargs)

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
            verbose=verbose,
            **kwargs,
        )

    def load_structure(self, filename_or_geom: str | Path | Atoms) -> Atoms:
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
        self, atoms: Atoms, output_file: str | Path, format: str | None = None
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
                raise RuntimeError(
                    f"Failed to save structure to {output_file}: {e}. "
                    f"Clean attempt also failed: {e2}"
                )

    def save_trajectory(
        self, atoms_list: list[Atoms], output_file: str | Path, format: str | None = None
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
        >>> result = explorer.run(mode="neb", npoints=7)
        >>> explorer.save_trajectory(result, "reaction_path.xyz")
        """
        output_file = Path(output_file)

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
                raise RuntimeError(
                    f"Failed to save trajectory to {output_file}: {e}. "
                    f"Clean attempt also failed: {e2}"
                )

    def _twoended_path_runner(self, atoms_list: list[Atoms], **kwargs: Any) -> Any:
        """General path runner that dispatches based on mode."""
        mode = kwargs.get("mode", "neb").lower()
        if mode == "interpolate":
            # Raw interpolation - no optimization
            from qme.core.twoended_strategies import path_generator

            # Remove dry_run from kwargs to prevent it from being passed to path_generator
            clean_kwargs = {k: v for k, v in kwargs.items() if k != "dry_run"}

            return path_generator(
                atoms_list,
                npoints=clean_kwargs.get("npoints", 11),
                method=clean_kwargs.get("method", "geodesic"),
                optimize_path=False,  # Raw interpolation only
                explorer=None,  # No explorer to prevent any optimization
            )
        elif mode == "cineb":
            from qme.core.twoended_strategies import twoended_cineb_runner

            return twoended_cineb_runner(atoms_list, **kwargs)
        else:
            # Default to NEB for any other mode
            from qme.core.twoended_strategies import twoended_neb_runner

            return twoended_neb_runner(atoms_list, **kwargs)

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
    ) -> dict[str, Any]:
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
            Dictionary containing frequencies, normal modes, thermodynamic
            properties, etc.
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
