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
from typing import Any, Dict, List, Optional, Sequence, Union

from ase import Atoms

from qme.core.calculator_setup import create_calculator
from qme.core.constraint_parser import parse_constraints
from qme.core.local_strategies import (
    local_minima_runner,
    local_ts_runner,
)
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


__all__ = ["Explorer"]
