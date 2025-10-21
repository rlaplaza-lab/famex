"""Local minima optimization strategy."""

from typing import Any

from ase import Atoms

from qme.core.strategies.local.helpers import _get_local_optimizer_class
from qme.core.strategy import REGISTRY, BaseStrategy, StrategyMetadata
from qme.core.strategy_utils import StrategyUtils
from qme.logging_utils import get_qme_logger

logger = get_qme_logger(__name__)


class LocalMinimaStrategy(BaseStrategy):
    """Local minima optimization strategy."""

    metadata = StrategyMetadata(
        name="minima:local",
        target="minima",
        strategy="local",
        description="Local minima optimization (ASE/LBFGS or SELLA)",
        aliases=["minima", "local:minima", "local-minima"],
        requires_multiple_structures=False,
    )

    def run(
        self, atoms_list: list[Atoms], fmax: float = 0.05, steps: int = 1000, **kwargs
    ) -> dict[str, Any]:
        """Run local minima optimization.

        Parameters
        ----------
        atoms_list : list[Atoms]
            List of structures to optimize
        fmax : float, default=0.05
            Force convergence threshold
        steps : int, default=1000
            Maximum optimization steps
        **kwargs
            Additional keyword arguments

        Returns
        -------
        dict[str, Any]
            Standardized result dictionary
        """
        self.validate_inputs(atoms_list)

        local_optimizer_name = kwargs.get("local_optimizer_name", "sella")
        verbose = kwargs.get("verbose", 1)

        # Log minima optimization start
        if verbose >= 1:
            logger.info(f"Starting local minima optimization with {local_optimizer_name}")
            logger.info(f"Force threshold: {fmax} eV/Å, Max steps: {steps}")

        opt_class = _get_local_optimizer_class(local_optimizer_name)
        # Accept either a single Atoms instance or a list of them
        single_input = False
        if not isinstance(atoms_list, (list, tuple)):
            single_input = True
            atoms_iter = [atoms_list]
        else:
            atoms_iter = atoms_list
            # If it's a single-element list, treat as single input
            if len(atoms_list) == 1:
                single_input = True

        results = []
        step_counts = []
        converged_flags = []

        for atoms in atoms_iter:
            # CRITICAL FIX: Make a copy of atoms before optimization to prevent in-place modifications
            # that corrupt the coordinate system for subsequent Hessian calculations
            atoms_copy = atoms.copy()

            self.explorer._create_and_attach_calculator(atoms_copy)
            self.explorer._apply_constraints(atoms_copy)
            opt_kwargs = getattr(self.explorer, "optimizer_kwargs", {}) or {}
            # Add verbosity control
            opt_kwargs = dict(opt_kwargs)
            opt_kwargs.setdefault("verbose", getattr(self.explorer, "verbose", 1))

            if local_optimizer_name.lower() == "sella":
                # Sella-specific kwargs for minima search
                opt_kwargs.setdefault("internal", True)
                opt_kwargs.setdefault("order", 0)
                # Check if Hessian is provided in explorer
                if (
                    hasattr(self.explorer, "initial_hessian")
                    and self.explorer.initial_hessian is not None
                ):
                    opt_kwargs["hessian"] = self.explorer.initial_hessian

            opt = opt_class(atoms_copy, **opt_kwargs)
            opt.run(fmax=fmax, steps=steps)

            # Get step count and convergence status using helpers
            steps_taken = StrategyUtils.get_step_count(opt)
            converged = StrategyUtils.get_convergence_status(opt, atoms_copy)

            results.append(atoms_copy)
            step_counts.append(steps_taken)
            converged_flags.append(converged)

        # Log completion
        if verbose >= 1:
            if single_input and results:
                converged = bool(converged_flags[0])
                steps_taken = step_counts[0]
                logger.info(
                    f"Minima optimization completed: converged={converged}, steps={steps_taken}"
                )
            else:
                total_converged = sum(converged_flags)
                total_structures = len(converged_flags)
                logger.info(
                    f"Minima optimization completed: {total_converged}/{total_structures} structures converged"
                )

        if single_input and results:
            return self.prepare_result(
                results[0],
                steps_taken=step_counts[0],
                converged=bool(converged_flags[0]),
            )
        else:
            return self.prepare_result(
                results,
                steps_taken=step_counts,
                converged=[bool(c) for c in converged_flags],
            )


# Register the strategy
REGISTRY.register(LocalMinimaStrategy)
