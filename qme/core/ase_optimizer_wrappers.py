"""
ASE optimizer wrappers with verbosity control for QME.

This module provides wrapper classes for ASE optimizers (LBFGS, BFGS, FIRE)
and Sella to add consistent verbosity control using QME's logging system.
"""

from typing import IO, Any

import numpy as np
from ase import Atoms
from ase.optimize.optimize import Optimizer

from qme.logging_utils import get_qme_logger, setup_qme_logging

logger = get_qme_logger(__name__)


class VerboseOptimizerWrapper(Optimizer):
    """
    Base wrapper class for ASE optimizers with verbosity control.

    This wrapper adds QME-style verbosity control to any ASE optimizer
    by intercepting the logfile parameter and managing output based on
    the verbose level.
    """

    def __init__(
        self,
        atoms: Atoms,
        wrapped_optimizer_class: type[Optimizer],
        logfile: IO | str = "-",
        trajectory: str | None = None,
        verbose: int = 1,
        **kwargs: Any,
    ):
        """Initialize the verbose optimizer wrapper.

        Parameters
        ----------
        atoms : Atoms
            The Atoms object to optimize.
        wrapped_optimizer_class : type[Optimizer]
            The ASE optimizer class to wrap.
        logfile : Union[IO, str]
            File object or filename for logging. Use '-' for stdout.
        trajectory : Optional[str]
            Trajectory file to store optimization path.
        verbose : int
            Verbosity level for optimization output:
            - 0: Quiet (minimal output)
            - 1: Normal (default, shows progress)
            - 2: Verbose (detailed information)
        **kwargs
            Additional arguments passed to the wrapped optimizer.
        """
        # Store verbosity level
        self.verbose = verbose

        # Set up QME logging if not already configured
        setup_qme_logging(verbosity=verbose, force=True)

        # Set up logging based on verbosity
        if verbose == 0:
            # Quiet mode: suppress ASE logging by using None logfile
            logfile = None
        elif verbose == 1:
            # Normal mode: use provided logfile or default
            pass
        else:
            # Verbose mode: use provided logfile or default
            pass

        # Initialize the wrapped optimizer
        self.wrapped_optimizer = wrapped_optimizer_class(
            atoms, restart=None, logfile=logfile, trajectory=trajectory, **kwargs
        )

        # Copy important attributes from wrapped optimizer
        self.atoms = self.wrapped_optimizer.atoms
        self.fmax = self.wrapped_optimizer.fmax
        self.nsteps = self.wrapped_optimizer.nsteps
        self.max_steps = self.wrapped_optimizer.max_steps

        # Ensure calculator is properly attached to both atoms objects
        if hasattr(atoms, "calc") and atoms.calc is not None:
            self.atoms.calc = atoms.calc
            self.wrapped_optimizer.atoms.calc = atoms.calc
        else:
            # If no calculator, try to get it from the original atoms
            if hasattr(atoms, "calc"):
                self.atoms.calc = atoms.calc
                self.wrapped_optimizer.atoms.calc = atoms.calc

        if self.verbose >= 1:
            optimizer_name = wrapped_optimizer_class.__name__
            logger.info(f"Initialized {optimizer_name} optimizer with verbosity control")

    def run(self, fmax: float = 0.05, steps: int = 1000) -> bool:
        """Run the optimization with verbosity control."""
        if self.verbose >= 1:
            optimizer_name = self.wrapped_optimizer.__class__.__name__
            logger.info(f"Starting {optimizer_name} optimization")
            logger.info(f"Convergence criterion: fmax = {fmax} eV/Å")
            logger.info(f"Maximum steps: {steps}")

        # Run the wrapped optimizer
        result = self.wrapped_optimizer.run(fmax=fmax, steps=steps)

        if self.verbose >= 1:
            if result:
                logger.info("Optimization converged!")
            else:
                logger.warning(f"Optimization stopped after {steps} steps without converging")
                forces = self.atoms.get_forces()
                logger.warning(f"Final max force: {np.max(np.abs(forces)):.6f} eV/Å")

        return result

    def get_number_of_steps(self) -> int:
        """Get the number of optimization steps taken."""
        return self.wrapped_optimizer.get_number_of_steps()

    def converged(self, forces: np.ndarray) -> bool:
        """Check if optimization has converged."""
        return self.wrapped_optimizer.converged(forces)

    def log(self, forces: np.ndarray) -> None:
        """Log optimization step."""
        return self.wrapped_optimizer.log(forces)

    def call_observers(self) -> None:
        """Call observers."""
        return self.wrapped_optimizer.call_observers()

    def dump(self, data: Any) -> None:
        """Dump optimizer state."""
        return self.wrapped_optimizer.dump(data)

    def load(self) -> None:
        """Load optimizer state."""
        return self.wrapped_optimizer.load()


class VerboseLBFGS(VerboseOptimizerWrapper):
    """LBFGS optimizer with verbosity control."""

    def __init__(
        self,
        atoms: Atoms,
        logfile: IO | str = "-",
        trajectory: str | None = None,
        verbose: int = 1,
        **kwargs: Any,
    ):
        from ase.optimize.lbfgs import LBFGS

        super().__init__(
            atoms=atoms,
            wrapped_optimizer_class=LBFGS,
            logfile=logfile,
            trajectory=trajectory,
            verbose=verbose,
            **kwargs,
        )


class VerboseBFGS(VerboseOptimizerWrapper):
    """BFGS optimizer with verbosity control."""

    def __init__(
        self,
        atoms: Atoms,
        logfile: IO | str = "-",
        trajectory: str | None = None,
        verbose: int = 1,
        **kwargs: Any,
    ):
        from ase.optimize import BFGS

        super().__init__(
            atoms=atoms,
            wrapped_optimizer_class=BFGS,
            logfile=logfile,
            trajectory=trajectory,
            verbose=verbose,
            **kwargs,
        )


class VerboseFIRE(VerboseOptimizerWrapper):
    """FIRE optimizer with verbosity control."""

    def __init__(
        self,
        atoms: Atoms,
        logfile: IO | str = "-",
        trajectory: str | None = None,
        verbose: int = 1,
        **kwargs: Any,
    ):
        from ase.optimize import FIRE

        super().__init__(
            atoms=atoms,
            wrapped_optimizer_class=FIRE,
            logfile=logfile,
            trajectory=trajectory,
            verbose=verbose,
            **kwargs,
        )


class VerboseSella(VerboseOptimizerWrapper):
    """Sella optimizer with verbosity control."""

    def __init__(
        self,
        atoms: Atoms,
        logfile: IO | str = "-",
        trajectory: str | None = None,
        verbose: int = 1,
        **kwargs: Any,
    ):
        from sella import Sella

        super().__init__(
            atoms=atoms,
            wrapped_optimizer_class=Sella,
            logfile=logfile,
            trajectory=trajectory,
            verbose=verbose,
            **kwargs,
        )
