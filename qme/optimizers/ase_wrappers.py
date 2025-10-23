"""ASE optimizer wrappers with verbosity control for QME.

This module provides wrapper classes for ASE optimizers (LBFGS, BFGS, FIRE)
and Sella to add consistent verbosity control using QME's logging system.
"""

from typing import IO, Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.optimize.optimize import Optimizer

from qme.logging_utils import get_qme_logger, setup_qme_logging

logger = get_qme_logger(__name__)


class ProfilerCalculatorWrapper(Calculator):
    """Wrapper for ASE calculators that tracks energy and force calls for profiling."""

    def __init__(self, calculator: Calculator, profiler: Any) -> None:
        """Initialize the profiler calculator wrapper.

        Parameters
        ----------
        calculator : Calculator
            The ASE calculator to wrap
        profiler : Any
            Performance profiler instance

        """
        super().__init__()
        self.calculator = calculator
        self.profiler = profiler

        # Copy calculator properties to ensure proper delegation
        self._name = getattr(calculator, "name", "wrapped")
        # Copy implemented_properties to ensure ASE property checks work correctly
        self.implemented_properties = getattr(
            calculator,
            "implemented_properties",
            ["energy", "forces"],
        ).copy()

    def calculate(self, atoms=None, properties=None, system_changes=None):
        """Calculate properties and track calls in profiler."""
        if properties is None:
            properties = ["energy"]
        # Track energy calls
        if "energy" in properties:
            self.profiler.increment_call("energy")

        # Track force calls
        if "forces" in properties:
            self.profiler.increment_call("forces")

        # Track hessian calls
        if "hessian" in properties:
            self.profiler.increment_call("hessian")

        # Delegate to wrapped calculator
        return self.calculator.calculate(atoms, properties, system_changes)

    @property
    def name(self):
        """Get calculator name."""
        return self._name

    def __getattr__(self, name):
        """Delegate attribute access to wrapped calculator."""
        if name in ("calculator", "profiler", "_name"):
            # Prevent recursion when accessing our own attributes
            return object.__getattribute__(self, name)
        return getattr(self.calculator, name)

    def get_property(self, name, atoms, allow_calculation=True):
        """Get a property from the calculator and track calls in profiler."""
        # Track the call in profiler
        if name == "energy":
            self.profiler.increment_call("energy")
        elif name == "forces":
            self.profiler.increment_call("forces")
        elif name == "hessian":
            self.profiler.increment_call("hessian")

        return self.calculator.get_property(name, atoms, allow_calculation)

    def get_potential_energy(self, atoms, force_consistent=False):
        """Get potential energy and track call in profiler."""
        self.profiler.increment_call("energy")
        return self.calculator.get_potential_energy(atoms, force_consistent)

    def get_forces(self, atoms):
        """Get forces and track call in profiler."""
        self.profiler.increment_call("forces")
        return self.calculator.get_forces(atoms)

    def get_stress(self, atoms):
        """Get stress and track call in profiler."""
        # Note: stress calls are not tracked separately, but could be added if needed
        return self.calculator.get_stress(atoms)

    def get_hessian(self, atoms):
        """Get Hessian and track call in profiler."""
        self.profiler.increment_call("hessian")
        if hasattr(self.calculator, "get_hessian"):
            return self.calculator.get_hessian(atoms)
        msg = f"Calculator {type(self.calculator).__name__} does not support Hessian calculation"
        raise AttributeError(
            msg,
        )

    def check_state(self, atoms, tol=1e-15):
        """Check calculator state (delegate to wrapped calculator)."""
        return self.calculator.check_state(atoms, tol)


class VerboseOptimizerWrapper(Optimizer):
    """Base wrapper class for ASE optimizers with verbosity control.

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
        profiler: Any = None,
        **kwargs: Any,
    ) -> None:
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
        # Store verbosity level and profiler
        self.verbose = verbose
        self.profiler = profiler

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
            atoms,
            restart=None,
            logfile=logfile,
            trajectory=trajectory,
            **kwargs,
        )

        # Copy important attributes from wrapped optimizer
        self.atoms = self.wrapped_optimizer.atoms
        self.fmax = self.wrapped_optimizer.fmax
        self.nsteps = self.wrapped_optimizer.nsteps
        self.max_steps = self.wrapped_optimizer.max_steps

        # Ensure calculator is properly attached to both atoms objects
        if hasattr(atoms, "calc") and atoms.calc is not None:
            # Wrap calculator with profiler if available
            if self.profiler is not None:
                self.atoms.calc = ProfilerCalculatorWrapper(atoms.calc, self.profiler)
                self.wrapped_optimizer.atoms.calc = self.atoms.calc
            else:
                self.atoms.calc = atoms.calc
                self.wrapped_optimizer.atoms.calc = atoms.calc
        # If no calculator, try to get it from the original atoms
        elif hasattr(atoms, "calc"):
            if self.profiler is not None:
                self.atoms.calc = ProfilerCalculatorWrapper(atoms.calc, self.profiler)
                self.wrapped_optimizer.atoms.calc = self.atoms.calc
            else:
                self.atoms.calc = atoms.calc
                self.wrapped_optimizer.atoms.calc = atoms.calc

        if self.verbose >= 2:
            optimizer_name = wrapped_optimizer_class.__name__
            logger.info(f"Initialized {optimizer_name} optimizer with verbosity control")

    def run(self, fmax: float = 0.05, steps: int = 1000) -> bool:
        """Run the optimization with verbosity control."""
        if self.verbose >= 2:
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
    ) -> None:
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
    ) -> None:
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
    ) -> None:
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
    ) -> None:
        from sella import Sella

        super().__init__(
            atoms=atoms,
            wrapped_optimizer_class=Sella,
            logfile=logfile,
            trajectory=trajectory,
            verbose=verbose,
            **kwargs,
        )
