"""ASE optimizer wrappers with verbosity control for QME.

This module provides wrapper classes for ASE optimizers (LBFGS, BFGS, FIRE)
and Sella to add consistent verbosity control using QME's logging system.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from typing import IO, Any, TextIO, cast

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.optimize.optimize import Optimizer

from qme.utils.logging import get_qme_logger

logger = get_qme_logger(__name__)


class LoggingFile:
    """File-like object that routes output to QME logger.

    This allows ASE optimizer output to be captured and routed through
    the QME logging system, respecting verbosity levels.
    """

    def __init__(self) -> None:
        """Initialize the logging file.

        The logger level is determined by the QME logging system's
        verbosity configuration, so this will automatically respect
        the current verbosity level.
        """
        import logging
        import sys

        self.log_level = logging.INFO  # Use INFO level for optimizer steps
        self.buffer = ""
        # Check current logger level - if it's WARNING or above, suppress output
        self.should_output = logger.getEffectiveLevel() <= logging.INFO
        self.stdout: TextIO = sys.stdout

    def write(self, text: str) -> int:
        """Write text to stdout, but only if verbosity allows it.

        Parameters
        ----------
        text : str
            Text to write

        Returns
        -------
        int
            Number of characters written
        """
        if not text:
            return 0

        # If verbosity is 0 (logger level is WARNING), suppress output
        if not self.should_output:
            return len(text)  # Consume the text but don't output it

        # At verbosity 1+, write directly to stdout (clean output, no logger prefix)
        # Accumulate in buffer until we have a complete line
        self.buffer += text

        # Process complete lines
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            if line.strip():  # Only output non-empty lines
                self.stdout.write(line.strip() + "\n")

        return len(text)

    def flush(self) -> None:
        """Flush any remaining buffer content."""
        if self.should_output and self.buffer.strip():
            self.stdout.write(self.buffer.strip() + "\n")
            self.buffer = ""
            self.stdout.flush()

    def close(self) -> None:
        """Close the file (flush remaining content)."""
        self.flush()

    def __enter__(self) -> LoggingFile:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()


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

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] | None = None,
    ) -> dict[str, Any]:
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
        # ASE calculators return Any (untyped), but we know the return type matches the protocol
        return self.calculator.calculate(atoms, properties, system_changes)  # type: ignore[no-any-return]

    @property
    def name(self) -> str:
        """Get calculator name."""
        return self._name

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to wrapped calculator."""
        if name in ("calculator", "profiler", "_name"):
            # Prevent recursion when accessing our own attributes
            return object.__getattribute__(self, name)
        return getattr(self.calculator, name)

    def get_property(
        self, name: str, atoms: Atoms | None = None, allow_calculation: bool = True
    ) -> Any:
        """Get a property from the calculator and track calls in profiler."""
        # Track the call in profiler
        if name == "energy":
            self.profiler.increment_call("energy")
        elif name == "forces":
            self.profiler.increment_call("forces")
        elif name == "hessian":
            self.profiler.increment_call("hessian")

        return self.calculator.get_property(name, atoms, allow_calculation)

    def get_potential_energy(
        self, atoms: Atoms | None = None, force_consistent: bool = False
    ) -> float:
        """Get potential energy and track call in profiler."""
        self.profiler.increment_call("energy")
        # ASE calculators return Any (untyped), but we know it's float
        return self.calculator.get_potential_energy(atoms, force_consistent)  # type: ignore[no-any-return]

    def get_forces(self, atoms: Atoms | None = None) -> np.ndarray:
        """Get forces and track call in profiler."""
        self.profiler.increment_call("forces")
        # ASE calculators return Any (untyped), but we know it's np.ndarray
        return self.calculator.get_forces(atoms)  # type: ignore[no-any-return]

    def get_stress(self, atoms: Atoms | None = None) -> np.ndarray:
        """Get stress and track call in profiler."""
        # Note: stress calls are not tracked separately, but could be added if needed
        # ASE calculators return Any (untyped), but we know it's np.ndarray
        return self.calculator.get_stress(atoms)  # type: ignore[no-any-return]

    def get_hessian(self, atoms: Atoms | None = None) -> np.ndarray:
        """Get Hessian and track call in profiler."""
        self.profiler.increment_call("hessian")
        if hasattr(self.calculator, "get_hessian"):
            # ASE calculators return Any (untyped), but we know it's np.ndarray
            return self.calculator.get_hessian(atoms)  # type: ignore[no-any-return]
        msg = f"Calculator {type(self.calculator).__name__} does not support Hessian calculation"
        raise AttributeError(
            msg,
        )

    def check_state(self, atoms: Atoms, tol: float = 1e-15) -> bool:
        """Check calculator state (delegate to wrapped calculator)."""
        # ASE calculators return Any (untyped), but we know it's bool
        return self.calculator.check_state(atoms, tol)  # type: ignore[no-any-return]


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
        logfile: IO[Any] | TextIO | str | None = "-",
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
        logfile : IO | str
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
        self._logging_file = None  # Store reference to logging file for cleanup

        # Set up logging based on verbosity
        # Route ASE optimizer output through our logging system
        if verbose == 0:
            # Quiet mode: suppress ASE logging by using None logfile
            logfile = None
        elif verbose >= 1:
            # Normal/verbose mode: route through logging system
            # Create a logging file wrapper - it will use INFO level,
            # which respects the logger's verbosity configuration
            # (At verbosity 0, logger level is WARNING, so INFO won't print)
            logging_file = LoggingFile()
            self._logging_file = logging_file  # Store for cleanup
            # If user provided a specific logfile, use it; otherwise use our logger
            if logfile is None or logfile == "-":
                logfile = logging_file  # type: ignore[assignment]
            # If user provided a file path or file object, keep it as-is
            # (but we could also wrap it to add logging - maybe in future)

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

    # ASE Optimizer.run() signature varies by optimizer; this wrapper matches common signature
    def run(self, fmax: float = 0.05, steps: int = 1000) -> bool:  # type: ignore[override]
        """Run the optimization with verbosity control."""
        if self.verbose >= 2:
            optimizer_name = self.wrapped_optimizer.__class__.__name__
            logger.info(f"Starting {optimizer_name} optimization")
            logger.info(f"Convergence criterion: fmax = {fmax} eV/Å")
            logger.info(f"Maximum steps: {steps}")

        # Run the wrapped optimizer
        # If we're using LoggingFile, redirect stdout to it to capture print() statements
        # This prevents duplicate output from ASE optimizers
        if self._logging_file is not None:
            # LoggingFile implements write() method required by redirect_stdout
            # but doesn't implement full IO protocol, so we cast it
            with redirect_stdout(cast(TextIO, self._logging_file)):
                result = self.wrapped_optimizer.run(fmax=fmax, steps=steps)
            self._logging_file.flush()
        else:
            result = self.wrapped_optimizer.run(fmax=fmax, steps=steps)

        if self.verbose >= 1:
            if result:
                logger.info("Optimization converged!")
            else:
                actual_steps = self.wrapped_optimizer.get_number_of_steps()
                # Get the actual stop reason from wrapped optimizer if available
                scipy_reason = ""
                if (
                    hasattr(self.wrapped_optimizer, "_scipy_result")
                    and self.wrapped_optimizer._scipy_result
                ):
                    scipy_reason = str(getattr(self.wrapped_optimizer._scipy_result, "message", ""))

                logger.warning(
                    f"Optimization stopped after {actual_steps} steps without converging"
                )
                if scipy_reason and "Maximum number of iterations" not in scipy_reason:
                    logger.warning(f"  SciPy reason: {scipy_reason}")
                logger.warning(f"  (Max outer iterations limit: {steps})")
                forces = self.atoms.get_forces()
                logger.warning(f"Final max force: {np.max(np.abs(forces)):.6f} eV/Å")

        return result  # type: ignore[no-any-return]

    def get_number_of_steps(self) -> int:
        """Get the number of optimization steps taken."""
        return self.wrapped_optimizer.get_number_of_steps()  # type: ignore[no-any-return]

    def converged(self, forces: np.ndarray) -> bool:
        """Check if optimization has converged."""
        return self.wrapped_optimizer.converged(forces)  # type: ignore[no-any-return]

    def log(self, forces: np.ndarray) -> None:
        """Log optimization step."""
        return self.wrapped_optimizer.log(forces)  # type: ignore[no-any-return]

    def call_observers(self) -> None:
        """Call observers."""
        return self.wrapped_optimizer.call_observers()  # type: ignore[no-any-return]

    def dump(self, data: Any) -> None:
        """Dump optimizer state."""
        return self.wrapped_optimizer.dump(data)  # type: ignore[no-any-return]

    def load(self) -> None:
        """Load optimizer state."""
        return self.wrapped_optimizer.load()  # type: ignore[no-any-return]


class VerboseLBFGS(VerboseOptimizerWrapper):
    """LBFGS optimizer with verbosity control."""

    def __init__(
        self,
        atoms: Atoms,
        logfile: IO[Any] | TextIO | str | None = "-",
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
        logfile: IO[Any] | TextIO | str | None = "-",
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
        logfile: IO[Any] | TextIO | str | None = "-",
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
        logfile: IO[Any] | TextIO | str | None = "-",
        trajectory: str | None = None,
        verbose: int = 1,
        profiler: Any = None,
        **kwargs: Any,
    ) -> None:
        from sella import Sella

        # Add numerical stability parameters for Sella to prevent SVD failures
        # Only set if not already provided by user
        kwargs.setdefault("eta", 1e-4)  # Hessian regularization (prevents singular matrices)

        super().__init__(
            atoms=atoms,
            wrapped_optimizer_class=Sella,
            logfile=logfile,
            trajectory=trajectory,
            verbose=verbose,
            profiler=profiler,
            **kwargs,
        )
