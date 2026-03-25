"""ASE optimizer wrappers with verbosity control for QME."""

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
    def __init__(self) -> None:
        import logging
        import sys

        self.log_level = logging.INFO
        self.buffer = ""
        self.should_output = logger.getEffectiveLevel() <= logging.INFO
        self.stdout: TextIO = sys.stdout

    def write(self, text: str) -> int:
        if not text:
            return 0

        if not self.should_output:
            return len(text)

        self.buffer += text

        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            if line.strip():
                self.stdout.write(line.strip() + "\n")

        return len(text)

    def flush(self) -> None:
        if self.should_output and self.buffer.strip():
            self.stdout.write(self.buffer.strip() + "\n")
            self.buffer = ""
            self.stdout.flush()

    def close(self) -> None:
        self.flush()

    def __enter__(self) -> LoggingFile:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class ProfilerCalculatorWrapper(Calculator):
    def __init__(self, calculator: Calculator, profiler: Any) -> None:
        super().__init__()
        self.calculator = calculator
        self.profiler = profiler
        self._name = getattr(calculator, "name", "wrapped")
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
        if properties is None:
            properties = ["energy"]
        if "energy" in properties:
            self.profiler.increment_call("energy")
        if "forces" in properties:
            self.profiler.increment_call("forces")
        if "hessian" in properties:
            self.profiler.increment_call("hessian")

        return self.calculator.calculate(atoms, properties, system_changes)  # type: ignore[no-any-return]

    @property
    def name(self) -> str:
        return self._name

    def __getattr__(self, name: str) -> Any:
        if name in ("calculator", "profiler", "_name"):
            return object.__getattribute__(self, name)
        return getattr(self.calculator, name)

    def get_property(
        self, name: str, atoms: Atoms | None = None, allow_calculation: bool = True
    ) -> Any:
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
        self.profiler.increment_call("energy")
        return self.calculator.get_potential_energy(atoms, force_consistent)  # type: ignore[no-any-return]

    def get_forces(self, atoms: Atoms | None = None) -> np.ndarray:
        self.profiler.increment_call("forces")
        return self.calculator.get_forces(atoms)  # type: ignore[no-any-return]

    def get_stress(self, atoms: Atoms | None = None) -> np.ndarray:
        return self.calculator.get_stress(atoms)  # type: ignore[no-any-return]

    def get_hessian(self, atoms: Atoms | None = None) -> np.ndarray:
        self.profiler.increment_call("hessian")
        if hasattr(self.calculator, "get_hessian"):
            return self.calculator.get_hessian(atoms)  # type: ignore[no-any-return]
        msg = f"Calculator {type(self.calculator).__name__} does not support Hessian calculation"
        raise AttributeError(
            msg,
        )

    def check_state(self, atoms: Atoms, tol: float = 1e-15) -> bool:
        return self.calculator.check_state(atoms, tol)  # type: ignore[no-any-return]


class VerboseOptimizerWrapper(Optimizer):
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
        self.verbose = verbose
        self.profiler = profiler
        self._logging_file = None

        if verbose == 0:
            logfile = None
        elif verbose >= 1:
            logging_file = LoggingFile()
            self._logging_file = logging_file
            if logfile is None or logfile == "-":
                logfile = logging_file  # type: ignore[assignment]
        self.wrapped_optimizer = wrapped_optimizer_class(
            atoms,
            restart=None,
            logfile=logfile,
            trajectory=trajectory,
            **kwargs,
        )

        self.atoms = self.wrapped_optimizer.atoms
        self.fmax = self.wrapped_optimizer.fmax
        self.nsteps = self.wrapped_optimizer.nsteps
        self.max_steps = self.wrapped_optimizer.max_steps

        if hasattr(atoms, "calc") and atoms.calc is not None or hasattr(atoms, "calc"):
            if self.profiler is not None:
                self.atoms.calc = ProfilerCalculatorWrapper(atoms.calc, self.profiler)
                self.wrapped_optimizer.atoms.calc = self.atoms.calc
            else:
                self.atoms.calc = atoms.calc
                self.wrapped_optimizer.atoms.calc = atoms.calc

        if self.verbose >= 2:
            optimizer_name = wrapped_optimizer_class.__name__
            logger.info(f"Initialized {optimizer_name} optimizer with verbosity control")

    def run(self, fmax: float = 0.05, steps: int = 1000) -> bool:  # type: ignore[override]
        if self.verbose >= 2:
            optimizer_name = self.wrapped_optimizer.__class__.__name__
            logger.info(f"Starting {optimizer_name} optimization")
            logger.info(f"Convergence criterion: fmax = {fmax} eV/Å")
            logger.info(f"Maximum steps: {steps}")
        if self._logging_file is not None:
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
        return self.wrapped_optimizer.get_number_of_steps()  # type: ignore[no-any-return]

    def converged(self, forces: np.ndarray) -> bool:
        return self.wrapped_optimizer.converged(forces)  # type: ignore[no-any-return]

    def log(self, forces: np.ndarray) -> None:
        return self.wrapped_optimizer.log(forces)  # type: ignore[no-any-return]

    def call_observers(self) -> None:
        return self.wrapped_optimizer.call_observers()  # type: ignore[no-any-return]

    def dump(self, data: Any) -> None:
        return self.wrapped_optimizer.dump(data)  # type: ignore[no-any-return]

    def load(self) -> None:
        return self.wrapped_optimizer.load()  # type: ignore[no-any-return]


class VerboseLBFGS(VerboseOptimizerWrapper):
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

        kwargs.setdefault("eta", 1e-4)

        super().__init__(
            atoms=atoms,
            wrapped_optimizer_class=Sella,
            logfile=logfile,
            trajectory=trajectory,
            verbose=verbose,
            profiler=profiler,
            **kwargs,
        )
