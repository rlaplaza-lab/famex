"""ASE optimizer wrappers with verbosity control for FAMEX."""

from __future__ import annotations

import logging
from contextlib import redirect_stdout
from typing import IO, Any, TextIO, cast

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.optimize.optimize import Optimizer

from famex.optimizers.messages import (
    MSG_CONVERGENCE_CRITERION_FMAX,
    MSG_INITIALIZED_OPTIMIZER_VERBOSITY,
    MSG_MAXIMUM_STEPS,
    MSG_OPTIMIZATION_CONVERGED,
    MSG_OPTIMIZATION_STOPPED,
    MSG_STARTING_NAMED_OPTIMIZATION,
)
from famex.utils.logging import get_famex_logger

logger = get_famex_logger(__name__)


class LoggingFile:
    def __init__(self) -> None:
        self.buffer = ""
        self.should_output = logger.getEffectiveLevel() <= logging.INFO

    def write(self, text: str) -> int:
        if not text:
            return 0

        if not self.should_output:
            return len(text)

        self.buffer += text

        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            if line.strip():
                logger.info("%s", line.strip())

        return len(text)

    def flush(self) -> None:
        if self.should_output and self.buffer.strip():
            logger.info("%s", self.buffer.strip())
            self.buffer = ""

    def close(self) -> None:
        self.flush()

    def __enter__(self) -> LoggingFile:
        return self

    def __exit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
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
        raise AttributeError(msg)

    def check_state(self, atoms: Atoms, tol: float = 1e-15) -> bool:
        return self.calculator.check_state(atoms, tol)  # type: ignore[no-any-return]


class SymmetryCalculatorWrapper(Calculator):
    """Project forces and Hessians onto a resolved molecular point group."""

    def __init__(self, calculator: Calculator, molecular_symmetry: Any) -> None:
        super().__init__()
        self.calculator = calculator
        self.molecular_symmetry = molecular_symmetry
        self._name = getattr(calculator, "name", "wrapped")
        self.implemented_properties = getattr(
            calculator,
            "implemented_properties",
            ["energy", "forces"],
        ).copy()

    @property
    def name(self) -> str:
        return self._name

    def __getattr__(self, name: str) -> Any:
        if name in ("calculator", "molecular_symmetry", "_name"):
            return object.__getattribute__(self, name)
        attr = getattr(self.calculator, name)
        # Forward get_hessian only when the backend has it, so property probes
        # (hasattr / has_calculator_property) stay accurate for FD fallbacks.
        if name == "get_hessian" and callable(attr):

            def get_hessian(atoms: Atoms | None = None) -> np.ndarray:
                hessian = np.asarray(attr(atoms), dtype=np.float64)
                if not self.molecular_symmetry.should_project:
                    return hessian
                return np.asarray(
                    self.molecular_symmetry.project_hessian(
                        self._atoms_for_projection(atoms), hessian
                    ),
                    dtype=np.float64,
                )

            return get_hessian
        return attr

    def _atoms_for_projection(self, atoms: Atoms | None) -> Atoms:
        if atoms is not None:
            return atoms
        calc_atoms = getattr(self.calculator, "atoms", None)
        if calc_atoms is None:
            raise RuntimeError("No atoms available for point-group projection")
        return calc_atoms  # type: ignore[no-any-return]

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] | None = None,
    ) -> Any:
        result = self.calculator.calculate(atoms, properties, system_changes)
        results = getattr(self.calculator, "results", None)
        if isinstance(results, dict) and self.molecular_symmetry.should_project:
            target = self._atoms_for_projection(atoms)
            if results.get("forces") is not None:
                results["forces"] = self.molecular_symmetry.project_forces(
                    target, results["forces"]
                )
            if results.get("hessian") is not None:
                results["hessian"] = self.molecular_symmetry.project_hessian(
                    target, results["hessian"]
                )
        return result

    def get_property(
        self, name: str, atoms: Atoms | None = None, allow_calculation: bool = True
    ) -> Any:
        if name == "forces":
            return self.get_forces(atoms)
        if name == "hessian" and hasattr(self.calculator, "get_hessian"):
            return self.get_hessian(atoms)
        return self.calculator.get_property(name, atoms, allow_calculation)

    def get_potential_energy(
        self, atoms: Atoms | None = None, force_consistent: bool = False
    ) -> float:
        return self.calculator.get_potential_energy(atoms, force_consistent)  # type: ignore[no-any-return]

    def get_forces(self, atoms: Atoms | None = None) -> np.ndarray:
        forces = np.asarray(self.calculator.get_forces(atoms), dtype=np.float64)
        if not self.molecular_symmetry.should_project:
            return forces
        return np.asarray(
            self.molecular_symmetry.project_forces(self._atoms_for_projection(atoms), forces),
            dtype=np.float64,
        )

    def get_stress(self, atoms: Atoms | None = None) -> np.ndarray:
        return self.calculator.get_stress(atoms)  # type: ignore[no-any-return]

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

        if getattr(atoms, "calc", None) is not None and self.profiler is not None:
            self.atoms.calc = ProfilerCalculatorWrapper(atoms.calc, self.profiler)

        if self.verbose >= 2:
            optimizer_name = wrapped_optimizer_class.__name__
            logger.info(MSG_INITIALIZED_OPTIMIZER_VERBOSITY, optimizer_name)

    def run(self, fmax: float = 0.05, steps: int = 1000) -> bool:
        if self.verbose >= 2:
            optimizer_name = self.wrapped_optimizer.__class__.__name__
            logger.info(MSG_STARTING_NAMED_OPTIMIZATION, optimizer_name)
            logger.info(MSG_CONVERGENCE_CRITERION_FMAX, fmax)
            logger.info(MSG_MAXIMUM_STEPS, steps)
        if self._logging_file is not None:
            with redirect_stdout(cast(TextIO, self._logging_file)):
                result = self.wrapped_optimizer.run(fmax=fmax, steps=steps)
            self._logging_file.flush()
        else:
            result = self.wrapped_optimizer.run(fmax=fmax, steps=steps)

        if self.verbose >= 1:
            if result:
                logger.info(MSG_OPTIMIZATION_CONVERGED)
            else:
                actual_steps = self.wrapped_optimizer.get_number_of_steps()
                scipy_reason = ""
                if (
                    hasattr(self.wrapped_optimizer, "_scipy_result")
                    and self.wrapped_optimizer._scipy_result
                ):
                    scipy_reason = str(getattr(self.wrapped_optimizer._scipy_result, "message", ""))

                logger.warning(MSG_OPTIMIZATION_STOPPED, actual_steps)
                if scipy_reason and "Maximum number of iterations" not in scipy_reason:
                    logger.warning("  SciPy reason: %s", scipy_reason)
                logger.warning("  (Max outer iterations limit: %s)", steps)
                forces = self.atoms.get_forces()
                logger.warning("Final max force: %.6f eV/Å", np.max(np.abs(forces)))

        return result  # type: ignore[no-any-return]

    def get_number_of_steps(self) -> int:
        return self.wrapped_optimizer.get_number_of_steps()  # type: ignore[no-any-return]

    def converged(self, forces: Any = None, *, gradient: Any = None) -> bool:
        if gradient is None:
            if forces is None:
                gradient = self.atoms.get_forces().ravel()
            else:
                gradient = np.asarray(forces).ravel()
        return bool(self.wrapped_optimizer.converged(gradient))

    def log(self, forces: np.ndarray) -> None:
        self.wrapped_optimizer.log(forces)

    def call_observers(self) -> None:
        self.wrapped_optimizer.call_observers()

    def dump(self, data: Any) -> None:
        self.wrapped_optimizer.dump(data)

    def load(self) -> None:
        self.wrapped_optimizer.load()


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


class VerboseSellaAnalytical(VerboseSella):
    """Sella optimizer that uses analytical Hessians from the attached calculator."""

    def __init__(
        self,
        atoms: Atoms,
        logfile: IO[Any] | TextIO | str | None = "-",
        trajectory: str | None = None,
        verbose: int = 1,
        profiler: Any = None,
        **kwargs: Any,
    ) -> None:
        from famex.optimizers.sella_utils import (
            make_analytical_hessian_function,
            validate_calculator_supports_hessian,
        )

        if "hessian_function" not in kwargs:
            validate_calculator_supports_hessian(atoms.calc)
            kwargs["hessian_function"] = make_analytical_hessian_function()

        super().__init__(
            atoms=atoms,
            logfile=logfile,
            trajectory=trajectory,
            verbose=verbose,
            profiler=profiler,
            **kwargs,
        )


def get_verbose_ase_optimizer_class(name: str) -> type[Optimizer] | None:
    """Return a verbosity-wrapped ASE optimizer class for a normalized name, if known."""
    normalized = (name or "").lower()
    if normalized in ("lbfgs", "l-bfgs", "l_bfgs"):
        return VerboseLBFGS
    if normalized == "bfgs":
        return VerboseBFGS
    if normalized == "fire":
        return VerboseFIRE
    if normalized == "sella":
        return VerboseSella
    if normalized in ("sella-analytical", "sella_analytical", "sellaanalytical"):
        return VerboseSellaAnalytical
    return None
