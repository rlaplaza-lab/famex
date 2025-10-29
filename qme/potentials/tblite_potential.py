"""TBLite Semi-empirical Quantum Chemistry integration for ASE.

This module implements a TBLite calculator integration using the TBLite library
for semi-empirical quantum chemistry calculations with xTB methods.
"""

import contextlib
import os
import sys
from collections.abc import Sequence
from typing import Any

from ase import Atoms

from qme.backends.dependencies import deps
from qme.potentials.base_potential import BasePotential


@contextlib.contextmanager
def suppress_tblite_output():
    """Context manager to suppress TBLite verbose output when QME verbosity < 3.

    This redirects stdout and stderr at the file descriptor level using os.dup2()
    to catch output from TBLite's Fortran/C backend that bypasses Python's
    stdout/stderr redirection. This silences verbose SCF cycle tables, timings,
    and dictionary dumps.

    TBLite output is suppressed at verbosity levels 0, 1, and 2, and only
    appears at verbosity 3 or higher (if supported in the future).
    """
    # Always suppress TBLite output since verbosity 3+ is not currently
    # supported in QME's logging system. This ensures TBLite remains quiet
    # even at verbosity 2 (DEBUG level).
    # Use os.dup2() for file descriptor-level redirection to catch C/Fortran output
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        # Get file descriptor numbers for stdout and stderr
        stdout_fd = sys.stdout.fileno()
        stderr_fd = sys.stderr.fileno()

        # Save copies of original file descriptors before redirecting
        saved_stdout_fd = os.dup(stdout_fd)
        saved_stderr_fd = os.dup(stderr_fd)

        # Redirect at file descriptor level (catches C/Fortran output)
        os.dup2(devnull_fd, stdout_fd)
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        # Restore original file descriptors
        if "saved_stdout_fd" in locals():
            os.dup2(saved_stdout_fd, stdout_fd)
            os.close(saved_stdout_fd)
        if "saved_stderr_fd" in locals():
            os.dup2(saved_stderr_fd, stderr_fd)
            os.close(saved_stderr_fd)
        os.close(devnull_fd)


class TBLitePotential(BasePotential):
    """TBLite semi-empirical quantum chemistry calculator.

    This calculator provides access to TBLite's xTB methods (GFN1-xTB, GFN2-xTB)
    for fast semi-empirical quantum chemistry calculations. TBLite is particularly
    useful for large systems and rapid screening calculations.
    """

    implemented_properties = ["energy", "energies", "forces", "charges", "dipole", "stress"]

    def __init__(
        self,
        method: str = "GFN2-xTB",
        charge: int | None = None,
        multiplicity: int | None = None,
        accuracy: float = 1.0,
        electronic_temperature: float = 300.0,
        max_iterations: int = 250,
        initial_guess: str = "sad",
        mixer_damping: float = 0.4,
        electric_field: list[float] | None = None,
        spin_polarization: float | None = None,
        solvation: tuple[str, ...] | None = None,
        cache_api: bool = True,
        verbosity: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize TBLite potential calculator.

        Parameters
        ----------
        method : str, default "GFN2-xTB"
            TBLite method to use. Available options:
            - "GFN1-xTB": Fast method for large systems
            - "GFN2-xTB": More accurate method (default)
        charge : int, optional
            Total charge of the system
        multiplicity : int, optional
            Total spin multiplicity of the system
        accuracy : float, default 1.0
            Numerical accuracy of the calculation
        electronic_temperature : float, default 300.0
            Electronic temperature in Kelvin
        max_iterations : int, default 250
            Maximum iterations for self-consistent evaluation
        initial_guess : str, default "sad"
            Initial guess for wavefunction ("sad" or "eeq")
        mixer_damping : float, default 0.4
            Damping parameter for self-consistent mixer
        electric_field : list[float], optional
            Uniform electric field vector (in V/A)
        spin_polarization : float, optional
            Spin polarization (scaling factor)
        solvation : tuple[str, ...], optional
            Solvation model to use (see TBLite docs for details)
        cache_api : bool, default True
            Reuse generated API objects (recommended)
        verbosity : int, default 0
            Set verbosity of printout (0=quiet, 1=normal, 2=verbose).
            Note: This is automatically adjusted based on QME verbosity level.
        **kwargs
            Additional arguments passed to BasePotential

        """
        # Store TBLite-specific parameters
        self.method = method
        self.charge = charge
        self.multiplicity = multiplicity
        self.accuracy = accuracy
        self.electronic_temperature = electronic_temperature
        self.max_iterations = max_iterations
        self.initial_guess = initial_guess
        self.mixer_damping = mixer_damping
        self.electric_field = electric_field
        self.spin_polarization = spin_polarization
        self.solvation = solvation
        self.cache_api = cache_api
        self.verbosity = verbosity

        # Placeholder for the underlying calculator implementation
        self._calc = None

        super().__init__(backend="tblite", **kwargs)

    def _load_calculator(self) -> None:
        """Load the TBLite calculator implementation."""
        # Skip if already loaded
        if hasattr(self, "_calc") and self._calc is not None:
            return

        from qme.utils.ml_warnings import quiet_backend_loading

        # Don't show model info - let the outer context handle it
        with quiet_backend_loading("tblite", self.method, None, None, show_model_info=False):
            try:
                # Check TBLite availability
                if not deps.has("tblite"):
                    msg = "TBLite is required for TBLite backend. Install with: pip install tblite"
                    raise ImportError(
                        msg,
                    )

                # Import TBLite ASE calculator
                from tblite.ase import TBLite

                # Always suppress tblite verbosity unless explicitly set very high
                # TBLite output should only appear at verbosity 3+, which isn't
                # currently supported in QME, so we always set it to 0
                # Create TBLite calculator with all parameters
                # According to docs: TBLite(atoms: Atoms | None = None, **kwargs)
                calc_kwargs = {
                    "method": self.method,
                    "accuracy": self.accuracy,
                    "electronic_temperature": self.electronic_temperature,
                    "max_iterations": self.max_iterations,
                    "initial_guess": self.initial_guess,
                    "mixer_damping": self.mixer_damping,
                    "cache_api": self.cache_api,
                    "verbosity": 0,  # Set to 0 for quiet operation
                }

                # Add optional parameters if provided
                if self.charge is not None:
                    calc_kwargs["charge"] = self.charge
                if self.multiplicity is not None:
                    calc_kwargs["multiplicity"] = self.multiplicity
                if self.electric_field is not None:
                    calc_kwargs["electric_field"] = self.electric_field
                if self.spin_polarization is not None:
                    calc_kwargs["spin_polarization"] = self.spin_polarization
                if self.solvation is not None:
                    calc_kwargs["solvation"] = self.solvation

                self._calc = TBLite(**calc_kwargs)

            except ImportError as e:
                msg = f"TBLite not available ({e}). Install with: pip install tblite"
                raise ImportError(msg)
            except (ValueError, AttributeError, RuntimeError) as e:
                msg = f"Failed to initialize TBLite calculator: {e}"
                raise RuntimeError(msg)

    def _get_backend_name(self) -> str:
        """Get the backend name for this calculator."""
        return "tblite"

    def _set_tblite_verbosity(self, verbosity: int) -> None:
        """Set verbosity on the underlying tblite calculator.

        Parameters
        ----------
        verbosity : int
            Verbosity level (0=quiet, 1=normal, 2=verbose)
        """
        if self._calc is None:
            return

        # Try to set verbosity using the tblite Calculator.set() API
        if hasattr(self._calc, "set"):
            try:
                self._calc.set("verbosity", verbosity)
                return
            except (AttributeError, ValueError, RuntimeError):
                pass

        # Fallback: try accessing underlying calculator if available
        for attr_name in ["_calc", "calculator"]:
            if hasattr(self._calc, attr_name):
                calc_obj = getattr(self._calc, attr_name)
                if hasattr(calc_obj, "set"):
                    try:
                        calc_obj.set("verbosity", verbosity)
                        return
                    except (AttributeError, ValueError, RuntimeError):
                        pass

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: Sequence[str] | None = None,
        system_changes: Any = None,
    ) -> None:
        """Calculate properties using the TBLite calculator."""
        # Common setup
        super().calculate(atoms, properties, system_changes)

        # Ensure calculator is loaded
        if self._calc is None:
            self._load_calculator()

        if self._calc is None:
            msg = "Failed to load TBLite calculator"
            raise RuntimeError(msg)

        # Delegate to the underlying TBLite calculator
        # Suppress verbose output unless QME verbosity is 3 or higher
        # Note: verbosity is set once during calculator initialization, not here
        try:
            with suppress_tblite_output():
                self._calc.calculate(self.atoms, properties, system_changes)
        except (AttributeError, RuntimeError) as e:
            msg = f"TBLite calculation failed: {e}"
            raise RuntimeError(msg)

        # Extract results from underlying calculator
        # Match UMA pattern: directly access results for requested properties
        if properties is None:
            properties = self.implemented_properties

        # Extract all requested properties explicitly to ensure they're populated
        for prop in properties:
            if prop in self.implemented_properties and hasattr(self._calc, "results"):
                try:
                    self.results[prop] = self._calc.results[prop]
                except (KeyError, AttributeError):
                    # Try .get() as fallback, but this shouldn't happen for valid calculations
                    self.results[prop] = self._calc.results.get(prop, None)

    def get_potential_energy(self, atoms=None, force_consistent=False):
        """Get potential energy."""
        if atoms is not None:
            self.atoms = atoms
        # Ensure calculator is loaded
        return super().get_potential_energy(atoms, force_consistent)

    def get_forces(self, atoms=None):
        """Get forces on atoms."""
        if atoms is not None:
            self.atoms = atoms
        # Ensure calculator is loaded
        return super().get_forces(atoms)

    def get_charges(self, atoms=None):
        """Get Mulliken charges (if supported)."""
        if atoms is not None:
            self.atoms = atoms
        # Ensure calculator is loaded
        if self._calc is None:
            self._load_calculator()

        if self._calc is not None and hasattr(self._calc, "get_charges"):
            return self._calc.get_charges(atoms)
        msg = "Charge calculation not supported by this TBLite method"
        raise NotImplementedError(msg)

    def get_dipole_moment(self, atoms=None):
        """Get dipole moment (if supported)."""
        if atoms is not None:
            self.atoms = atoms
        # Ensure calculator is loaded
        if self._calc is None:
            self._load_calculator()

        if self._calc is not None and hasattr(self._calc, "get_dipole_moment"):
            return self._calc.get_dipole_moment(atoms)
        msg = "Dipole moment calculation not supported by this TBLite method"
        raise NotImplementedError(msg)

    def get_stress(self, atoms=None):
        """Get stress tensor (if supported)."""
        if atoms is not None:
            self.atoms = atoms
        # Ensure calculator is loaded
        if self._calc is None:
            self._load_calculator()

        if self._calc is not None and hasattr(self._calc, "get_stress"):
            return self._calc.get_stress(atoms)
        msg = "Stress calculation not supported by this TBLite method"
        raise NotImplementedError(msg)


def get_tblite_calculator(
    method: str = "GFN2-xTB",
    charge: int | None = None,
    multiplicity: int | None = None,
    **kwargs: Any,
) -> TBLitePotential:
    """Factory function to create TBLite calculator.

    Parameters
    ----------
    method : str, default "GFN2-xTB"
        TBLite method to use
    charge : int, optional
        Total charge of the system
    multiplicity : int, optional
        Total spin multiplicity of the system
    **kwargs
        Additional arguments passed to TBLitePotential

    Returns:
    -------
    TBLitePotential
        Configured TBLite calculator instance

    Examples:
    --------
    >>> calc = get_tblite_calculator()  # Uses GFN2-xTB
    >>> calc = get_tblite_calculator(method="GFN1-xTB")
    >>> calc = get_tblite_calculator(charge=-1, multiplicity=2)

    """
    return TBLitePotential(
        method=method,
        charge=charge,
        multiplicity=multiplicity,
        **kwargs,
    )
