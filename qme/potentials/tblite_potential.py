"""TBLite Semi-empirical Quantum Chemistry integration for ASE.

This module implements a TBLite calculator integration using the TBLite library
for semi-empirical quantum chemistry calculations with xTB methods.
"""

from collections.abc import Sequence
from typing import Any

from ase import Atoms

from qme.dependencies import deps
from qme.potentials.base_potential import BasePotential


class TBLitePotential(BasePotential):
    """TBLite semi-empirical quantum chemistry calculator.

    This calculator provides access to TBLite's xTB methods (GFN1-xTB, GFN2-xTB)
    for fast semi-empirical quantum chemistry calculations. TBLite is particularly
    useful for large systems and rapid screening calculations.
    """

    implemented_properties = ["energy", "forces", "charges", "dipole", "stress"]

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
        verbosity: int = 1,
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
        verbosity : int, default 1
            Set verbosity of printout
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

        from qme.logging_utils import quiet_backend_loading

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

                # Create TBLite calculator with all parameters
                calc_kwargs = {
                    "method": self.method,
                    "accuracy": self.accuracy,
                    "electronic_temperature": self.electronic_temperature,
                    "max_iterations": self.max_iterations,
                    "initial_guess": self.initial_guess,
                    "mixer_damping": self.mixer_damping,
                    "cache_api": self.cache_api,
                    "verbosity": self.verbosity,
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
        try:
            self._calc.calculate(self.atoms, properties, system_changes)
        except (AttributeError, RuntimeError) as e:
            msg = f"TBLite calculation failed: {e}"
            raise RuntimeError(msg)

        # Copy results from underlying calculator
        try:
            self.results = self._calc.results.copy()
        except Exception:
            # If underlying calculator does not use .results dict, attempt to
            # extract common properties
            if properties is None:
                properties = self.implemented_properties
            if "energy" in properties and hasattr(self._calc, "results"):
                self.results["energy"] = getattr(self._calc.results, "energy", None)

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

    Returns
    -------
    TBLitePotential
        Configured TBLite calculator instance

    Examples
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
