"""
MACE Machine Learning Potential integration for ASE.

This module implements a MACE calculator integration using the MACE-OMOL-0
foundation model for molecular systems, transition metals, and cations.
"""

from typing import Optional

from .base_potential import BasePotential
from .dependencies import deps


class MACEPotential(BasePotential):
    """
    MACE potential calculator using foundation models.

    This calculator provides access to MACE foundation models, particularly
    the MACE-OMOL-0 model which is excellent for molecules, transition metals,
    and cations with charge/spin embedding capabilities.
    """

    def __init__(
        self, model_name: Optional[str] = None, device: Optional[str] = None, **kwargs
    ):
        """
        Initialize MACE potential calculator.

        Parameters:
        -----------
        model_name : str, optional
            MACE model to use. Defaults to "mace-omol-0"
            Available options:
            - "mace-omol-0": Large model for molecules/transition metals/cations
            - "mace-mp-0": Materials Project models (small, medium, large)
            - "mace-off23": Organic chemistry models (small, medium, large)
        device : str, optional
            Device to run computations on ('cpu', 'cuda'). Auto-detected if None.
        **kwargs : dict
            Additional arguments passed to Calculator
        """
        if model_name is None:
            model_name = "mace-omol-0"

        super().__init__(model_name=model_name, device=device, **kwargs)

    def _load_calculator(self):
        """Load the MACE calculator implementation."""
        from .logging_utils import quiet_backend_loading

        if not deps.has("torch"):
            raise ImportError(
                "PyTorch is required for MACE backend. "
                "Install with: pip install torch"
            )

        with quiet_backend_loading("mace", self.model_name, None, self.device):
            try:
                # Try to import MACE calculators
                if self.model_name == "mace-omol-0":
                    from mace.calculators import mace_omol

                    self._calc = mace_omol(device=self.device or "cpu")
                elif self.model_name and self.model_name.startswith("mace-mp"):
                    from mace.calculators import mace_mp

                    # Extract model size from model name (e.g., mace-mp-medium -> medium)
                    model_size = self.model_name.replace("mace-mp-", "") or "medium"
                    self._calc = mace_mp(model=model_size, device=self.device or "cpu")
                elif self.model_name and self.model_name.startswith("mace-off"):
                    from mace.calculators import mace_off

                    # Extract model size from model name (e.g., mace-off-medium -> medium)
                    model_size = self.model_name.replace("mace-off-", "") or "medium"
                    self._calc = mace_off(model=model_size, device=self.device or "cpu")
                else:
                    # Default to MACE-OMOL for unknown model names
                    from mace.calculators import mace_omol

                    self._calc = mace_omol(device=self.device or "cpu")

            except ImportError as e:
                deps.warn_fallback(
                    "mace",
                    f"MACE not available ({e}). Install with: pip install mace-torch",
                )
                # Fall back to mock calculator
                from .mock_calculator import MockCalculator

                self._calc = MockCalculator(backend="mace")

    def _get_backend_name(self) -> str:
        """Get the backend name for this calculator."""
        return "mace"

    def calculate(self, atoms=None, properties=None, system_changes=None):
        """Calculate properties using the MACE calculator."""
        if atoms is not None:
            self.atoms = atoms

        if properties is None:
            properties = self.implemented_properties

        # Delegate to the underlying MACE calculator
        self._calc.calculate(atoms, properties, system_changes)

        # Copy results from underlying calculator
        self.results = self._calc.results.copy()

    def get_potential_energy(self, atoms=None, force_consistent=False):
        """Get potential energy."""
        if atoms is not None:
            self.atoms = atoms
        return self._calc.get_potential_energy(atoms, force_consistent)

    def get_forces(self, atoms=None):
        """Get forces on atoms."""
        if atoms is not None:
            self.atoms = atoms
        return self._calc.get_forces(atoms)

    def get_stress(self, atoms=None):
        """Get stress tensor (if supported)."""
        if atoms is not None:
            self.atoms = atoms
        if hasattr(self._calc, "get_stress"):
            return self._calc.get_stress(atoms)
        else:
            msg = "Stress calculation not supported by this MACE model"
            raise NotImplementedError(msg)


def get_mace_calculator(
    model_name: Optional[str] = None, device: Optional[str] = None, **kwargs
) -> MACEPotential:
    """
    Factory function to create MACE calculator.

    Parameters:
    -----------
    model_name : str, optional
        MACE model to use. Defaults to "mace-omol-0"
    device : str, optional
        Device for computations ('cpu', 'cuda'). Auto-detected if None.
    **kwargs : dict
        Additional arguments passed to MACEPotential

    Returns:
    --------
    MACEPotential
        Configured MACE calculator instance

    Examples:
    ---------
    >>> calc = get_mace_calculator()  # Uses MACE-OMOL-0
    >>> calc = get_mace_calculator(model_name="mace-mp-medium")
    >>> calc = get_mace_calculator(model_name="mace-off-large", device="cuda")
    """
    return MACEPotential(model_name=model_name, device=device, **kwargs)
