"""
UMA Machine Learning Potential integration for ASE.
"""

from typing import Optional

from ase.calculators.calculator import all_changes

from .base_potential import BasePotential
from .dependencies import HAS_FAIRCHEM, HAS_TORCH, deps, torch


class UMAPotential(BasePotential):
    """
    ASE Calculator interface for UMA (Universal Model for Atoms) potential.

    This calculator provides an interface to use UMA machine learning potentials
    for molecular property prediction and geometry optimization.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self, model_name: str = "uma-s-1p1", device: Optional[str] = None, **kwargs
    ):
        """
        Initialize UMA potential calculator.

        Parameters:
        -----------
        model_name : str
            Name of the UMA model to load (default: "uma-s-1p1")
        device : str, optional
            Device to run computations on ('cpu', 'cuda'). Auto-detected if None.
        """

        if not HAS_TORCH:
            raise ImportError(
                "PyTorch is required for UMA potentials. "
                "Install with: pip install torch"
            )

        if not HAS_FAIRCHEM:
            raise ImportError(
                "fairchem-core is required for UMA potentials. "
                "Install with: pip install fairchem-core"
            )

        # Set device
        if device is None:
            device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # Initialize base class
        super().__init__(model_name=model_name, device=device, **kwargs)

        # UMA-specific attributes
        self.predictor = None
        self.fairchem_calc = None

    def _load_calculator(self):
        """Load the UMA model from fairchem v2 API."""
        try:
            # Get fairchem v2 dependencies
            pretrained_mlip = deps.get("fairchem_pretrained_mlip")
            FAIRChemCalculator = deps.get("fairchem_calculator")

            if not pretrained_mlip or not FAIRChemCalculator:
                raise RuntimeError("FairChem v2 components not available")

            # Load UMA model using v2 API
            device_str = str(self.device)
            self.predictor = pretrained_mlip.get_predict_unit(
                self.model_name, device=device_str
            )

            # Create fairchem calculator for internal use
            # Default to 'omol' task for molecular systems
            self.fairchem_calc = FAIRChemCalculator(self.predictor, task_name="omol")

        except Exception as e:
            raise RuntimeError(
                f"Failed to load UMA model '{self.model_name}'. "
                f"Error: {e}. Please check the model name or installation."
            )

    def calculate(
        self,
        atoms=None,
        properties=["energy", "forces"],
        system_changes=all_changes,
    ):
        """Calculate properties using UMA potential."""

        super().calculate(atoms, properties, system_changes)

        if atoms is None:
            raise ValueError("atoms cannot be None")

        # Ensure calculator is loaded
        if self.fairchem_calc is None:
            self._load_calculator()

        # Use the fairchem v2 calculator directly
        # Calculate properties using the fairchem calculator directly
        self.fairchem_calc.calculate(atoms, properties, system_changes)

        # Extract results from the fairchem calculator
        if "energy" in properties:
            self.results["energy"] = self.fairchem_calc.results["energy"]

        if "forces" in properties:
            self.results["forces"] = self.fairchem_calc.results["forces"]

    def _get_backend_name(self) -> str:
        """Get the backend name for UMA."""
        return "uma"


def get_uma_calculator(model_name: str = "uma-s-1p1", **kwargs) -> UMAPotential:
    """
    Convenience function to get UMA calculator.

    Parameters:
    -----------
    model_name : str
        Name of UMA model to use
    **kwargs :
        Additional arguments passed to UMAPotential

    Returns:
    --------
    UMAPotential
        Configured UMA calculator
    """
    return UMAPotential(model_name=model_name, **kwargs)
