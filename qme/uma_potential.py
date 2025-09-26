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
        self,
        model_name: str = "uma-s-1p1",
        device: Optional[str] = None,
        default_charge: int = 0,
        default_spin: int = 1,
        **kwargs,
    ):
        """
        Initialize UMA potential calculator.

        Parameters:
        -----------
        model_name : str
            Name of the UMA model to load (default: "uma-s-1p1")
        device : str, optional
            Device to run computations on ('cpu', 'cuda'). Auto-detected if None.
        default_charge : int
            Default charge to use if not specified in atoms.info (default: 0)
        default_spin : int
            Default spin multiplicity to use if not specified in atoms.info (default: 1)
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
        self.default_charge = default_charge
        self.default_spin = default_spin

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

            # Try to force consistent precision to avoid dtype mismatches
            if hasattr(self.predictor, "model"):
                # Force model to float32 precision to avoid mixed precision issues
                if hasattr(self.predictor.model, "float"):
                    self.predictor.model.float()

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

        # Set default charge and spin if not already set to avoid warnings
        if "charge" not in atoms.info:
            atoms.info["charge"] = self.default_charge
        if "spin" not in atoms.info:
            atoms.info["spin"] = self.default_spin

        # Ensure calculator is loaded
        if self.fairchem_calc is None:
            self._load_calculator()

        if self.fairchem_calc is None:
            raise RuntimeError("Failed to load UMA calculator")

        # Use the fairchem v2 calculator directly
        # Calculate properties using the fairchem calculator directly
        try:
            self.fairchem_calc.calculate(atoms, properties, system_changes)
        except RuntimeError as e:
            if "expected scalar type Double but found Float" in str(
                e
            ) or "mat1 and mat2 must have the same dtype, but got Double and Float" in str(
                e
            ):
                # Try to set model to use float32 precision consistently
                if self.predictor is not None and hasattr(self.predictor, "model"):
                    # Force model to float32
                    if hasattr(self.predictor.model, "float"):
                        self.predictor.model.float()

                    # Also try to set the model to double precision if that's what it needs
                    if "expected scalar type Double but found Float" in str(e):
                        if hasattr(self.predictor.model, "double"):
                            self.predictor.model.double()

                    # Retry calculation
                    self.fairchem_calc.calculate(atoms, properties, system_changes)
                else:
                    raise RuntimeError(
                        f"UMA model precision mismatch. {e}. "
                        "This may be due to model expecting different tensor precision."
                    )
            else:
                raise

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
