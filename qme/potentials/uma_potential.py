"""
UMA Machine Learning Potential integration for ASE.
"""

from typing import Optional

from ase.calculators.calculator import all_changes

from qme.dependencies import deps
from qme.potentials.base_potential import BasePotential


class UMAPotential(BasePotential):
    """ASE Calculator interface for UMA (Universal Model for Atoms) potential.

    This calculator provides an interface to use UMA machine learning potentials
    for molecular property prediction and geometry optimization.

    Parameters
    ----------
    model_name : str, default "uma-s-1p1"
        Name of the UMA model to load
    device : str, optional
        Device to run computations on ('cpu', 'cuda'). Auto-detected if None.
    default_charge : int, default 0
        Default charge to use if not specified in atoms.info
    default_spin : int, default 1
        Default spin multiplicity to use if not specified in atoms.info
    **kwargs
        Additional arguments passed to BasePotential
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

        Parameters
        ----------
        model_name : str, default "uma-s-1p1"
            Name of the UMA model to load
        device : str, optional
            Device to run computations on ('cpu', 'cuda'). Auto-detected if None.
        default_charge : int, default 0
            Default charge to use if not specified in atoms.info
        default_spin : int, default 1
            Default spin multiplicity to use if not specified in atoms.info
        """

        # Don't check dependencies here - let _load_calculator handle it
        # This avoids early imports that might interfere with fairchem

        # Set device
        if device is None:
            device = "cpu"  # Default device, will be auto-detected later if needed

        # Initialize UMA-specific attributes first
        self.predictor = None
        self._calc = None
        self.default_charge = default_charge
        self.default_spin = default_spin

        # Initialize base class (this will call _load_calculator)
        super().__init__(model_name=model_name, device=device, **kwargs)

    def _load_calculator(self):
        """Load the UMA model from fairchem v2 API."""
        # Skip if already loaded
        if hasattr(self, "_calc") and self._calc is not None:
            return

        from qme.logging_utils import quiet_backend_loading

        # Don't show model info - let the outer context handle it
        with quiet_backend_loading(
            "uma", self.model_name, None, self.device, show_model_info=False
        ):
            try:
                # Check fairchem availability without forcing PyTorch import
                if not deps.has("fairchem"):
                    raise ImportError(
                        "fairchem-core is required for UMA potentials. "
                        "Install with: pip install fairchem-core"
                    )

                # Use the dependency system to get fairchem components (lazy-loaded)
                pretrained_mlip = deps.get("fairchem_pretrained_mlip")
                FAIRChemCalculator = deps.get("fairchem_calculator")

                if not pretrained_mlip or not FAIRChemCalculator:
                    raise RuntimeError(
                        "FairChem v2 components not available"
                    )  # Load UMA model using v2 API
                # Ensure model_name is not None
                model_name = self.model_name or "uma-s-1p1"

                # Ensure device is compatible
                if self.device == "cuda":
                    device_param = "cuda"
                else:
                    device_param = "cpu"

                self.predictor = pretrained_mlip.get_predict_unit(model_name, device=device_param)

                # Try to force consistent precision to avoid dtype mismatches
                if hasattr(self.predictor, "model"):
                    # Force model to float32 precision to avoid mixed precision issues
                    if hasattr(self.predictor.model, "float"):
                        self.predictor.model.float()

                # Create fairchem calculator for internal use
                # Default to 'omol' task for molecular systems
                self._calc = FAIRChemCalculator(self.predictor, task_name="omol")

            except Exception as e:
                # If anything goes wrong while initializing the UMA model, raise a clear error
                raise RuntimeError(
                    f"Failed to load UMA model '{self.model_name}'. Error: {e}. "
                    f"Make sure fairchem-core is properly installed and the model is available."
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
        if self._calc is None:
            self._load_calculator()

        if self._calc is None:
            raise RuntimeError("Failed to load UMA calculator")

        # Use the underlying calculator directly
        try:
            self._calc.calculate(atoms, properties, system_changes)
        except RuntimeError as e:
            if "expected scalar type Double but found Float" in str(
                e
            ) or "mat1 and mat2 must have the same dtype, but got Double and Float" in str(e):
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
                    self._calc.calculate(atoms, properties, system_changes)
                else:
                    raise RuntimeError(
                        f"UMA model precision mismatch. {e}. "
                        "This may be due to model expecting different tensor precision."
                    )
            else:
                raise

        # Extract results from the underlying calculator
        if "energy" in properties:
            try:
                self.results["energy"] = self._calc.results["energy"]
            except Exception:
                self.results["energy"] = self.results.get("energy")

        if "forces" in properties:
            try:
                self.results["forces"] = self._calc.results["forces"]
            except Exception:
                self.results["forces"] = self.results.get("forces")

    def _get_backend_name(self) -> str:
        """Get the backend name for UMA."""
        return "uma"

    def get_potential_energy(self, atoms=None, force_consistent: bool = False):
        """Get potential energy (ASE-compatible)."""
        return super().get_potential_energy(atoms, force_consistent)

    def get_forces(self, atoms=None):
        """Get forces (ASE-compatible)."""
        return super().get_forces(atoms)


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
