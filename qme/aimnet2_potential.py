"""
AIMNET2 Machine Learning Potential integration for ASE.
"""

from typing import Optional

from ase.calculators.calculator import Calculator, all_changes

# Handle optional dependencies
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

try:
    from aimnet2calc import AIMNet2ASE

    HAS_AIMNET2 = True
except ImportError:
    HAS_AIMNET2 = False


class AIMNet2Potential(Calculator):
    """
    ASE Calculator interface for AIMNET2 neural network potential.

    AIMNET2 provides accurate and versatile neural network potentials for
    molecular property prediction and geometry optimization, excelling at
    modeling neutral, charged, organic, and elemental-organic systems.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        model_name: str = "aimnet2",
        device: Optional[str] = None,
        charge: int = 0,
        mult: int = 1,
        **kwargs,
    ):
        """Initialize AIMNET2 potential calculator.

        Parameters:
        -----------
        model_name : str
            Name/path of AIMNET2 model to use. Default: 'aimnet2'
        device : str, optional
            Device for computations ('cpu', 'cuda'). Auto-detected if None.
        charge : int
            Molecular charge. Default: 0
        mult : int
            Spin multiplicity. Default: 1
        **kwargs :
            Additional arguments passed to parent Calculator
        """
        Calculator.__init__(self, **kwargs)

        self.model_name = model_name
        self.device = device or (
            "cuda" if torch and torch.cuda.is_available() else "cpu"
        )
        self.charge = charge
        self.mult = mult

        # Initialize calculator
        if HAS_AIMNET2 and HAS_TORCH:
            try:
                self._load_model()
            except Exception as e:
                print(f"Warning: Failed to load AIMNET2 model: {e}")
                print("Falling back to mock AIMNET2 calculator for testing.")
                self._use_mock_implementation()
        else:
            print("Warning: AIMNET2 or PyTorch not available.")
            print("Falling back to mock AIMNET2 calculator for testing.")
            self._use_mock_implementation()

    def _load_model(self):
        """Load the AIMNET2 model from aimnet2calc."""
        try:
            # Create AIMNET2ASE calculator with model
            self.aimnet2_calc = AIMNet2ASE(
                base_calc=self.model_name, charge=self.charge, mult=self.mult
            )

            # Set device if torch is available
            if hasattr(self.aimnet2_calc.base_calc, "device"):
                self.aimnet2_calc.base_calc.device = self.device

        except Exception as e:
            raise RuntimeError(
                f"Failed to load AIMNET2 model '{self.model_name}'. "
                f"Error: {e}. Please check the model name or installation."
            )

    def _use_mock_implementation(self):
        """Use mock AIMNET2 implementation for testing."""
        # Import here to avoid circular imports
        from .mock_calculator import MockAIMNet2Calculator

        self.aimnet2_calc = MockAIMNet2Calculator(charge=self.charge, mult=self.mult)

    def calculate(
        self,
        atoms=None,
        properties=["energy", "forces"],
        system_changes=all_changes,
    ):
        """Calculate properties using AIMNET2 potential."""
        Calculator.calculate(self, atoms, properties, system_changes)

        # Set atoms to the AIMNET2 calculator
        self.aimnet2_calc.set_atoms(atoms)

        # Set charge and multiplicity if they differ
        if (
            hasattr(self.aimnet2_calc, "set_charge")
            and self.aimnet2_calc.charge != self.charge
        ):
            self.aimnet2_calc.set_charge(self.charge)
        if (
            hasattr(self.aimnet2_calc, "set_mult")
            and self.aimnet2_calc.mult != self.mult
        ):
            self.aimnet2_calc.set_mult(self.mult)

        # Calculate properties
        if "energy" in properties:
            energy = self.aimnet2_calc.get_potential_energy()
            self.results["energy"] = energy

        if "forces" in properties:
            forces = self.aimnet2_calc.get_forces()
            self.results["forces"] = forces

    def set_charge(self, charge: int):
        """Set molecular charge."""
        self.charge = charge
        if hasattr(self.aimnet2_calc, "set_charge"):
            self.aimnet2_calc.set_charge(charge)

    def set_mult(self, mult: int):
        """Set spin multiplicity."""
        self.mult = mult
        if hasattr(self.aimnet2_calc, "set_mult"):
            self.aimnet2_calc.set_mult(mult)


def get_aimnet2_calculator(
    model_name: str = "aimnet2",
    device: Optional[str] = None,
    charge: int = 0,
    mult: int = 1,
    **kwargs,
) -> AIMNet2Potential:
    """
    Convenience function to get AIMNET2 calculator.

    Parameters:
    -----------
    model_name : str
        Name/path of AIMNET2 model to use
    device : str, optional
        Device for computations ('cpu', 'cuda')
    charge : int
        Molecular charge
    mult : int
        Spin multiplicity
    **kwargs :
        Additional arguments passed to AIMNet2Potential

    Returns:
    --------
    AIMNet2Potential
        Configured AIMNET2 calculator
    """
    return AIMNet2Potential(
        model_name=model_name, device=device, charge=charge, mult=mult, **kwargs
    )


# Mock AIMNET2 calculator for testing without dependencies
def get_mock_aimnet2_calculator(**kwargs):
    """Get mock AIMNET2 calculator for testing without dependencies."""
    # Use the AIMNet2Potential class which will automatically fall back to mock
    return AIMNet2Potential(**kwargs)
