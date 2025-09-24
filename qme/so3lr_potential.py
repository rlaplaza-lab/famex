"""
SO3LR Neural Network Potential integration for ASE.

SO3LR is an open source neural network potential with SO(3) invariant architecture.
This module provides ASE Calculator interface for SO3LR models.
"""

import warnings
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
    # This would be the actual SO3LR import when available
    # For now, we'll implement a mock interface
    import so3lr

    HAS_SO3LR = True
except ImportError:
    HAS_SO3LR = False
    so3lr = None


class SO3LRPotential(Calculator):
    """
    ASE Calculator interface for SO3LR neural network potential.

    SO3LR provides SO(3) invariant neural network potentials for molecular
    property prediction and geometry optimization.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = "so3lr-small",
        device: Optional[str] = None,
        use_mock: bool = False,
        **kwargs,
    ):
        """
        Initialize SO3LR potential calculator.

        Parameters:
        -----------
        model_path : str, optional
            Path to trained SO3LR model file
        model_name : str
            Name of pre-trained SO3LR model (default: "so3lr-small")
        device : str, optional
            Device to run computations on ('cpu', 'cuda'). Auto-detected if None.
        use_mock : bool, optional
            Force use of mock implementation even if PyTorch is available
        """

        Calculator.__init__(self, **kwargs)

        # Store parameters
        self.model_path = model_path
        self.model_name = model_name
        self.use_mock = use_mock

        # Check if we should use mock mode
        if not HAS_TORCH and not use_mock:
            raise ImportError(
                "PyTorch and SO3LR are required for SO3LR potentials. "
                "Install SO3LR with: git clone https://github.com/general-molecular-simulations/so3lr.git && cd so3lr && pip install ., "
                "or use use_mock=True for testing"
            )

        if HAS_TORCH and not use_mock:
            # Initialize PyTorch-related attributes only if PyTorch is available
            if device is None:
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
            else:
                self.device = torch.device(device)
        else:
            # Mock mode - don't use torch
            self.device = "cpu"  # Just a string for mock mode

        # Initialize model
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the SO3LR model."""
        # Check if we should use mock mode (either forced or no PyTorch/SO3LR)
        if self.use_mock or not HAS_TORCH or not HAS_SO3LR:
            if not HAS_TORCH:
                warnings.warn(
                    "PyTorch not available. Falling back to mock SO3LR implementation for testing.",
                    UserWarning,
                )
            elif not HAS_SO3LR:
                warnings.warn(
                    "SO3LR package not available. Install with: pip install so3lr\n"
                    "Falling back to mock SO3LR implementation for testing.",
                    UserWarning,
                )
            self._use_mock_implementation()
            return

        try:
            if self.model_path:
                # Load model from file
                self.model = so3lr.load_model(self.model_path)
            else:
                # Load pre-trained model
                self.model = so3lr.get_pretrained_model(self.model_name)

            self.model.to(self.device)
            self.model.eval()

        except Exception as e:
            raise RuntimeError(
                f"Failed to load SO3LR model '{self.model_name}'. "
                f"Error: {e}. Please check the model name or path."
            )

    def _use_mock_implementation(self):
        """Use mock SO3LR implementation for testing."""
        from .mock_calculator import MockSO3LRCalculator

        # Create a wrapper that makes MockSO3LRCalculator behave like SO3LR model
        class MockSO3LRModel:
            def __init__(self, device):
                self.device = device
                self.mock_calc = MockSO3LRCalculator()

            def to(self, device):
                self.device = device
                return self

            def eval(self):
                pass

            def __call__(self, data):
                # Convert data format to atoms-like object for mock calculator
                from ase import Atoms

                positions = data["positions"]
                # Create dummy atoms object (we only need positions for mock)
                # Use hydrogen atoms as default since mass doesn't affect the mock calculation
                atoms = Atoms("H" * len(positions), positions=positions)

                # Use mock calculator
                atoms.calc = self.mock_calc
                energy = atoms.get_potential_energy()
                forces = atoms.get_forces()

                return {"energy": energy, "forces": forces}

        self.model = MockSO3LRModel(self.device)

    def calculate(
        self,
        atoms=None,
        properties=["energy", "forces"],
        system_changes=all_changes,
    ):
        """Calculate properties using SO3LR potential."""

        Calculator.calculate(self, atoms, properties, system_changes)

        # Convert atoms to format expected by SO3LR
        data = self._atoms_to_data(atoms)

        # Run prediction
        if self.use_mock or not HAS_TORCH:
            # Mock mode - no need for torch.no_grad()
            outputs = self.model(data)
        else:
            # Real SO3LR mode
            with torch.no_grad():
                outputs = self.model(data)

        # Extract results
        if "energy" in properties:
            energy = outputs["energy"]
            if energy is not None:
                # Handle tensor conversion properly
                if hasattr(energy, "item"):
                    self.results["energy"] = energy.item()
                elif hasattr(energy, "cpu"):
                    self.results["energy"] = float(energy.cpu().numpy().item())
                else:
                    # Mock mode - energy is already a scalar
                    self.results["energy"] = float(energy)

        if "forces" in properties:
            forces = outputs["forces"]
            if forces is not None:
                if hasattr(forces, "cpu"):
                    self.results["forces"] = forces.cpu().numpy()
                else:
                    # Mock mode - forces are already numpy arrays
                    self.results["forces"] = forces

    def _atoms_to_data(self, atoms):
        """Convert ASE Atoms object to format expected by SO3LR model."""

        # If using mock mode, we don't need PyTorch tensors
        if self.use_mock or not HAS_TORCH:
            # Return data in a format that works with mock implementation
            import numpy as np

            data = {
                "positions": atoms.positions,  # Keep as numpy arrays
                "atomic_numbers": atoms.numbers,
                "n_atoms": len(atoms),
            }
            if atoms.pbc.any():
                data["cell"] = atoms.cell.array
            return data

        if not HAS_TORCH:
            raise ImportError("PyTorch is required")

        positions = torch.tensor(
            atoms.positions, dtype=torch.float32, device=self.device
        )
        atomic_numbers = torch.tensor(
            atoms.numbers, dtype=torch.long, device=self.device
        )

        # Create data dictionary expected by SO3LR models
        data = {
            "positions": positions,
            "atomic_numbers": atomic_numbers,
            "n_atoms": len(atoms),
        }

        # Add cell information if periodic
        if atoms.pbc.any():
            cell = torch.tensor(
                atoms.cell.array, dtype=torch.float32, device=self.device
            )
            data["cell"] = cell
            data["pbc"] = torch.tensor(atoms.pbc, device=self.device)

        return data


def get_so3lr_calculator(
    model_path: Optional[str] = None, model_name: str = "so3lr-small", **kwargs
) -> SO3LRPotential:
    """
    Convenience function to get SO3LR calculator.

    Parameters:
    -----------
    model_path : str, optional
        Path to trained SO3LR model file
    model_name : str
        Name of pre-trained SO3LR model
    **kwargs :
        Additional arguments passed to SO3LRPotential

    Returns:
    --------
    SO3LRPotential
        Configured SO3LR calculator
    """
    return SO3LRPotential(model_path=model_path, model_name=model_name, **kwargs)


# Mock SO3LR calculator for testing without dependencies
def get_mock_so3lr_calculator(**kwargs):
    """
    Get mock SO3LR calculator for testing.

    This function now returns a standardized MockSO3LRCalculator instead of
    an SO3LRPotential with mock enabled. This provides better separation
    between the real potential class and the mock implementation.

    Parameters
    ----------
    **kwargs
        Keyword arguments passed to MockSO3LRCalculator

    Returns
    -------
    MockSO3LRCalculator
        Mock calculator instance that simulates SO3LR behavior
    """
    from .mock_calculator import MockSO3LRCalculator

    return MockSO3LRCalculator(**kwargs)
