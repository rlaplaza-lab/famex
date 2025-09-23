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
        """

        if not HAS_TORCH:
            raise ImportError(
                "PyTorch is required for SO3LR potentials. "
                "Install with: pip install torch"
            )

        Calculator.__init__(self, **kwargs)

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Store model configuration
        self.model_path = model_path
        self.model_name = model_name
        self.model = None

        self._load_model()

    def _load_model(self):
        """Load the SO3LR model."""
        if not HAS_SO3LR:
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

        # Create a simple mock model that behaves like SO3LR
        class MockSO3LRModel:
            def __init__(self, device):
                self.device = device
                self.cutoff = 5.0  # Cutoff radius in Angstroms

            def to(self, device):
                self.device = device
                return self

            def eval(self):
                pass

            def __call__(self, data):
                # Mock SO3LR inference - returns simple harmonic-like energies
                positions = data["positions"]
                n_atoms = len(positions)

                # Simple Lennard-Jones-like potential
                energy = 0.0
                forces = torch.zeros_like(positions)

                for i in range(n_atoms):
                    for j in range(i + 1, n_atoms):
                        r_vec = positions[j] - positions[i]
                        r = torch.norm(r_vec)

                        if r < self.cutoff:
                            # Simple LJ-like potential:
                            # 4*epsilon*[(sigma/r)^12 - (sigma/r)^6]
                            sigma = 1.5  # Angstroms
                            epsilon = 0.1  # eV

                            sigma_over_r = sigma / r
                            sigma6 = sigma_over_r**6
                            sigma12 = sigma6**2

                            energy += 4 * epsilon * (sigma12 - sigma6)

                            # Force calculation
                            force_magnitude = 24 * epsilon * (2 * sigma12 - sigma6) / r
                            force_vec = force_magnitude * r_vec / r

                            forces[i] -= force_vec
                            forces[j] += force_vec

                return {"energy": energy.unsqueeze(0), "forces": forces}

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
        with torch.no_grad():
            outputs = self.model(data)

        # Extract results
        if "energy" in properties:
            energy = outputs["energy"]
            if energy is not None:
                # Handle tensor conversion properly
                if hasattr(energy, "item"):
                    self.results["energy"] = energy.item()
                else:
                    self.results["energy"] = float(energy.cpu().numpy().item())

        if "forces" in properties:
            forces = outputs["forces"]
            if forces is not None:
                self.results["forces"] = forces.cpu().numpy()

    def _atoms_to_data(self, atoms):
        """Convert ASE Atoms object to format expected by SO3LR model."""

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

    Returns:
    --------
    SO3LRPotential
        SO3LR calculator with mock implementation
    """
    calc = SO3LRPotential(**kwargs)
    calc._use_mock_implementation()
    return calc
