"""
UMA Machine Learning Potential integration for ASE.
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
    from fairchem.core.common.utils import build_config
    from fairchem.core.models import model_registry
    from fairchem.core.trainers import ForcesTrainer

    HAS_FAIRCHEM = True
except ImportError:
    HAS_FAIRCHEM = False


class UMAPotential(Calculator):
    """
    ASE Calculator interface for UMA (Universal Model for Atoms) potential.

    This calculator provides an interface to use UMA machine learning potentials
    for molecular property prediction and geometry optimization.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self, model_name: str = "uma-4m", device: Optional[str] = None, **kwargs
    ):
        """
        Initialize UMA potential calculator.

        Parameters:
        -----------
        model_name : str
            Name of the UMA model to load (default: "uma-4m")
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

        Calculator.__init__(self, **kwargs)

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load UMA model
        self.model_name = model_name
        self.model = None
        self.trainer = None
        self._load_model()

    def _load_model(self):
        """Load the UMA model from fairchem."""
        try:
            # Try to load UMA model from fairchem registry
            if self.model_name in model_registry.MODEL_REGISTRY:
                model_class = model_registry.MODEL_REGISTRY[self.model_name]
                self.model = model_class()
            else:
                # Fallback: try to load as a pre-trained checkpoint
                config = build_config({"model": self.model_name, "trainer": "forces"})
                self.trainer = ForcesTrainer(**config)
                self.model = self.trainer.model

            self.model.to(self.device)
            self.model.eval()

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

        Calculator.calculate(self, atoms, properties, system_changes)

        # Convert atoms to format expected by UMA
        data = self._atoms_to_data(atoms)

        # Run prediction
        with torch.no_grad():
            if self.trainer is not None:
                outputs = self.trainer.predict(data)
            else:
                outputs = self.model(data)

        # Extract results
        if "energy" in properties:
            energy = outputs.get("energy", outputs.get("total_energy"))
            if energy is not None:
                self.results["energy"] = float(energy.cpu().numpy())

        if "forces" in properties:
            forces = outputs.get("forces")
            if forces is not None:
                self.results["forces"] = forces.cpu().numpy()

    def _atoms_to_data(self, atoms):
        """Convert ASE Atoms object to format expected by UMA model."""

        if not HAS_TORCH:
            raise ImportError("PyTorch is required")

        positions = torch.tensor(
            atoms.positions, dtype=torch.float32, device=self.device
        )
        atomic_numbers = torch.tensor(
            atoms.numbers, dtype=torch.long, device=self.device
        )

        # Create data dictionary expected by fairchem models
        data = {
            "pos": positions,
            "atomic_numbers": atomic_numbers,
            "natoms": torch.tensor([len(atoms)], device=self.device),
            "batch": torch.zeros(len(atoms), dtype=torch.long, device=self.device),
        }

        # Add cell information if periodic
        if atoms.pbc.any():
            cell = torch.tensor(
                atoms.cell.array, dtype=torch.float32, device=self.device
            )
            data["cell"] = cell.unsqueeze(0)
            data["pbc"] = torch.tensor(atoms.pbc, device=self.device)

        return data


def get_uma_calculator(model_name: str = "uma-4m", **kwargs) -> UMAPotential:
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
