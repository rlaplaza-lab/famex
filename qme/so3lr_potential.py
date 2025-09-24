"""
SO3LR Neural Network Potential integration for ASE.

SO3LR is an open source neural network potential with SO(3) invariant architecture.
This module provides ASE Calculator interface for SO3LR models.
"""

from typing import Optional

from ase.calculators.calculator import Calculator, all_changes

from .dependencies import HAS_SO3LR, HAS_TORCH, deps, torch


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

        Calculator.__init__(self, **kwargs)

        # Store parameters
        self.model_path = model_path
        self.model_name = model_name

        # Check dependencies
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch is required for SO3LR potentials. "
                "Install with: pip install torch"
            )

        if not HAS_SO3LR:
            raise ImportError(
                "SO3LR is required for SO3LR potentials. "
                "Install SO3LR with: git clone "
                "https://github.com/general-molecular-simulations/so3lr.git && "
                "cd so3lr && pip install ."
            )

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize model
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the SO3LR model."""
        # Get SO3LR module
        so3lr = deps.get("so3lr")
        if so3lr is None:
            raise RuntimeError("SO3LR module not available")

        if self.model_path:
            # Load model from file
            self.model = so3lr.load_model(self.model_path)
        else:
            # Load pre-trained model
            self.model = so3lr.get_pretrained_model(self.model_name)

        self.model.to(self.device)
        self.model.eval()

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
                elif hasattr(energy, "cpu"):
                    self.results["energy"] = float(energy.cpu().numpy().item())

        if "forces" in properties:
            forces = outputs["forces"]
            if forces is not None:
                self.results["forces"] = forces.cpu().numpy()

    def _atoms_to_data(self, atoms):
        """Convert ASE Atoms object to format expected by SO3LR model."""
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

    def get_calculator(self):
        """Get the calculator instance.

        For SO3LR, this returns self since SO3LRPotential is itself the calculator.

        Returns:
        --------
        SO3LRPotential
            The calculator instance that can be used with ASE
        """
        return self


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
