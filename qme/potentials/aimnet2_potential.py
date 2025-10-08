"""
AIMNET2 Machine Learning Potential integration for ASE.

This module implements a native AIMNet2 calculator without external dependencies,
based on the AIMNet2 repository implementation.
"""

import os
from typing import Any, Dict, Optional

import numpy as np
import requests
from ase.calculators.calculator import all_changes
from torch_cluster import radius_graph

from qme.dependencies import deps
from qme.potentials.base_potential import BasePotential

# Lazy torch import - will be None until needed
_torch = None


def _get_torch():
    """Get torch module, importing it lazily."""
    global _torch
    if _torch is None:
        _torch = deps.require("torch", purpose="AIMNet2 calculations")
    return _torch


# Create a module-level torch that's lazy
class _LazyTorch:
    def __getattr__(self, name):
        torch = _get_torch()
        return getattr(torch, name)


torch = _LazyTorch()

# Model registry - mapping model names to download URLs
MODEL_REGISTRY = {
    "aimnet2": "aimnet2/aimnet2_wb97m_0",
    "aimnet2_wb97m": "aimnet2/aimnet2_wb97m_0",
    "aimnet2_b973c": "aimnet2/aimnet2_b973c_0",
    "aimnet2-qr": "aimnet2-qr/aimnet2-qr_b97md4_qzvp_2",
}


def get_model_path(model_name: str) -> str:
    """Get the path to a model file, downloading if necessary.

    Parameters
    ----------
    model_name : str
        Name of the model or path to model file

    Returns
    -------
    str
        Path to the model file

    Notes
    -----
    If model_name is a file path, returns it directly. Otherwise, looks up
    the model in the registry and downloads it if necessary.
    """
    from qme.potentials.model_cache import download_and_cache_model

    # Direct file path
    if os.path.isfile(model_name):
        print(f"Found model file: {model_name}")
        return model_name

    # Check aliases
    if model_name in MODEL_REGISTRY:
        model_path = MODEL_REGISTRY[model_name]
    else:
        model_path = model_name

    # Add .jpt extension if needed
    if not model_path.endswith(".jpt"):
        model_path = model_path + ".jpt"

    # Try to use cached model first
    try:
        model_url = f"https://github.com/zubatyuk/aimnet-model-zoo/raw/main/{model_path}"
        cached_path = download_and_cache_model(model_name, model_url)
        return str(cached_path)
    except Exception as e:
        # Fallback to old behavior if caching fails
        print(f"Model caching failed, using fallback: {e}")

        # Create local assets directory
        assets_dir = os.path.join(os.path.dirname(__file__), "assets")
        model_dir = os.path.dirname(model_path)
        if model_dir:
            os.makedirs(os.path.join(assets_dir, model_dir), exist_ok=True)

        local_path = os.path.join(assets_dir, model_path)

        if os.path.isfile(local_path):
            print(f"Found model file: {local_path}")
            return local_path

        # Download from model zoo
        url = f"https://github.com/zubatyuk/aimnet-model-zoo/raw/main/{model_path}"
        print(f"Downloading model from {url}")

        try:
            response = requests.get(url)
            response.raise_for_status()

            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(response.content)

            print(f"Saved to {local_path}")
            return local_path

        except Exception as e:
            raise RuntimeError(f"Failed to download model {model_name}: {e}")


def sparse_nb_to_dense_half(idx, natom, max_nb):
    """Convert sparse neighbor list to dense format (from aimnet2calc)"""

    dense_nb = np.full((natom + 1, max_nb), natom, dtype=np.int32)
    last_idx = np.zeros((natom,), dtype=np.int32)
    for k in range(idx.shape[0]):
        i, j = idx[k]
        il, jl = last_idx[i], last_idx[j]
        dense_nb[i, il] = j
        dense_nb[j, jl] = i
        last_idx[i] += 1
        last_idx[j] += 1
    return dense_nb


def nblist_torch_cluster(coord, cutoff, mol_idx=None, max_nb=256):
    """Generate neighbor list using torch_cluster (from aimnet2calc)"""

    device = coord.device
    assert coord.ndim == 2, f"Expected 2D tensor for coord, got {coord.ndim}D"
    assert coord.shape[0] < 2147483646, "Too many atoms, max supported is 2147483646"

    max_num_neighbors = max_nb
    while True:
        sparse_nb = radius_graph(coord, batch=mol_idx, r=cutoff, max_num_neighbors=max_nb).to(
            torch.int32
        )
        nnb = torch.unique(sparse_nb[0], return_counts=True)[1]
        if nnb.numel() == 0:
            break
        max_num_neighbors = nnb.max().item()
        if max_num_neighbors < max_nb:
            break
        max_nb *= 2

    # Convert to dense format
    sparse_nb_half = sparse_nb[:, sparse_nb[0] > sparse_nb[1]]
    dense_nb = sparse_nb_to_dense_half(
        sparse_nb_half.mT.cpu().numpy(), coord.shape[0], max_num_neighbors
    )
    dense_nb = torch.as_tensor(dense_nb, device=device)
    return dense_nb


def maybe_pad_dim0(a, N, value=0.0):
    """Pad tensor in dimension 0 if needed (from aimnet2calc)"""
    _shape_diff = N - a.shape[0]
    assert _shape_diff == 0 or _shape_diff == 1, "Invalid shape"
    if _shape_diff == 1:
        a = pad_dim0(a, value=value)
    return a


def pad_dim0(a, value=0.0):
    """Pad tensor in dimension 0 (from aimnet2calc)"""
    shapes = [0] * ((a.ndim - 1) * 2) + [0, 1]
    a = torch.nn.functional.pad(a, shapes, mode="constant", value=value)
    return a


def maybe_unpad_dim0(a, N):
    """Unpad tensor in dimension 0 if needed (from aimnet2calc)"""
    _shape_diff = a.shape[0] - N
    assert _shape_diff == 0 or _shape_diff == 1, "Invalid shape"
    if _shape_diff == 1:
        a = a[:-1]
    return a


def generate_neighbor_list_torch_cluster(coord, cutoff, mol_idx=None, max_nb=256):
    """Generate neighbor list using torch_cluster radius_graph."""

    device = coord.device
    max_num_neighbors = 0

    # Generate sparse neighbor list using torch_cluster
    while True:
        sparse_nb = radius_graph(coord, batch=mol_idx, r=cutoff, max_num_neighbors=max_nb).to(
            torch.int32
        )
        nnb = torch.unique(sparse_nb[0], return_counts=True)[1]
        if nnb.numel() == 0:
            max_num_neighbors = 0
            break
        max_num_neighbors = nnb.max().item()
        if max_num_neighbors < max_nb:
            break
        max_nb *= 2

    # Convert to dense format using the half neighbor list
    sparse_nb_half = sparse_nb[:, sparse_nb[0] > sparse_nb[1]]
    dense_nb = sparse_nb_to_dense_half(
        sparse_nb_half.mT.cpu().numpy(), coord.shape[0], max_num_neighbors
    )
    dense_nb = torch.as_tensor(dense_nb, device=device)

    return dense_nb


def generate_neighbor_list_numpy(positions: np.ndarray, cutoff: float, max_neighbors: int = 128):
    """
    Fallback neighbor list generation using numpy.
    """
    n_atoms = len(positions)

    # Calculate pairwise distances
    distances = np.linalg.norm(positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=2)

    # Find neighbors within cutoff (excluding self-interactions with 0.1 Å threshold)
    neighbors = (distances < cutoff) & (distances > 0.1)

    # Count neighbors for each atom
    neighbor_counts = np.sum(neighbors, axis=1)
    max_nb_actual = np.max(neighbor_counts) if len(neighbor_counts) > 0 else 0
    max_nb = min(max_neighbors, max_nb_actual)

    # Create dense neighbor matrix with padding (N+1 x max_nb)
    nbmat = np.full((n_atoms + 1, max_nb), n_atoms, dtype=np.int64)

    for i in range(n_atoms):
        neighbor_indices = np.where(neighbors[i])[0]
        n_neighbors = min(len(neighbor_indices), max_nb)
        if n_neighbors > 0:
            nbmat[i, :n_neighbors] = neighbor_indices[:n_neighbors]

    return nbmat


class NativeAIMNet2Calculator:
    """
    Native AIMNet2 calculator implementation.
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        if not deps.has("torch"):
            raise ImportError("PyTorch is required for AIMNet2 calculations")

        # torch is now available globally as a lazy import

        self.device = device
        self.model = torch.jit.load(model_path, map_location=device)
        self.cutoff = self.model.cutoff

        # Check for long-range Coulomb support
        self.lr = hasattr(self.model, "cutoff_lr")
        self.cutoff_lr = getattr(self.model, "cutoff_lr", float("inf"))

        # Input/output key specifications
        self.keys_in = {
            "coord": torch.float,
            "numbers": torch.int,
            "charge": torch.float,
        }
        self.keys_in_optional = {
            "mult": torch.float,
            "mol_idx": torch.int,
            "nbmat": torch.int,
            "cell": torch.float,
        }
        self.keys_out = ["energy", "charges", "forces"]
        self.atom_feature_keys = ["coord", "numbers", "charges", "forces"]

        # State variables
        self._batch = None
        self._saved_for_grad = None

    def to_input_tensors(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert input data to PyTorch tensors."""
        ret = {}

        # Required keys
        for k in self.keys_in:
            if k not in data:
                raise ValueError(f"Missing required key '{k}' in input data")
            ret[k] = torch.as_tensor(data[k], device=self.device, dtype=self.keys_in[k]).detach()

        # Optional keys
        for k in self.keys_in_optional:
            if k in data and data[k] is not None:
                ret[k] = torch.as_tensor(
                    data[k], device=self.device, dtype=self.keys_in_optional[k]
                ).detach()

        # Ensure scalar tensors have shape (1,)
        for k, v in ret.items():
            if v.ndim == 0:
                ret[k] = v.unsqueeze(0)

        return ret

    def make_nbmat(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate neighbor lists following AIMNet2 repo logic."""
        # No PBC support in our implementation, so always use torch_cluster
        if "nbmat" not in data:
            data["nbmat"] = generate_neighbor_list_torch_cluster(
                data["coord"], self.cutoff, data.get("mol_idx"), max_nb=128
            )

            # Generate long-range neighbor list if model has long-range capabilities
            if self.lr:
                if "nbmat_lr" not in data:
                    data["nbmat_lr"] = generate_neighbor_list_torch_cluster(
                        data["coord"], self.cutoff_lr, data.get("mol_idx"), max_nb=1024
                    )
                data["cutoff_lr"] = torch.tensor(self.cutoff_lr, device=self.device)

        return data

    def prepare_mol_idx(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare molecule index for single molecule."""
        if "mol_idx" not in data:
            n_atoms = data["coord"].shape[0]
            data["mol_idx"] = torch.zeros(n_atoms, device=self.device, dtype=torch.int64)
        return data

    def pad_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Pad input tensors to match neighbor matrix dimensions using AIMNet2 logic."""
        if "nbmat" in data:
            N = data["nbmat"].shape[0]  # This includes the padding row, so it's N+1
            # Pad coord, numbers, and mol_idx to match the neighbor matrix size
            data["coord"] = maybe_pad_dim0(data["coord"], N)
            data["numbers"] = maybe_pad_dim0(data["numbers"], N)
            # For mol_idx, pad with the last molecule index
            last_mol_idx = data["mol_idx"][-1] if data["mol_idx"].numel() > 0 else 0
            data["mol_idx"] = maybe_pad_dim0(data["mol_idx"], N, value=last_mol_idx)

        return data

    def unpad_output(self, data: Dict[str, Any], original_n_atoms: int) -> Dict[str, Any]:
        """Remove padding from output tensors using AIMNet2 logic."""
        atom_feature_keys = ["coord", "numbers", "charges", "forces"]
        for k in atom_feature_keys:
            if k in data:
                data[k] = maybe_unpad_dim0(data[k], original_n_atoms)
        return data

    def set_grad_tensors(self, data: Dict[str, Any], forces: bool = False) -> Dict[str, Any]:
        """Set up gradients for force calculation."""
        self._saved_for_grad = {}
        if forces:
            data["coord"].requires_grad_(True)
            self._saved_for_grad["coord"] = data["coord"]
        return data

    def calculate_forces(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate forces using automatic differentiation."""
        if "forces" not in data and self._saved_for_grad and "coord" in self._saved_for_grad:
            energy = data["energy"].sum()
            grad = torch.autograd.grad(energy, self._saved_for_grad["coord"], create_graph=False)[0]
            data["forces"] = -grad
        return data

    def __call__(self, data: Dict[str, Any], forces: bool = False) -> Dict[str, Any]:
        """
        Calculate energy and optionally forces.

        Parameters:
        -----------
        data : dict
            Input data with keys 'coord', 'numbers', 'charge', optionally 'mult'
        forces : bool
            Whether to calculate forces

        Returns:
        --------
        dict
            Results with 'energy' and optionally 'forces'
        """
        # Store original number of atoms for unpadding later
        original_n_atoms = len(data["coord"]) if "coord" in data else 0

        # Convert to tensors
        data = self.to_input_tensors(data)

        # Prepare molecule indices
        data = self.prepare_mol_idx(data)

        # Generate neighbor list
        data = self.make_nbmat(data)

        # Pad inputs to match neighbor list dimensions
        data = self.pad_input(data)

        # Set up gradients if forces needed
        if forces:
            data = self.set_grad_tensors(data, forces=forces)

        # Run model inference
        with torch.jit.optimized_execution(False):
            data = self.model(data)

        # Calculate forces if requested
        if forces:
            data = self.calculate_forces(data)

        # Unpad outputs
        data = self.unpad_output(data, original_n_atoms)

        # Filter output keys
        result = {}
        for k, v in data.items():
            if k in self.keys_out:
                result[k] = v

        return result


class AIMNet2Potential(BasePotential):
    """ASE Calculator interface for AIMNET2 neural network potential.

    AIMNET2 provides accurate and versatile neural network potentials for
    molecular property prediction and geometry optimization, excelling at
    modeling neutral, charged, organic, and elemental-organic systems.

    Parameters
    ----------
    model_name : str, default "aimnet2"
        Name/path of AIMNET2 model to use
    device : str, optional
        Device for computations ('cpu', 'cuda'). Auto-detected if None.
    charge : int, default 0
        Total charge of the system
    mult : int, default 1
        Spin multiplicity (2S + 1)
    **kwargs
        Additional arguments passed to BasePotential
    """

    def __init__(
        self,
        model_name: str = "aimnet2",
        device: Optional[str] = None,
        charge: int = 0,
        mult: int = 1,
        **kwargs,
    ):
        """Initialize AIMNET2 potential calculator.

        Parameters
        ----------
        model_name : str, default "aimnet2"
            Name/path of AIMNET2 model to use
        device : str, optional
            Device for computations ('cpu', 'cuda'). Auto-detected if None.
        charge : int, default 0
            Molecular charge
        mult : int, default 1
            Spin multiplicity
        **kwargs
            Additional arguments passed to parent Calculator
        """

        # Check dependencies
        if not deps.has("torch"):
            raise ImportError(
                "PyTorch is required for AIMNET2 potentials. " "Install with: pip install torch"
            )

        # Set device if not provided
        if device is None:
            from qme.utils.device import get_optimal_device

            device = get_optimal_device()

        # Initialize base class
        super().__init__(model_name=model_name, device=device, **kwargs)

        # AIMNet2-specific attributes
        self.charge = charge
        self.mult = mult

        # Ensure results dict exists for ASE-style API
        self.results = {}

    # ASE-compatible properties (class attribute like other potentials)
    implemented_properties = ["energy", "forces"]

    def _load_calculator(self):
        """Load the AIMNET2 model directly."""
        from qme.logging_utils import quiet_backend_loading

        try:
            # Ensure model_name is not None
            if self.model_name is None:
                self.model_name = "aimnet2"

            # Ensure device is not None
            if self.device is None:
                self.device = "cpu"

            model_path = get_model_path(self.model_name)

            with quiet_backend_loading("aimnet2", self.model_name, model_path, self.device):
                self._calc = NativeAIMNet2Calculator(model_path, device=self.device)

        except Exception as e:
            raise RuntimeError(
                f"Failed to load AIMNET2 model '{self.model_name}'. "
                f"Error: {e}. Please check the model name or installation."
            )

    def calculate(
        self,
        atoms=None,
        properties=["energy", "forces"],
        system_changes=all_changes,
    ):
        """Calculate properties using AIMNET2 potential."""
        super().calculate(atoms, properties, system_changes)

        # Use self.atoms if atoms is None (standard ASE behavior)
        if atoms is None:
            atoms = self.atoms

        if atoms is None:
            raise ValueError("No atoms provided for calculation")

        # Prepare input data
        data = {
            "coord": atoms.positions,
            "numbers": atoms.numbers,
            "charge": float(self.charge),
            "mult": float(self.mult),
        }

        # Calculate with forces if requested
        # Ensure backend loaded
        if self._calc is None:
            self._load_calculator()

        forces_needed = "forces" in properties
        results = self._calc(data, forces=forces_needed)

        # Convert results to numpy arrays and store
        if "energy" in properties:
            energy = results["energy"].detach().cpu().numpy()
            if energy.ndim > 0:
                energy = float(energy.item()) if energy.size == 1 else float(energy[0])
            self.results["energy"] = energy

        if "forces" in properties and "forces" in results:
            forces = results["forces"].detach().cpu().numpy()
            self.results["forces"] = forces

    def set_charge(self, charge: int):
        """Set molecular charge."""
        self.charge = charge

    def set_mult(self, mult: int):
        """Set spin multiplicity."""
        self.mult = mult

    def _get_backend_name(self) -> str:
        """Get the backend name for AIMNet2."""
        return "aimnet2"

    def get_potential_energy(self, atoms=None, force_consistent: bool = False):
        """Get potential energy (ASE-compatible)."""
        if atoms is not None:
            self.atoms = atoms

        # Ensure calculator is loaded
        return super().get_potential_energy(atoms, force_consistent)

    def get_forces(self, atoms=None):
        """Get forces (ASE-compatible)."""
        if atoms is not None:
            self.atoms = atoms

        return super().get_forces(atoms)


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
