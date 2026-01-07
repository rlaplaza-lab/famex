"""AIMNET2 Machine Learning Potential integration for ASE.

This module implements a native AIMNet2 calculator without external dependencies,
based on the AIMNet2 repository implementation.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import requests
from ase.calculators.calculator import all_changes

try:
    # Optional dependency. We can fall back to a NumPy neighbor list when
    # torch_cluster is unavailable or its CUDA kernels are incompatible.
    from torch_cluster import radius_graph  # type: ignore
except Exception:  # pragma: no cover
    radius_graph = None

from qme.backends.dependencies import deps
from qme.potentials.base_potential import BasePotential
from qme.utils.logging import get_qme_logger
from qme.utils.path_security import PathSecurityError, is_safe_relative_path, validate_safe_path

if TYPE_CHECKING:
    from ase import Atoms

logger = get_qme_logger(__name__)

# Lazy torch import - will be None until needed
_torch = None


def _get_torch() -> Any:
    """Get torch module, importing it lazily."""
    global _torch
    if _torch is None:
        _torch = deps.require("torch", purpose="AIMNet2 calculations")
    return _torch


# Create a module-level torch that's lazy
class _LazyTorch:
    """Lazy proxy for torch module to defer import until needed."""

    def __getattr__(self, name: str) -> Any:
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
    try:
        from qme.potentials.model_cache import (  # type: ignore[import-not-found]
            download_and_cache_model,
        )
    except ImportError:
        download_and_cache_model = None

    # Security check: reject path traversal attempts and absolute paths
    if not is_safe_relative_path(model_name):
        if os.path.isabs(model_name):
            msg = f"Absolute paths not allowed for model: {model_name}"
            raise PathSecurityError(msg)
        # Check for path traversal patterns
        if ".." in model_name or "~" in model_name:
            msg = f"Unsafe model path detected (path traversal attempt): {model_name}"
            raise PathSecurityError(msg)

    # Direct file path
    if os.path.isfile(model_name):
        logger.info(f"Found model file: {model_name}")
        return model_name

    # Check aliases
    model_path = MODEL_REGISTRY.get(model_name, model_name)

    # Add .jpt extension if needed
    if not model_path.endswith(".jpt"):
        model_path = model_path + ".jpt"

    # Try to use cached model first
    if download_and_cache_model is not None:
        try:
            model_url = f"https://github.com/zubatyuk/aimnet-model-zoo/raw/main/{model_path}"
            cached_path = download_and_cache_model(model_name, model_url)
            return str(cached_path)
        except (OSError, ValueError, TypeError) as e:
            # File system/cache errors during caching - fallback to old behavior
            logger.debug(f"Model caching failed, using fallback: {e}")

    # Create local assets directory (fallback behavior)
    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    assets_dir_path = Path(assets_dir)

    # SECURITY: Validate model_path doesn't escape assets directory
    # Resolve relative to assets_dir to prevent path traversal
    try:
        local_path = validate_safe_path(
            assets_dir_path / model_path,
            base_dir=assets_dir_path,
            must_exist=False,
            allow_absolute=False,
        )
    except Exception as e:
        msg = f"Invalid model path {model_path}: {e}"
        raise RuntimeError(msg) from e

    # Create parent directory if needed
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.is_file():
        logger.info(f"Found model file: {local_path}")
        return str(local_path)

    # Download from model zoo
    url = f"https://github.com/zubatyuk/aimnet-model-zoo/raw/main/{model_path}"
    logger.info(f"Downloading model from {url}")

    try:
        response = requests.get(url)
        response.raise_for_status()

        with open(local_path, "wb") as f:
            f.write(response.content)

        logger.info(f"Saved to {local_path}")
        return str(local_path)

    except requests.RequestException as e:
        # Network errors (connection, timeout, HTTP errors)
        msg = (
            f"Failed to download model {model_name}: network error. "
            f"Error: {e}. Check your internet connection and the model URL."
        )
        raise RuntimeError(msg) from e
    except OSError as e:
        # File system errors (permissions, disk space, etc.)
        msg = (
            f"Failed to download model {model_name}: file system error. "
            f"Error: {e}. Check file permissions and available disk space."
        )
        raise RuntimeError(msg) from e
    except (ValueError, TypeError) as e:
        # Data format or configuration errors
        msg = (
            f"Failed to download model {model_name}: invalid data format. "
            f"Error: {e}. The downloaded model may be corrupted."
        )
        raise RuntimeError(msg) from e


def sparse_nb_to_dense_half(idx: np.ndarray, natom: int, max_nb: int) -> np.ndarray:
    """Convert sparse neighbor list to dense format (from aimnet2calc)."""
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


def nblist_torch_cluster(coord: Any, cutoff: float, mol_idx: Any = None, max_nb: int = 256) -> Any:
    """Generate neighbor list using torch_cluster (from aimnet2calc)."""
    if radius_graph is None:
        raise ImportError("torch_cluster is not available")
    device = coord.device
    assert coord.ndim == 2, f"Expected 2D tensor for coord, got {coord.ndim}D"
    assert coord.shape[0] < 2147483646, "Too many atoms, max supported is 2147483646"

    max_num_neighbors = max_nb
    while True:
        sparse_nb = radius_graph(coord, batch=mol_idx, r=cutoff, max_num_neighbors=max_nb).to(
            torch.int32,
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
        sparse_nb_half.mT.cpu().numpy(),
        coord.shape[0],
        max_num_neighbors,
    )
    return torch.as_tensor(dense_nb, device=device)


def maybe_pad_dim0(a: Any, n: int, value: float = 0.0) -> Any:
    """Pad tensor in dimension 0 if needed (from aimnet2calc)."""
    _shape_diff = n - a.shape[0]
    assert _shape_diff in {0, 1}, "Invalid shape"
    if _shape_diff == 1:
        a = pad_dim0(a, value=value)
    return a


def pad_dim0(a: Any, value: float = 0.0) -> Any:
    """Pad tensor in dimension 0 (from aimnet2calc)."""
    shapes = [0] * ((a.ndim - 1) * 2) + [0, 1]
    return torch.nn.functional.pad(a, shapes, mode="constant", value=value)


def maybe_unpad_dim0(a: Any, n: int) -> Any:
    """Unpad tensor in dimension 0 if needed (from aimnet2calc)."""
    _shape_diff = a.shape[0] - n
    assert _shape_diff in {0, 1}, "Invalid shape"
    if _shape_diff == 1:
        a = a[:-1]
    return a


def generate_neighbor_list_torch_cluster(
    coord: Any, cutoff: float, mol_idx: Any = None, max_nb: int = 256
) -> Any:
    """Generate neighbor list using torch_cluster radius_graph."""
    if radius_graph is None:
        raise ImportError("torch_cluster is not available")
    device = coord.device
    max_num_neighbors = 0

    # Generate sparse neighbor list using torch_cluster
    while True:
        sparse_nb = radius_graph(coord, batch=mol_idx, r=cutoff, max_num_neighbors=max_nb).to(
            torch.int32,
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
        sparse_nb_half.mT.cpu().numpy(),
        coord.shape[0],
        max_num_neighbors,
    )
    return torch.as_tensor(dense_nb, device=device)


def generate_neighbor_list_numpy(
    positions: np.ndarray, cutoff: float, max_neighbors: int = 128
) -> np.ndarray:
    """Fallback neighbor list generation using numpy."""
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
    """Native AIMNet2 calculator implementation."""

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        if not deps.has("torch"):
            msg = "PyTorch is required for AIMNet2 calculations"
            raise ImportError(msg)

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
        self._saved_for_grad: dict[str, Any] | None = None
        self._warned_nbmat_fallback = False

    def to_input_tensors(self, data: dict[str, Any]) -> dict[str, Any]:
        """Convert input data to PyTorch tensors."""
        ret = {}

        # Required keys
        for k in self.keys_in:
            if k not in data:
                msg = f"Missing required key '{k}' in input data"
                raise ValueError(msg)
            ret[k] = torch.as_tensor(data[k], device=self.device, dtype=self.keys_in[k]).detach()

        # Optional keys
        for k in self.keys_in_optional:
            if k in data and data[k] is not None:
                ret[k] = torch.as_tensor(
                    data[k],
                    device=self.device,
                    dtype=self.keys_in_optional[k],
                ).detach()

        # Ensure scalar tensors have shape (1,)
        for k, v in ret.items():
            if v.ndim == 0:
                ret[k] = v.unsqueeze(0)

        return ret

    def make_nbmat(self, data: dict[str, Any]) -> dict[str, Any]:
        """Generate neighbor lists following AIMNet2 repo logic."""
        # No PBC support in our implementation.
        #
        # IMPORTANT: AIMNet2 must work on very new GPUs where optional CUDA extensions
        # (e.g., torch_cluster) may not ship compatible kernels yet. To keep the backend
        # robust, we default to a pure-NumPy neighbor list (O(N^2), but small molecules
        # make this fine for BH28). If torch_cluster is available and working, users can
        # still benefit elsewhere, but we won't require it for correctness.
        if "nbmat" not in data:
            coord_cpu = data["coord"].detach().cpu().numpy()
            nbmat_np = generate_neighbor_list_numpy(
                coord_cpu,
                float(self.cutoff),
                max_neighbors=128,
            )
            data["nbmat"] = torch.as_tensor(
                nbmat_np,
                device=self.device,
                dtype=torch.int32,
            )

            # Generate long-range neighbor list if model has long-range capabilities
            if self.lr:
                if "nbmat_lr" not in data:
                    nbmat_lr_np = generate_neighbor_list_numpy(
                        coord_cpu,
                        float(self.cutoff_lr),
                        max_neighbors=1024,
                    )
                    data["nbmat_lr"] = torch.as_tensor(
                        nbmat_lr_np,
                        device=self.device,
                        dtype=torch.int32,
                    )
                data["cutoff_lr"] = torch.tensor(self.cutoff_lr, device=self.device)

        return data

    def prepare_mol_idx(self, data: dict[str, Any]) -> dict[str, Any]:
        """Prepare molecule index for single molecule."""
        if "mol_idx" not in data:
            n_atoms = data["coord"].shape[0]
            data["mol_idx"] = torch.zeros(n_atoms, device=self.device, dtype=torch.int64)
        return data

    def pad_input(self, data: dict[str, Any]) -> dict[str, Any]:
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

    def unpad_output(self, data: dict[str, Any], original_n_atoms: int) -> dict[str, Any]:
        """Remove padding from output tensors using AIMNet2 logic."""
        atom_feature_keys = ["coord", "numbers", "charges", "forces"]
        for k in atom_feature_keys:
            if k in data:
                data[k] = maybe_unpad_dim0(data[k], original_n_atoms)
        return data

    def set_grad_tensors(self, data: dict[str, Any], forces: bool = False) -> dict[str, Any]:
        """Set up gradients for force calculation."""
        self._saved_for_grad = {}
        if forces:
            data["coord"].requires_grad_(True)
            self._saved_for_grad["coord"] = data["coord"]
        return data

    def calculate_forces(self, data: dict[str, Any]) -> dict[str, Any]:
        """Calculate forces using automatic differentiation."""
        if "forces" not in data and self._saved_for_grad and "coord" in self._saved_for_grad:
            energy = data["energy"].sum()
            grad = torch.autograd.grad(energy, self._saved_for_grad["coord"], create_graph=False)[0]
            data["forces"] = -grad
        return data

    def __call__(self, data: dict[str, Any], forces: bool = False) -> dict[str, Any]:
        """Calculate energy and optionally forces.

        Parameters
        ----------
        data : dict
            Input data with keys 'coord', 'numbers', 'charge', optionally 'mult'
        forces : bool
            Whether to calculate forces

        Returns
        -------
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
        device: str | None = None,
        charge: int = 0,
        mult: int = 1,
        **kwargs: Any,
    ) -> None:
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
            msg = "PyTorch is required for AIMNET2 potentials. Install with: pip install torch"
            raise ImportError(
                msg,
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

    def _load_calculator(self) -> None:
        """Load the AIMNET2 model directly."""
        from qme.utils.ml_warnings import quiet_backend_loading

        try:
            # Ensure model_name is not None
            if self.model_name is None:
                self.model_name = "aimnet2"

            # Ensure device is not None
            if self.device is None:
                self.device = "cpu"

            model_path = get_model_path(self.model_name)

            # Don't show model info - let the outer context handle it
            with quiet_backend_loading(
                "aimnet2",
                self.model_name,
                model_path,
                self.device,
                show_model_info=False,
            ):
                self._calc = NativeAIMNet2Calculator(model_path, device=self.device)

        except ImportError as e:
            # Missing dependencies
            msg = (
                f"Failed to load AIMNET2 model '{self.model_name}': missing required dependencies. "
                f"Error: {e}. Install torch and ensure all dependencies are available."
            )
            raise ImportError(msg) from e
        except (ValueError, TypeError, KeyError) as e:
            # Configuration or model format errors
            msg = (
                f"Failed to load AIMNET2 model '{self.model_name}': invalid model configuration. "
                f"Error: {e}. Check that the model name is correct and the model format is valid."
            )
            raise ValueError(msg) from e
        except OSError as e:
            # File system errors
            msg = (
                f"Failed to load AIMNET2 model '{self.model_name}': file access error. "
                f"Error: {e}. Check file permissions and ensure model files are accessible."
            )
            raise RuntimeError(msg) from e
        except RuntimeError as e:
            # Runtime errors from PyTorch/backend
            msg = (
                f"Failed to load AIMNET2 model '{self.model_name}': runtime error. "
                f"Error: {e}. This may indicate a device/GPU issue or model incompatibility."
            )
            raise RuntimeError(msg) from e

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: Sequence[str] | None = None,
        system_changes: Any = all_changes,
    ) -> None:
        """Calculate properties using AIMNET2 potential."""
        if properties is None:
            properties = ["energy", "forces"]
        super().calculate(atoms, properties, system_changes)

        # Use self.atoms if atoms is None (standard ASE behavior)
        if atoms is None:
            atoms = self.atoms

        if atoms is None:
            msg = "No atoms provided for calculation"
            raise ValueError(msg)

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
            self._load_calculator()  # type: ignore[unreachable]
        # After _load_calculator() returns without exception, _calc is guaranteed to be set
        assert self._calc is not None

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

    def set_charge(self, charge: int) -> None:
        """Set molecular charge."""
        self.charge = charge

    def set_mult(self, mult: int) -> None:
        """Set spin multiplicity."""
        self.mult = mult

    def _get_backend_name(self) -> str:
        """Get the backend name for AIMNet2."""
        return "aimnet2"

    def get_potential_energy(self, atoms: Any = None, force_consistent: bool = False) -> float:
        """Get potential energy (ASE-compatible)."""
        if atoms is not None:
            self.atoms = atoms

        # Ensure calculator is loaded
        return super().get_potential_energy(atoms, force_consistent)

    def get_forces(self, atoms: Atoms | None = None) -> np.ndarray:
        """Get forces (ASE-compatible)."""
        from typing import cast

        import numpy as np

        if atoms is not None:
            self.atoms = atoms

        forces = super().get_forces(atoms)
        if forces is None:
            msg = "Forces calculation returned None"
            raise RuntimeError(msg)
        return cast(np.ndarray, forces)


def get_aimnet2_calculator(
    model_name: str = "aimnet2",
    device: str | None = None,
    charge: int = 0,
    mult: int = 1,
    **kwargs: Any,
) -> AIMNet2Potential:
    """Get AIMNET2 calculator.

    Parameters
    ----------
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

    Returns
    -------
    AIMNet2Potential
        Configured AIMNET2 calculator

    """
    return AIMNet2Potential(
        model_name=model_name,
        device=device,
        charge=charge,
        mult=mult,
        **kwargs,
    )
