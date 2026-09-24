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

from famex.backends.dependencies import deps
from famex.potentials._load_utils import raise_backend_load_error
from famex.potentials.base_potential import BasePotential
from famex.utils.logging import get_famex_logger
from famex.utils.path_security import PathSecurityError, is_safe_relative_path, validate_safe_path

if TYPE_CHECKING:
    from ase import Atoms

logger = get_famex_logger(__name__)

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

    # Create local assets directory
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
        self._padded_coord: Any | None = None

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

    def _run_model(
        self,
        data: dict[str, Any],
        forces: bool = False,
        create_graph: bool = False,
    ) -> dict[str, Any]:
        """Run the AIMNet2 inference pipeline and optionally compute forces."""
        # Store original number of atoms for unpadding later
        original_n_atoms = len(data["coord"]) if "coord" in data else 0

        data = self.to_input_tensors(data)
        data = self.prepare_mol_idx(data)
        data = self.make_nbmat(data)
        data = self.pad_input(data)

        if forces:
            data["coord"].requires_grad_(True)
            self._padded_coord = data["coord"]

        with torch.jit.optimized_execution(False):
            data = self.model(data)

        if forces:
            energy = data["energy"].sum()
            grad = torch.autograd.grad(energy, data["coord"], create_graph=create_graph)[0]
            data["forces"] = -grad

        data = self.unpad_output(data, original_n_atoms)
        return data

    def calculate_hessian(self, data: dict[str, Any]) -> dict[str, Any]:
        """Calculate the analytical Hessian using double-backward autograd.

        Returns input dict updated with ``hessian`` (3N x 3N torch Tensor, eV/A^2).
        """
        data = self._run_model(data, forces=True, create_graph=True)

        forces = data["forces"]
        n = forces.shape[0]
        nf = 3 * n

        padded_coord = self._padded_coord
        forces_flat = forces.reshape(-1)
        rows = []
        for i in range(nf):
            (g,) = torch.autograd.grad(
                forces_flat[i].sum(),
                padded_coord,
                retain_graph=True,
                allow_unused=True,
            )
            if g is None:
                g = torch.zeros_like(padded_coord)
            rows.append((-g[:n]).reshape(-1))

        data["hessian"] = torch.stack(rows)
        return data

    def __call__(
        self,
        data: dict[str, Any],
        forces: bool = False,
        hessian: bool = False,
    ) -> dict[str, Any]:
        """Calculate energy and optionally forces / Hessian.

        Parameters
        ----------
        data : dict
            Input data with keys 'coord', 'numbers', 'charge', optionally 'mult'
        forces : bool
            Whether to calculate forces
        hessian : bool
            Whether to calculate the analytical Hessian (forces implied)

        Returns
        -------
        dict
            Results with 'energy' and optionally 'forces' and 'hessian'
        """
        if hessian:
            data = self.calculate_hessian(data)
        elif forces:
            data = self._run_model(data, forces=True, create_graph=False)
        else:
            data = self._run_model(data, forces=False, create_graph=False)

        out_keys = list(self.keys_out)
        if hessian:
            out_keys.append("hessian")

        result: dict[str, Any] = {}
        for k, v in data.items():
            if k in out_keys:
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
        if not deps.has("torch"):
            msg = "PyTorch is required for AIMNET2 potentials. Install with: pip install torch"
            raise ImportError(
                msg,
            )

        if device is None:
            from famex.utils.device import get_optimal_device

            device = get_optimal_device()

        self._calc: Any | None = None
        self.charge = charge
        self.mult = mult

        super().__init__(
            backend="aimnet2",
            model_name=model_name,
            device=device,
            **kwargs,
        )

    # ASE-compatible properties (class attribute like other potentials)
    implemented_properties = ["energy", "forces", "hessian"]

    def _load_calculator(self) -> None:
        """Load the AIMNET2 model directly."""
        from famex.utils.ml_warnings import quiet_backend_loading

        try:
            if self.model_name is None:
                self.model_name = "aimnet2"

            if self.device is None:
                self.device = "cpu"

            model_path = get_model_path(self.model_name)

            with quiet_backend_loading(
                "aimnet2",
                self.model_name,
                model_path,
                self.device,
                show_model_info=False,
            ):
                self._calc = NativeAIMNet2Calculator(model_path, device=self.device)

        except (ImportError, ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
            raise_backend_load_error("aimnet2", self.model_name, exc)

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

        atoms = atoms if atoms is not None else self.atoms
        if atoms is None:
            msg = "No atoms provided for calculation"
            raise ValueError(msg)

        data = {
            "coord": atoms.positions,
            "numbers": atoms.numbers,
            "charge": float(self.charge),
            "mult": float(self.mult),
        }

        hessian_needed = "hessian" in properties
        forces_needed = hessian_needed or "forces" in properties
        results = self._require_calc()(data, forces=forces_needed, hessian=hessian_needed)

        if "energy" in properties:
            energy = results["energy"].detach().cpu().numpy()
            if energy.ndim > 0:
                energy = float(energy.item()) if energy.size == 1 else float(energy[0])
            self.results["energy"] = energy

        if "forces" in properties and "forces" in results:
            self.results["forces"] = results["forces"].detach().cpu().numpy()

        if hessian_needed and "hessian" in results:
            hessian = results["hessian"].detach().cpu().numpy()
            self.results["hessian"] = 0.5 * (hessian + hessian.T)

    def set_charge(self, charge: int) -> None:
        """Set molecular charge."""
        self.charge = charge

    def set_mult(self, mult: int) -> None:
        """Set spin multiplicity."""
        self.mult = mult

    def get_forces(self, atoms: Atoms | None = None) -> np.ndarray:
        """Get forces."""
        forces = super().get_forces(atoms)
        if forces is None:
            msg = "Forces calculation returned None"
            raise RuntimeError(msg)
        return np.asarray(forces)

    def get_hessian(self, atoms: Atoms | None = None) -> np.ndarray:
        """Get analytical Hessian matrix (3N x 3N) from AIMNet2's autograd."""
        if atoms is not None:
            self.atoms = atoms

        calc = self._require_calc()

        if self.atoms is None:
            msg = "No atoms available for Hessian calculation"
            raise ValueError(msg)

        data = {
            "coord": self.atoms.positions,
            "numbers": self.atoms.numbers,
            "charge": float(self.charge),
            "mult": float(self.mult),
        }

        results = calc(data, forces=True, hessian=True)

        hessian: np.ndarray = np.asarray(
            results["hessian"].detach().cpu().numpy(), dtype=np.float64
        )
        expected_shape = (3 * len(self.atoms), 3 * len(self.atoms))
        if hessian.shape != expected_shape:
            if hessian.size == expected_shape[0] * expected_shape[1]:
                hessian = hessian.reshape(expected_shape)
            else:
                msg = f"Hessian has unexpected shape {hessian.shape}, expected {expected_shape}"
                raise ValueError(msg)

        hessian = 0.5 * (hessian + hessian.T)
        self.results["hessian"] = hessian
        return hessian

    def get_property(
        self, prop: str, atoms: Atoms | None = None, allow_calculation: bool = True
    ) -> Any:
        """Get a specific property (energy, forces, hessian) from the calculator."""
        if atoms is not None:
            self.atoms = atoms

        if prop == "energy":
            return self.get_potential_energy(atoms)
        if prop == "forces":
            return self.get_forces(atoms)
        if prop == "hessian":
            return self.get_hessian(atoms)
        msg = f"Property '{prop}' not supported by AIMNet2Potential"
        raise KeyError(msg)
