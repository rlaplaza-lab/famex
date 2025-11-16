"""UMA Machine Learning Potential integration for ASE."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from ase.calculators.calculator import all_changes

from qme.backends.dependencies import deps
from qme.potentials.base_potential import BasePotential
from qme.utils.logging import get_qme_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch
    from ase import Atoms
else:
    torch = None  # type: ignore[assignment, misc]

logger = get_qme_logger(__name__)


class UMAPotential(BasePotential):
    """ASE Calculator interface for UMA (Universal Model for Atoms) potential.

    This calculator provides an interface to use UMA machine learning potentials
    for molecular property prediction and geometry optimization.

    Supports analytical Hessian calculations via double back-propagation through
    the neural network for efficient frequency analysis.

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

    implemented_properties = ["energy", "forces", "hessian"]

    def __init__(
        self,
        model_name: str = "uma-s-1p1",
        device: str | None = None,
        default_charge: int = 0,
        default_spin: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize UMA potential calculator.

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
        self.predictor: Any = None
        self._calc: Any = None
        self.default_charge = default_charge
        self.default_spin = default_spin

        # Initialize base class (this will call _load_calculator)
        super().__init__(model_name=model_name, device=device, **kwargs)

    def _set_model_precision(self, precision: str = "float32") -> None:
        """Set model precision to avoid dtype mismatches.

        Parameters
        ----------
        precision : str
            Precision to set ('float32' or 'double')
        """
        if self.predictor is not None:
            if hasattr(self.predictor, "model"):
                model = self.predictor.model
                if precision == "float32" and hasattr(model, "float"):
                    model.float()
                elif precision == "double" and hasattr(model, "double"):
                    model.double()

    def _load_calculator(self) -> None:
        """Load the UMA model from fairchem v2 API."""
        # Skip if already loaded
        if hasattr(self, "_calc"):
            if self._calc is not None:
                return

        from qme.utils.ml_warnings import quiet_backend_loading

        # Don't show model info - let the outer context handle it
        with quiet_backend_loading(
            "uma",
            self.model_name,
            None,
            self.device,
            show_model_info=False,
        ):
            try:
                # Check fairchem availability without forcing PyTorch import
                if not deps.has("fairchem"):
                    msg = (
                        "fairchem-core is required for UMA potentials. "
                        "Install with: pip install fairchem-core"
                    )
                    raise ImportError(
                        msg,
                    )

                # Use the dependency system to get fairchem components (lazy-loaded)
                pretrained_mlip = deps.get("fairchem_pretrained_mlip")
                FAIRChemCalculator = deps.get("fairchem_calculator")

                if not pretrained_mlip or not FAIRChemCalculator:
                    msg = "FairChem v2 components not available"
                    raise RuntimeError(
                        msg,
                    )  # Load UMA model using v2 API
                # Ensure model_name is not None
                model_name = self.model_name or "uma-s-1p1"

                # Ensure device is compatible
                device_param = "cuda" if self.device == "cuda" else "cpu"

                self.predictor = pretrained_mlip.get_predict_unit(model_name, device=device_param)

                # Try to force consistent precision to avoid dtype mismatches
                self._set_model_precision("float32")

                # Create fairchem calculator for internal use
                # Default to 'omol' task for molecular systems
                self._calc = FAIRChemCalculator(self.predictor, task_name="omol")

            except ImportError as e:
                # Missing dependencies
                logger.error(
                    "Failed to load UMA model '%s': missing required dependencies. Error: %s",
                    self.model_name,
                    e,
                )
                msg = (
                    f"Failed to load UMA model '{self.model_name}': missing required dependencies. "
                    f"Error: {e}. Install fairchem-core and ensure all dependencies are available."
                )
                raise ImportError(msg) from e
            except (ValueError, TypeError, KeyError) as e:
                # Configuration or model format errors
                logger.error(
                    "Failed to load UMA model '%s': invalid model configuration. Error: %s",
                    self.model_name,
                    e,
                )
                msg = (
                    f"Failed to load UMA model '{self.model_name}': invalid model configuration. "
                    f"Error: {e}. Check that the model name is correct and the model format is valid."
                )
                raise ValueError(msg) from e
            except OSError as e:
                # File system errors
                logger.error(
                    "Failed to load UMA model '%s': file access error. Error: %s",
                    self.model_name,
                    e,
                )
                msg = (
                    f"Failed to load UMA model '{self.model_name}': file access error. "
                    f"Error: {e}. Check file permissions and ensure model files are accessible."
                )
                raise RuntimeError(msg) from e
            except RuntimeError as e:
                # Runtime errors from PyTorch/backend
                logger.error(
                    "Failed to load UMA model '%s': runtime error. Error: %s",
                    self.model_name,
                    e,
                )
                msg = (
                    f"Failed to load UMA model '{self.model_name}': runtime error. "
                    f"Error: {e}. This may indicate a device/GPU issue or model incompatibility."
                )
                raise RuntimeError(msg) from e

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: Sequence[str] | None = None,
        system_changes: Any = all_changes,
    ) -> None:
        """Calculate properties using UMA potential."""
        super().calculate(atoms, properties, system_changes)

        if atoms is None:
            msg = "atoms cannot be None"
            raise ValueError(msg)

        # Set default charge and spin if not already set to avoid warnings (ensure Python integers)
        if "charge" not in atoms.info:
            atoms.info["charge"] = int(self.default_charge)
        else:
            atoms.info["charge"] = int(atoms.info["charge"])
        if "spin" not in atoms.info:
            atoms.info["spin"] = int(self.default_spin)
        else:
            atoms.info["spin"] = int(atoms.info["spin"])

        # Ensure calculator is loaded
        if self._calc is None:
            self._load_calculator()
            # After loading, _calc should be set (or exception was raised)
            if self._calc is None:
                logger.error("Failed to load UMA calculator")
                msg = "Failed to load UMA calculator"
                raise RuntimeError(msg)

        # Use the underlying calculator directly
        try:
            self._calc.calculate(atoms, properties, system_changes)
        except RuntimeError as e:
            if "expected scalar type Double but found Float" in str(
                e,
            ) or "mat1 and mat2 must have the same dtype, but got Double and Float" in str(e):
                logger.debug("UMA dtype mismatch detected, adjusting precision and retrying")
                # Try to set model to use consistent precision
                if "expected scalar type Double but found Float" in str(e):
                    self._set_model_precision("double")
                else:
                    self._set_model_precision("float32")

                # Retry calculation
                self._calc.calculate(atoms, properties, system_changes)
            else:
                logger.exception("Unexpected error during UMA calculation")
                raise

        # Extract results from the underlying calculator
        if properties is not None:
            if "energy" in properties:
                self.results["energy"] = self._calc.results["energy"]

            if "forces" in properties:
                self.results["forces"] = self._calc.results["forces"]

    def _get_backend_name(self) -> str:
        """Get the backend name for UMA."""
        return "uma"

    def get_potential_energy(
        self,
        atoms: Atoms | None = None,
        force_consistent: bool = False,
    ) -> float:
        """Get potential energy (ASE-compatible)."""
        return super().get_potential_energy(atoms, force_consistent)

    def get_forces(self, atoms: Atoms | None = None) -> np.ndarray | None:
        """Get forces (ASE-compatible)."""
        return super().get_forces(atoms)

    def get_hessian(
        self,
        atoms: Atoms | None = None,
        method: str = "auto",
        symmetrize: bool = True,
    ) -> np.ndarray:
        """Get analytical Hessian matrix.

        Returns the Hessian matrix (3N x 3N) computed using PyTorch's automatic
        differentiation. Multiple computation methods are available to balance
        numerical stability and performance.

        Parameters
        ----------
        atoms : Atoms, optional
            Atoms object to calculate Hessian for
        method : str, default "auto"
            Method to use for Hessian computation:
            - 'vmap': Vector-Jacobian products with vectorization (original MACE-style)
            - 'double_backward': Direct double-backward from energy
            - 'fairchem' / 'fairchem_vmap': Match FairChem PR #1361 implementation (vmap variant)
            - 'fairchem_loop': FairChem PR style but without vmap
            - 'auto': Automatically select best method (currently 'double_backward')
        symmetrize : bool, default True
            Whether to symmetrize the Hessian by averaging with its transpose.
            The Hessian must be symmetric by definition, so this can reduce numerical noise.

        Returns
        -------
        np.ndarray
            Hessian matrix of shape (3N, 3N) in eV/Å² units where N is the number of atoms
        """
        if atoms is not None:
            self.atoms = atoms

        # Ensure calculator is loaded
        if self._calc is None:
            self._load_calculator()
            # After loading, _calc should be set (or exception was raised)
            if self._calc is None:
                logger.error("Failed to load UMA calculator for Hessian calculation")
                msg = "Failed to load UMA calculator"
                raise RuntimeError(msg)

        # Set default charge and spin if not already set (ensure Python integers)
        # BasePotential.calculate sets self.atoms, so it should not be None here
        assert self.atoms is not None, "atoms should be set by base class calculate method"
        if self.atoms is not None:
            if "charge" not in self.atoms.info:
                self.atoms.info["charge"] = int(self.default_charge)
            else:
                self.atoms.info["charge"] = int(self.atoms.info["charge"])
            if "spin" not in self.atoms.info:
                self.atoms.info["spin"] = int(self.default_spin)
            else:
                self.atoms.info["spin"] = int(self.atoms.info["spin"])

        try:
            import torch
            from fairchem.core.datasets import data_list_collater
            from fairchem.core.datasets.atomic_data import AtomicData

            # Ensure we have the predictor loaded
            if self.predictor is None:
                msg = "UMA predictor not loaded. Cannot calculate analytical Hessian."
                raise RuntimeError(msg)

            # Get device from predictor
            device = next(self.predictor.model.parameters()).device

            # Select method after handling 'auto'
            if method == "auto":
                method = "double_backward"  # Default to double_backward for better stability

            # Create AtomicData from current atoms
            atoms_copy = self.atoms.copy()
            atoms_copy.info["charge"] = int(self.atoms.info.get("charge", self.default_charge))
            atoms_copy.info["spin"] = int(self.atoms.info.get("spin", self.default_spin))

            # Convert to AtomicData format
            data = AtomicData.from_ase(
                atoms_copy,
                task_name="omol",
                r_edges=False,
                r_data_keys=["spin", "charge"],
            ).to(device)

            # Create batch
            batch = data_list_collater([data], otf_graph=True).to(device)

            # Enable gradients on positions - this is the key step
            # Note: otf_graph=True doesn't automatically set this, it's required for autograd
            # Do NOT detach/clone as that breaks the computation graph connection
            batch.pos.requires_grad_(True)

            # Match FairChem's approach: only set head.training = True
            # Do NOT set model.train() or manually disable dropout
            # This preserves the computation graph and matches the reference implementation
            # Match standalone helper exactly: direct access pattern
            model_module = self.predictor.model.module
            energy_wrapper = model_module.output_heads["energyandforcehead"]
            prev_head_training = energy_wrapper.head.training
            # Match standalone helper: set training=True for all methods
            # This ensures forces are computed with computation graph when needed
            # For double_backward, we'll compute forces ourselves from energy
            energy_wrapper.head.training = True

            # Compute energy and forces
            result = self.predictor.predict(batch)
            energy = result["energy"]

            # For double_backward, we need energy with computation graph
            # When head.training=True, predict() computes forces which might consume the graph
            # So we need to ensure we can still use the energy graph
            # The key is that we don't access result["forces"] for double_backward
            if method == "double_backward":
                # Direct double-backward approach
                # Energy should still have computation graph even if forces were computed
                # because we're not accessing result["forces"], so the graph isn't consumed
                hessian_tensor = self._compute_hessian_double_backward(energy, batch.pos)
            elif method == "vmap":
                # VJP approach (original)
                # Forces = -∂E/∂r
                forces = -torch.autograd.grad(
                    energy,
                    batch.pos,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                hessian_tensor = self._compute_hessian_vmap(forces, batch.pos)
            elif method in {"fairchem", "fairchem_vmap"}:
                hessian_tensor = self._compute_hessian_fairchem_style(
                    result["forces"],
                    batch.pos,
                    use_vmap=True,
                )
            elif method == "fairchem_loop":
                hessian_tensor = self._compute_hessian_fairchem_style(
                    result["forces"],
                    batch.pos,
                    use_vmap=False,
                )
            else:
                msg = (
                    "Unknown Hessian computation method: "
                    f"{method}. Use 'vmap', 'double_backward', 'fairchem', 'fairchem_loop', or 'auto'"
                )
                raise ValueError(msg)

            # Restore head training state (we only modified head.training, not model.train())
            energy_wrapper.head.training = prev_head_training

            n_atoms = len(self.atoms)
            expected_shape = (3 * n_atoms, 3 * n_atoms)

            # Ensure shape is correct
            if hessian_tensor.shape != expected_shape:
                # Try reshaping if total size matches
                expected_size = expected_shape[0] * expected_shape[1]
                if hessian_tensor.numel() == expected_size:
                    hessian_tensor = hessian_tensor.reshape(expected_shape)
                else:
                    msg = (
                        f"Hessian tensor has unexpected shape {hessian_tensor.shape}, "
                        f"expected {expected_shape}"
                    )
                    raise ValueError(msg)

            # Convert to numpy array
            hessian_np = hessian_tensor.detach().cpu().numpy()

            # Final shape check
            if hessian_np.shape != expected_shape:
                msg = f"Hessian has unexpected shape {hessian_np.shape}, expected {expected_shape}"
                raise ValueError(msg)

            # Optional symmetry check before symmetrization
            if symmetrize:
                import numpy as np

                # Check asymmetry before symmetrization (for diagnostics)
                asymmetry = np.abs(hessian_np - hessian_np.T)
                max_asymmetry = np.max(asymmetry)
                if max_asymmetry > 1e-5:  # Warn if significant asymmetry
                    import warnings

                    warnings.warn(
                        f"Hessian asymmetry detected (max deviation: {max_asymmetry:.2e}). "
                        "This suggests numerical noise. Symmetrization will be applied.",
                        UserWarning,
                        stacklevel=2,
                    )

            # Apply symmetrization if requested
            if symmetrize:
                hessian_np = self._symmetrize_hessian(hessian_np)

            return hessian_np

        except ImportError as e:
            msg = f"PyTorch is required for analytical Hessian calculation. Install PyTorch: {e}"
            raise ImportError(msg) from e
        except (ValueError, RuntimeError) as e:
            # Computation errors (invalid shapes, device mismatches, etc.)
            msg = (
                f"Failed to calculate UMA analytical Hessian: {e}. "
                f"This may indicate a device mismatch, invalid structure data, or computational error."
            )
            raise RuntimeError(msg) from e
        except TypeError as e:
            # Type errors (wrong tensor types, etc.)
            msg = (
                f"Failed to calculate UMA analytical Hessian: {e}. "
                f"This may indicate a type mismatch in tensor operations."
            )
            raise TypeError(msg) from e

    def _compute_hessian_fairchem_style(
        self,
        forces: torch.Tensor,
        positions: torch.Tensor,
        *,
        use_vmap: bool,
    ) -> torch.Tensor:
        """Replicate FairChem PR #1361 Hessian logic using PyTorch autograd."""
        import torch

        forces_flat = forces.view(-1)
        num_dofs = forces_flat.shape[0]
        dtype = forces_flat.dtype
        device = forces_flat.device

        def grad_wrt_positions(vec: torch.Tensor) -> torch.Tensor:
            grad_pos = torch.autograd.grad(
                -forces_flat,
                positions,
                grad_outputs=vec,
                retain_graph=True,
                allow_unused=False,
                create_graph=False,
            )[0]
            return grad_pos.reshape(-1)

        identity = torch.eye(num_dofs, dtype=dtype, device=device)

        if use_vmap and hasattr(torch, "vmap"):
            try:
                chunk_size = 1 if num_dofs < 64 else 16
                hessian = torch.vmap(
                    grad_wrt_positions,
                    in_dims=0,
                    out_dims=0,
                    chunk_size=chunk_size,
                )(identity)
            except RuntimeError:
                use_vmap = False

        if not use_vmap:
            rows = []
            for idx in range(num_dofs):
                vec = identity[idx]
                rows.append(grad_wrt_positions(vec))
            hessian = torch.stack(rows, dim=0)

        return cast(torch.Tensor, hessian)

    def _compute_hessian_vmap(self, forces: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Compute Hessian using vector-Jacobian products (VJP) with vectorization.

        This follows MACE's proven approach for efficient Hessian computation.

        Parameters
        ----------
        forces : torch.Tensor
            Forces array of shape (N, 3) where N is number of atoms
        positions : torch.Tensor
            Positions array of shape (N, 3) where N is number of atoms

        Returns
        -------
        torch.Tensor
            Hessian matrix of shape (3N, 3N)

        Raises
        ------
        RuntimeError
            If computation fails

        """
        import torch

        # Use current dtype from model outputs to avoid dtype mismatch
        forces_flatten = forces.view(-1)
        num_elements = forces_flatten.shape[0]
        n_atoms = forces.shape[0]

        def get_vjp(v: torch.Tensor) -> torch.Tensor:
            """Compute vector-Jacobian product for a single unit vector."""
            grad_output = torch.autograd.grad(
                -1 * forces_flatten,
                positions,
                grad_outputs=v,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )[0]
            # Flatten to (3N,) to match MACE's output shape
            return grad_output.view(-1)

        I_N = torch.eye(num_elements, dtype=forces.dtype, device=forces.device)

        # Try using vmap for efficient vectorized computation
        try:
            chunk_size = 1 if num_elements < 64 else 16
            # vmap over each row of identity matrix to compute gradient of each force component
            hessian = torch.vmap(get_vjp, in_dims=0, out_dims=0, chunk_size=chunk_size)(I_N)
        except RuntimeError:
            # Fallback to loop-based implementation if vmap fails
            hessian = self._compute_hessian_loop(forces, positions)

        if hessian is None:
            return torch.zeros(
                (3 * n_atoms, 3 * n_atoms),
                dtype=forces.dtype,
                device=forces.device,
            )

        # mypy doesn't understand that torch.vmap returns the same type as the function
        # Since get_vjp returns torch.Tensor and _compute_hessian_loop returns torch.Tensor,
        # hessian is guaranteed to be torch.Tensor here
        from typing import cast

        return cast(torch.Tensor, hessian)

    def _compute_hessian_loop(self, forces: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Compute Hessian using loop-based VJP (fallback for large systems).

        Parameters
        ----------
        forces : torch.Tensor
            Forces array of shape (N, 3)
        positions : torch.Tensor
            Positions array of shape (N, 3)

        Returns
        -------
        torch.Tensor
            Hessian matrix of shape (3N, 3N)

        """
        # Keep dtype consistent with model graph
        forces_flatten = forces.view(-1)
        num_elements = forces_flatten.shape[0]

        hessian = []
        for i in range(num_elements):
            # Create unit vector in position i
            v = torch.zeros(num_elements, dtype=forces.dtype, device=forces.device)
            v[i] = 1.0

            # Compute gradient of i-th force component with respect to positions
            hess_row = torch.autograd.grad(
                outputs=forces_flatten,
                inputs=positions,
                grad_outputs=-v,  # Negative sign for Hessian = -∂forces/∂positions
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )[0]
            # torch.autograd.grad with allow_unused=False should not return None
            hess_row = hess_row.detach()
            # Flatten to (3N,)
            hessian.append(hess_row.view(-1))

        return torch.stack(hessian)

    def _symmetrize_hessian(self, hessian: np.ndarray) -> np.ndarray:
        """Symmetrize Hessian matrix by averaging with its transpose.

        The Hessian matrix must be symmetric by definition (H = H^T), so this
        operation reduces numerical noise from accumulated floating-point errors.

        Parameters
        ----------
        hessian : np.ndarray
            Hessian matrix of shape (3N, 3N)

        Returns
        -------
        np.ndarray
            Symmetrized Hessian matrix of shape (3N, 3N)
        """
        from typing import cast

        return cast(np.ndarray, 0.5 * (hessian + hessian.T))

    def _compute_hessian_double_backward(
        self,
        energy: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Hessian using direct double-backward from energy.

        This method computes the Hessian by taking the gradient of forces
        with respect to positions in a single efficient backward pass.
        This can be more numerically stable than the VJP approach which
        requires multiple separate gradient computations.

        Parameters
        ----------
        energy : torch.Tensor
            Energy scalar tensor with gradients enabled
        positions : torch.Tensor
            Positions tensor of shape (N, 3) with gradients enabled

        Returns
        -------
        torch.Tensor
            Hessian matrix of shape (3N, 3N)

        Raises
        ------
        RuntimeError
            If computation fails
        """
        import torch

        # Keep dtype consistent with model graph (likely float32)
        n_atoms = positions.shape[0]
        num_elements = 3 * n_atoms

        # Compute forces = -∂E/∂r (with create_graph=True for second derivatives)
        forces = -torch.autograd.grad(
            energy,
            positions,
            create_graph=True,
            retain_graph=True,
        )[0]

        # Flatten forces to (3N,)
        forces_flat = forces.view(-1)

        # Compute Hessian = ∂²E/∂r² = -∂forces/∂positions
        # Compute each row of the Hessian by backward on each force component
        hessian_list = []
        for i in range(num_elements):
            # Compute gradient of i-th force component w.r.t. positions
            hess_row = torch.autograd.grad(
                forces_flat[i],
                positions,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )[0]
            # torch.autograd.grad with allow_unused=False should not return None
            # Negative sign: Hessian = -∂forces/∂positions
            hessian_list.append((-hess_row).view(-1))

        hessian = torch.stack(hessian_list)

        return hessian

    def get_property(
        self, prop: str, atoms: Atoms | None = None, allow_calculation: bool = True
    ) -> Any:
        """Get a specific property from the calculator.

        This method is used by ASE's property system and frequency analysis.

        Parameters
        ----------
        prop : str
            Property name ('energy', 'forces', 'hessian', etc.)
        atoms : Atoms, optional
            Atoms object to calculate property for

        Returns
        -------
        Any
            The requested property
        """
        if atoms is not None:
            self.atoms = atoms

        if prop == "energy":
            return self.get_potential_energy(atoms)
        elif prop == "forces":
            return self.get_forces(atoms)
        elif prop == "hessian":
            return self.get_hessian(atoms)
        else:
            msg = f"Property '{prop}' not supported by UMAPotential"
            raise KeyError(msg)


def get_uma_calculator(model_name: str = "uma-s-1p1", **kwargs: Any) -> UMAPotential:
    """Get UMA calculator.

    Parameters
    ----------
    model_name : str, default "uma-s-1p1"
        Name of UMA model to use
    **kwargs : Any
        Additional arguments passed to UMAPotential

    Returns
    -------
    UMAPotential
        Configured UMA calculator

    Examples
    --------
    >>> # Get default UMA calculator
    >>> calc = get_uma_calculator()

    >>> # Get specific model
    >>> calc = get_uma_calculator("uma-s-1p1", device="cuda")

    """
    return UMAPotential(model_name=model_name, **kwargs)
