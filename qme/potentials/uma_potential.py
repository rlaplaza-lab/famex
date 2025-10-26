"""UMA Machine Learning Potential integration for ASE."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ase.calculators.calculator import all_changes

from qme.backends.dependencies import deps
from qme.potentials.base_potential import BasePotential

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    import torch
    from ase import Atoms


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
        self.predictor = None
        self._calc = None
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
        if self.predictor is not None and hasattr(self.predictor, "model"):
            model = self.predictor.model
            if precision == "float32" and hasattr(model, "float"):
                model.float()
            elif precision == "double" and hasattr(model, "double"):
                model.double()

    def _load_calculator(self) -> None:
        """Load the UMA model from fairchem v2 API."""
        # Skip if already loaded
        if hasattr(self, "_calc") and self._calc is not None:
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

            except Exception as e:
                # If anything goes wrong while initializing the UMA model, raise a clear error
                msg = (
                    f"Failed to load UMA model '{self.model_name}'. Error: {e}. "
                    f"Make sure fairchem-core is properly installed and the model is available."
                )
                raise RuntimeError(
                    msg,
                )

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

        if self._calc is None:
            msg = "Failed to load UMA calculator"
            raise RuntimeError(msg)

        # Use the underlying calculator directly
        try:
            self._calc.calculate(atoms, properties, system_changes)
        except RuntimeError as e:
            if "expected scalar type Double but found Float" in str(
                e,
            ) or "mat1 and mat2 must have the same dtype, but got Double and Float" in str(e):
                # Try to set model to use consistent precision
                if "expected scalar type Double but found Float" in str(e):
                    self._set_model_precision("double")
                else:
                    self._set_model_precision("float32")

                # Retry calculation
                self._calc.calculate(atoms, properties, system_changes)
            else:
                raise

        # Extract results from the underlying calculator
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

    def get_hessian(self, atoms: Atoms | None = None) -> np.ndarray:
        """Get analytical Hessian matrix.

        Returns the Hessian matrix (3N x 3N) computed using PyTorch's automatic
        differentiation via vector-Jacobian products (VJP). This approach follows
        the proven MACE implementation for better memory efficiency and stability.
        The Hessian is returned directly from the VJP computation without symmetrization.

        Parameters
        ----------
        atoms : Atoms, optional
            Atoms object to calculate Hessian for

        Returns:
        -------
        np.ndarray
            Hessian matrix of shape (3N, 3N) in eV/Å² units where N is the number of atoms
        """
        if atoms is not None:
            self.atoms = atoms

        # Ensure calculator is loaded
        if self._calc is None:
            self._load_calculator()

        if self._calc is None:
            msg = "Failed to load UMA calculator"
            raise RuntimeError(msg)

        # Set default charge and spin if not already set (ensure Python integers)
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
            batch.pos = batch.pos.detach().clone().requires_grad_(True)

            # Set model to training mode for Hessian calculation
            self.predictor.model.train()

            # Disable dropout layers
            for module in self.predictor.model.modules():
                if hasattr(module, "p") and hasattr(module, "training"):
                    module.p = 0.0

            # Compute energy and forces
            result = self.predictor.predict(batch)
            energy = result["energy"].double()

            # Forces = -∂E/∂r
            forces = -torch.autograd.grad(
                energy,
                batch.pos,
                create_graph=True,
                retain_graph=True,
            )[0]

            # Compute Hessian using VJP approach (MACE-style)
            hessian_tensor = self._compute_hessian_vmap(forces, batch.pos)

            # Set model back to eval mode
            self.predictor.model.eval()

            # VJP now returns (3N, 3N) directly
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

            return hessian_np

        except ImportError as e:
            msg = f"PyTorch is required for analytical Hessian calculation. Install PyTorch: {e}"
            raise ImportError(msg) from e
        except Exception as e:
            msg = f"Failed to calculate UMA analytical Hessian: {e}"
            raise RuntimeError(msg) from e

    def _compute_hessian_vmap(self, forces: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Compute Hessian using vector-Jacobian products (VJP) with vectorization.

        This follows MACE's proven approach for efficient Hessian computation.

        Parameters
        ----------
        forces : torch.Tensor
            Forces array of shape (N, 3) where N is number of atoms
        positions : torch.Tensor
            Positions array of shape (N, 3) where N is number of atoms

        Returns:
        -------
        torch.Tensor
            Hessian matrix of shape (3N, 3N)

        Raises:
        ------
        RuntimeError
            If computation fails

        """
        import torch

        forces_flatten = forces.view(-1)
        num_elements = forces_flatten.shape[0]
        n_atoms = forces.shape[0]

        def get_vjp(v):
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

        return hessian

    def _compute_hessian_loop(self, forces: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Compute Hessian using loop-based VJP (fallback for large systems).

        Parameters
        ----------
        forces : torch.Tensor
            Forces array of shape (N, 3)
        positions : torch.Tensor
            Positions array of shape (N, 3)

        Returns:
        -------
        torch.Tensor
            Hessian matrix of shape (3N, 3N)

        """
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
            hess_row = hess_row.detach()

            if hess_row is None:
                hessian.append(torch.zeros_like(positions).view(-1))
            else:
                # Flatten to (3N,)
                hessian.append(hess_row.view(-1))

        return torch.stack(hessian)

    def get_property(self, prop: str, atoms: Atoms | None = None) -> Any:
        """Get a specific property from the calculator.

        This method is used by ASE's property system and frequency analysis.

        Parameters
        ----------
        prop : str
            Property name ('energy', 'forces', 'hessian', etc.)
        atoms : Atoms, optional
            Atoms object to calculate property for

        Returns:
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
    """Convenience function to get UMA calculator.

    Parameters
    ----------
    model_name : str, default "uma-s-1p1"
        Name of UMA model to use
    **kwargs : Any
        Additional arguments passed to UMAPotential

    Returns:
    -------
    UMAPotential
        Configured UMA calculator

    Examples:
    --------
    >>> # Get default UMA calculator
    >>> calc = get_uma_calculator()

    >>> # Get specific model
    >>> calc = get_uma_calculator("uma-s-1p1", device="cuda")

    """
    return UMAPotential(model_name=model_name, **kwargs)
