"""UMA Machine Learning Potential integration for ASE."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from ase.calculators.calculator import all_changes

from famex.backends.constants import DEFAULT_UMA_MODEL, FAIRCHEM_INSTALL
from famex.backends.dependencies import deps
from famex.potentials._load_utils import raise_backend_load_error
from famex.potentials.base_potential import BasePotential
from famex.utils.logging import get_famex_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ase import Atoms

logger = get_famex_logger(__name__)


class UMAPotential(BasePotential):
    """ASE calculator for UMA (Universal Model for Atoms) via fairchem-core."""

    implemented_properties = ["energy", "forces", "hessian"]

    def __init__(
        self,
        model_name: str = DEFAULT_UMA_MODEL,
        device: str | None = None,
        default_charge: int = 0,
        default_spin: int = 1,
        **kwargs: Any,
    ) -> None:
        if device is None:
            device = "cpu"

        self.predictor: Any = None
        self._calc: Any = None
        self.default_charge = default_charge
        self.default_spin = default_spin

        super().__init__(model_name=model_name, device=device, **kwargs)

    def _set_model_precision(self, precision: str = "float32") -> None:
        if self.predictor is None:
            return
        if not hasattr(self.predictor, "model"):
            return
        model = self.predictor.model
        if precision == "float32" and hasattr(model, "float"):
            model.float()
        elif precision == "double" and hasattr(model, "double"):
            model.double()

    def _load_calculator(self) -> None:
        if getattr(self, "_calc", None) is not None:
            return

        from famex.utils.ml_warnings import quiet_backend_loading

        with quiet_backend_loading(
            "uma",
            self.model_name,
            None,
            self.device,
            show_model_info=False,
        ):
            try:
                if not deps.has("fairchem"):
                    msg = (
                        f"fairchem-core is required for UMA potentials. "
                        f"Install with: pip install {FAIRCHEM_INSTALL}"
                    )
                    raise ImportError(msg)

                pretrained_mlip = deps.get("fairchem_pretrained_mlip")
                fairchem_calculator = deps.get("fairchem_calculator")
                if not pretrained_mlip or not fairchem_calculator:
                    raise RuntimeError("FairChem v2 components not available")

                model_name = self.model_name or DEFAULT_UMA_MODEL
                device_param = "cuda" if self.device == "cuda" else "cpu"
                self.predictor = pretrained_mlip.get_predict_unit(model_name, device=device_param)
                self._set_model_precision("float32")
                self._calc = fairchem_calculator(self.predictor, task_name="omol")
            except (ImportError, ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
                raise_backend_load_error("UMA", self.model_name, exc)

    def _ensure_uma_calc(self) -> Any:
        if self._calc is None:
            self._load_calculator()
        if self._calc is None:
            msg = "Failed to load UMA calculator"
            logger.error(msg)
            raise RuntimeError(msg)
        return self._calc

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: Sequence[str] | None = None,
        system_changes: Any = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        if atoms is None:
            raise ValueError("atoms cannot be None")

        self._set_atoms_charge_spin(atoms, self.default_charge, self.default_spin)
        calc = self._ensure_uma_calc()

        try:
            calc.calculate(atoms, properties, system_changes)
        except RuntimeError as exc:
            message = str(exc)
            if "expected scalar type Double but found Float" in message:
                self._set_model_precision("double")
                calc.calculate(atoms, properties, system_changes)
            elif "mat1 and mat2 must have the same dtype, but got Double and Float" in message:
                self._set_model_precision("float32")
                calc.calculate(atoms, properties, system_changes)
            else:
                logger.exception("Unexpected error during UMA calculation")
                raise

        if properties is not None:
            if "energy" in properties:
                self.results["energy"] = calc.results["energy"]
            if "forces" in properties:
                self.results["forces"] = calc.results["forces"]

    def get_hessian(
        self,
        atoms: Atoms | None = None,
        method: str = "auto",
        symmetrize: bool = True,
    ) -> np.ndarray:
        """Return the analytical Hessian (3N x 3N) in eV/Å²."""
        if atoms is not None:
            self.atoms = atoms

        self._ensure_uma_calc()
        assert self.atoms is not None
        self._set_atoms_charge_spin(self.atoms, self.default_charge, self.default_spin)

        try:
            from fairchem.core.datasets import data_list_collater
            from fairchem.core.datasets.atomic_data import AtomicData

            if self.predictor is None:
                raise RuntimeError("UMA predictor not loaded. Cannot calculate analytical Hessian.")

            device = next(self.predictor.model.parameters()).device
            if method == "auto":
                method = "double_backward"

            atoms_copy = self.atoms.copy()
            atoms_copy.info["charge"] = int(self.atoms.info.get("charge", self.default_charge))
            atoms_copy.info["spin"] = int(self.atoms.info.get("spin", self.default_spin))

            data = AtomicData.from_ase(
                atoms_copy,
                task_name="omol",
                r_edges=False,
                r_data_keys=["spin", "charge"],
            ).to(device)
            batch = data_list_collater([data], otf_graph=True).to(device)
            # otf_graph does not enable position gradients; required for autograd Hessian
            batch.pos.requires_grad_(True)

            model_module = self.predictor.model.module
            energy_wrapper = model_module.output_heads["energyandforcehead"]
            prev_head_training = energy_wrapper.head.training
            energy_wrapper.head.training = True

            result = self.predictor.predict(batch)
            energy = result["energy"]

            if method == "double_backward":
                hessian_tensor = self._compute_hessian_double_backward(energy, batch.pos)
            elif method == "vmap":
                forces = -torch.autograd.grad(
                    energy,
                    batch.pos,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                hessian_tensor = self._hessian_from_forces_vjp(forces, batch.pos, use_vmap=True)
            elif method in {"fairchem", "fairchem_vmap"}:
                hessian_tensor = self._hessian_from_forces_vjp(
                    result["forces"],
                    batch.pos,
                    use_vmap=True,
                )
            elif method == "fairchem_loop":
                hessian_tensor = self._hessian_from_forces_vjp(
                    result["forces"],
                    batch.pos,
                    use_vmap=False,
                )
            else:
                raise ValueError(
                    f"Unknown Hessian computation method: {method}. "
                    "Use 'vmap', 'double_backward', 'fairchem', 'fairchem_loop', or 'auto'"
                )

            energy_wrapper.head.training = prev_head_training

            n_atoms = len(self.atoms)
            expected_shape = (3 * n_atoms, 3 * n_atoms)
            if hessian_tensor.shape != expected_shape:
                expected_size = expected_shape[0] * expected_shape[1]
                if hessian_tensor.numel() == expected_size:
                    hessian_tensor = hessian_tensor.reshape(expected_shape)
                else:
                    raise ValueError(
                        f"Hessian tensor has unexpected shape {hessian_tensor.shape}, "
                        f"expected {expected_shape}"
                    )

            hessian_np = hessian_tensor.detach().cpu().numpy()
            if hessian_np.shape != expected_shape:
                raise ValueError(
                    f"Hessian has unexpected shape {hessian_np.shape}, expected {expected_shape}"
                )

            if symmetrize:
                asymmetry = np.abs(hessian_np - hessian_np.T)
                max_asymmetry = np.max(asymmetry)
                if max_asymmetry > 1e-5:
                    warnings.warn(
                        f"Hessian asymmetry detected (max deviation: {max_asymmetry:.2e}). "
                        "Symmetrization will be applied.",
                        UserWarning,
                        stacklevel=2,
                    )
                hessian_np = cast(np.ndarray, 0.5 * (hessian_np + hessian_np.T))

            return hessian_np

        except ImportError as exc:
            raise ImportError(
                f"PyTorch is required for analytical Hessian calculation. Install PyTorch: {exc}"
            ) from exc
        except (ValueError, RuntimeError) as exc:
            raise RuntimeError(f"Failed to calculate UMA analytical Hessian: {exc}") from exc
        except TypeError as exc:
            raise TypeError(f"Failed to calculate UMA analytical Hessian: {exc}") from exc

    def _hessian_from_forces_vjp(
        self,
        forces: torch.Tensor,
        positions: torch.Tensor,
        *,
        use_vmap: bool,
    ) -> torch.Tensor:
        forces_flat = forces.view(-1)
        num_dofs = forces_flat.shape[0]

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

        identity = torch.eye(num_dofs, dtype=forces_flat.dtype, device=forces_flat.device)

        if use_vmap and hasattr(torch, "vmap"):
            try:
                chunk_size = 1 if num_dofs < 64 else 16
                return cast(
                    torch.Tensor,
                    torch.vmap(
                        grad_wrt_positions,
                        in_dims=0,
                        out_dims=0,
                        chunk_size=chunk_size,
                    )(identity),
                )
            except RuntimeError:
                use_vmap = False

        rows = [grad_wrt_positions(identity[idx]) for idx in range(num_dofs)]
        return torch.stack(rows, dim=0)

    def _compute_hessian_double_backward(
        self,
        energy: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        n_atoms = positions.shape[0]
        num_elements = 3 * n_atoms

        forces = -torch.autograd.grad(
            energy,
            positions,
            create_graph=True,
            retain_graph=True,
        )[0]
        forces_flat = forces.view(-1)

        hessian_list = []
        for i in range(num_elements):
            hess_row = torch.autograd.grad(
                forces_flat[i],
                positions,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )[0]
            hessian_list.append((-hess_row).view(-1))

        return torch.stack(hessian_list)

    def get_property(
        self, prop: str, atoms: Atoms | None = None, allow_calculation: bool = True
    ) -> Any:
        if atoms is not None:
            self.atoms = atoms

        if prop == "energy":
            return self.get_potential_energy(atoms)
        if prop == "forces":
            return self.get_forces(atoms)
        if prop == "hessian":
            return self.get_hessian(atoms)
        raise KeyError(f"Property '{prop}' not supported by UMAPotential")
