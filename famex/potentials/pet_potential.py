"""PET (UPET) Machine Learning Potential integration for ASE.

This module implements integration with lab-cosmo's UPET universal interatomic
potentials based on the Point Edge Transformer architecture.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from famex.backends.constants import DEFAULT_PET_MODEL
from famex.backends.dependencies import deps
from famex.potentials.base_potential import BasePotential
from famex.utils.logging import get_famex_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ase import Atoms

logger = get_famex_logger(__name__)


def parse_pet_model_name(model_name: str | None) -> tuple[str, str]:
    """Parse FAMEX model_name into UPET model and version.

    Supports optional version suffix: ``pet-mad-s@1.5.0`` → ``("pet-mad-s", "1.5.0")``.
    """
    if model_name is None:
        return DEFAULT_PET_MODEL, "latest"
    if "@" in model_name:
        model, version = model_name.rsplit("@", 1)
        return model, version
    return model_name, "latest"


class PETPotential(BasePotential):
    """ASE Calculator interface for UPET neural network potentials.

    Wraps UPET's ``UPETCalculator`` for universal materials and molecular
    modeling with PET-MAD, PET-OAM, PET-OMat, and related model families.
    """

    implemented_properties = ["energy", "forces", "hessian"]

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        model_path: str | None = None,
        version: str = "latest",
        **kwargs: Any,
    ) -> None:
        """Initialize PET potential calculator.

        Parameters
        ----------
        model_name : str, optional
            UPET model identifier (e.g. ``pet-mad-s``, ``pet-oam-xl``).
            An optional version suffix is supported: ``pet-mad-s@1.5.0``.
        device : str, optional
            Device for computations (``cpu``, ``cuda``). Auto-detected if None.
        model_path : str, optional
            Path to a local UPET checkpoint file. When set, ``model_name`` and
            ``version`` are ignored by UPET (parsed from filename).
        version : str, default "latest"
            UPET model version. Overridden by ``@`` suffix in ``model_name``.
        **kwargs
            Additional arguments passed to BasePotential.
        """
        self._calc: Any | None = None
        self._hessian_calc: Any | None = None
        self.model_path = model_path

        parsed_model, parsed_version = parse_pet_model_name(model_name)
        if model_name is not None and "@" in model_name:
            version = parsed_version
        self.pet_model = parsed_model
        self.pet_version = version
        # A checkpoint path passed as model_name loads locally and skips the
        # Hugging Face repository listing that UPET otherwise does on startup.
        if self.model_path is None and Path(self.pet_model).is_file():
            self.model_path = self.pet_model

        super().__init__(
            backend="pet",
            model_name=model_name or DEFAULT_PET_MODEL,
            device=device,
            **kwargs,
        )

    def _load_calculator(self) -> None:
        """Load the UPET calculator implementation."""
        if self._calc is not None:
            return

        from famex.utils.ml_warnings import quiet_backend_loading

        if not deps.has("torch"):
            msg = "PyTorch is required for PET backend. Install with: pip install torch"
            raise ImportError(msg)

        if not deps.has("upet"):
            msg = "upet is required for PET backend. Install with: pip install upet"
            raise ImportError(msg)

        with quiet_backend_loading(
            "pet",
            self.pet_model,
            self.model_path,
            self.device,
            show_model_info=False,
        ):
            try:
                from upet.calculator import UPETCalculator

                device = self.device
                if self.model_path is not None:
                    self._calc = UPETCalculator(checkpoint_path=self.model_path, device=device)
                else:
                    self._calc = UPETCalculator(
                        model=self.pet_model,
                        version=self.pet_version,
                        device=device,
                    )
            except ImportError as exc:
                logger.error("UPET not available: %s. Install with: pip install upet", exc)
                msg = f"UPET not available ({exc}). Install with: pip install upet"
                raise ImportError(msg) from exc

    def _ensure_hessian_calculator(self) -> Any:
        """Load an unscripted model; TorchScripted ASE path cannot second-differentiate."""
        if self._hessian_calc is not None:
            return self._hessian_calc

        from metatomic_ase import MetatomicCalculator
        from upet._models import _get_upet_exported_atomistic_model

        from famex.utils.ml_warnings import quiet_backend_loading

        with quiet_backend_loading(
            "pet",
            self.pet_model,
            self.model_path,
            self.device,
            show_model_info=False,
        ):
            if self.model_path is not None:
                model = _get_upet_exported_atomistic_model(checkpoint_path=self.model_path)
            else:
                base_model, size = self.pet_model.rsplit("-", 1)
                model = _get_upet_exported_atomistic_model(
                    model=base_model,
                    size=size,
                    version=self.pet_version,
                )

            for parameter in model.parameters():
                parameter.requires_grad_(False)
            model = model.eval()
            self._hessian_calc = MetatomicCalculator(
                model,
                device=self.device,
                do_gradients_with_energy=False,
            )

        return self._hessian_calc

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: Sequence[str] | None = None,
        system_changes: Any = None,
    ) -> None:
        """Calculate properties using the UPET calculator."""
        super().calculate(atoms, properties, system_changes)

        calc = self._require_calc()
        calc.calculate(self.atoms, properties, system_changes)
        self.results = calc.results.copy()

    def get_hessian(self, atoms: Atoms | None = None) -> np.ndarray:
        """Analytical Hessian (3N x 3N, eV/Å²) via double-backward on energy."""
        if atoms is not None:
            self.atoms = atoms
        if self.atoms is None:
            msg = "No atoms provided for Hessian calculation"
            raise ValueError(msg)

        hessian = self._compute_analytical_hessian(self.atoms)
        self.results["hessian"] = hessian
        return hessian

    def _compute_analytical_hessian(self, atoms: Atoms) -> np.ndarray:
        import torch
        from metatomic.torch import ModelEvaluationOptions, System
        from metatomic_ase._calculator import _ase_to_torch_data, _get_ase_input

        hessian_calc = self._ensure_hessian_calculator()
        types, positions, cell, pbc = _ase_to_torch_data(
            atoms=atoms,
            dtype=hessian_calc._dtype,
            device=hessian_calc._device,
        )
        positions = positions.clone().detach().requires_grad_(True)
        system = System(types, positions, cell, pbc)
        input_system = hessian_calc._nl_calculators.compute(systems=[system])[0]

        for name, option in hessian_calc._model.requested_inputs(use_new_names=True).items():
            input_system.add_data(
                name,
                _get_ase_input(
                    atoms,
                    name,
                    option,
                    dtype=hessian_calc._dtype,
                    device=hessian_calc._device,
                ),
            )

        outputs = hessian_calc._ase_properties_to_metatensor_outputs(
            properties=["energy"],
            calculate_forces=False,
            calculate_stress=False,
            calculate_stresses=False,
        )
        predictions = hessian_calc._model(
            systems=[input_system],
            options=ModelEvaluationOptions(length_unit="angstrom", outputs=outputs),
            check_consistency=False,
        )
        energy = predictions[hessian_calc._energy_key].block().values.sum()
        if energy.grad_fn is None:
            msg = "PET energy has no autograd graph; cannot form an analytical Hessian"
            raise RuntimeError(msg)

        forces = -torch.autograd.grad(energy, positions, create_graph=True, retain_graph=True)[0]
        rows = [
            (-torch.autograd.grad(component, positions, retain_graph=True)[0]).reshape(-1)
            for component in forces.reshape(-1)
        ]
        hessian_np = np.asarray(torch.stack(rows).detach().cpu().numpy(), dtype=np.float64)
        return 0.5 * (hessian_np + hessian_np.T)

    def get_property(
        self, prop: str, atoms: Atoms | None = None, allow_calculation: bool = True
    ) -> Any:
        """Get energy, forces, or the analytical Hessian."""
        del allow_calculation
        if atoms is not None:
            self.atoms = atoms
        if prop == "energy":
            return self.get_potential_energy(atoms)
        if prop == "forces":
            return self.get_forces(atoms)
        if prop == "hessian":
            return self.get_hessian(atoms)
        msg = f"Property '{prop}' not supported by PETPotential"
        raise KeyError(msg)
