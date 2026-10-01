"""Orb Machine Learning Potential integration for ASE.

This module implements integration with Orbital Materials' Orb models,
providing universal forcefields for molecular and materials calculations.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import all_changes

from famex.backends.dependencies import deps
from famex.potentials._load_utils import raise_backend_load_error
from famex.potentials.base_potential import BasePotential
from famex.utils.logging import get_famex_logger

logger = get_famex_logger(__name__)

_ORB_INSTALL = 'pip install "orb-models>=0.7.0"'

# Public model names mapped onto pretrained loaders. orb-models>=0.7.0
# loaders return (model, atoms_adapter); older releases return the model.
_ORB_LOADERS = {
    "orb-v3-conservative-omol": "orb_v3_conservative_omol",
    "orb-v3-conservative-inf-omat": "orb_v3_conservative_inf_omat",
    "orb-v2": "orb_v2",
    "orb-v3-omol": "orb_v3_conservative_omol",
    "orb-v3-omat": "orb_v3_conservative_inf_omat",
    "omol": "orb_v3_conservative_omol",
    "omat": "orb_v3_conservative_inf_omat",
    "orbmol-v2": "orbmol_v2",
    "orbmol_v2": "orbmol_v2",
}


def _import_orb_calculator() -> Any:
    try:
        from orb_models.forcefield.inference.calculator import ORBCalculator
    except ImportError:
        from orb_models.forcefield.calculator import ORBCalculator
    return ORBCalculator


def _resolve_orb_loader(pretrained: Any, model_name: str) -> tuple[Any, str]:
    """Return the pretrained loader and the canonical model name."""
    if model_name not in _ORB_LOADERS:
        return pretrained.orb_v3_conservative_omol, "orb-v3-conservative-omol"
    attr = _ORB_LOADERS[model_name]
    loader = getattr(pretrained, attr, None)
    if loader is None:
        msg = (
            f"Orb model '{model_name}' requires orb-models>=0.7.0 on Python 3.12+. "
            f"Install with: {_ORB_INSTALL}"
        )
        raise ImportError(msg)
    return loader, model_name


class OrbPotential(BasePotential):
    """ASE Calculator interface for Orb neural network potential.

    Orb provides universal neural network potentials for molecular and materials
    property prediction and geometry optimization. This implementation uses the
    OrbMol variant which requires charge and spin multiplicity specification.

    Parameters
    ----------
    model_name : str, default "orb-v3-conservative-omol"
        Name of Orb model to use. Available models:
        - "orbmol-v2": OrbMol-v2 with learnable electrostatics (orb-models>=0.7.0)
        - "orb-v3-conservative-omol": Conservative molecular model (default)
        - "orb-v3-conservative-inf-omat": Inference materials model
        - "orb-v2": Orb v2 model
    device : str, optional
        Device for computations ('cpu', 'cuda'). Auto-detected if None.
    charge : int, default 0
        Total charge of the system
    spin : int, default 1
        Spin multiplicity (2S + 1)
    **kwargs
        Additional arguments passed to BasePotential

    """

    def __init__(
        self,
        model_name: str = "orb-v3-conservative-omol",
        device: str | None = None,
        charge: int = 0,
        spin: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize Orb potential calculator."""
        if not deps.has("orb_models"):
            msg = f"orb-models is required for Orb potentials. Install with: {_ORB_INSTALL}"
            raise ImportError(msg)

        if not deps.has("torch"):
            msg = "PyTorch is required for Orb potentials. Install with: pip install torch"
            raise ImportError(msg)

        if device is None:
            from famex.utils.device import get_optimal_device

            device = get_optimal_device()

        self._calc: Any | None = None
        self.charge = charge
        self.spin = spin

        super().__init__(
            backend="orb",
            model_name=model_name,
            device=device,
            implemented_properties=["energy", "forces", "hessian"],
            **kwargs,
        )

    def _load_calculator(self) -> None:
        """Load the Orb model and create calculator."""
        from famex.utils.ml_warnings import quiet_backend_loading

        try:
            from orb_models.forcefield import pretrained

            orb_calculator = _import_orb_calculator()
            if self.model_name is None:
                self.model_name = "orb-v3-conservative-omol"
            model_loader, self.model_name = _resolve_orb_loader(pretrained, self.model_name)

            with quiet_backend_loading(
                "orb",
                self.model_name,
                "pretrained",
                self.device,
                show_model_info=False,
            ):
                try:
                    # Eager mode keeps the force graph differentiable. torch.compile
                    # on this model rejects the second backward used for the Hessian.
                    loaded = model_loader(device=self.device, compile=False)
                except TypeError:
                    loaded = model_loader(device=self.device)
                atoms_adapter = None
                if isinstance(loaded, tuple):
                    orbff, atoms_adapter = loaded[0], loaded[1]
                else:
                    orbff = loaded
                calc_kwargs: dict[str, Any] = {"device": self.device}
                if atoms_adapter is not None:
                    calc_kwargs["atoms_adapter"] = atoms_adapter
                self._calc = orb_calculator(orbff, **calc_kwargs)

                import torch

                if hasattr(torch._dynamo, "config"):
                    torch._dynamo.config.disable = True

        except (ImportError, ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
            raise_backend_load_error("orb", self.model_name, exc)

    def _apply_charge_spin(self) -> None:
        if self.atoms is not None:
            self.atoms.info["charge"] = self.charge
            self.atoms.info["spin"] = self.spin

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: Sequence[str] | None = None,
        system_changes: Any = all_changes,
    ) -> None:
        """Calculate properties using Orb potential."""
        super().calculate(atoms, properties, system_changes)

        if self.atoms is None:
            msg = "No atoms provided for calculation"
            raise ValueError(msg)

        self._apply_charge_spin()
        calc = self._require_calc()
        calc.calculate(self.atoms, properties, system_changes)

        if hasattr(calc, "results") and isinstance(calc.results, dict):
            self.results = calc.results.copy()

    def set_charge(self, charge: int) -> None:
        """Set molecular charge."""
        self.charge = charge

    def set_spin(self, spin: int) -> None:
        """Set spin multiplicity."""
        self.spin = spin

    def get_potential_energy(
        self,
        atoms: Atoms | None = None,
        force_consistent: bool = False,
    ) -> float:
        """Get potential energy."""
        if atoms is not None:
            self.atoms = atoms
        self._apply_charge_spin()
        return super().get_potential_energy(atoms, force_consistent)

    def get_forces(self, atoms: Atoms | None = None) -> Any:
        """Get forces."""
        if atoms is not None:
            self.atoms = atoms
        self._apply_charge_spin()
        return super().get_forces(atoms)

    def get_hessian(self, atoms: Atoms | None = None) -> np.ndarray:
        """Analytical Hessian (3N x 3N, eV/Å²) for conservative Orb models.

        Forces are minus the energy gradient, so the Hessian is minus the
        Jacobian of those forces. Direct (non-conservative) Orb models have no
        energy graph and fall back to finite differences.
        """
        import torch

        if atoms is not None:
            self.atoms = atoms
        if self.atoms is None:
            msg = "No atoms provided for Hessian calculation"
            raise ValueError(msg)

        self._apply_charge_spin()
        calc = self._require_calc()
        if not getattr(calc, "conservative", False):
            from famex.analysis.hessian import HessianCalculator

            hessian_calc = HessianCalculator(self.atoms, self, verbose=0)
            return hessian_calc.calculate_numerical_hessian()

        import orb_models.forcefield.models.conservative_regressor as orb_regressor

        batch = calc.adapter.from_ase_atoms(
            atoms=self.atoms,
            max_num_neighbors=calc.max_num_neighbors,
            edge_method=calc.edge_method,
            half_supercell=calc.half_supercell,
            device=calc.device,
        )
        original = orb_regressor.compute_gradient_forces_and_stress

        def _keep_force_graph(*args: Any, **kwargs: Any) -> Any:
            kwargs["training"] = True
            return original(*args, **kwargs)

        orb_regressor.compute_gradient_forces_and_stress = _keep_force_graph
        was_training = calc.model.training
        calc.model.eval()
        try:
            prediction = calc.model(batch)
            forces = prediction[calc.model.grad_forces_name]
            positions = batch.node_features["positions"]
            rows = [
                (-torch.autograd.grad(component, positions, retain_graph=True)[0]).reshape(-1)
                for component in forces.reshape(-1)
            ]
            hessian = torch.stack(rows)
        finally:
            orb_regressor.compute_gradient_forces_and_stress = original
            calc.model.train(was_training)

        hessian_np = np.asarray(hessian.detach().cpu().numpy(), dtype=np.float64)
        hessian_np = 0.5 * (hessian_np + hessian_np.T)
        self.results["hessian"] = hessian_np
        return hessian_np

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
        msg = f"Property '{prop}' not supported by OrbPotential"
        raise KeyError(msg)
