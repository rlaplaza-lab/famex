"""Orb Machine Learning Potential integration for ASE.

This module implements integration with Orbital Materials' Orb models,
providing universal forcefields for molecular and materials calculations.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ase import Atoms
from ase.calculators.calculator import all_changes

from famex.backends.dependencies import deps
from famex.potentials._load_utils import raise_backend_load_error
from famex.potentials.base_potential import BasePotential
from famex.utils.logging import get_famex_logger

logger = get_famex_logger(__name__)


class OrbPotential(BasePotential):
    """ASE Calculator interface for Orb neural network potential.

    Orb provides universal neural network potentials for molecular and materials
    property prediction and geometry optimization. This implementation uses the
    OrbMol variant which requires charge and spin multiplicity specification.

    Parameters
    ----------
    model_name : str, default "orb-v3-conservative-omol"
        Name of Orb model to use. Available models:
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
            msg = "orb-models is required for Orb potentials. Install with: pip install orb-models"
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
            implemented_properties=["energy", "forces"],
            **kwargs,
        )

    def _load_calculator(self) -> None:
        """Load the Orb model and create calculator."""
        from famex.utils.ml_warnings import quiet_backend_loading

        try:
            from orb_models.forcefield import pretrained
            from orb_models.forcefield.calculator import ORBCalculator

            if self.model_name is None:
                self.model_name = "orb-v3-conservative-omol"

            if self.device is None:
                self.device = "cpu"

            model_registry = {
                "orb-v3-conservative-omol": pretrained.orb_v3_conservative_omol,
                "orb-v3-conservative-inf-omat": pretrained.orb_v3_conservative_inf_omat,
                "orb-v2": pretrained.orb_v2,
                "orb-v3-omol": pretrained.orb_v3_conservative_omol,
                "orb-v3-omat": pretrained.orb_v3_conservative_inf_omat,
                "omol": pretrained.orb_v3_conservative_omol,
                "omat": pretrained.orb_v3_conservative_inf_omat,
            }

            if self.model_name in model_registry:
                model_loader = model_registry[self.model_name]
            else:
                model_loader = pretrained.orb_v3_conservative_omol
                self.model_name = "orb-v3-conservative-omol"

            with quiet_backend_loading(
                "orb",
                self.model_name,
                "pretrained",
                self.device,
                show_model_info=False,
            ):
                orbff = model_loader(device=self.device)
                self._calc = ORBCalculator(orbff, device=self.device)

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
