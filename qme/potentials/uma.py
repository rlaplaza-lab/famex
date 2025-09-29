"""
UMA potential facade moved into qme.potentials
"""

from typing import Optional

from .. import dependencies as _deps
from .base import BasePotential


class UMAPotential(BasePotential):
    def __init__(
        self,
        model_name: str = "uma-s-1p1",
        device: Optional[str] = None,
        default_charge: int = 0,
        default_spin: int = 1,
        **kwargs,
    ):
        self.predictor = None
        self.fairchem_calc = None
        self.default_charge = default_charge
        self.default_spin = default_spin
        super().__init__(model_name=model_name, device=device, **kwargs)

    def _load_calculator(self):
        pass

    def calculate(
        self, atoms=None, properties=["energy", "forces"], system_changes=None
    ):
        super().calculate(atoms, properties, system_changes)

    def _get_backend_name(self) -> str:
        return "uma"


def get_uma_calculator(model_name: str = "uma-s-1p1", **kwargs):
    return UMAPotential(model_name=model_name, **kwargs)
