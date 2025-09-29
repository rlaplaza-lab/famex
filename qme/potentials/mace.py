"""
MACE potential facade moved to qme.potentials
"""

from typing import Optional

from .. import dependencies as _deps
from .base import BasePotential


class MACEPotential(BasePotential):
    def __init__(
        self, model_name: Optional[str] = None, device: Optional[str] = None, **kwargs
    ):
        if model_name is None:
            model_name = "mace-omol-0"
        super().__init__(model_name=model_name, device=device, **kwargs)

    def _load_calculator(self):
        # Implementation remains in the original file; this facade will be
        # replaced by a full move in the next iterations.
        pass

    def _get_backend_name(self) -> str:
        return "mace"


def get_mace_calculator(
    model_name: Optional[str] = None, device: Optional[str] = None, **kwargs
):
    return MACEPotential(model_name=model_name, device=device, **kwargs)
