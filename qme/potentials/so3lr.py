"""
SO3LR potential facade moved into qme.potentials
"""

from typing import Optional

import numpy as np

from .. import dependencies as _deps
from .base import BasePotential


class SO3LRPotential(BasePotential):
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = "so3lr-small",
        device: Optional[str] = None,
        **kwargs,
    ):
        if not _deps.deps.has("so3lr"):
            raise ImportError("SO3LR dependency missing")
        self.model_path = model_path
        self.calculator = None
        super().__init__(model_name=model_name, device=device, **kwargs)

    def _load_calculator(self):
        pass

    def calculate(
        self, atoms=None, properties=["energy", "forces"], system_changes=None
    ):
        if atoms is None:
            return
        super().calculate(atoms, properties, system_changes)

    def _get_backend_name(self) -> str:
        return "so3lr"


def get_so3lr_calculator(
    model_path: Optional[str] = None,
    model_name: str = "so3lr-small",
    device: Optional[str] = None,
    **kwargs,
):
    return SO3LRPotential(
        model_path=model_path, model_name=model_name, device=device, **kwargs
    )
