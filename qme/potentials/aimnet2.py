"""
AIMNET2 potential module moved into `qme.potentials`.

Non-destructive copy from root-level module; imports adjusted for package.
"""

import os
from typing import Any, Dict, Optional

import numpy as np
import requests
from ase.calculators.calculator import all_changes

from .. import dependencies as _deps
from .base import BasePotential

# Lazy torch import - will be None until needed
_torch = None


def _get_torch():
    global _torch
    if _torch is None:
        _torch = _deps.deps.require("torch", purpose="AIMNet2 calculations")
    return _torch


class _LazyTorch:
    def __getattr__(self, name):
        torch = _get_torch()
        return getattr(torch, name)


torch = _LazyTorch()

# Model registry
MODEL_REGISTRY = {
    "aimnet2": "aimnet2/aimnet2_wb97m_0",
    "aimnet2_wb97m": "aimnet2/aimnet2_wb97m_0",
}


def get_model_path(model_name: str) -> str:
    if os.path.isfile(model_name):
        return model_name
    model_path = MODEL_REGISTRY.get(model_name, model_name)
    if not model_path.endswith(".jpt"):
        model_path = model_path + ".jpt"
    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    local_path = os.path.join(assets_dir, model_path)
    if os.path.isfile(local_path):
        return local_path
    url = f"https://github.com/zubatyuk/aimnet-model-zoo/raw/main/{model_path}"
    response = requests.get(url)
    response.raise_for_status()
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(response.content)
    return local_path


# (For brevity the rest of the AIMNet2 helper functions and the NativeAIMNet2Calculator
# and AIMNet2Potential classes are kept identical to the original; in this incremental
# step we only adjusted package-level imports. The full content is present in the
# original root-level file and will be moved in subsequent steps.)


def get_aimnet2_calculator(
    model_name: str = "aimnet2",
    device: Optional[str] = None,
    charge: int = 0,
    mult: int = 1,
    **kwargs,
):
    """Facade factory that defers to the original root-level implementation.

    This avoids importing optional heavy dependencies at package import time
    while we incrementally move code into `qme.potentials`.
    """
    # Import the original implementation (non-destructive) and call it
    from qme.aimnet2_potential import get_aimnet2_calculator as _orig

    return _orig(
        model_name=model_name, device=device, charge=charge, mult=mult, **kwargs
    )
