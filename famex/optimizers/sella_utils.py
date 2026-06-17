"""Utilities for SELLA optimizer integration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from ase import Atoms

SELLA_ANALYTICAL_ALIASES = frozenset(
    {
        "sella-analytical",
        "sella_analytical",
        "sellaanalytical",
    }
)


def is_sella_analytical_optimizer(name: str) -> bool:
    return (name or "").lower() in SELLA_ANALYTICAL_ALIASES


def is_sella_optimizer(name: str) -> bool:
    normalized = (name or "").lower()
    return normalized == "sella" or normalized in SELLA_ANALYTICAL_ALIASES


def validate_calculator_supports_hessian(calculator: Any) -> None:
    from famex.analysis.utils import has_calculator_property

    if calculator is None:
        msg = (
            "sella-analytical requires a calculator with analytical Hessian support. "
            "Attach a calculator to the atoms before optimization."
        )
        raise ValueError(msg)

    if not has_calculator_property(calculator, "hessian"):
        calc_name = type(calculator).__name__
        msg = (
            f"Calculator '{calc_name}' does not provide an analytical Hessian. "
            "Use 'sella' for finite-difference Hessians, or choose a backend that "
            "implements get_hessian / the 'hessian' property (e.g. uma, mace, tblite)."
        )
        raise ValueError(msg)


def make_analytical_hessian_function() -> Callable[[Atoms], np.ndarray]:
    """Return a Sella-compatible callback that reads Hessians from atoms.calc."""

    def hessian_function(atoms: Atoms) -> np.ndarray:
        from famex.analysis.utils import get_calculator_property

        calc = atoms.calc
        validate_calculator_supports_hessian(calc)
        hessian = get_calculator_property(calc, "hessian", atoms)
        return np.asarray(hessian, dtype=float)

    return hessian_function
