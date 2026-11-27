"""SciPy-based optimizers with Hessian support for FAMEX."""

from __future__ import annotations

from typing import Any

from ase import Atoms
from scipy.optimize import minimize

from famex.optimizers._exceptions import ConvergedError
from famex.optimizers.hessian_optimizer_base import HessianOptimizerBase

__all__ = [
    "ConvergedError",
    "HessianOptimizerBase",
    "NewtonCG",
    "SciPyHessianOptimizer",
    "TrustExact",
    "TrustKrylov",
    "TrustNCG",
    "minimize",
]

SciPyHessianOptimizer = HessianOptimizerBase


class _NamedSciPyHessianOptimizer(HessianOptimizerBase):
    """Boilerplate subclass selecting a fixed SciPy ``method``."""

    method_name: str = "trust-krylov"

    def __init__(self, atoms: Atoms, **kwargs: Any) -> None:
        super().__init__(atoms=atoms, method=self.method_name, **kwargs)


class TrustKrylov(_NamedSciPyHessianOptimizer):
    """Trust-Krylov optimizer for ASE."""

    def __init__(
        self,
        atoms: Atoms,
        trust_radius: float | None = None,
        max_trust_radius: float | None = None,
        **kwargs: Any,
    ) -> None:
        ts_search = bool(kwargs.get("ts_search", False))
        if trust_radius is None:
            trust_radius = 0.05 if ts_search else 1.0
        if max_trust_radius is None:
            max_trust_radius = 0.15 if ts_search else 10.0
        super().__init__(
            atoms,
            trust_radius=trust_radius,
            max_trust_radius=max_trust_radius,
            **kwargs,
        )


class TrustNCG(_NamedSciPyHessianOptimizer):
    """Trust-NCG optimizer for ASE."""

    method_name = "trust-ncg"


class TrustExact(_NamedSciPyHessianOptimizer):
    """Trust-Exact optimizer for ASE."""

    method_name = "trust-exact"


class NewtonCG(_NamedSciPyHessianOptimizer):
    """Newton-CG optimizer for ASE."""

    method_name = "Newton-CG"
