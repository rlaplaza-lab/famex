from __future__ import annotations

import numpy as np
from ase import Atoms

from qme.analysis.hessian import HessianCalculator
from qme.potentials.mock_potential import MockCalculator


def build_h2(d: float = 0.74) -> Atoms:
    # Simple H2 molecule aligned on x-axis
    positions = np.array([[0.0, 0.0, 0.0], [d, 0.0, 0.0]])
    atoms = Atoms("H2", positions=positions)
    return atoms


def test_richardson_improves_accuracy_over_central():
    atoms = build_h2(0.74)
    calc = MockCalculator(force_constant=1.0)

    # Reference with very small step
    hess_ref = HessianCalculator(
        atoms,
        calc,
        delta=1e-4,
        method="central",
        richardson=False,
        indices=None,
        verbose=0,
    ).calculate_numerical_hessian()

    # Baseline central with practical step
    hess_central = HessianCalculator(
        atoms,
        calc,
        delta=0.02,
        method="central",
        richardson=False,
        indices=None,
        verbose=0,
    ).calculate_numerical_hessian()

    # Richardson with two deltas (delta2 defaults to delta/2)
    hess_rich = HessianCalculator(
        atoms,
        calc,
        delta=0.02,
        method="central",
        richardson=True,
        delta2=None,
        indices=None,
        verbose=0,
    ).calculate_numerical_hessian()

    # Compare errors to reference
    err_central = np.linalg.norm(hess_central - hess_ref)
    err_rich = np.linalg.norm(hess_rich - hess_ref)

    # Richardson should be strictly better
    assert err_rich < err_central * 0.8  # expect noticeable improvement

    # Also check elementwise max error improves
    max_err_central = np.max(np.abs(hess_central - hess_ref))
    max_err_rich = np.max(np.abs(hess_rich - hess_ref))
    assert max_err_rich < max_err_central
