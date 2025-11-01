#!/usr/bin/env python3
"""Compare Hessian accuracy for different finite difference schemes.

This script demonstrates the accuracy improvements achieved by using higher-order
finite difference schemes (5-point) and Richardson extrapolation for Hessian
calculations. It compares:
1. 3-point central difference (standard, O(h²))
2. 5-point central difference (O(h⁴))
3. 3-point + Richardson extrapolation
4. 5-point + Richardson extrapolation (O(h⁶) when combined)

The comparison uses a harmonic potential where the analytical Hessian is known
exactly, allowing precise error analysis.
"""

import os

import numpy as np
from ase import Atoms

# Disable ASE GUI to prevent popup windows
os.environ["DISPLAY"] = ""
os.environ["MPLBACKEND"] = "Agg"

from qme.analysis.frequency import HessianCalculator


class HarmonicCalculator:
    """Mock calculator for harmonic potential: E = 0.5 * k * Σr²."""

    def __init__(self, k: float = 1.0) -> None:
        """Initialize with force constant k."""
        self.k = k

    def get_forces(self, atoms: Atoms) -> np.ndarray:
        """Compute harmonic forces: F = -k * r."""
        return -self.k * atoms.positions

    def get_hessian(self, atoms: Atoms) -> np.ndarray:
        """Compute analytical harmonic Hessian: H = k * I."""
        n_atoms = len(atoms)
        return self.k * np.eye(3 * n_atoms)


def compare_hessian_methods(atoms: Atoms, delta: float = 0.05) -> None:
    """Compare different Hessian calculation methods.

    Parameters
    ----------
    atoms : Atoms
        Molecular system to analyze
    delta : float
        Step size for finite differences
    """
    calc = HarmonicCalculator(k=1.0)
    hessian_analytical = calc.get_hessian(atoms)

    print(f"Comparing Hessian methods for {len(atoms)} atoms")
    print("=" * 80)
    print(f"Step size: {delta} Å")
    print()

    # 1. 3-point central difference
    hc_3point = HessianCalculator(atoms, calc, delta=delta, method="central", verbose=0)
    hessian_3point = hc_3point.calculate_numerical_hessian()
    error_3point = np.max(np.abs(hessian_3point - hessian_analytical))

    # 2. 5-point central difference
    hc_5point = HessianCalculator(atoms, calc, delta=delta, method="5point", verbose=0)
    hessian_5point = hc_5point.calculate_numerical_hessian()
    error_5point = np.max(np.abs(hessian_5point - hessian_analytical))

    # 3. 3-point + Richardson
    hc_3point_rich = HessianCalculator(
        atoms, calc, delta=delta, method="central", richardson=True, verbose=0
    )
    hessian_3point_rich = hc_3point_rich.calculate_numerical_hessian()
    error_3point_rich = np.max(np.abs(hessian_3point_rich - hessian_analytical))

    # 4. 5-point + Richardson
    hc_5point_rich = HessianCalculator(
        atoms, calc, delta=delta, method="5point", richardson=True, verbose=0
    )
    hessian_5point_rich = hc_5point_rich.calculate_numerical_hessian()
    error_5point_rich = np.max(np.abs(hessian_5point_rich - hessian_analytical))

    # Print results
    methods = [
        ("3-point central", error_3point),
        ("5-point central", error_5point),
        ("3-point + Richardson", error_3point_rich),
        ("5-point + Richardson", error_5point_rich),
    ]

    print("Method                         | Max Error")
    print("-" * 80)
    for method_name, error in methods:
        print(f"{method_name:30s} | {error:.2e}")
    print()

    # Calculate speedup factors
    force_calls = {
        "3-point": 6 * len(atoms),
        "5-point": 15 * len(atoms),
        "3-point + Richardson": 12 * len(atoms),
        "5-point + Richardson": 30 * len(atoms),
    }

    print("Computational Cost (force evaluations):")
    print("-" * 80)
    for method, calls in force_calls.items():
        print(f"{method:30s} | {calls}")

    print()
    print("Accuracy improvements:")
    print("-" * 80)
    print(f"5-point vs 3-point: {error_3point / error_5point:.2f}x better")
    print(f"5-point + Richardson vs 3-point: {error_3point / error_5point_rich:.2e}x better")
    print()

    # For harmonic potential, all should be exact to machine precision
    # The value depends on the step size and numerical precision
    print("Note: For harmonic potentials, all methods achieve very high accuracy.")
    print("The accuracy differences become significant for non-quadratic potentials.")


def main():
    """Main function."""
    # Create test molecules
    water = Atoms(
        symbols="OHH",
        positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.24, 0.93, 0.0],
        ],
    )

    methane = Atoms(
        symbols="CHHHH",
        positions=[
            [0.0, 0.0, 0.0],
            [1.09, 0.0, 0.0],
            [-0.36, 1.03, 0.0],
            [-0.36, -0.51, 0.89],
            [-0.36, -0.51, -0.89],
        ],
    )

    print("\n" + "=" * 80)
    print("WATER (3 atoms)")
    print("=" * 80)
    compare_hessian_methods(water, delta=0.05)

    print("\n" + "=" * 80)
    print("METHANE (5 atoms)")
    print("=" * 80)
    compare_hessian_methods(methane, delta=0.05)

    # Test with different step sizes
    print("\n" + "=" * 80)
    print("CONVERGENCE ANALYSIS (Water, varying step sizes)")
    print("=" * 80)
    print()

    deltas = [0.1, 0.05, 0.01, 0.005]
    calc = HarmonicCalculator(k=1.0)
    hessian_analytical = calc.get_hessian(water)

    print("Step Size | 3-point Error | 5-point Error | Improvement")
    print("-" * 80)
    for delta in deltas:
        hc_3 = HessianCalculator(water, calc, delta=delta, method="central", verbose=0)
        hc_5 = HessianCalculator(water, calc, delta=delta, method="5point", verbose=0)

        hessian_3 = hc_3.calculate_numerical_hessian()
        hessian_5 = hc_5.calculate_numerical_hessian()

        error_3 = np.max(np.abs(hessian_3 - hessian_analytical))
        error_5 = np.max(np.abs(hessian_5 - hessian_analytical))

        improvement = error_3 / error_5 if error_5 > 0 else np.inf
        print(f"{delta:8.3f} | {error_3:13.2e} | {error_5:13.2e} | {improvement:11.2f}x")


if __name__ == "__main__":
    main()
