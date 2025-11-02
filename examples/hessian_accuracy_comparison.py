#!/usr/bin/env python3
"""Compare Hessian accuracy for different finite difference schemes.

This script demonstrates the accuracy improvements achieved by using higher-order
finite difference schemes and Richardson extrapolation for Hessian calculations.
It compares:
1. 3-point central difference (standard, O(h²))
2. 5-point central difference (O(h⁴))
3. 7-point central difference (O(h⁶))
4. 3-point + Richardson extrapolation
5. 5-point + Richardson extrapolation (O(h⁶) when combined)
6. 7-point + Richardson extrapolation (O(h⁸) when combined)
7. Energy-based FD (4N² energy evals)
8. Adaptive 5-point + Richardson (automatically optimizes step size)
9. Analytical MACE Hessian (if available)

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
from qme.analysis.hessian_energy import EnergyBasedHessianCalculator

try:
    import qme

    has_mace = True
except ImportError:
    has_mace = False


class HarmonicCalculator:
    """Mock calculator for harmonic potential: E = 0.5 * k * Σr²."""

    def __init__(self, k: float = 1.0) -> None:
        """Initialize with force constant k."""
        self.k = k

    def get_forces(self, atoms: Atoms) -> np.ndarray:
        """Compute harmonic forces: F = -k * r."""
        return -self.k * atoms.positions

    def get_potential_energy(self, atoms: Atoms) -> float:
        """Compute harmonic potential energy: E = 0.5 * k * Σr²."""
        return 0.5 * self.k * np.sum(atoms.positions**2)

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

    # 5. 7-point central difference
    hc_7point = HessianCalculator(atoms, calc, delta=delta, method="7point", verbose=0)
    hessian_7point = hc_7point.calculate_numerical_hessian()
    error_7point = np.max(np.abs(hessian_7point - hessian_analytical))

    # 6. 7-point + Richardson
    hc_7point_rich = HessianCalculator(
        atoms, calc, delta=delta, method="7point", richardson=True, verbose=0
    )
    hessian_7point_rich = hc_7point_rich.calculate_numerical_hessian()
    error_7point_rich = np.max(np.abs(hessian_7point_rich - hessian_analytical))

    # 7. Adaptive 5-point + Richardson
    hc_adaptive = HessianCalculator(
        atoms,
        calc,
        delta=delta,
        method="5point",
        richardson=True,
        adaptive_delta=True,
        max_iterations=3,
        verbose=0,
    )
    hessian_adaptive = hc_adaptive.calculate_numerical_hessian()
    error_adaptive = np.max(np.abs(hessian_adaptive - hessian_analytical))

    # 8. Energy-based FD
    ebc = EnergyBasedHessianCalculator(atoms, calc, delta=delta, verbose=0)
    hessian_energy = ebc.calculate_energy_hessian()
    error_energy = np.max(np.abs(hessian_energy - hessian_analytical))

    # Print results
    methods = [
        ("3-point central", error_3point),
        ("5-point central", error_5point),
        ("7-point central", error_7point),
        ("3-point + Richardson", error_3point_rich),
        ("5-point + Richardson", error_5point_rich),
        ("7-point + Richardson", error_7point_rich),
        ("Adaptive 5-point + Richardson", error_adaptive),
        ("Energy-based FD", error_energy),
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
        "7-point": 18 * len(atoms),
        "3-point + Richardson": 12 * len(atoms),
        "5-point + Richardson": 30 * len(atoms),
        "7-point + Richardson": 36 * len(atoms),
        "Adaptive 5-point + Richardson": "~30-90 * N (adaptive)",
        "Energy-based FD": f"{4 * len(atoms) ** 2} energy evals",
    }

    print("Computational Cost:")
    print("-" * 80)
    for method, calls in force_calls.items():
        print(f"{method:30s} | {calls}")

    print()
    print("Accuracy improvements:")
    print("-" * 80)
    print(f"5-point vs 3-point: {error_3point / error_5point:.2f}x better")
    print(f"7-point vs 3-point: {error_3point / error_7point:.2e}x better")
    print(f"5-point + Richardson vs 3-point: {error_3point / error_5point_rich:.2e}x better")
    print(f"7-point + Richardson vs 3-point: {error_3point / error_7point_rich:.2e}x better")
    print(f"Energy-based FD vs 3-point: {error_3point / error_energy:.2e}x better")
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

    # Optional: Compare with MACE analytical Hessian if available
    if has_mace:
        try:
            print("\n" + "=" * 80)
            print("MACE ANALYTICAL HESSIAN COMPARISON")
            print("=" * 80)
            print("\nComparing FD methods against MACE analytical Hessian...")

            # Create MACE calculator
            mace_calc = qme.get_mace_calculator(model="small", device="cpu")
            water.calc = mace_calc

            # Get analytical Hessian
            from qme.analysis.frequency import FrequencyAnalysis

            freq_analysis = FrequencyAnalysis(water, mace_calc, verbose=0)
            hessian_mace = freq_analysis.calculate_hessian(method="direct")

            print("\nComputing FD Hessians and comparing...")

            # Compare a few key methods
            methods_to_test = [
                ("3-point", "central", False),
                ("5-point", "5point", False),
                ("7-point", "7point", False),
                ("5-point + Richardson", "5point", True),
                ("7-point + Richardson", "7point", True),
            ]

            print(f"\n{'Method':30s} {'Max Error (eV/Å²)':>20s}")
            print("-" * 80)

            for name, method, use_richardson in methods_to_test:
                hc = HessianCalculator(
                    water,
                    mace_calc,
                    delta=0.01,
                    method=method,
                    richardson=use_richardson,
                    verbose=0,
                )
                hessian_fd = hc.calculate_numerical_hessian()
                error = np.max(np.abs(hessian_fd - hessian_mace))
                print(f"{name:30s} {error:20.2e}")

            print("\nNote: Analytical Hessian from MACE is not necessarily 'true',")
            print("but it represents the MLIP's best estimate without FD errors.")

        except Exception as e:
            print(f"\n⚠ MACE comparison skipped: {e}")


if __name__ == "__main__":
    main()
