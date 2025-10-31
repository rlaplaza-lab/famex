"""Diagnostic script to investigate Richardson extrapolation issues."""

from io import StringIO
from urllib.request import urlopen

import numpy as np
from ase.io import read

import qme
from qme.analysis.frequency import HessianCalculator


def main():
    # Simple water first
    print("=" * 80)
    print("WATER MOLECULE (small system)")
    print("=" * 80)
    from ase import Atoms as ASEAtoms

    water = ASEAtoms(symbols="OHH", positions=[[0, 0, 0], [0.96, 0, 0], [0.24, 0.93, 0]])
    calc = qme.get_uma_calculator(model_name="uma-s-1p1")
    calc.ensure_loaded()
    water.calc = calc
    water.info["charge"] = 0
    water.info["spin"] = 1

    H_a = water.calc.get_hessian(water, method="double_backward", symmetrize=True)
    H_01 = HessianCalculator(
        water, water.calc, delta=0.01, method="central", verbose=0
    ).calculate_numerical_hessian()
    H_005 = HessianCalculator(
        water, water.calc, delta=0.005, method="central", verbose=0
    ).calculate_numerical_hessian()
    H_rich = HessianCalculator(
        water, water.calc, delta=0.01, method="central", richardson=True, verbose=0
    ).calculate_numerical_hessian()
    H_rich_manual = (4.0 * H_005 - H_01) / 3.0

    print(f"Analytical norm: {np.linalg.norm(H_a):.6e}")
    print(f"H_0.01 norm: {np.linalg.norm(H_01):.6e}")
    print(f"H_0.005 norm: {np.linalg.norm(H_005):.6e}")
    print(f"H_richardson norm: {np.linalg.norm(H_rich):.6e}")
    print(f"H_richardson_manual norm: {np.linalg.norm(H_rich_manual):.6e}")

    print("\nRMS errors vs analytical:")
    print(f"  delta=0.01: {np.sqrt(np.mean((H_a - H_01) ** 2)):.6e}")
    print(f"  delta=0.005: {np.sqrt(np.mean((H_a - H_005) ** 2)):.6e}")
    print(f"  Richardson: {np.sqrt(np.mean((H_a - H_rich) ** 2)):.6e}")
    print(f"  Richardson_manual: {np.sqrt(np.mean((H_a - H_rich_manual) ** 2)):.6e}")

    print(f"\nImplementation matches manual: {np.allclose(H_rich, H_rich_manual, atol=1e-6)}")
    print(f"Max diff: {np.max(np.abs(H_rich - H_rich_manual)):.6e}")

    # Beta-carotene - check if 0.01 itself is problematic
    print("\n" + "=" * 80)
    print("BETA-CAROTENE (large system)")
    print("=" * 80)
    url = (
        "https://github.com/lvpp/sigma/raw/"
        "cf6ef53a5a9ffef0459b7d2dfe495ebd8d6244c8/geometry/bp86-d2svp/BETACAROTENE.xyz"
    )
    with urlopen(url) as r:
        atoms = read(StringIO(r.read().decode("utf-8")), format="xyz")

    atoms.calc = calc
    atoms.info.setdefault("charge", 0)
    atoms.info.setdefault("spin", 1)

    H_a_large = atoms.calc.get_hessian(atoms, method="double_backward", symmetrize=True)
    print(f"Analytical norm: {np.linalg.norm(H_a_large):.6e}")

    # Only compute 0.01 and 0.005 to understand the issue
    print("\nComputing FD at 0.01 (this will take time)...")
    H_01_large = HessianCalculator(
        atoms, atoms.calc, delta=0.01, method="central", verbose=0
    ).calculate_numerical_hessian()
    print(f"H_0.01 norm: {np.linalg.norm(H_01_large):.6e}")
    print(f"RMS error vs analytical: {np.sqrt(np.mean((H_a_large - H_01_large) ** 2)):.6e}")

    print("\nComputing FD at 0.005 (this will take time)...")
    H_005_large = HessianCalculator(
        atoms, atoms.calc, delta=0.005, method="central", verbose=0
    ).calculate_numerical_hessian()
    print(f"H_0.005 norm: {np.linalg.norm(H_005_large):.6e}")
    print(f"RMS error vs analytical: {np.sqrt(np.mean((H_a_large - H_005_large) ** 2)):.6e}")

    # Manual Richardson
    H_rich_manual_large = (4.0 * H_005_large - H_01_large) / 3.0
    print("\nRichardson manual:")
    print(f"  Norm: {np.linalg.norm(H_rich_manual_large):.6e}")
    print(
        f"  RMS error vs analytical: {np.sqrt(np.mean((H_a_large - H_rich_manual_large) ** 2)):.6e}"
    )

    # Check if the issue is that 0.01 FD itself is very wrong
    print("\nDifference between FD methods:")
    print(f"  ||H_0.01 - H_0.005||: {np.linalg.norm(H_01_large - H_005_large):.6e}")

    # Check element-wise - are there huge differences?
    diff_01_005 = np.abs(H_01_large - H_005_large)
    print(f"  Max element diff: {np.max(diff_01_005):.6e}")
    print(f"  Mean element diff: {np.mean(diff_01_005):.6e}")

    # If 0.01 is very wrong, Richardson will amplify it
    print("\nConclusion:")
    if np.linalg.norm(H_01_large - H_005_large) > 100:
        print("  PROBLEM: H_0.01 and H_0.005 differ greatly!")
        print("  This suggests delta=0.01 is too large for this system.")
        print("  Richardson extrapolation amplifies the error in H_0.01.")
        print("  Solution: Use smaller base delta (e.g., 0.001) for Richardson.")
    else:
        print("  H_0.01 and H_0.005 are similar - Richardson should work.")


if __name__ == "__main__":
    main()
