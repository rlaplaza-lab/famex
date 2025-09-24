#!/usr/bin/env python3

import numpy as np
from ase.build import molecule

from qme import QMEOptimizer


def test_force_values():
    """Test what kind of forces our mock calculators generate."""

    # Test with each backend
    backends = ["so3lr", "uma", "aimnet2"]

    for backend in backends:
        print(f"\n=== Testing {backend.upper()} backend ===")

        # Create optimizer
        optimizer = QMEOptimizer(use_mock=True, backend=backend)

        # Test with H2 molecule
        h2 = molecule("H2")
        positions = h2.get_positions()
        positions[1][0] += 0.2  # Stretch bond
        h2.set_positions(positions)
        h2.calc = optimizer.calculator

        # Get energy and forces
        energy = h2.get_potential_energy()
        forces = h2.get_forces()

        print(f"Energy: {energy:.6f} eV")
        print(f"Max force: {np.max(np.abs(forces)):.6f} eV/Å")
        print(f"Forces shape: {forces.shape}")
        print(f"Forces:\n{forces}")

        # Check for problematic values
        if np.any(np.isnan(forces)) or np.any(np.isinf(forces)):
            print("WARNING: NaN or Inf values in forces!")

        if np.max(np.abs(forces)) > 100:
            print("WARNING: Very large forces detected!")

        # Test Hessian condition by computing second derivatives
        # (approximate numerical Hessian)
        try:
            from ase.optimize.bfgs import BFGS

            opt = BFGS(h2)
            # Try to prepare first step
            pos = h2.get_positions()
            forces = h2.get_forces()
            opt.prepare_step(pos, forces)
            print("BFGS step preparation: OK")
        except Exception as e:
            print(f"BFGS step preparation failed: {e}")


if __name__ == "__main__":
    test_force_values()
