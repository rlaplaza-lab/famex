#!/usr/bin/env python3
"""
Demonstration of optimizer verbosity control in QME.

This example shows how to use the new verbose parameter to control
the amount of output from QME optimizers through the Explorer class.
"""

import numpy as np
from ase.build import molecule

import qme
from qme import Explorer


def main():
    """Demonstrate different verbosity levels with various optimizers."""
    print("QME Optimizer Verbosity Control Demo")
    print("=" * 50)

    # Create a test molecule
    h2o = molecule("H2O")
    h2o.positions += np.random.RandomState(42).normal(0, 0.05, h2o.positions.shape)

    # Test different optimizers with different verbosity levels
    optimizers = ["lbfgs", "bfgs", "fire", "trust-krylov", "trust-ncg"]

    for opt_name in optimizers:
        print(f"\n{opt_name.upper()} Optimizer:")
        print("-" * 30)

        # Quiet mode (verbose=0)
        print("  Quiet mode (verbose=0) - No output:")
        exp = Explorer(h2o, backend="mock", local_optimizer=opt_name, verbose=0)
        result = exp.run(mode="minima", fmax=0.5, steps=2)

        # Normal mode (verbose=1)
        print("  Normal mode (verbose=1) - Essential information:")
        exp = Explorer(h2o, backend="mock", local_optimizer=opt_name, verbose=1)
        result = exp.run(mode="minima", fmax=0.5, steps=2)

        # Verbose mode (verbose=2)
        print("  Verbose mode (verbose=2) - Detailed information:")
        exp = Explorer(h2o, backend="mock", local_optimizer=opt_name, verbose=2)
        result = exp.run(mode="minima", fmax=0.5, steps=2)

    print("\n" + "=" * 50)
    print("Demo completed! All optimizers now support verbosity control:")
    print("- verbose=0: Quiet mode (minimal output)")
    print("- verbose=1: Normal mode (essential information)")
    print("- verbose=2: Verbose mode (detailed information)")
    print("=" * 50)


if __name__ == "__main__":
    main()
