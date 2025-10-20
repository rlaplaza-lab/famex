#!/usr/bin/env python3
"""
Comprehensive demonstration of optimizer verbosity control in QME.

This example shows how to use the new verbose parameter to control
the amount of output from all QME optimizers, including:
- SciPy Hessian-based optimizers (TrustKrylov, TrustNCG, TrustExact, NewtonCG)
- ASE optimizers (LBFGS, BFGS, FIRE)
- Sella optimizer
- Integration with Explorer class
"""

import os
import sys

import numpy as np
from ase.build import molecule

import qme

# Add the parent directory to the path so we can import from qme.core
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qme import Explorer
from qme.core.ase_optimizer_wrappers import VerboseBFGS, VerboseFIRE, VerboseLBFGS
from qme.core.scipy_optimizers import NewtonCG, TrustExact, TrustKrylov, TrustKrylovTS, TrustNCG


def test_scipy_optimizers():
    """Test SciPy Hessian-based optimizers."""
    print("SciPy Hessian-Based Optimizers")
    print("=" * 50)

    # Create a test molecule
    h2o = molecule("H2O")
    h2o.positions += np.random.RandomState(42).normal(0, 0.05, h2o.positions.shape)
    h2o.calc = qme.MockCalculator(backend="mock")

    optimizers = [
        ("TrustKrylov", TrustKrylov),
        ("TrustKrylovTS", TrustKrylovTS),
        ("TrustNCG", TrustNCG),
        ("TrustExact", TrustExact),
        ("NewtonCG", NewtonCG),
    ]

    for name, opt_class in optimizers:
        print(f"\n{name} - Quiet mode (verbose=0):")
        print("-" * 40)
        opt = opt_class(h2o.copy(), logfile="-", verbose=0)
        opt.run(fmax=0.5, steps=2)

        print(f"\n{name} - Normal mode (verbose=1):")
        print("-" * 40)
        opt = opt_class(h2o.copy(), logfile="-", verbose=1)
        opt.run(fmax=0.5, steps=2)


def test_ase_optimizers():
    """Test ASE optimizers with verbosity control."""
    print("\n\nASE Optimizers with Verbosity Control")
    print("=" * 50)

    # Create a test molecule
    h2o = molecule("H2O")
    h2o.positions += np.random.RandomState(42).normal(0, 0.05, h2o.positions.shape)
    h2o.calc = qme.MockCalculator(backend="mock")

    optimizers = [
        ("VerboseLBFGS", VerboseLBFGS),
        ("VerboseBFGS", VerboseBFGS),
        ("VerboseFIRE", VerboseFIRE),
    ]

    for name, opt_class in optimizers:
        print(f"\n{name} - Quiet mode (verbose=0):")
        print("-" * 40)
        opt = opt_class(h2o.copy(), logfile="-", verbose=0)
        opt.run(fmax=0.5, steps=2)

        print(f"\n{name} - Normal mode (verbose=1):")
        print("-" * 40)
        opt = opt_class(h2o.copy(), logfile="-", verbose=1)
        opt.run(fmax=0.5, steps=2)


def test_explorer_integration():
    """Test integration with Explorer class."""
    print("\n\nExplorer Integration")
    print("=" * 50)

    # Create a test molecule
    h2o = molecule("H2O")
    h2o.positions += np.random.RandomState(42).normal(0, 0.05, h2o.positions.shape)

    optimizers = ["lbfgs", "bfgs", "fire", "trust-krylov", "trust-ncg"]

    for opt_name in optimizers:
        print(f"\nExplorer with {opt_name.upper()} - Quiet mode (verbose=0):")
        print("-" * 40)
        exp = Explorer(h2o, backend="mock", local_optimizer=opt_name, verbose=0)
        exp.run(mode="minima", fmax=0.5, steps=2)

        print(f"\nExplorer with {opt_name.upper()} - Normal mode (verbose=1):")
        print("-" * 40)
        exp = Explorer(h2o, backend="mock", local_optimizer=opt_name, verbose=1)
        exp.run(mode="minima", fmax=0.5, steps=2)


def main():
    """Run all demonstrations."""
    print("QME Optimizer Verbosity Control - Comprehensive Demo")
    print("=" * 60)

    test_scipy_optimizers()
    test_ase_optimizers()
    test_explorer_integration()

    print("\n\n" + "=" * 60)
    print("Demo completed! All optimizers now support verbosity control:")
    print("- verbose=0: Quiet mode (minimal output)")
    print("- verbose=1: Normal mode (essential information)")
    print("- verbose=2: Verbose mode (detailed information)")
    print("=" * 60)


if __name__ == "__main__":
    main()
