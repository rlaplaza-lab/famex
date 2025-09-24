#!/usr/bin/env python3
"""
Debug script to test QME functionality and identify CI issues.
"""

import sys
import traceback


def test_basic_imports():
    """Test basic imports."""
    print("=== Testing Basic Imports ===")
    try:
        import qme

        print(f"✅ QME imported successfully, version: {qme.__version__}")
    except Exception as e:
        print(f"❌ QME import failed: {e}")
        return False

    try:
        from qme import QMEOptimizer

        print("✅ QMEOptimizer imported successfully")
    except Exception as e:
        print(f"❌ QMEOptimizer import failed: {e}")
        return False

    return True


def test_so3lr_imports():
    """Test SO3LR related imports."""
    print("\n=== Testing SO3LR Imports ===")
    try:
        from qme.so3lr_potential import SO3LRPotential, get_mock_so3lr_calculator

        print("✅ SO3LR potential modules imported successfully")
    except Exception as e:
        print(f"❌ SO3LR potential import failed: {e}")
        traceback.print_exc()
        return False

    return True


def test_so3lr_functionality():
    """Test SO3LR functionality in mock mode."""
    print("\n=== Testing SO3LR Functionality ===")
    try:
        from qme.so3lr_potential import get_mock_so3lr_calculator

        calc = get_mock_so3lr_calculator()
        print("✅ Mock SO3LR calculator created successfully")

        from ase.build import molecule

        atoms = molecule("H2")
        atoms.calc = calc

        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        print(
            f"✅ Mock SO3LR calculation successful - Energy: {energy}, Forces shape: {forces.shape}"
        )

    except Exception as e:
        print(f"❌ SO3LR functionality test failed: {e}")
        traceback.print_exc()
        return False

    return True


def test_qme_optimizer():
    """Test QME optimizer with SO3LR backend."""
    print("\n=== Testing QME Optimizer with SO3LR Backend ===")
    try:
        from qme import QMEOptimizer

        qme = QMEOptimizer(backend="so3lr", use_mock=True)
        print("✅ QME Optimizer with SO3LR backend created successfully")

        from ase.build import molecule

        atoms = molecule("H2")
        qme.atoms = atoms

        print(f"✅ Structure loaded: {len(atoms)} atoms")

    except Exception as e:
        print(f"❌ QME optimizer test failed: {e}")
        traceback.print_exc()
        return False

    return True


def main():
    """Run all debug tests."""
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")

    tests = [
        test_basic_imports,
        test_so3lr_imports,
        test_so3lr_functionality,
        test_qme_optimizer,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"Test {test.__name__} failed")
        except Exception as e:
            print(f"Test {test.__name__} crashed: {e}")
            traceback.print_exc()

    print(f"\n=== Summary ===")
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
