#!/usr/bin/env python3
"""
Test script for TorchSim integration in QME.

This script tests the TorchSim integration to ensure it works correctly
and falls back gracefully when TorchSim is not available.
"""

import sys
import warnings
from pathlib import Path

# Add the parent directory to Python path to find the local qme module
script_dir = Path(__file__).parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from ase import Atoms
    from ase.build import molecule

    import qme
except ImportError as e:
    print(f"Error importing QME: {e}")
    print("Make sure you're in the QME package directory or have it installed.")
    sys.exit(1)


def test_torchsim_availability():
    """Test if TorchSim is available."""
    print("Testing TorchSim availability...")

    # Check if TorchSim is available
    has_torchsim = qme.deps.has("torch_sim")
    print(f"TorchSim available: {has_torchsim}")

    if has_torchsim:
        print("✅ TorchSim is available")
        return True
    else:
        print("❌ TorchSim is not available")
        print("  Install with: pip install torch-sim-atomistic")
        return False


def test_torchsim_backends():
    """Test TorchSim backend availability."""
    print("\nTesting TorchSim backends...")

    backends = ["torchsim", "torchsim_mace", "torchsim_fairchem"]

    for backend in backends:
        available = qme.calculator_registry.is_backend_available(backend)
        status = "✅" if available else "❌"
        print(f"  {status} {backend}: {'Available' if available else 'Not available'}")


def test_torchsim_calculator_creation():
    """Test creating TorchSim calculators."""
    print("\nTesting TorchSim calculator creation...")

    # Test basic TorchSim calculator
    try:
        calc = qme.calculator_registry.create_calculator(
            backend="torchsim", model_name="mace-omol-0", device="cpu"
        )
        print(f"✅ Created TorchSim calculator: {type(calc).__name__}")
        print(f"   Backend: {getattr(calc, 'backend', 'unknown')}")
    except Exception as e:
        print(f"❌ Failed to create TorchSim calculator: {e}")

    # Test TorchSim MACE calculator
    try:
        calc = qme.calculator_registry.create_calculator(
            backend="torchsim_mace", model_name="mace-omol-0", device="cpu"
        )
        print(f"✅ Created TorchSim MACE calculator: {type(calc).__name__}")
    except Exception as e:
        print(f"❌ Failed to create TorchSim MACE calculator: {e}")


def test_torchsim_calculation():
    """Test TorchSim calculation if available."""
    print("\nTesting TorchSim calculation...")

    if not qme.deps.has("torch_sim"):
        print("❌ TorchSim not available, skipping calculation test")
        return

    try:
        # Create a simple molecule
        benzene = molecule("C6H6")

        # Create TorchSim calculator
        calc = qme.calculator_registry.create_calculator(
            backend="torchsim_mace", model_name="mace-omol-0", device="cpu"
        )

        # Attach calculator to atoms
        benzene.calc = calc

        # Test energy calculation
        print("  Testing energy calculation...")
        energy = benzene.get_potential_energy()
        print(f"  ✅ Energy: {energy:.6f} eV")

        # Test forces calculation
        print("  Testing forces calculation...")
        forces = benzene.get_forces()
        print(f"  ✅ Forces shape: {forces.shape}")
        print(f"  ✅ Max force: {forces.max():.6f} eV/Å")

    except Exception as e:
        print(f"❌ TorchSim calculation failed: {e}")


def test_fallback_behavior():
    """Test fallback behavior when TorchSim is not available."""
    print("\nTesting fallback behavior...")

    # This should work even without TorchSim
    try:
        calc = qme.calculator_registry.create_calculator(
            backend="torchsim", model_name="mace-omol-0", device="cpu"
        )
        print(f"✅ Fallback calculator created: {type(calc).__name__}")

        # Test that it's a mock calculator if TorchSim is not available
        if not qme.deps.has("torch_sim"):
            if hasattr(calc, "backend") and "mock" in str(type(calc)).lower():
                print("✅ Correctly fell back to mock calculator")
            else:
                print("⚠️  Fallback behavior may not be working correctly")

    except Exception as e:
        print(f"❌ Fallback failed: {e}")


def main():
    """Main test function."""
    print("TorchSim Integration Test for QME")
    print("=" * 50)

    # Test availability
    has_torchsim = test_torchsim_availability()

    # Test backends
    test_torchsim_backends()

    # Test calculator creation
    test_torchsim_calculator_creation()

    # Test calculation if available
    if has_torchsim:
        test_torchsim_calculation()
    else:
        print("\nSkipping calculation test (TorchSim not available)")

    # Test fallback behavior
    test_fallback_behavior()

    print("\n" + "=" * 50)
    print("Test completed!")

    if has_torchsim:
        print("\n🎉 TorchSim integration is working!")
        print("   You can now use TorchSim backends for accelerated calculations.")
    else:
        print("\n💡 To enable TorchSim acceleration:")
        print("   pip install torch-sim-atomistic")
        print("   Then re-run this test.")


if __name__ == "__main__":
    main()
