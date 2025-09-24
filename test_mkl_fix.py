#!/usr/bin/env python3
"""
Test script to verify MKL threading fix for QME.

This script tests whether the Intel MKL threading layer fix resolves
symbol lookup errors in conda environments.

Usage:
    python test_mkl_fix.py

Or with environment variable:
    MKL_THREADING_LAYER=GNU python test_mkl_fix.py
"""

import os
import sys


def test_mkl_fix():
    """Test if MKL threading issues are resolved."""
    print("Testing QME MKL compatibility...")
    print(f"Python version: {sys.version}")

    # Check environment variable
    mkl_layer = os.environ.get("MKL_THREADING_LAYER", "Not set")
    print(f"MKL_THREADING_LAYER: {mkl_layer}")

    try:
        # This import chain typically triggers the MKL error
        print("Importing QME...")
        import qme

        print("✓ QME imported successfully")

        print("Testing AIMNET2 backend...")
        optimizer = qme.QMEOptimizer(backend="aimnet2")
        print("✓ AIMNET2 backend created successfully")

        print("\n✓ All tests passed! QME should work correctly.")
        return True

    except Exception as e:
        print(f"\n✗ Error encountered: {e}")
        print("\nTry running with: MKL_THREADING_LAYER=GNU python test_mkl_fix.py")
        return False


if __name__ == "__main__":
    success = test_mkl_fix()
    sys.exit(0 if success else 1)
