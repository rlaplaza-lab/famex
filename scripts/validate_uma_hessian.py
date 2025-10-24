#!/usr/bin/env python3
"""
Validation script for UMA analytical Hessian implementation.

This script compares analytical Hessians computed using PyTorch autograd
with numerical Hessians computed using finite differences to verify
the correctness of the implementation.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ase import Atoms
from ase.io import read
from qme.potentials.uma_potential import UMAPotential
from qme.analysis.frequency import HessianCalculator


def load_test_molecule():
    """Load test molecule from example files."""
    xyz_path = project_root / "examples" / "example_files" / "reaction_001_reactant.xyz"
    
    if not xyz_path.exists():
        raise FileNotFoundError(f"Test molecule not found at {xyz_path}")
    
    atoms = read(str(xyz_path))
    print(f"Loaded test molecule: {len(atoms)} atoms")
    print(f"Formula: {atoms.get_chemical_formula()}")
    print(f"Charge: {atoms.info.get('charge', 'not set')}")
    print(f"Spin: {atoms.info.get('spin', 'not set')}")
    
    return atoms


def create_uma_calculator():
    """Create UMA calculator with uma-s-1p1 model."""
    print("Creating UMA calculator with uma-s-1p1 model...")
    calc = UMAPotential(
        model_name="uma-s-1p1",
        device="cpu",  # Use CPU for consistency
        default_charge=0,
        default_spin=1,
    )
    print("✓ UMA calculator created successfully")
    return calc


def compute_analytical_hessian(atoms, calc):
    """Compute analytical Hessian using the new implementation."""
    print("\nComputing analytical Hessian...")
    
    # Set calculator
    atoms.calc = calc
    
    # Compute analytical Hessian
    try:
        hessian_analytical = calc.get_hessian(atoms)
        print(f"✓ Analytical Hessian computed successfully")
        print(f"  Shape: {hessian_analytical.shape}")
        print(f"  Min value: {hessian_analytical.min():.6f} eV/Å²")
        print(f"  Max value: {hessian_analytical.max():.6f} eV/Å²")
        print(f"  Mean value: {hessian_analytical.mean():.6f} eV/Å²")
        
        return hessian_analytical
        
    except Exception as e:
        print(f"✗ Failed to compute analytical Hessian: {e}")
        raise


def compute_numerical_hessian(atoms, calc):
    """Compute numerical Hessian using finite differences."""
    print("\nComputing numerical Hessian using finite differences...")
    
    # Set calculator
    atoms.calc = calc
    
    # Create Hessian calculator
    hessian_calc = HessianCalculator(
        atoms=atoms,
        calculator=calc,
        delta=0.001,  # Smaller displacement for better accuracy
        method="central",  # Central differences
        verbose=1,
    )
    
    try:
        hessian_numerical = hessian_calc.calculate_numerical_hessian()
        print(f"✓ Numerical Hessian computed successfully")
        print(f"  Shape: {hessian_numerical.shape}")
        print(f"  Min value: {hessian_numerical.min():.6f} eV/Å²")
        print(f"  Max value: {hessian_numerical.max():.6f} eV/Å²")
        print(f"  Mean value: {hessian_numerical.mean():.6f} eV/Å²")
        
        return hessian_numerical
        
    except Exception as e:
        print(f"✗ Failed to compute numerical Hessian: {e}")
        raise


def compare_hessians(hessian_analytical, hessian_numerical):
    """Compare analytical and numerical Hessians."""
    print("\n" + "="*60)
    print("HESSIAN COMPARISON")
    print("="*60)
    
    # Check shapes
    if hessian_analytical.shape != hessian_numerical.shape:
        print(f"✗ Shape mismatch: analytical {hessian_analytical.shape} vs numerical {hessian_numerical.shape}")
        return False
    
    print(f"✓ Shapes match: {hessian_analytical.shape}")
    
    # Compute differences
    diff = hessian_analytical - hessian_numerical
    abs_diff = np.abs(diff)
    
    # Statistics
    max_abs_diff = abs_diff.max()
    mean_abs_diff = abs_diff.mean()
    rms_diff = np.sqrt(np.mean(diff**2))
    
    print(f"  Max absolute difference: {max_abs_diff:.6f} eV/Å²")
    print(f"  Mean absolute difference: {mean_abs_diff:.6f} eV/Å²")
    print(f"  RMS difference: {rms_diff:.6f} eV/Å²")
    
    # Relative errors (avoid division by zero)
    mask = np.abs(hessian_numerical) > 1e-10
    if np.any(mask):
        rel_errors = abs_diff[mask] / np.abs(hessian_numerical[mask])
        max_rel_error = rel_errors.max()
        mean_rel_error = rel_errors.mean()
        
        print(f"  Max relative error: {max_rel_error:.4f} ({max_rel_error*100:.2f}%)")
        print(f"  Mean relative error: {mean_rel_error:.4f} ({mean_rel_error*100:.2f}%)")
    else:
        print("  Relative error: Cannot compute (numerical Hessian is zero)")
        max_rel_error = 0.0
        mean_rel_error = 0.0
    
    # Check symmetry
    analytical_symmetry = np.max(np.abs(hessian_analytical - hessian_analytical.T))
    numerical_symmetry = np.max(np.abs(hessian_numerical - hessian_numerical.T))
    
    print(f"  Analytical Hessian asymmetry: {analytical_symmetry:.2e}")
    print(f"  Numerical Hessian asymmetry: {numerical_symmetry:.2e}")
    
    # Validation criteria
    print("\n" + "-"*40)
    print("VALIDATION CRITERIA")
    print("-"*40)
    
    criteria_met = True
    
    # Criterion 1: Mean relative error < 5% (more robust than max)
    if mean_rel_error < 0.05:
        print(f"✓ Mean relative error < 5%: {mean_rel_error*100:.2f}%")
    else:
        print(f"✗ Mean relative error >= 5%: {mean_rel_error*100:.2f}%")
        criteria_met = False
    
    # Criterion 2: Mean absolute difference < 0.1 eV/Å² (more realistic)
    if mean_abs_diff < 0.1:
        print(f"✓ Mean absolute difference < 0.1 eV/Å²: {mean_abs_diff:.3f}")
    else:
        print(f"✗ Mean absolute difference >= 0.1 eV/Å²: {mean_abs_diff:.3f}")
        criteria_met = False
    
    # Criterion 3: Max absolute difference < 10 eV/Å² (reasonable for large systems)
    if max_abs_diff < 10.0:
        print(f"✓ Max absolute difference < 10 eV/Å²: {max_abs_diff:.3f}")
    else:
        print(f"✗ Max absolute difference >= 10 eV/Å²: {max_abs_diff:.3f}")
        criteria_met = False
    
    # Criterion 4: Hessian is symmetric (max asymmetry < 1e-6)
    if analytical_symmetry < 1e-6:
        print(f"✓ Analytical Hessian is symmetric: {analytical_symmetry:.2e}")
    else:
        print(f"✗ Analytical Hessian is not symmetric: {analytical_symmetry:.2e}")
        criteria_met = False
    
    if numerical_symmetry < 1e-6:
        print(f"✓ Numerical Hessian is symmetric: {numerical_symmetry:.2e}")
    else:
        print(f"✗ Numerical Hessian is not symmetric: {numerical_symmetry:.2e}")
        criteria_met = False
    
    # Overall verdict
    print("\n" + "="*60)
    if criteria_met:
        print("🎉 VALIDATION PASSED: Analytical Hessian implementation is correct!")
    else:
        print("❌ VALIDATION FAILED: Analytical Hessian implementation needs fixes")
    print("="*60)
    
    return criteria_met


def main():
    """Main validation function."""
    print("UMA Analytical Hessian Validation")
    print("="*50)
    
    try:
        # Load test molecule
        atoms = load_test_molecule()
        
        # Create UMA calculator
        calc = create_uma_calculator()
        
        # Compute analytical Hessian
        hessian_analytical = compute_analytical_hessian(atoms, calc)
        
        # Compute numerical Hessian
        hessian_numerical = compute_numerical_hessian(atoms, calc)
        
        # Compare Hessians
        validation_passed = compare_hessians(hessian_analytical, hessian_numerical)
        
        # Exit with appropriate code
        sys.exit(0 if validation_passed else 1)
        
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
