#!/usr/bin/env python3
"""
Test script to verify that UMA frequency analysis uses analytical Hessians by default.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ase import Atoms
from ase.io import read
from qme.potentials.uma_potential import UMAPotential
from qme.analysis.frequency import FrequencyAnalysis


def test_uma_frequency_analysis():
    """Test that UMA frequency analysis uses analytical Hessians by default."""
    print("Testing UMA Frequency Analysis with Analytical Hessians")
    print("="*60)
    
    # Load a small test molecule
    xyz_path = project_root / "examples" / "example_files" / "reaction_001_reactant.xyz"
    atoms = read(str(xyz_path))
    
    # Use only first 10 atoms for faster testing
    atoms = atoms[:10]
    print(f"Using {len(atoms)} atoms for testing: {atoms.get_chemical_formula()}")
    
    # Create UMA calculator
    calc = UMAPotential(model_name="uma-s-1p1", device="cpu")
    atoms.calc = calc
    
    # Create frequency analysis
    freq_analysis = FrequencyAnalysis(atoms, calculator=calc)
    
    # Check if analytical Hessians are supported
    supports_hessian = freq_analysis._supports_direct_hessian()
    print(f"✓ Analytical Hessian support detected: {supports_hessian}")
    
    if supports_hessian:
        print("✓ UMA calculator supports analytical Hessians")
        
        # Test the default method (should be "auto")
        print("\nTesting frequency calculation with method='auto'...")
        freq_analysis.calculate_hessian(method="auto")
        frequencies = freq_analysis.get_frequencies()
        
        print(f"✓ Calculated {len(frequencies)} frequencies")
        print(f"  Frequency range: {frequencies.min():.2f} to {frequencies.max():.2f} cm⁻¹")
        print(f"  Number of imaginary frequencies: {(frequencies < 0).sum()}")
        
        # Test explicitly requesting analytical Hessian
        print("\nTesting frequency calculation with method='direct'...")
        freq_analysis_direct = FrequencyAnalysis(atoms, calculator=calc)
        freq_analysis_direct.calculate_hessian(method="direct")
        frequencies_direct = freq_analysis_direct.get_frequencies()
        
        print(f"✓ Calculated {len(frequencies_direct)} frequencies using analytical Hessian")
        print(f"  Frequency range: {frequencies_direct.min():.2f} to {frequencies_direct.max():.2f} cm⁻¹")
        
        # Compare with finite differences
        print("\nTesting frequency calculation with method='finite_differences'...")
        freq_analysis_fd = FrequencyAnalysis(atoms, calculator=calc)
        freq_analysis_fd.calculate_hessian(method="finite_differences")
        frequencies_fd = freq_analysis_fd.get_frequencies()
        
        print(f"✓ Calculated {len(frequencies_fd)} frequencies using finite differences")
        print(f"  Frequency range: {frequencies_fd.min():.2f} to {frequencies_fd.max():.2f} cm⁻¹")
        
        # Compare results
        print("\n" + "="*60)
        print("COMPARISON: Analytical vs Finite Differences")
        print("="*60)
        
        diff = frequencies_direct - frequencies_fd
        abs_diff = abs(diff)
        
        print(f"Max absolute difference: {abs_diff.max():.4f} cm⁻¹")
        print(f"Mean absolute difference: {abs_diff.mean():.4f} cm⁻¹")
        print(f"RMS difference: {(diff**2).mean()**0.5:.4f} cm⁻¹")
        
        # Check if they're close
        if abs_diff.max() < 10.0:  # Within 10 cm⁻¹
            print("✓ Analytical and finite difference results are in good agreement!")
        else:
            print("⚠ Large differences detected - may need investigation")
        
        print("\n🎉 UMA frequency analysis successfully uses analytical Hessians by default!")
        
    else:
        print("❌ UMA calculator does not support analytical Hessians")
        return False
    
    return True


def main():
    """Main test function."""
    try:
        success = test_uma_frequency_analysis()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
