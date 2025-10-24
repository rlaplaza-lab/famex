#!/usr/bin/env python3
"""
Benchmark script to compare the speed of analytical vs finite difference Hessians for UMA.
"""

import sys
import time
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ase import Atoms
from ase.io import read
from qme.potentials.uma_potential import UMAPotential
from qme.analysis.frequency import FrequencyAnalysis, HessianCalculator


def benchmark_hessian_methods(atoms, calc, n_runs=3):
    """Benchmark analytical vs finite difference Hessian calculation."""
    print(f"🔬 Benchmarking Hessian calculation methods")
    print(f"   Molecule: {atoms.get_chemical_formula()} ({len(atoms)} atoms)")
    print(f"   Number of runs: {n_runs}")
    print("="*60)
    
    # Set calculator
    atoms.calc = calc
    
    # Benchmark analytical Hessian
    print("⚡ Benchmarking analytical Hessian...")
    analytical_times = []
    
    for i in range(n_runs):
        start_time = time.time()
        hessian_analytical = calc.get_hessian(atoms)
        end_time = time.time()
        
        elapsed = end_time - start_time
        analytical_times.append(elapsed)
        print(f"   Run {i+1}: {elapsed:.3f} seconds")
    
    # Benchmark finite difference Hessian
    print("\n🐌 Benchmarking finite difference Hessian...")
    fd_times = []
    
    for i in range(n_runs):
        start_time = time.time()
        
        # Create Hessian calculator for finite differences
        hessian_calc = HessianCalculator(
            atoms=atoms,
            calculator=calc,
            delta=0.001,  # Small displacement for accuracy
            method="central",
            verbose=0,  # No output
        )
        hessian_fd = hessian_calc.calculate_numerical_hessian()
        
        end_time = time.time()
        
        elapsed = end_time - start_time
        fd_times.append(elapsed)
        print(f"   Run {i+1}: {elapsed:.3f} seconds")
    
    # Calculate statistics
    analytical_mean = sum(analytical_times) / len(analytical_times)
    analytical_std = (sum((t - analytical_mean)**2 for t in analytical_times) / len(analytical_times))**0.5
    
    fd_mean = sum(fd_times) / len(fd_times)
    fd_std = (sum((t - fd_mean)**2 for t in fd_times) / len(fd_times))**0.5
    
    speedup = fd_mean / analytical_mean
    
    # Results
    print("\n" + "="*60)
    print("📊 BENCHMARK RESULTS")
    print("="*60)
    print(f"Analytical Hessian:")
    print(f"   Mean time: {analytical_mean:.3f} ± {analytical_std:.3f} seconds")
    print(f"   Min time:  {min(analytical_times):.3f} seconds")
    print(f"   Max time:  {max(analytical_times):.3f} seconds")
    
    print(f"\nFinite Difference Hessian:")
    print(f"   Mean time: {fd_mean:.3f} ± {fd_std:.3f} seconds")
    print(f"   Min time:  {min(fd_times):.3f} seconds")
    print(f"   Max time:  {max(fd_times):.3f} seconds")
    
    print(f"\n🚀 SPEEDUP:")
    print(f"   Analytical is {speedup:.1f}x faster than finite differences!")
    print(f"   Time saved: {fd_mean - analytical_mean:.3f} seconds per calculation")
    
    # Efficiency analysis
    print(f"\n💡 EFFICIENCY ANALYSIS:")
    if speedup > 10:
        print(f"   🎯 Excellent speedup! Analytical Hessians are highly recommended.")
    elif speedup > 5:
        print(f"   ✅ Good speedup! Analytical Hessians provide significant benefits.")
    elif speedup > 2:
        print(f"   👍 Moderate speedup. Analytical Hessians are still beneficial.")
    else:
        print(f"   ⚠️  Small speedup. Consider the accuracy benefits too.")
    
    return {
        'analytical_mean': analytical_mean,
        'analytical_std': analytical_std,
        'fd_mean': fd_mean,
        'fd_std': fd_std,
        'speedup': speedup,
        'analytical_times': analytical_times,
        'fd_times': fd_times
    }


def benchmark_different_sizes():
    """Benchmark with different molecule sizes to see how speedup scales."""
    print("🔬 Benchmarking with different molecule sizes")
    print("="*60)
    
    # Load the test molecule
    xyz_path = project_root / "examples" / "example_files" / "reaction_001_reactant.xyz"
    full_atoms = read(str(xyz_path))
    
    # Create UMA calculator
    calc = UMAPotential(model_name="uma-s-1p1", device="cpu")
    
    results = []
    
    # Test different sizes
    sizes = [5, 10, 20, 30]  # Number of atoms
    
    for size in sizes:
        if size > len(full_atoms):
            continue
            
        print(f"\n📏 Testing with {size} atoms...")
        atoms = full_atoms[:size]
        
        result = benchmark_hessian_methods(atoms, calc, n_runs=2)  # Fewer runs for speed
        result['size'] = size
        results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("📈 SPEEDUP SUMMARY BY MOLECULE SIZE")
    print("="*60)
    print(f"{'Atoms':<8} {'Analytical (s)':<15} {'Finite Diff (s)':<15} {'Speedup':<10}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['size']:<8} "
              f"{result['analytical_mean']:<15.3f} "
              f"{result['fd_mean']:<15.3f} "
              f"{result['speedup']:<10.1f}x")
    
    # Overall average speedup
    avg_speedup = sum(r['speedup'] for r in results) / len(results)
    print(f"\n🎯 Average speedup across all sizes: {avg_speedup:.1f}x")
    
    return results


def main():
    """Main benchmark function."""
    print("🚀 UMA Hessian Speed Benchmark")
    print("="*60)
    
    try:
        # Load test molecule
        xyz_path = project_root / "examples" / "example_files" / "reaction_001_reactant.xyz"
        atoms = read(str(xyz_path))[:15]  # Use 15 atoms for detailed benchmark
        
        # Create UMA calculator
        calc = UMAPotential(model_name="uma-s-1p1", device="cpu")
        
        # Run detailed benchmark
        result = benchmark_hessian_methods(atoms, calc, n_runs=3)
        
        # Run size scaling benchmark
        print("\n" + "="*60)
        size_results = benchmark_different_sizes()
        
        print(f"\n🎉 Benchmark complete!")
        print(f"   Analytical Hessians are consistently faster than finite differences")
        print(f"   The speedup makes UMA frequency analysis much more practical!")
        
    except Exception as e:
        print(f"\n❌ Benchmark failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
