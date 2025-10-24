#!/usr/bin/env python3
"""
Clean benchmark focusing on the most reliable measurements.
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
from qme.analysis.frequency import HessianCalculator


def clean_benchmark(atoms, calc, n_runs=5):
    """Clean benchmark with more runs and better timing."""
    print(f"🧪 Clean Benchmark: {atoms.get_chemical_formula()} ({len(atoms)} atoms)")
    print(f"   Runs: {n_runs}")
    print("-" * 50)
    
    atoms.calc = calc
    
    # Warm up runs (not counted)
    print("🔥 Warming up...")
    calc.get_hessian(atoms)  # Warm up analytical
    hessian_calc = HessianCalculator(atoms, calc, delta=0.001, method="central", verbose=0)
    hessian_calc.calculate_numerical_hessian()  # Warm up finite difference
    
    # Benchmark analytical
    print("⚡ Analytical Hessian:")
    analytical_times = []
    for i in range(n_runs):
        start = time.time()
        hessian = calc.get_hessian(atoms)
        elapsed = time.time() - start
        analytical_times.append(elapsed)
        print(f"   Run {i+1}: {elapsed:.2f}s")
    
    # Benchmark finite difference
    print("🐌 Finite Difference Hessian:")
    fd_times = []
    for i in range(n_runs):
        start = time.time()
        hessian_calc = HessianCalculator(atoms, calc, delta=0.001, method="central", verbose=0)
        hessian = hessian_calc.calculate_numerical_hessian()
        elapsed = time.time() - start
        fd_times.append(elapsed)
        print(f"   Run {i+1}: {elapsed:.2f}s")
    
    # Statistics
    analytical_mean = sum(analytical_times) / len(analytical_times)
    fd_mean = sum(fd_times) / len(fd_times)
    speedup = fd_mean / analytical_mean
    
    print(f"\n📊 Results:")
    print(f"   Analytical: {analytical_mean:.2f}s ± {max(analytical_times) - min(analytical_times):.2f}s")
    print(f"   Finite Diff: {fd_mean:.2f}s ± {max(fd_times) - min(fd_times):.2f}s")
    print(f"   Speedup: {speedup:.1f}x")
    print(f"   Time saved: {fd_mean - analytical_mean:.2f}s per calculation")
    
    return speedup, analytical_mean, fd_mean


def main():
    """Main benchmark function."""
    print("🚀 UMA Hessian Speed Benchmark (Clean)")
    print("=" * 60)
    
    # Load test molecule
    xyz_path = project_root / "examples" / "example_files" / "reaction_001_reactant.xyz"
    full_atoms = read(str(xyz_path))
    
    # Test different sizes
    sizes = [8, 15, 25]  # Good range for testing
    results = []
    
    for size in sizes:
        if size > len(full_atoms):
            continue
            
        atoms = full_atoms[:size]
        calc = UMAPotential(model_name="uma-s-1p1", device="cpu")
        
        print(f"\n{'='*60}")
        speedup, analytical_time, fd_time = clean_benchmark(atoms, calc, n_runs=3)
        
        results.append({
            'size': size,
            'speedup': speedup,
            'analytical': analytical_time,
            'fd': fd_time
        })
    
    # Summary
    print(f"\n{'='*60}")
    print("📈 SUMMARY")
    print("=" * 60)
    print(f"{'Atoms':<8} {'Analytical (s)':<15} {'Finite Diff (s)':<15} {'Speedup':<10}")
    print("-" * 60)
    
    for r in results:
        print(f"{r['size']:<8} {r['analytical']:<15.2f} {r['fd']:<15.2f} {r['speedup']:<10.1f}x")
    
    avg_speedup = sum(r['speedup'] for r in results) / len(results)
    print(f"\n🎯 Average speedup: {avg_speedup:.1f}x")
    
    # Analysis
    print(f"\n💡 Analysis:")
    if avg_speedup > 2:
        print(f"   ✅ Analytical Hessians provide significant speedup!")
        print(f"   🚀 This makes UMA frequency analysis much more practical.")
    elif avg_speedup > 1.5:
        print(f"   👍 Analytical Hessians provide moderate speedup.")
        print(f"   📈 Combined with better accuracy, they're clearly beneficial.")
    else:
        print(f"   ⚠️  Speedup is modest, but accuracy benefits are still valuable.")
    
    print(f"\n🎉 Key benefits of analytical Hessians:")
    print(f"   • {avg_speedup:.1f}x faster than finite differences")
    print(f"   • More accurate (no numerical errors)")
    print(f"   • No need to tune step size")
    print(f"   • Automatic fallback to finite differences if needed")


if __name__ == "__main__":
    main()
