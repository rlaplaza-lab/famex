#!/usr/bin/env python3
"""
TorchSim Performance Benchmark for QME.

This script benchmarks the performance of TorchSim backends compared to
standard ASE-based backends, measuring speedup and accuracy.

Usage:
    python torchsim_performance_benchmark.py [--backends BACKEND1,BACKEND2,...] [--device DEVICE]
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from ase import Atoms
from ase.build import molecule

import qme
from qme.dependencies import deps


def create_test_molecules() -> List[Atoms]:
    """Create test molecules of varying sizes for benchmarking."""
    molecules = []

    # Small molecules
    molecules.append(("H2O", molecule("H2O")))
    molecules.append(("NH3", molecule("NH3")))
    molecules.append(("CH4", molecule("CH4")))

    # Medium molecules
    molecules.append(("C6H6", molecule("C6H6")))
    molecules.append(("C2H6", molecule("C2H6")))
    molecules.append(("CH3OH", molecule("CH3OH")))

    # Larger molecules
    molecules.append(("C8H10", molecule("C8H10")))  # p-xylene
    molecules.append(("C10H8", molecule("C10H8")))  # naphthalene

    return molecules


def benchmark_single_calculation(
    atoms: Atoms, backend: str, model_name: Optional[str] = None, device: str = "cpu"
) -> Dict[str, Any]:
    """Benchmark a single energy/force calculation."""
    results = {
        "backend": backend,
        "model_name": model_name,
        "device": device,
        "n_atoms": len(atoms),
        "formula": atoms.get_chemical_formula(),
        "available": False,
        "error": None,
        "timings": {},
        "energies": {},
        "forces": {},
    }

    try:
        # Check if backend is available
        if not qme.calculator_registry.is_backend_available(backend):
            results["error"] = f"Backend {backend} not available"
            return results

        results["available"] = True

        # Create calculator
        calc = qme.calculator_registry.create_calculator(
            backend=backend,
            model_name=model_name,
            device=device,
        )

        # Attach to atoms
        atoms.calc = calc

        # Benchmark energy calculation
        print(f"  Testing energy calculation...")
        start_time = time.perf_counter()
        energy = atoms.get_potential_energy()
        energy_time = time.perf_counter() - start_time

        results["timings"]["energy"] = energy_time
        results["energies"]["single"] = float(energy)

        # Benchmark forces calculation
        print(f"  Testing forces calculation...")
        start_time = time.perf_counter()
        forces = atoms.get_forces()
        forces_time = time.perf_counter() - start_time

        results["timings"]["forces"] = forces_time
        results["forces"]["shape"] = forces.shape
        results["forces"]["max"] = float(forces.max())
        results["forces"]["rms"] = float(np.sqrt(np.mean(forces**2)))

        # Benchmark multiple calculations (to test consistency)
        print(f"  Testing multiple calculations...")
        energies = []
        start_time = time.perf_counter()

        for i in range(5):  # 5 calculations
            energy = atoms.get_potential_energy()
            energies.append(float(energy))

        multi_time = time.perf_counter() - start_time

        results["timings"]["multi_energy"] = multi_time
        results["timings"]["avg_energy"] = multi_time / 5
        results["energies"]["multiple"] = energies
        results["energies"]["std"] = float(np.std(energies))

        print(f"    Energy: {energy:.6f} eV ({energy_time:.4f}s)")
        print(f"    Forces: {forces.shape} ({forces_time:.4f}s)")
        print(f"    Multi: {multi_time:.4f}s total, {multi_time/5:.4f}s avg")

    except Exception as e:
        results["error"] = str(e)
        print(f"    Error: {e}")

    return results


def benchmark_optimization(
    atoms: Atoms, backend: str, model_name: Optional[str] = None, device: str = "cpu"
) -> Dict[str, Any]:
    """Benchmark optimization performance."""
    results = {
        "backend": backend,
        "model_name": model_name,
        "device": device,
        "n_atoms": len(atoms),
        "formula": atoms.get_chemical_formula(),
        "available": False,
        "error": None,
        "optimization": {},
    }

    try:
        # Check if backend is available
        if not qme.calculator_registry.is_backend_available(backend):
            results["error"] = f"Backend {backend} not available"
            return results

        results["available"] = True

        # Create QME optimizer
        qme_opt = qme.QMEOptimizer(
            backend=backend,
            model_name=model_name,
            device=device,
        )

        # Load structure
        qme_opt.load_structure(atoms)

        # Benchmark optimization
        print(f"  Testing optimization...")
        start_time = time.perf_counter()

        opt_result = qme_opt.optimize_minimum(
            optimizer="BFGS",
            fmax=0.05,
            steps=50,  # Limited steps for benchmarking
        )

        opt_time = time.perf_counter() - start_time

        results["optimization"]["time"] = opt_time
        results["optimization"]["converged"] = opt_result.get("converged", False)
        results["optimization"]["steps"] = opt_result.get("steps_taken", 0)
        results["optimization"]["final_energy"] = opt_result.get("final_energy", 0.0)
        results["optimization"]["max_force"] = opt_result.get("max_force", 0.0)

        print(f"    Time: {opt_time:.4f}s")
        print(f"    Converged: {opt_result.get('converged', False)}")
        print(f"    Steps: {opt_result.get('steps_taken', 0)}")

    except Exception as e:
        results["error"] = str(e)
        print(f"    Error: {e}")

    return results


def run_performance_benchmark(
    backends: List[str], device: str = "cpu", molecules: Optional[List[Atoms]] = None
) -> Dict[str, Any]:
    """Run comprehensive performance benchmark."""
    print("TorchSim Performance Benchmark")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Backends: {', '.join(backends)}")
    print()

    if molecules is None:
        molecules = create_test_molecules()

    all_results = {
        "device": device,
        "backends": backends,
        "molecules": [],
        "summary": {},
    }

    for name, atoms in molecules:
        print(f"\n{'='*60}")
        print(f"Testing molecule: {name} ({atoms.get_chemical_formula()})")
        print(f"Atoms: {len(atoms)}")
        print(f"{'='*60}")

        molecule_results = {
            "name": name,
            "formula": atoms.get_chemical_formula(),
            "n_atoms": len(atoms),
            "backends": {},
        }

        for backend in backends:
            print(f"\nBackend: {backend}")
            print("-" * 40)

            # Determine model name based on backend
            model_name = None
            if backend == "torchsim_mace":
                model_name = "mace-omol-0"
            elif backend == "torchsim_fairchem":
                model_name = "equiformer_v2_31M_s2ef_all_md"
            elif backend == "mace":
                model_name = "mace-omol-0"
            elif backend == "uma":
                model_name = "uma-m-1p1"

            # Test single calculations
            calc_results = benchmark_single_calculation(
                atoms, backend, model_name, device
            )

            # Test optimization (only for available backends)
            opt_results = None
            if calc_results["available"]:
                opt_results = benchmark_optimization(atoms, backend, model_name, device)

            molecule_results["backends"][backend] = {
                "calculation": calc_results,
                "optimization": opt_results,
            }

        all_results["molecules"].append(molecule_results)

    # Generate summary
    print(f"\n{'='*60}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*60}")

    summary = generate_performance_summary(all_results)
    all_results["summary"] = summary

    return all_results


def generate_performance_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate performance summary and comparisons."""
    summary = {
        "backend_comparison": {},
        "speedup_analysis": {},
        "accuracy_analysis": {},
    }

    # Collect timing data
    timing_data = {}
    for molecule in results["molecules"]:
        for backend, data in molecule["backends"].items():
            if backend not in timing_data:
                timing_data[backend] = {
                    "energy_times": [],
                    "forces_times": [],
                    "opt_times": [],
                    "n_atoms": [],
                }

            calc_data = data["calculation"]
            if calc_data["available"]:
                timing_data[backend]["energy_times"].append(
                    calc_data["timings"]["energy"]
                )
                timing_data[backend]["forces_times"].append(
                    calc_data["timings"]["forces"]
                )
                timing_data[backend]["n_atoms"].append(calc_data["n_atoms"])

                opt_data = data["optimization"]
                if opt_data and opt_data["available"]:
                    timing_data[backend]["opt_times"].append(
                        opt_data["optimization"]["time"]
                    )

    # Calculate averages
    for backend, data in timing_data.items():
        if data["energy_times"]:
            summary["backend_comparison"][backend] = {
                "avg_energy_time": np.mean(data["energy_times"]),
                "avg_forces_time": np.mean(data["forces_times"]),
                "avg_opt_time": (
                    np.mean(data["opt_times"]) if data["opt_times"] else None
                ),
                "n_tests": len(data["energy_times"]),
            }

    # Calculate speedups (TorchSim vs standard backends)
    torchsim_backends = [b for b in results["backends"] if b.startswith("torchsim")]
    standard_backends = [b for b in results["backends"] if not b.startswith("torchsim")]

    for torchsim_backend in torchsim_backends:
        if torchsim_backend in summary["backend_comparison"]:
            torchsim_time = summary["backend_comparison"][torchsim_backend][
                "avg_energy_time"
            ]

            speedups = {}
            for standard_backend in standard_backends:
                if standard_backend in summary["backend_comparison"]:
                    standard_time = summary["backend_comparison"][standard_backend][
                        "avg_energy_time"
                    ]
                    speedup = standard_time / torchsim_time if torchsim_time > 0 else 0
                    speedups[standard_backend] = speedup

            summary["speedup_analysis"][torchsim_backend] = speedups

    # Print summary
    print("\nBackend Performance Comparison:")
    print("-" * 50)
    print(f"{'Backend':<20} {'Energy (s)':<12} {'Forces (s)':<12} {'Opt (s)':<12}")
    print("-" * 50)

    for backend, data in summary["backend_comparison"].items():
        opt_time = f"{data['avg_opt_time']:.4f}" if data["avg_opt_time"] else "N/A"
        print(
            f"{backend:<20} {data['avg_energy_time']:<12.4f} {data['avg_forces_time']:<12.4f} {opt_time:<12}"
        )

    print("\nTorchSim Speedup Analysis:")
    print("-" * 50)
    for torchsim_backend, speedups in summary["speedup_analysis"].items():
        print(f"\n{torchsim_backend}:")
        for standard_backend, speedup in speedups.items():
            print(f"  vs {standard_backend}: {speedup:.2f}x speedup")

    return summary


def save_results(results: Dict[str, Any], output_file: str):
    """Save results to JSON file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    """Main function to run the performance benchmark."""
    parser = argparse.ArgumentParser(description="TorchSim Performance Benchmark")
    parser.add_argument(
        "--backends",
        type=str,
        default="mace,torchsim_mace,uma,torchsim_fairchem",
        help="Comma-separated list of backends to benchmark",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for calculations",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="torchsim_performance_results.json",
        help="Output file for results",
    )
    parser.add_argument(
        "--molecules",
        type=str,
        default="H2O,NH3,CH4,C6H6",
        help="Comma-separated list of molecules to test",
    )

    args = parser.parse_args()

    # Parse backends
    backends = [b.strip() for b in args.backends.split(",")]

    # Parse molecules
    molecule_names = [m.strip() for m in args.molecules.split(",")]
    all_molecules = create_test_molecules()
    molecules = [
        (name, atoms) for name, atoms in all_molecules if name in molecule_names
    ]

    if not molecules:
        print(
            f"No valid molecules found. Available: {[name for name, _ in all_molecules]}"
        )
        return 1

    print(f"Testing molecules: {[name for name, _ in molecules]}")

    # Run benchmark
    results = run_performance_benchmark(backends, args.device, molecules)

    # Save results
    save_results(results, args.output)

    return 0


if __name__ == "__main__":
    exit(main())
