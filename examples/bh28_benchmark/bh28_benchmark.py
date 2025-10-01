#!/usr/bin/env python3
"""
Comprehensive BH28 Benchmark Suite

This unified benchmark evaluates QME backends on the BH28 database of 28 diverse
chemical reaction barrier heights with reference values from high-level quantum
chemistry calculations (CCSDT(Q)/CBS level).

The benchmark:
1. Optimizes all reactant minima using various QME backends
2. Optimizes all transition states (when SELLA is available)
3. Calculates barrier heights from optimized structures
4. Compares accuracy against reference values
5. Provides comprehensive performance analysis

Usage:
    python bh28_benchmark.py --quicker           # Single reaction (fastest)
    python bh28_benchmark.py --quick             # Representative subset
    python bh28_benchmark.py --backends uma mace # Specific backends
    python bh28_benchmark.py                     # Full benchmark

Reference: A. Karton, J. Phys. Chem. A 2019, 123, 6720-6729
"""

import argparse
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import Dict, List

import numpy as np
from ase import Atoms
from ase.io import read

import qme
from qme.dependencies import HAS_SELLA, deps

# Suppress verbose logging from dependencies early
logging.getLogger("jax").setLevel(logging.WARNING)
logging.getLogger("numexpr").setLevel(logging.WARNING)
logging.getLogger("ase").setLevel(logging.WARNING)

# Try to suppress JAX startup messages
os.environ["JAX_LOG_LEVEL"] = "ERROR"


@contextmanager
def suppress_verbose_output():
    """Context manager to capture and suppress verbose ASE/SELLA output."""
    # Capture stdout/stderr
    old_stdout, old_stderr = sys.stdout, sys.stderr
    stdout_buffer, stderr_buffer = StringIO(), StringIO()

    # Temporarily disable verbose logging
    old_level = logging.getLogger().getEffectiveLevel()
    logging.getLogger().setLevel(logging.ERROR)

    try:
        sys.stdout, sys.stderr = stdout_buffer, stderr_buffer
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        logging.getLogger().setLevel(old_level)


class BH28Benchmark:
    """Comprehensive benchmark suite for QME on the BH28 database."""

    def __init__(
        self, dataset_dir: str = "bh28_dataset", output_dir: str = "benchmark_results"
    ):
        """Initialize comprehensive benchmark."""
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"BH28 dataset directory not found: {dataset_dir}")

        # Load reference barrier heights from JSON
        ref_file = self.dataset_dir / "reference_barrier_heights.json"
        with open(ref_file, "r") as f:
            self.reference_barriers = json.load(f)

        # Define bimolecular reactions (require multiple reactants)
        self.bimolecular_reactions = {
            # CADBH reactions - all involve small molecule + C2H4
            "CADBH_1": ["CADBH_1_min.xyz", "CADBH_c2h4.xyz"],
            "CADBH_2": ["CADBH_2_min.xyz", "CADBH_c2h4.xyz"],
            "CADBH_3": ["CADBH_3_min.xyz", "CADBH_c2h4.xyz"],
            "CADBH_4": ["CADBH_4_min.xyz", "CADBH_c2h4.xyz"],
            "CADBH_5": ["CADBH_5_min.xyz", "CADBH_c2h4.xyz"],
            "CADBH_6": ["CADBH_6_min.xyz", "CADBH_c2h4.xyz"],
            "CADBH_7": ["CADBH_7_min.xyz", "CADBH_c2h4.xyz"],
            "CADBH_8": ["CADBH_8_min.xyz", "CADBH_c2h4.xyz"],
            "CADBH_9": ["CADBH_9_min.xyz", "CADBH_c2h4.xyz"],
            # BHPERI_6 - Diels-Alder reaction
            "BHPERI_6": ["BHPERI_6_min.xyz", "CADBH_c2h4.xyz"],
        }

        # Quick subset for testing (representative reactions)
        self.quick_reactions = [
            "BHDIV_3",
            "PXBH_3",
            "CADBH_2",
            "CRBH_1",
            "PXBH_2",
            "BHPERI_2",
        ]

        # Quicker subset for very fast testing (single reaction)
        self.quicker_reactions = ["BHDIV_3"]

        # Results storage
        self.results = {}

        print("🧪 Comprehensive BH28 Benchmark Initialized")
        print(f"📁 Dataset: {self.dataset_dir}")
        print(f"📊 Output: {self.output_dir}")
        print(f"🔬 Available reactions: {len(self.reference_barriers)}")
        print(
            "⚠️  Note: SO3LR has known molecular size limitations and may skip some reactions"
        )

    def load_structure(self, filename: str) -> Atoms:
        """Load molecular structure from XYZ file."""
        filepath = self.dataset_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Structure file not found: {filepath}")

        atoms = read(str(filepath))
        return atoms[0] if isinstance(atoms, list) else atoms

    def get_reactants(self, reaction_name: str) -> List[Atoms]:
        """Get all reactant structures for a reaction."""
        if reaction_name in self.bimolecular_reactions:
            # Bimolecular reaction - load multiple reactants
            reactant_files = self.bimolecular_reactions[reaction_name]
            return [self.load_structure(filename) for filename in reactant_files]
        else:
            # Unimolecular reaction - load single minimum structure
            min_file = f"{reaction_name}_min.xyz"
            return [self.load_structure(min_file)]

    def get_transition_state(self, reaction_name: str) -> Atoms:
        """Get transition state structure for a reaction."""
        ts_file = f"{reaction_name}_ts.xyz"
        return self.load_structure(ts_file)

    def get_available_backends(self) -> List[str]:
        """Get list of available QME backends (excluding mock)."""
        available = []
        if deps.has("fairchem"):
            available.append("uma")
        if deps.has("so3lr"):
            available.append("so3lr")
        if deps.has("aimnet2"):
            available.append("aimnet2")
        if deps.has("mace"):
            available.append("mace")

        # Add TorchSim backends if available
        if deps.has("torch_sim"):
            available.append("torchsim")
            available.append("torchsim_mace")
            if deps.has("fairchem"):  # TorchSim Fairchem needs fairchem
                available.append("torchsim_fairchem")

        if not available:
            raise RuntimeError(
                "No ML backends available! Please install at least one ML backend."
                "The mock backend is excluded from benchmarking as it cannot optimize "
                "transition states."
            )

        return available

    def optimize_structures(self, reactions: List[str], backends: List[str]) -> Dict:
        """Optimize all structures (minima and TS) for given reactions and backends."""
        print(f"\n{'='*80}")
        print("🔬 STRUCTURE OPTIMIZATION")
        print(f"{'='*80}")

        results = {}

        for backend in backends:
            print(f"\n🔧 Backend: {backend.upper()}")
            backend_results = {}

            # Get default model for backend (use None for auto-detection)
            model_name = None

            try:
                for reaction in reactions:
                    print(f"\n  📍 {reaction}")
                    reaction_data = {}

                    # Create fresh optimizer for each reaction (ensures consistency across backends)
                    with suppress_verbose_output():
                        optimizer = qme.QMEOptimizer(
                            backend=backend, model_name=model_name
                        )

                    try:
                        # 1. Optimize reactant minima
                        reactants = self.get_reactants(reaction)
                        is_bimolecular = reaction in self.bimolecular_reactions

                        # Brief molecular information
                        reactant_info = [r.get_chemical_formula() for r in reactants]
                        if len(reactant_info) > 1:
                            print(f"    🧪 Reactants: {' + '.join(reactant_info)}")
                        else:
                            print(f"    🧪 Reactant: {reactant_info[0]}")

                        start_time = time.time()
                        optimized_reactants = []
                        reactant_energies = []

                        for i, reactant in enumerate(reactants):
                            with suppress_verbose_output():
                                result = optimizer.optimize_minimum(
                                    atoms=reactant,
                                    optimizer="LBFGS",
                                    steps=500,
                                    fmax=0.01,
                                )
                            optimized_reactants.append(result["optimized_atoms"])
                            reactant_energies.append(result["final_energy"])

                            reactant_label = (
                                f"reactant_{i+1}" if len(reactants) > 1 else "reactant"
                            )
                            print(
                                f"    ✅ {reactant_label}: {result['final_energy']:.6f} eV "
                                f"({result['steps_taken']} steps)"
                            )

                        total_reactant_energy = sum(reactant_energies)
                        minima_time = time.time() - start_time

                        reaction_data.update(
                            {
                                "is_bimolecular": is_bimolecular,
                                "optimized_reactants": optimized_reactants,
                                "reactant_energies": reactant_energies,
                                "total_reactant_energy": total_reactant_energy,
                                "minima_time": minima_time,
                                "minima_success": True,
                            }
                        )

                        # 2. Optimize transition state
                        if HAS_SELLA:
                            try:
                                ts_atoms = self.get_transition_state(reaction)
                                start_time = time.time()

                                with suppress_verbose_output():
                                    ts_result = optimizer.find_transition_state(
                                        atoms=ts_atoms, steps=500, fmax=0.01
                                    )

                                ts_time = time.time() - start_time
                                reaction_data.update(
                                    {
                                        "optimized_ts": ts_result["ts_atoms"],
                                        "ts_energy": ts_result["final_energy"],
                                        "ts_time": ts_time,
                                        "ts_steps": ts_result["steps_taken"],
                                        "ts_success": True,
                                        "ts_converged": ts_result.get(
                                            "converged", True
                                        ),
                                    }
                                )

                                status = (
                                    "✅" if ts_result.get("converged", True) else "⚠️"
                                )
                                print(
                                    f"    {status} TS: {ts_result['final_energy']:.6f} eV "
                                    f"({ts_result['steps_taken']} steps)"
                                )

                            except Exception as e:
                                print(f"    ❌ TS optimization failed: {str(e)}")
                                # Fall back to single-point energy
                                try:
                                    ts_atoms = self.get_transition_state(reaction)
                                    ts_atoms.calc = optimizer.calculator
                                    with suppress_verbose_output():
                                        ts_energy = ts_atoms.get_potential_energy()
                                    reaction_data.update(
                                        {
                                            "ts_energy": ts_energy,
                                            "ts_success": True,
                                            "ts_method": "single_point",
                                        }
                                    )
                                    print(
                                        f"    📊 TS (single-point fallback): {ts_energy:.6f} eV"
                                    )
                                except Exception as e2:
                                    print(
                                        f"    ❌ TS single-point also failed: {str(e2)}"
                                    )
                                    reaction_data.update(
                                        {"ts_success": False, "ts_error": str(e)}
                                    )
                        else:
                            print(
                                "    ⚠️  SELLA not available - using single-point TS energy"
                            )
                            try:
                                ts_atoms = self.get_transition_state(reaction)
                                ts_atoms.calc = optimizer.calculator
                                with suppress_verbose_output():
                                    ts_energy = ts_atoms.get_potential_energy()
                                reaction_data.update(
                                    {
                                        "ts_energy": ts_energy,
                                        "ts_success": True,
                                        "ts_method": "single_point",
                                    }
                                )
                                print(f"    📊 TS (single-point): {ts_energy:.6f} eV")
                            except Exception as e:
                                print(f"    ❌ TS single-point failed: {str(e)}")
                                reaction_data.update(
                                    {"ts_success": False, "ts_error": str(e)}
                                )

                        # 3. Calculate barrier height from optimized structures
                        if reaction_data.get("ts_success") and reaction_data.get(
                            "minima_success"
                        ):
                            barrier_height = (
                                reaction_data["ts_energy"] - total_reactant_energy
                            )
                            ref_barrier = self.reference_barriers[reaction]
                            error = barrier_height - ref_barrier
                            relative_error = (
                                (error / ref_barrier) * 100
                                if ref_barrier != 0
                                else float("inf")
                            )

                            reaction_data.update(
                                {
                                    "barrier_height": barrier_height,
                                    "reference_barrier": ref_barrier,
                                    "absolute_error": error,
                                    "relative_error": relative_error,
                                    "barrier_success": True,
                                }
                            )

                            print(
                                f"    📊 Barrier: {barrier_height:.3f} eV | "
                                f"Ref: {ref_barrier:.3f} eV | "
                                f"Error: {error:+.3f} eV ({relative_error:+.1f}%)"
                            )

                    except Exception as e:
                        error_msg = str(e)
                        if (
                            backend == "so3lr"
                            and "vmap got inconsistent sizes" in error_msg
                        ):
                            print(
                                "    ⚠️  Skipped: SO3LR molecular size incompatibility"
                            )
                            reaction_data = {
                                "success": False,
                                "error": "SO3LR molecular size incompatibility",
                                "skipped": True,
                            }
                        else:
                            print(f"    ❌ Failed: {error_msg}")
                            reaction_data = {"success": False, "error": error_msg}

                    backend_results[reaction] = reaction_data

            except Exception as e:
                print(f"❌ Backend {backend} initialization failed: {str(e)}")
                backend_results = {"backend_error": str(e)}

            results[backend] = backend_results

        self.results = results
        return results

    def analyze_performance(self, backends: List[str]) -> Dict:
        """Analyze performance metrics across backends."""
        print(f"\n{'='*80}")
        print("📊 PERFORMANCE ANALYSIS")
        print(f"{'='*80}")

        analysis = {}

        for backend in backends:
            print(f"\n🔧 Backend: {backend.upper()}")
            backend_data = self.results.get(backend, {})

            # Collect successful barrier calculations
            successful_barriers = []
            minima_times = []
            ts_times = []
            skipped_count = 0
            failed_count = 0

            for reaction, data in backend_data.items():
                if isinstance(data, dict):
                    if data.get("skipped"):
                        skipped_count += 1
                    elif not data.get("success", True):
                        failed_count += 1
                    else:
                        if data.get("barrier_success"):
                            successful_barriers.append(data)
                        if data.get("minima_success"):
                            minima_times.append(data.get("minima_time", 0))
                        if data.get("ts_success") and "ts_time" in data:
                            ts_times.append(data["ts_time"])

            # Calculate barrier height statistics
            if successful_barriers:
                errors = [data["absolute_error"] for data in successful_barriers]
                rel_errors = [
                    data["relative_error"]
                    for data in successful_barriers
                    if abs(data["relative_error"]) != float("inf")
                ]

                barrier_stats = {
                    "count": len(successful_barriers),
                    "mae": np.mean(np.abs(errors)),
                    "rmse": np.sqrt(np.mean(np.array(errors) ** 2)),
                    "max_error": np.max(np.abs(errors)),
                    "mean_rel_error": np.mean(np.abs(rel_errors)) if rel_errors else 0,
                    "std_error": np.std(errors),
                }

                print(f"  📈 Barrier Heights ({barrier_stats['count']} reactions):")
                print(f"     MAE:  {barrier_stats['mae']:.3f} eV")
                print(f"     RMSE: {barrier_stats['rmse']:.3f} eV")
                print(f"     Max Error: {barrier_stats['max_error']:.3f} eV")
                print(f"     Mean Rel. Error: {barrier_stats['mean_rel_error']:.1f}%")
            else:
                barrier_stats = {"count": 0}
                print("  ❌ No successful barrier height calculations")

            # Report skipped and failed reactions
            if skipped_count > 0:
                print(
                    f"  ⚠️  Skipped reactions (known incompatibilities): {skipped_count}"
                )
            if failed_count > 0:
                print(f"  ❌ Failed reactions: {failed_count}")

            # Calculate timing statistics
            timing_stats = {
                "minima_count": len(minima_times),
                "minima_total_time": sum(minima_times),
                "minima_avg_time": np.mean(minima_times) if minima_times else 0,
                "ts_count": len(ts_times),
                "ts_total_time": sum(ts_times),
                "ts_avg_time": np.mean(ts_times) if ts_times else 0,
            }

            print("  ⏱️  Timing:")
            if timing_stats["minima_count"] > 0:
                print(
                    f"     Minima: {timing_stats['minima_avg_time']:.1f}s avg "
                    f"({timing_stats['minima_total_time']:.1f}s total)"
                )
            if timing_stats["ts_count"] > 0:
                print(
                    f"     TS: {timing_stats['ts_avg_time']:.1f}s avg "
                    f"({timing_stats['ts_total_time']:.1f}s total)"
                )

            analysis[backend] = {
                "barrier_statistics": barrier_stats,
                "timing_statistics": timing_stats,
            }

        # Cross-backend comparison
        print("\n🏆 BACKEND COMPARISON")
        print(
            f"{'Backend':<12} {'Success':<8} {'MAE (eV)':<10} {'RMSE (eV)':<11} {'Avg Time':<10}"
        )
        print("-" * 65)

        for backend in backends:
            stats = analysis.get(backend, {}).get("barrier_statistics", {})
            timing = analysis.get(backend, {}).get("timing_statistics", {})

            backend_reactions = self.results.get(backend, {})
            total_attempted = len(
                [
                    r
                    for r in backend_reactions
                    if isinstance(backend_reactions[r], dict)
                    and not backend_reactions[r].get("skipped", False)
                ]
            )
            success_rate = f"{stats.get('count', 0)}/{total_attempted}"
            mae = f"{stats.get('mae', 0):.3f}" if stats.get("count", 0) > 0 else "N/A"
            rmse = f"{stats.get('rmse', 0):.3f}" if stats.get("count", 0) > 0 else "N/A"
            avg_time = (
                f"{timing.get('minima_avg_time', 0):.1f}s"
                if timing.get("minima_count", 0) > 0
                else "N/A"
            )

            print(
                f"{backend:<12} {success_rate:<8} {mae:<10} {rmse:<11} {avg_time:<10}"
            )

        return analysis

    def _convert_to_serializable(self, obj):
        """Recursively convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {
                key: self._convert_to_serializable(value) for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif hasattr(obj, "get_chemical_formula"):  # ASE Atoms object
            return {
                "formula": obj.get_chemical_formula(),
                "positions": obj.positions.tolist(),
                "symbols": obj.get_chemical_symbols(),
            }
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        else:
            return str(obj)

    def save_results(self, filename: str = "bh28_benchmark_results.json"):
        """Save all results to JSON file."""
        output_file = self.output_dir / filename
        serializable_results = self._convert_to_serializable(self.results)

        with open(output_file, "w") as f:
            json.dump(serializable_results, f, indent=2)

        print(f"\n💾 Results saved to: {output_file}")

    def run_benchmark(self, backends: List[str], reactions: List[str]):
        """Run the complete benchmark suite."""
        print("\n🚀 Starting Comprehensive BH28 Benchmark")
        print(f"🔬 Backends: {', '.join(backends)}")
        print(
            f"⚗️  Reactions: {len(reactions)} ({', '.join(reactions[:3])}{
                '...' if len(reactions) > 3 else ''})"
        )

        start_time = time.time()

        # Optimize all structures and calculate barriers
        self.optimize_structures(reactions, backends)

        # Analyze performance
        self.analyze_performance(backends)

        # Save results
        self.save_results()

        total_time = time.time() - start_time
        print(f"\n🏁 Benchmark completed in {total_time:.1f} seconds")

        return self.results


def main():
    """Main entry point for the comprehensive benchmark."""
    parser = argparse.ArgumentParser(
        description="Comprehensive BH28 Benchmark Suite for QME"
    )

    parser.add_argument(
        "--backends",
        nargs="+",
        default=None,
        help="QME backends to test (default: all available)",
    )

    parser.add_argument(
        "--reactions",
        nargs="+",
        default=None,
        help="Specific reactions to test (default: all 28 reactions)",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark with representative subset of reactions",
    )

    parser.add_argument(
        "--quicker",
        action="store_true",
        help="Run quicker benchmark with single reaction for very fast testing",
    )

    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze existing results without running new calculations",
    )

    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Output directory for results (default: benchmark_results)",
    )

    args = parser.parse_args()

    # Initialize benchmark
    benchmark = BH28Benchmark(output_dir=args.output_dir)

    # Determine backends to test
    available_backends = benchmark.get_available_backends()
    backends = args.backends if args.backends else available_backends

    # Validate backends
    invalid_backends = [b for b in backends if b not in available_backends]
    if invalid_backends:
        print(f"❌ Invalid backends: {invalid_backends}")
        print(f"✅ Available backends: {available_backends}")
        return 1

    # Determine reactions to test
    if args.quicker:
        reactions = benchmark.quicker_reactions
    elif args.quick:
        reactions = benchmark.quick_reactions
    elif args.reactions:
        reactions = args.reactions
    else:
        reactions = list(benchmark.reference_barriers.keys())

    # Validate reactions
    invalid_reactions = [r for r in reactions if r not in benchmark.reference_barriers]
    if invalid_reactions:
        print(f"❌ Invalid reactions: {invalid_reactions}")
        return 1

    if args.analyze:
        # Load existing results and analyze
        results_file = Path(args.output_dir) / "bh28_benchmark_results.json"
        if results_file.exists():
            with open(results_file, "r") as f:
                benchmark.results = json.load(f)
            benchmark.analyze_performance(backends)
        else:
            print(f"❌ No existing results found at {results_file}")
            return 1
    else:
        # Run full benchmark
        benchmark.run_benchmark(backends, reactions)
        print("\n✨ Comprehensive benchmark completed successfully!")

    return 0


if __name__ == "__main__":
    exit(main())
