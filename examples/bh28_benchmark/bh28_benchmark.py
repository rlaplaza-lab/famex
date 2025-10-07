#!/usr/bin/env python3
"""
QME BH28 Benchmark - Chemical Accuracy Evaluation

This benchmark evaluates QME backends on the BH28 database of 28 diverse
chemical reaction barrier heights with reference values from high-level quantum
chemistry calculations (CCSDT(Q)/CBS level).

Usage:
    conda run -n py312 python bh28_benchmark.py [--quick|--quicker]
    conda run -n py312 python bh28_benchmark.py --backends uma,mace
    conda run -n py312 python bh28_benchmark.py --analyze

Features:
    - Optimizes reactant minima using various QME backends
    - Optimizes transition states (when SELLA is available)
    - Calculates barrier heights from optimized structures
    - Compares accuracy against reference values
    - Provides comprehensive performance analysis

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

# Import QME components
try:
    from qme.calculator_registry import calculator_registry
    from qme.core.explorer import Explorer
    from qme.dependencies import HAS_SELLA
except ImportError as e:
    print(f"❌ Error importing QME: {e}")
    print("   Please ensure QME is installed and accessible")
    sys.exit(1)

# Import device utilities
from device_utils import get_optimal_device, print_device_info

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

    def __init__(self, dataset_dir: str = "bh28_dataset", output_dir: str = "benchmark_results"):
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

        print("=" * 80)
        print("QME BH28 Benchmark - Chemical Accuracy Evaluation")
        print("=" * 80)
        print(f"Dataset: {self.dataset_dir}")
        print(f"Output: {self.output_dir}")
        print(f"Available reactions: {len(self.reference_barriers)}")
        print("Note: SO3LR has known molecular size limitations and may skip some reactions")

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
        ml_backends = [
            "aimnet2",
            "uma",
            "so3lr",
            "mace",
            "torchsim_mace",
            "torchsim_uma",
        ]

        for backend in ml_backends:
            if calculator_registry.is_backend_available(backend):
                available.append(backend)

        if not available:
            raise RuntimeError(
                "No ML backends available! Please install at least one ML backend. "
                "The mock backend is excluded from benchmarking as it cannot optimize "
                "transition states."
            )

        return available

    def filter_available_backends(
        self, requested_backends: List[str], verbose: bool = False
    ) -> List[str]:
        """Filter requested backends to only available ones."""
        available = []
        for backend in requested_backends:
            if calculator_registry.is_backend_available(backend):
                available.append(backend)
            elif verbose:
                print(f"Warning: Backend '{backend}' not available, skipping")
        return available

    def print_backend_summary(self, backends: List[str], title: str = "Available Backends"):
        """Print a formatted summary of backends."""
        print(f"\n📋 {title}")
        print("-" * 50)
        for i, backend in enumerate(backends, 1):
            print(f"  {i}. {backend}")
        print(f"Total: {len(backends)} backends")

    def optimize_structures(self, reactions: List[str], backends: List[str]) -> Dict:
        """Optimize all structures (minima and TS) for given reactions and backends."""
        print(f"\n{'='*80}")
        print("STRUCTURE OPTIMIZATION")
        print(f"{'='*80}")

        results = {}

        for backend in backends:
            print(f"\nBackend: {backend.upper()}")
            print("-" * 60)
            backend_results = {}

            # Get default model for backend (use None for auto-detection)
            model_name = None

            try:
                for reaction in reactions:
                    print(f"\n  📍 {reaction}")
                    reaction_data = {}

                    # Create fresh optimizer for each reaction (ensures consistency across backends)

                    try:
                        # 1. Optimize reactant minima
                        reactants = self.get_reactants(reaction)
                        is_bimolecular = reaction in self.bimolecular_reactions

                        # Brief molecular information
                        reactant_info = [r.get_chemical_formula() for r in reactants]
                        if len(reactant_info) > 1:
                            print(f"  Reactants: {' + '.join(reactant_info)}")
                        else:
                            print(f"  Reactant: {reactant_info[0]}")

                        start_time = time.time()
                        optimized_reactants = []
                        reactant_energies = []

                        for i, reactant in enumerate(reactants):
                            with suppress_verbose_output():
                                optimizer = Explorer(
                                    atoms=reactant,
                                    backend=backend,
                                    model_name=model_name,
                                )
                                result = optimizer.run(
                                    mode="minima",
                                    local_optimizer_name="LBFGS",
                                    steps=500,
                                    fmax=0.01,
                                )
                            # Normalize result shape (Explorer.run returns [dict] for local)
                            if isinstance(result, list) and len(result) == 1:
                                result = result[0]
                            if isinstance(result, dict):
                                optimized = result.get("optimized_atoms", reactant)
                                steps_taken = result.get("steps_taken", None)
                            else:
                                optimized = result
                                steps_taken = None

                            # Compute energy from atoms to avoid relying on dict keys
                            try:
                                energy_val = float(optimized.get_potential_energy())
                            except Exception:
                                # Attach calculator if missing and retry
                                optimizer._create_and_attach_calculator(optimized)
                                energy_val = float(optimized.get_potential_energy())

                            optimized_reactants.append(optimized)
                            reactant_energies.append(energy_val)

                            reactant_label = f"reactant_{i+1}" if len(reactants) > 1 else "reactant"
                            if steps_taken is None:
                                print(f"  ✅ {reactant_label}: {energy_val:.6f} eV")
                            else:
                                print(
                                    f"  ✅ {reactant_label}: {energy_val:.6f} eV "
                                    f"({steps_taken} steps)"
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
                                    ts_optimizer = Explorer(
                                        atoms=ts_atoms,
                                        backend=backend,
                                        model_name=model_name,
                                    )
                                    ts_result = ts_optimizer.run(mode="ts", steps=500, fmax=0.01)

                                ts_time = time.time() - start_time

                                # Normalize TS result
                                if isinstance(ts_result, list) and len(ts_result) == 1:
                                    ts_result = ts_result[0]
                                if isinstance(ts_result, dict):
                                    ts_atoms_opt = ts_result.get("optimized_atoms", ts_atoms)
                                    ts_steps = ts_result.get("steps_taken", None)
                                    ts_conv = bool(ts_result.get("converged", True))
                                else:
                                    ts_atoms_opt = ts_result
                                    ts_steps = None
                                    ts_conv = True

                                # Compute energy
                                try:
                                    ts_energy_val = float(ts_atoms_opt.get_potential_energy())
                                except Exception:
                                    ts_optimizer._create_and_attach_calculator(ts_atoms_opt)
                                    ts_energy_val = float(ts_atoms_opt.get_potential_energy())

                                reaction_data.update(
                                    {
                                        "optimized_ts": ts_atoms_opt,
                                        "ts_energy": ts_energy_val,
                                        "ts_time": ts_time,
                                        "ts_steps": ts_steps,
                                        "ts_success": True,
                                        "ts_converged": ts_conv,
                                    }
                                )

                                status = "✅" if ts_conv else "⚠️"
                                if ts_steps is None:
                                    print(f"  {status} TS: {ts_energy_val:.6f} eV")
                                else:
                                    print(
                                        f"  {status} TS: {ts_energy_val:.6f} eV "
                                        f"({ts_steps} steps)"
                                    )

                            except Exception as e:
                                print(f"  ❌ TS optimization failed: {str(e)}")
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
                                    print(f"  📊 TS (single-point fallback): {ts_energy:.6f} eV")
                                except Exception as e2:
                                    print(f"  ❌ TS single-point also failed: {str(e2)}")
                                    reaction_data.update({"ts_success": False, "ts_error": str(e)})
                        else:
                            print("  ⚠️  SELLA not available - using single-point TS energy")
                            try:
                                ts_atoms = self.get_transition_state(reaction)
                                # Attach calculator for single-point evaluation
                                try:
                                    ts_atoms.calc = optimizer.atoms_list[0].calc
                                except Exception:
                                    optimizer._create_and_attach_calculator(ts_atoms)
                                with suppress_verbose_output():
                                    ts_energy = ts_atoms.get_potential_energy()
                                reaction_data.update(
                                    {
                                        "ts_energy": ts_energy,
                                        "ts_success": True,
                                        "ts_method": "single_point",
                                    }
                                )
                                print(f"  📊 TS (single-point): {ts_energy:.6f} eV")
                            except Exception as e:
                                print(f"  ❌ TS single-point failed: {str(e)}")
                                reaction_data.update({"ts_success": False, "ts_error": str(e)})

                        # 3. Calculate barrier height from optimized structures
                        if reaction_data.get("ts_success") and reaction_data.get("minima_success"):
                            barrier_height = reaction_data["ts_energy"] - total_reactant_energy
                            ref_barrier = self.reference_barriers[reaction]
                            error = barrier_height - ref_barrier
                            relative_error = (
                                (error / ref_barrier) * 100 if ref_barrier != 0 else float("inf")
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
                                f"  📊 Barrier: {barrier_height:.3f} eV | "
                                f"Ref: {ref_barrier:.3f} eV | "
                                f"Error: {error:+.3f} eV ({relative_error:+.1f}%)"
                            )

                    except Exception as e:
                        error_msg = str(e)
                        if backend == "so3lr" and "vmap got inconsistent sizes" in error_msg:
                            print("  ⚠️  Skipped: SO3LR molecular size incompatibility")
                            reaction_data = {
                                "success": False,
                                "error": "SO3LR molecular size incompatibility",
                                "skipped": True,
                            }
                        else:
                            print(f"  ❌ Failed: {error_msg}")
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
        print("PERFORMANCE ANALYSIS")
        print(f"{'='*80}")

        analysis = {}

        for backend in backends:
            print(f"\nBackend: {backend.upper()}")
            print("-" * 50)
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

                print(f"Barrier Heights ({barrier_stats['count']} reactions):")
                print(f"  MAE:  {barrier_stats['mae']:.3f} eV")
                print(f"  RMSE: {barrier_stats['rmse']:.3f} eV")
                print(f"  Max Error: {barrier_stats['max_error']:.3f} eV")
                print(f"  Mean Rel. Error: {barrier_stats['mean_rel_error']:.1f}%")
            else:
                barrier_stats = {"count": 0}
                print("❌ No successful barrier height calculations")

            # Report skipped and failed reactions
            if skipped_count > 0:
                print(f"⚠️  Skipped reactions (known incompatibilities): {skipped_count}")
            if failed_count > 0:
                print(f"❌ Failed reactions: {failed_count}")

            # Calculate timing statistics
            timing_stats = {
                "minima_count": len(minima_times),
                "minima_total_time": sum(minima_times),
                "minima_avg_time": np.mean(minima_times) if minima_times else 0,
                "ts_count": len(ts_times),
                "ts_total_time": sum(ts_times),
                "ts_avg_time": np.mean(ts_times) if ts_times else 0,
            }

            print("Timing:")
            if timing_stats["minima_count"] > 0:
                print(
                    f"  Minima: {timing_stats['minima_avg_time']:.1f}s avg "
                    f"({timing_stats['minima_total_time']:.1f}s total)"
                )
            if timing_stats["ts_count"] > 0:
                print(
                    f"  TS: {timing_stats['ts_avg_time']:.1f}s avg "
                    f"({timing_stats['ts_total_time']:.1f}s total)"
                )

            analysis[backend] = {
                "barrier_statistics": barrier_stats,
                "timing_statistics": timing_stats,
            }

        # Cross-backend comparison
        print("\nBACKEND COMPARISON")
        print("-" * 65)
        print(f"{'Backend':<12} {'Success':<8} {'MAE (eV)':<10} {'RMSE (eV)':<11} {'Avg Time':<10}")
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

            print(f"{backend:<12} {success_rate:<8} {mae:<10} {rmse:<11} {avg_time:<10}")

        return analysis

    def _convert_to_serializable(self, obj):
        """Recursively convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
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

        print(f"\nResults saved to: {output_file}")

    def run_benchmark(self, backends: List[str], reactions: List[str]):
        """Run the complete benchmark suite."""
        print("\nStarting BH28 Benchmark")
        print(f"Backends: {', '.join(backends)}")
        print(
            f"Reactions: {len(reactions)} "
            f"({', '.join(reactions[:3])}{'...' if len(reactions) > 3 else ''})"
        )

        start_time = time.time()

        # Optimize all structures and calculate barriers
        self.optimize_structures(reactions, backends)

        # Analyze performance
        self.analyze_performance(backends)

        # Save results
        self.save_results()

        total_time = time.time() - start_time
        print(f"\nBenchmark completed in {total_time:.1f} seconds")

        return self.results


def main():
    """Main entry point for the benchmark."""
    parser = argparse.ArgumentParser(
        description="QME BH28 Benchmark - Chemical Accuracy Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  conda run -n py312 python bh28_benchmark.py
  conda run -n py312 python bh28_benchmark.py --backends aimnet2,uma
  conda run -n py312 python bh28_benchmark.py --quick
  conda run -n py312 python bh28_benchmark.py --quicker
  conda run -n py312 python bh28_benchmark.py --analyze
        """,
    )

    parser.add_argument(
        "--backends",
        nargs="+",
        help="QME backends to test (default: all available)",
    )

    parser.add_argument(
        "--reactions",
        nargs="+",
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

    if args.backends:
        backends = benchmark.filter_available_backends(args.backends, verbose=True)
        if not backends:
            print("❌ No requested backends are available!")
            return 1
    else:
        backends = available_backends

    benchmark.print_backend_summary(backends, "Benchmarking Backends")

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
        print("\n✅ Benchmark completed successfully!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
