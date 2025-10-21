#!/usr/bin/env python3
"""
QME Zimmermann-93 Benchmark - Two-Ended Transition State Search

This benchmark runs two-ended (reactant → product) transition state searches
across available ML backends in QME using the standardized Explorer API.

Usage:
    python zimmermann93_benchmark.py [--quick|--quicker]
    python zimmermann93_benchmark.py --backends uma,mace
    python zimmermann93_benchmark.py --npoints 15

Features:
    - Two-ended transition state search evaluation using Explorer API
    - Geometry comparison with reference structures
    - Comprehensive backend performance analysis
"""

import json
import logging
import os
import sys
import time
import warnings
from contextlib import contextmanager
from io import StringIO
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read

# Import QME components
try:
    from qme import Explorer, calculator_registry
except ImportError as e:
    print(f"❌ Error importing QME: {e}")
    print("   Please ensure QME is installed and accessible")
    sys.exit(1)

# Import common interface
from qme.examples import QMEExampleInterface, create_standard_epilog

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# Quiet noisy backends
logging.getLogger("jax").setLevel(logging.WARNING)
logging.getLogger("numexpr").setLevel(logging.WARNING)
logging.getLogger("ase").setLevel(logging.WARNING)

os.environ.setdefault("JAX_LOG_LEVEL", "ERROR")


@contextmanager
def suppress_verbose_output():
    """Capture stdout/stderr and suppress global logging temporarily."""
    old_stdout, old_stderr = sys.stdout, sys.stderr
    stdout_buffer, stderr_buffer = StringIO(), StringIO()
    old_level = logging.getLogger().getEffectiveLevel()
    logging.getLogger().setLevel(logging.ERROR)
    try:
        sys.stdout, sys.stderr = stdout_buffer, stderr_buffer
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        logging.getLogger().setLevel(old_level)


def compute_rmsd(reference: np.ndarray, target: np.ndarray) -> float:
    """Compute RMSD between two coordinate arrays using Kabsch alignment.

    Both arrays must be shape (N,3) and represent the same atoms in the same order.
    """
    # Center both structures
    ref_cent = reference.mean(axis=0)
    tar_cent = target.mean(axis=0)
    ref = reference - ref_cent
    tar = target - tar_cent

    # Kabsch alignment
    C = np.dot(tar.T, ref)
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(np.dot(V, Wt)))
    D = np.diag([1.0, 1.0, d])
    U = np.dot(np.dot(V, D), Wt)

    aligned = np.dot(tar, U.T)
    rmsd = np.sqrt(np.mean(np.sum((aligned - ref) ** 2, axis=1)))
    return float(rmsd)


class Zimmermann93Benchmark:
    """Benchmark suite for two-ended TS search on Zimmermann-93 dataset."""

    def __init__(self, dataset_dir: str | None = None, output_dir: str = "benchmark_results"):
        # Use dataset in same directory by default
        if dataset_dir is None:
            dataset_dir = str(Path(__file__).parent / "zimmermann93_dataset")

        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        # Discover reactions by searching for reactant files
        self.reactions = []
        for p in sorted(self.dataset_dir.glob("reaction_*_reactant.*")):
            stem = p.stem
            # stem like 'reaction_001_reactant'
            rxn = stem.replace("_reactant", "")
            self.reactions.append(rxn)

        # Quick subset
        self.quick_reactions = self.reactions[:8]

        # Quicker subset for very fast testing (single reaction)
        self.quicker_reactions = self.reactions[:1]

        self.results: dict[str, dict] = {}

        print("=" * 80)
        print("QME Zimmermann-93 Benchmark - Two-Ended Transition State Search")
        print("=" * 80)
        print(f"Dataset dir: {self.dataset_dir}")
        print(f"Output dir: {self.output_dir}")
        print(f"Reactions discovered: {len(self.reactions)}")

    def load_structure(self, filename: str) -> Atoms:
        filepath = self.dataset_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Structure file not found: {filepath}")
        atoms = read(str(filepath))
        return atoms[0] if isinstance(atoms, list) else atoms

    def get_reactant_and_product(self, reaction_name: str) -> tuple[Atoms, Atoms]:
        reactant_file = f"{reaction_name}_reactant.xyz"
        product_file = f"{reaction_name}_product.xyz"
        return self.load_structure(reactant_file), self.load_structure(product_file)

    def get_reference_ts(self, reaction_name: str) -> Atoms:
        ts_file = f"{reaction_name}_ts.xyz"
        return self.load_structure(ts_file)

    def get_available_backends(self) -> list[str]:
        """Get list of available ML backends (excluding mock)."""
        available = []
        ml_backends = [
            "aimnet2",
            "uma",
            "so3lr",
            "mace",
            "orb",
            "torchsim_mace",
            "torchsim_uma",
        ]

        for backend in ml_backends:
            if calculator_registry.is_backend_available(backend):
                available.append(backend)

        return available

    def filter_available_backends(
        self, requested_backends: list[str], verbose: bool = False
    ) -> list[str]:
        """Filter requested backends to only available ones."""
        available = []
        for backend in requested_backends:
            if calculator_registry.is_backend_available(backend):
                available.append(backend)
            elif verbose:
                print(f"Warning: Backend '{backend}' not available, skipping")
        return available

    def print_backend_summary(self, backends: list[str], title: str = "Available Backends"):
        """Print a formatted summary of backends."""
        print(f"\n📋 {title}")
        print("-" * 50)
        for i, backend in enumerate(backends, 1):
            print(f"  {i}. {backend}")
        print(f"Total: {len(backends)} backends")

    def run_benchmark(
        self,
        backends: list[str],
        reactions: list[str],
        npoints: int = 11,
        fmax: float = 0.01,
        steps: int = 300,
        verbose: bool = False,
    ) -> dict:
        results: dict[str, dict] = {}

        for backend in backends:
            print(f"\nBackend: {backend.upper()}")
            print("-" * 60)
            backend_results: dict[str, dict] = {}

            try:
                for reaction in reactions:
                    print(f"  Reaction: {reaction}")
                    reaction_data: dict = {
                        "timings": {},
                        "optimization_results": {},
                        "frequency_results": {},
                    }

                    try:
                        # Load endpoints
                        load_start = time.perf_counter()
                        reactant, product = self.get_reactant_and_product(reaction)
                        load_time = time.perf_counter() - load_start
                        reaction_data["timings"]["structure_loading"] = load_time

                        # Initialize Explorer with both reactant and product
                        # for growing string method TS optimization
                        init_start = time.perf_counter()
                        explorer = Explorer(
                            atoms=[reactant, product],
                            backend=backend,
                            target="ts",
                            strategy="growing_string",
                            verbose=0,  # Suppress output since we're using suppress_verbose_output()
                        )
                        init_time = time.perf_counter() - init_start
                        reaction_data["timings"]["initialization"] = init_time

                        # Run growing string method TS optimization
                        opt_start = time.perf_counter()
                        with suppress_verbose_output():
                            ts_result = explorer.run(
                                fmax=fmax,
                                steps=steps,
                                npoints=npoints,
                                step_size=0.1,
                                distance_threshold=0.5,
                                optimize_endpoints=True,
                                refine_ts=True,
                            )
                        opt_time = time.perf_counter() - opt_start
                        reaction_data["timings"]["optimization"] = opt_time

                        # Normalize TS result
                        if isinstance(ts_result, list) and len(ts_result) == 1:
                            ts_result = ts_result[0]

                        if isinstance(ts_result, dict):
                            ts_opt_atoms = ts_result.get("optimized_atoms", reactant)
                            ts_success = bool(ts_result.get("converged", False))
                            steps_taken = ts_result.get("steps_taken", 0)
                        else:
                            ts_opt_atoms = ts_result
                            ts_success = True
                            steps_taken = 0

                        # Calculate average time per step
                        if steps_taken > 0:
                            avg_time_per_step = opt_time / steps_taken
                        else:
                            avg_time_per_step = None

                        # Get final energy and forces
                        if ts_opt_atoms is not None:
                            final_energy = float(ts_opt_atoms.get_potential_energy())
                            forces = ts_opt_atoms.get_forces()
                            max_force = float(np.max(np.abs(forces)))
                        else:
                            final_energy = None
                            max_force = float("inf")

                        reaction_data["optimization_results"] = {
                            "converged": ts_success,
                            "final_energy": final_energy,
                            "max_force": max_force,
                            "steps_taken": steps_taken,
                        }
                        reaction_data["timings"]["avg_time_per_step"] = avg_time_per_step

                        # Frequency analysis to verify TS character
                        if ts_opt_atoms is not None and ts_success:
                            freq_start = time.perf_counter()
                            try:
                                with suppress_verbose_output():
                                    freq_results = explorer.calculate_frequencies(
                                        delta=0.01,
                                        method="auto",
                                        temperature=298.15,
                                        save_hessian=False,
                                    )
                                freq_time = time.perf_counter() - freq_start
                                reaction_data["timings"]["frequency_analysis"] = freq_time
                                reaction_data["frequency_results"] = {
                                    "n_frequencies": len(freq_results["frequencies"]),
                                    "frequencies": freq_results["frequencies"][
                                        :10
                                    ],  # First 10 frequencies
                                    "zero_point_energy": freq_results["zero_point_energy"],
                                    "is_transition_state": freq_results["is_ts"],
                                    "method_used": freq_results["method_used"],
                                    "ts_analysis": freq_results.get("ts_analysis", {}),
                                }
                            except Exception as e:
                                reaction_data["timings"]["frequency_analysis"] = None
                                reaction_data["frequency_results"] = {"error": str(e)}
                        else:
                            reaction_data["timings"]["frequency_analysis"] = None
                            reaction_data["frequency_results"] = {
                                "skipped": "TS optimization failed"
                            }

                        # Compare geometry to reference TS
                        try:
                            ref_ts = self.get_reference_ts(reaction)
                            if ts_opt_atoms is not None:
                                ref_coords = ref_ts.get_positions()
                                opt_coords = ts_opt_atoms.get_positions()
                                rmsd = compute_rmsd(ref_coords, opt_coords)
                            else:
                                rmsd = float("nan")
                        except Exception:
                            rmsd = float("nan")

                        reaction_data.update(
                            {
                                "ts_result": ts_result,
                                "ts_success": ts_success,
                                "ts_rmsd_to_reference": rmsd,
                                "success": True,
                            }
                        )

                        # Calculate total time
                        total_time = sum(
                            v for v in reaction_data["timings"].values() if v is not None
                        )
                        reaction_data["timings"]["total"] = total_time

                        # Print status
                        freq_info = reaction_data["frequency_results"]
                        if "is_transition_state" in freq_info:
                            ts_verified = freq_info["is_transition_state"]
                            ts_status = "✅ Verified TS" if ts_verified else "⚠️ Not verified as TS"
                        else:
                            ts_status = "❓ Not checked"

                        print(f"    ✓ Optimization: {'Success' if ts_success else 'Failed'}")
                        print(f"    ✓ Convergence: {ts_success}")
                        print(f"    ✓ TS Character: {ts_status}")
                        print(f"    ✓ RMSD to ref: {rmsd:.4f} Å")
                        if verbose:
                            print(f"    ✓ Steps: {steps_taken}, Time: {opt_time:.2f}s")
                            if avg_time_per_step:
                                print(f"    ✓ Avg time/step: {avg_time_per_step:.4f}s")

                    except Exception as e:
                        reaction_data = {
                            "success": False,
                            "error": str(e),
                            "timings": {},
                            "optimization_results": {},
                            "frequency_results": {},
                        }
                        print(f"    ❌ Reaction failed: {e}")

                    backend_results[reaction] = reaction_data

            except Exception as e:
                print(f"❌ Backend initialization failed: {e}")
                backend_results = {"_backend_error": {"error": str(e)}}

            results[backend] = backend_results

        self.results = results
        return results

    def analyze_performance(self, backends: list[str]) -> dict:
        """Analyze performance metrics across backends with detailed statistics."""
        print(f"\n{'=' * 120}")
        print("DETAILED PERFORMANCE ANALYSIS")
        print(f"{'=' * 120}")

        analysis = {}

        for backend in backends:
            print(f"\nBackend: {backend.upper()}")
            print("-" * 60)
            backend_data = self.results.get(backend, {})

            # Collect successful TS calculations
            successful_ts = []
            failed_count = 0
            converged_count = 0
            ts_verified_count = 0
            timing_stats = {"total": [], "optimization": [], "frequency": []}
            step_stats = []

            for _reaction, data in backend_data.items():
                if isinstance(data, dict) and not data.get("skipped"):
                    if not data.get("success", True):
                        failed_count += 1
                    else:
                        # Check convergence
                        opt_results = data.get("optimization_results", {})
                        if opt_results.get("converged"):
                            converged_count += 1
                            successful_ts.append(data)

                            # Collect timing statistics
                            timings = data.get("timings", {})
                            if timings.get("total"):
                                timing_stats["total"].append(timings["total"])
                            if timings.get("optimization"):
                                timing_stats["optimization"].append(timings["optimization"])
                            if timings.get("frequency_analysis"):
                                timing_stats["frequency"].append(timings["frequency_analysis"])

                            # Collect step statistics
                            steps = opt_results.get("steps_taken", 0)
                            if steps > 0:
                                step_stats.append(steps)

                            # Check TS verification
                            freq_results = data.get("frequency_results", {})
                            if freq_results.get("is_transition_state"):
                                ts_verified_count += 1

            # Calculate statistics
            total_reactions = len(
                [r for r in backend_data.values() if isinstance(r, dict) and not r.get("skipped")]
            )

            if successful_ts:
                rmsds = [
                    data["ts_rmsd_to_reference"]
                    for data in successful_ts
                    if not np.isnan(data["ts_rmsd_to_reference"])
                ]

                ts_stats = {
                    "total_reactions": total_reactions,
                    "converged": converged_count,
                    "ts_verified": ts_verified_count,
                    "failed": failed_count,
                    "convergence_rate": (
                        (converged_count / total_reactions * 100) if total_reactions > 0 else 0
                    ),
                    "verification_rate": (
                        (ts_verified_count / converged_count * 100) if converged_count > 0 else 0
                    ),
                    "mean_rmsd": np.mean(rmsds) if rmsds else 0,
                    "max_rmsd": np.max(rmsds) if rmsds else 0,
                    "std_rmsd": np.std(rmsds) if rmsds else 0,
                }

                print("📊 CONVERGENCE STATISTICS:")
                print(f"  Total reactions: {ts_stats['total_reactions']}")
                conv_rate = ts_stats["convergence_rate"]
                print(f"  Converged: {ts_stats['converged']} ({conv_rate:.1f}%)")
                ver_rate = ts_stats["verification_rate"]
                print(f"  TS Verified: {ts_stats['ts_verified']} ({ver_rate:.1f}%)")
                print(f"  Failed: {ts_stats['failed']}")

                print("\n📏 GEOMETRY ACCURACY:")
                print(f"  Mean RMSD: {ts_stats['mean_rmsd']:.4f} Å")
                print(f"  Max RMSD:  {ts_stats['max_rmsd']:.4f} Å")
                print(f"  Std RMSD:  {ts_stats['std_rmsd']:.4f} Å")

                # Timing statistics
                if timing_stats["total"]:
                    print("\n⏱️ TIMING STATISTICS:")
                    total_mean = np.mean(timing_stats["total"])
                    total_std = np.std(timing_stats["total"])
                    print(f"  Total time: {total_mean:.2f} ± {total_std:.2f} s")
                    if timing_stats["optimization"]:
                        opt_mean = np.mean(timing_stats["optimization"])
                        opt_std = np.std(timing_stats["optimization"])
                        print(f"  Optimization: {opt_mean:.2f} ± {opt_std:.2f} s")
                    if timing_stats["frequency"]:
                        freq_mean = np.mean(timing_stats["frequency"])
                        freq_std = np.std(timing_stats["frequency"])
                        print(f"  Frequency analysis: {freq_mean:.2f} ± {freq_std:.2f} s")

                # Step statistics
                if step_stats:
                    print("\n🔄 OPTIMIZATION STEPS:")
                    print(f"  Mean steps: {np.mean(step_stats):.1f} ± {np.std(step_stats):.1f}")
                    print(f"  Min steps: {np.min(step_stats)}")
                    print(f"  Max steps: {np.max(step_stats)}")

            else:
                ts_stats = {
                    "total_reactions": total_reactions,
                    "converged": 0,
                    "ts_verified": 0,
                    "failed": failed_count,
                    "convergence_rate": 0,
                    "verification_rate": 0,
                }
                print("❌ No successful transition state calculations")

            analysis[backend] = {"ts_statistics": ts_stats}

        # Comprehensive summary table
        print(f"\n{'=' * 120}")
        print("COMPREHENSIVE BACKEND COMPARISON")
        print(f"{'=' * 120}")

        # Print legend
        print("📊 COLUMN DEFINITIONS:")
        print("   Conv    = Number of converged optimizations")
        print("   Rate    = Convergence rate (%)")
        print("   Verified = Number vibrationally verified as TS")
        print("   V-Rate  = TS verification rate (%)")
        print("   Mean RMSD = Average RMSD to reference (Å)")
        print("   Max RMSD = Maximum RMSD to reference (Å)")
        print(f"{'-' * 120}")

        # Header
        print(
            f"{'Backend':<12} {'Total':<6} {'Conv':<6} {'Rate':<6} "
            f"{'Verified':<9} {'V-Rate':<7} {'Mean RMSD':<10} {'Max RMSD':<10}"
        )
        print("=" * 120)

        # Results
        for backend in backends:
            stats = analysis.get(backend, {}).get("ts_statistics", {})

            print(
                f"{backend:<12} "
                f"{stats.get('total_reactions', 0):<6} "
                f"{stats.get('converged', 0):<6} "
                f"{stats.get('convergence_rate', 0):<6.1f} "
                f"{stats.get('ts_verified', 0):<9} "
                f"{stats.get('verification_rate', 0):<7.1f} "
                f"{stats.get('mean_rmsd', 0):<10.4f} "
                f"{stats.get('max_rmsd', 0):<10.4f}"
            )

        print("=" * 120)

        return analysis

    def save_results(self, filename: str = "zimmermann93_benchmark_results.json"):
        out = self.output_dir / filename

        # Convert numpy and other non-serializable objects
        def convert(obj):
            # Handle ASE Atoms-like objects
            try:
                from ase import Atoms as _Atoms

                if isinstance(obj, _Atoms):
                    return {
                        "formula": obj.get_chemical_formula(),
                        "positions": obj.get_positions().tolist(),
                        "symbols": obj.get_chemical_symbols(),
                    }
            except Exception:
                pass

            # Handle generic geometry-like objects that expose get_positions
            if hasattr(obj, "get_positions") and callable(obj.get_positions):
                try:
                    pos = obj.get_positions()
                    return {
                        "positions": np.array(pos).tolist(),
                    }
                except Exception:
                    pass

            # Handle numpy arrays and scalars
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)

            # Handle numpy boolean scalars
            try:
                if isinstance(obj, np.bool_):
                    return bool(obj)
            except Exception:
                pass

            # Handle None values in timing data
            if obj is None:
                return None

            # Recursively convert nested structures
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(i) for i in obj]
            if isinstance(obj, tuple):
                return [convert(i) for i in obj]

            return obj

        serializable = convert(self.results)
        with open(out, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nResults saved to: {out}")


def main():
    """Main entry point for the benchmark."""
    # Create standardized interface
    interface = QMEExampleInterface(
        name="Zimmermann-93 Benchmark",
        description="Two-Ended Transition State Search",
        epilog=create_standard_epilog("benchmark_quick"),
    )

    parser = interface.create_parser()

    # Add benchmark-specific arguments
    parser.add_argument(
        "--reactions",
        nargs="+",
        help="Specific reactions to test (default: all reactions)",
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
        "--output-dir",
        default="benchmark_results",
        help="Output directory for results (default: benchmark_results)",
    )
    parser.add_argument(
        "--npoints",
        type=int,
        default=11,
        help="Number of points in interpolated path (default: 11)",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=0.01,
        help="Force convergence criterion (default: 0.01)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="Maximum optimization steps (default: 300)",
    )

    args = parser.parse_args()

    # Set up logging based on verbosity level
    interface.setup_logging(args.verbose)

    # Initialize benchmark
    benchmark = Zimmermann93Benchmark(dataset_dir=None, output_dir=args.output_dir)

    interface.print_header("Two-Ended Transition State Search")

    # Determine backends to test
    if args.backends:
        requested_backends = [b.strip() for b in args.backends.split(",")]
        backends = interface.filter_available_backends(requested_backends, verbose=args.verbose)
        if not backends:
            interface.print_error("No requested backends are available!")
            return 1
    else:
        backends = interface.get_available_ml_backends()
        if not backends:
            interface.print_error("No ML backends available!")
            print("Please install at least one ML backend:")
            print("  - UMA: pip install fairchem-core")
            print("  - MACE: pip install mace-torch")
            print("  - AIMNet2: pip install aimnet2")
            print("  - SO3LR: pip install so3lr")
            return 1

    interface.print_backend_summary(backends, "Benchmarking Backends")

    if args.quicker:
        reactions = benchmark.quicker_reactions
    elif args.quick:
        reactions = benchmark.quick_reactions
    elif args.reactions:
        reactions = args.reactions
    else:
        reactions = benchmark.reactions

    invalid_rxn = [
        r for r in reactions if (benchmark.dataset_dir / f"{r}_reactant.xyz").exists() is False
    ]
    if invalid_rxn:
        interface.print_error(f"Invalid reactions: {invalid_rxn}")
        return 1

    # Print configuration
    config = {
        "NPoints": args.npoints,
        "Force max": args.fmax,
        "Max steps": args.steps,
        "Verbose": args.verbose,
        "Output": args.output_dir,
    }
    interface.print_configuration(config)

    benchmark.run_benchmark(
        backends,
        reactions,
        npoints=args.npoints,
        fmax=args.fmax,
        steps=args.steps,
        verbose=args.verbose,
    )

    # Analyze performance
    benchmark.analyze_performance(backends)

    # Save results
    benchmark.save_results()

    interface.print_success()
    return 0


if __name__ == "__main__":
    sys.exit(main())
