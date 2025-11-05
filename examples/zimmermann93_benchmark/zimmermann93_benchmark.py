#!/usr/bin/env python3
"""QME Zimmermann-93 Benchmark - Two-Ended Transition State Search.

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
from qme import Explorer, calculator_registry

# Import common interface
from qme.example_utils import QMEExampleInterface, create_standard_epilog

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
    V, _S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(np.dot(V, Wt)))
    D = np.diag([1.0, 1.0, d])
    U = np.dot(np.dot(V, D), Wt)

    aligned = np.dot(tar, U.T)
    rmsd = np.sqrt(np.mean(np.sum((aligned - ref) ** 2, axis=1)))
    return float(rmsd)


class Zimmermann93Benchmark:
    """Benchmark suite for two-ended TS search on Zimmermann-93 dataset."""

    def __init__(
        self, dataset_dir: str | None = None, output_dir: str = "benchmark_results"
    ) -> None:
        # Use dataset in same directory by default
        if dataset_dir is None:
            dataset_dir = str(Path(__file__).parent / "zimmermann93_dataset")

        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        if not self.dataset_dir.exists():
            msg = f"Dataset directory not found: {self.dataset_dir}"
            raise FileNotFoundError(msg)

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

    def load_structure(self, filename: str) -> Atoms:
        filepath = self.dataset_dir / filename
        if not filepath.exists():
            msg = f"Structure file not found: {filepath}"
            raise FileNotFoundError(msg)
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
        # Use the centralized backend availability system
        from qme.backends.availability import get_available_ml_backends

        return get_available_ml_backends()

    def filter_available_backends(
        self,
        requested_backends: list[str],
        verbose: bool = False,
    ) -> list[str]:
        """Filter requested backends to only available ones."""
        available = []
        for backend in requested_backends:
            if calculator_registry.is_backend_available(backend):
                available.append(backend)
            elif verbose:
                pass
        return available

    def print_backend_summary(self, backends: list[str], title: str = "Available Backends") -> None:
        """Print a formatted summary of backends."""

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
            backend_results: dict[str, dict] = {}

            try:
                for reaction in reactions:
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

                        # Handle TS result from Explorer.run() method
                        # The run() method returns a dictionary with standardized results
                        if isinstance(ts_result, dict):
                            ts_opt_atoms = ts_result.get("optimized_atoms", reactant)
                            ts_success = bool(ts_result.get("converged", False))
                            steps_taken = ts_result.get("steps_taken", 0)
                        else:
                            ts_opt_atoms = ts_result
                            ts_success = True
                            steps_taken = 0

                        # Calculate average time per step
                        avg_time_per_step = opt_time / steps_taken if steps_taken > 0 else None

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
                                "skipped": "TS optimization failed",
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
                            },
                        )

                        # Calculate total time
                        total_time = sum(
                            v for v in reaction_data["timings"].values() if v is not None
                        )
                        reaction_data["timings"]["total"] = total_time

                        # Print status
                        freq_info = reaction_data["frequency_results"]
                        if "is_transition_state" in freq_info:
                            freq_info["is_transition_state"]
                        else:
                            pass

                        if verbose and avg_time_per_step:
                            pass

                    except Exception as e:
                        reaction_data = {
                            "success": False,
                            "error": str(e),
                            "timings": {},
                            "optimization_results": {},
                            "frequency_results": {},
                        }

                    backend_results[reaction] = reaction_data

            except Exception as e:
                backend_results = {"_backend_error": {"error": str(e)}}

            results[backend] = backend_results

        self.results = results
        return results

    def analyze_performance(self, backends: list[str]) -> dict:
        """Analyze performance metrics across backends with detailed statistics."""
        analysis = {}

        for backend in backends:
            backend_data = self.results.get(backend, {})

            # Collect successful TS calculations
            successful_ts = []
            failed_count = 0
            converged_count = 0
            ts_verified_count = 0
            timing_stats = {"total": [], "optimization": [], "frequency": []}
            step_stats = []

            for data in backend_data.values():
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
                [r for r in backend_data.values() if isinstance(r, dict) and not r.get("skipped")],
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

                ts_stats["convergence_rate"]
                ts_stats["verification_rate"]

                # Timing statistics
                if timing_stats["total"]:
                    np.mean(timing_stats["total"])
                    np.std(timing_stats["total"])
                    if timing_stats["optimization"]:
                        np.mean(timing_stats["optimization"])
                        np.std(timing_stats["optimization"])
                    if timing_stats["frequency"]:
                        np.mean(timing_stats["frequency"])
                        np.std(timing_stats["frequency"])

                # Step statistics
                if step_stats:
                    pass

            else:
                ts_stats = {
                    "total_reactions": total_reactions,
                    "converged": 0,
                    "ts_verified": 0,
                    "failed": failed_count,
                    "convergence_rate": 0,
                    "verification_rate": 0,
                }

            analysis[backend] = {"ts_statistics": ts_stats}

        # Comprehensive summary table

        # Print legend

        # Header

        # Results
        for backend in backends:
            analysis.get(backend, {}).get("ts_statistics", {})

        return analysis

    def save_results(self, filename: str = "zimmermann93_benchmark_results.json") -> None:
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


def main() -> int:
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
    requested = [b.strip() for b in args.backends.split(",")] if args.backends else None
    _backend, backends = interface.select_backend(
        requested_backends=requested,
        verbose=args.verbose,
    )
    if not backends:
        interface.print_error("No ML backends available!")
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
