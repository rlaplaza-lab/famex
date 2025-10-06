#!/usr/bin/env python3
"""
QME Zimmermann-93 Benchmark - Two-Ended Transition State Search

This benchmark runs two-ended (reactant → product) transition state searches
across available ML backends in QME. For each reaction in the provided dataset:
  - Loads reactant and product geometries
  - Interpolates an initial path and (optionally) optimizes it with NEB-like procedure
  - Picks the highest-energy image as a TS guess and runs TS optimization
  - Compares the located TS geometry to the reference TS (RMSD after alignment)

Usage:
    conda run -n py312 python zimmermann93_benchmark.py [--quick|--quicker]
    conda run -n py312 python zimmermann93_benchmark.py --backends uma,mace
    conda run -n py312 python zimmermann93_benchmark.py --npoints 15

Features:
    - Two-ended transition state search evaluation
    - NEB-like path optimization capabilities
    - Geometry comparison with reference structures
    - Comprehensive backend performance analysis
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
from typing import Dict, List, Optional, Tuple

import numpy as np
from ase import Atoms
from ase.io import read

# Import QME components
try:
    from qme import Explorer, calculator_registry
    from qme.dependencies import HAS_SELLA
except ImportError as e:
    print(f"❌ Error importing QME: {e}")
    print("   Please ensure QME is installed and accessible")
    sys.exit(1)

# Import device utilities
from device_utils import get_optimal_device, print_device_info

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


def kabsch_align(reference: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, float]:
    """Align target to reference using the Kabsch algorithm and return (aligned, rmsd).

    Both arrays must be shape (N,3).
    """
    # Center
    ref_cent = reference.mean(axis=0)
    tar_cent = target.mean(axis=0)
    ref = reference - ref_cent
    tar = target - tar_cent

    # Covariance
    C = np.dot(tar.T, ref)
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(np.dot(V, Wt)))
    D = np.diag([1.0, 1.0, d])
    U = np.dot(np.dot(V, D), Wt)

    aligned = np.dot(tar, U.T)
    aligned += ref_cent

    rmsd = np.sqrt(np.mean(np.sum((aligned - reference) ** 2, axis=1)))
    return aligned, float(rmsd)


def compute_rmsd_flexible(ref_atoms, opt_atoms) -> float:
    """Compute RMSD between two ASE Atoms-like objects.

    Tries direct Kabsch when shapes and ordering match. If ordering differs but
    atom counts and element types match, performs a greedy per-element matching
    and then Kabsch-aligns the paired coordinates.

    Returns float('nan') when a reliable pairing cannot be determined.
    """
    try:
        ref_coords = np.array(ref_atoms.get_positions())
        opt_coords = np.array(opt_atoms.get_positions())
    except Exception:
        return float("nan")

    ref_syms = ref_atoms.get_chemical_symbols()
    opt_syms = opt_atoms.get_chemical_symbols()

    # Quick path: same shape and same symbol ordering
    if ref_coords.shape == opt_coords.shape and ref_syms == opt_syms:
        _, rmsd = kabsch_align(ref_coords, opt_coords)
        return rmsd

    # If different number of atoms, cannot compute
    if ref_coords.shape[0] != opt_coords.shape[0]:
        return float("nan")

    n = ref_coords.shape[0]

    # Attempt greedy per-element matching
    paired_indices = [-1] * n
    used_opt = set()

    for i, sym in enumerate(ref_syms):
        # candidate indices in opt with same symbol and not used
        candidates = [
            j for j, s in enumerate(opt_syms) if s == sym and j not in used_opt
        ]
        if not candidates:
            # cannot find a matching element type
            return float("nan")

        # choose closest by Euclidean distance (no rotation yet) as initial pairing
        dists = [np.linalg.norm(ref_coords[i] - opt_coords[j]) for j in candidates]
        jmin = candidates[int(np.argmin(dists))]
        paired_indices[i] = jmin
        used_opt.add(jmin)

    # Construct ordered arrays according to pairing
    opt_ordered = np.array([opt_coords[j] for j in paired_indices])
    ref_ordered = ref_coords.copy()

    # Final alignment and RMSD
    try:
        _, rmsd = kabsch_align(ref_ordered, opt_ordered)
        return rmsd
    except Exception:
        return float("nan")


class Zimmermann93Benchmark:
    """Benchmark suite for two-ended TS search on Zimmermann-93 dataset."""

    def __init__(
        self, dataset_dir: Optional[str] = None, output_dir: str = "benchmark_results"
    ):
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

        self.results: Dict[str, Dict] = {}

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

    def get_reactant_and_product(self, reaction_name: str) -> Tuple[Atoms, Atoms]:
        reactant_file = f"{reaction_name}_reactant.xyz"
        product_file = f"{reaction_name}_product.xyz"
        return self.load_structure(reactant_file), self.load_structure(product_file)

    def get_reference_ts(self, reaction_name: str) -> Atoms:
        ts_file = f"{reaction_name}_ts.xyz"
        return self.load_structure(ts_file)

    def get_available_backends(self) -> List[str]:
        """Get list of available ML backends (excluding mock)."""
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
                "No ML backends available for benchmarking. "
                "The mock backend is excluded as it cannot optimize transition states."
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

    def print_backend_summary(
        self, backends: List[str], title: str = "Available Backends"
    ):
        """Print a formatted summary of backends."""
        print(f"\n📋 {title}")
        print("-" * 50)
        for i, backend in enumerate(backends, 1):
            print(f"  {i}. {backend}")
        print(f"Total: {len(backends)} backends")

    def run_benchmark(
        self,
        backends: List[str],
        reactions: List[str],
        npoints: int = 11,
        interp_method: str = "geodesic",
        optimize_path: bool = True,
        fmax: float = 0.01,
        steps: int = 300,
    ) -> Dict:
        results: Dict[str, Dict] = {}

        for backend in backends:
            print(f"\nBackend: {backend.upper()}")
            print("-" * 60)
            backend_results: Dict[str, Dict] = {}

            # Use None for auto-detection of model
            model_name = None

            try:
                for reaction in reactions:
                    print(f"  Reaction: {reaction}")
                    reaction_data: Dict = {}

                    try:
                        with suppress_verbose_output():
                            # Will create optimizer per structure below

                            # Load endpoints
                            reactant, product = self.get_reactant_and_product(reaction)

                            # Build Reaction object and interpolate
                            from qme.core.reaction import Reaction

                            reaction_obj = Reaction(reactant, product)

                            # Set calculator for path generation/energies
                            try:
                                qme_opt = Explorer(
                                    atoms=reactant,
                                    backend=backend,
                                    model_name=model_name,
                                )
                                # Get calculator from the explorer
                                qme_calc = qme_opt.atoms_list[0].calc
                                reaction_obj.set_calculator(qme_calc)
                            except Exception:
                                # fallback to mock calculator
                                mock_opt = Explorer(atoms=reactant, backend="mock")
                                reaction_obj.set_calculator(mock_opt.atoms_list[0].calc)

                            path = reaction_obj.interpolate(
                                npoints=npoints,
                                method=interp_method,
                                optimize_path=optimize_path,
                                calculator=reaction_obj.calculator,
                            )

                            # Evaluate energies along path
                            energies = reaction_obj.calculate_path_energies(path)

                            # Pick highest energy image as TS guess
                            max_idx = int(np.nanargmax(energies))
                            ts_guess_geom = path[max_idx]
                            ts_guess_atoms = (
                                ts_guess_geom  # Geometry objects are ASE-compatible
                            )

                            # TS optimization will be done with fresh optimizer

                            ts_result = {}
                            if HAS_SELLA:
                                try:
                                    ts_optimizer = Explorer(
                                        atoms=ts_guess_atoms,
                                        backend=backend,
                                        model_name=model_name,
                                    )
                                    ts_result = ts_optimizer.run(
                                        mode="ts", fmax=fmax, steps=steps
                                    )
                                    ts_success = ts_result.get(
                                        "converged", False
                                    ) or ts_result.get("ts_converged", False)
                                    ts_opt_atoms = ts_result.get(
                                        "optimized_atoms"
                                    ) or ts_result.get("ts_atoms")
                                except Exception as e:
                                    ts_success = False
                                    ts_result = {"error": str(e)}
                                    ts_opt_atoms = None
                            else:
                                # No SELLA: fallback to single-point energy evaluation only
                                try:
                                    # Create optimizer for single-point calculation
                                    ts_optimizer = Explorer(
                                        atoms=ts_guess_atoms,
                                        backend=backend,
                                        model_name=model_name,
                                    )
                                    ts_guess_atoms.calc = ts_optimizer.calculator
                                    with suppress_verbose_output():
                                        energy = ts_guess_atoms.get_potential_energy()
                                    ts_result = {
                                        "ts_energy": energy,
                                        "method": "single_point",
                                        "converged": False,
                                    }
                                    ts_success = False
                                    ts_opt_atoms = ts_guess_atoms
                                except Exception as e:
                                    ts_result = {"error": str(e)}
                                    ts_success = False
                                    ts_opt_atoms = None

                            # Compare geometry to reference TS (use flexible RMSD matcher)
                            try:
                                ref_ts = self.get_reference_ts(reaction)
                                if ts_opt_atoms is not None:
                                    rmsd = compute_rmsd_flexible(ref_ts, ts_opt_atoms)
                                else:
                                    rmsd = float("nan")
                            except Exception:
                                rmsd = float("nan")

                            reaction_data.update(
                                {
                                    "path_max_idx": int(max_idx),
                                    "path_energies": energies,
                                    "ts_guess_energy": (
                                        float(energies[max_idx])
                                        if len(energies) > max_idx
                                        else None
                                    ),
                                    "ts_result": ts_result,
                                    "ts_success": bool(ts_success),
                                    "ts_rmsd_to_reference": rmsd,
                                }
                            )

                            print(
                                f"    ✓ TS guess index: {max_idx}, RMSD to ref: {rmsd:.4f} Å"
                            )

                    except Exception as e:
                        reaction_data = {"success": False, "error": str(e)}
                        print(f"    ❌ Reaction failed: {e}")

                    backend_results[reaction] = reaction_data

            except Exception as e:
                print(f"❌ Backend initialization failed: {e}")
                backend_results = {"_backend_error": {"error": str(e)}}

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

            # Collect successful TS calculations
            successful_ts = []
            skipped_count = 0
            failed_count = 0

            for reaction, data in backend_data.items():
                if isinstance(data, dict):
                    if data.get("skipped"):
                        skipped_count += 1
                    elif not data.get("success", True):
                        failed_count += 1
                    else:
                        if data.get("ts_success"):
                            successful_ts.append(data)

            # Calculate TS statistics
            if successful_ts:
                rmsds = [
                    data["ts_rmsd_to_reference"]
                    for data in successful_ts
                    if not np.isnan(data["ts_rmsd_to_reference"])
                ]

                ts_stats = {
                    "count": len(successful_ts),
                    "mean_rmsd": np.mean(rmsds) if rmsds else 0,
                    "max_rmsd": np.max(rmsds) if rmsds else 0,
                    "std_rmsd": np.std(rmsds) if rmsds else 0,
                }

                print(f"Transition States ({ts_stats['count']} reactions):")
                print(f"  Mean RMSD: {ts_stats['mean_rmsd']:.4f} Å")
                print(f"  Max RMSD:  {ts_stats['max_rmsd']:.4f} Å")
                print(f"  Std RMSD:  {ts_stats['std_rmsd']:.4f} Å")
            else:
                ts_stats = {"count": 0}
                print("❌ No successful transition state calculations")

            # Report skipped and failed reactions
            if skipped_count > 0:
                print(f"⚠️  Skipped reactions: {skipped_count}")
            if failed_count > 0:
                print(f"❌ Failed reactions: {failed_count}")

            analysis[backend] = {"ts_statistics": ts_stats}

        # Cross-backend comparison
        print("\nBACKEND COMPARISON")
        print("-" * 50)
        print(f"{'Backend':<12} {'Success':<8} {'Mean RMSD':<12} {'Max RMSD':<12}")
        print("-" * 50)

        for backend in backends:
            stats = analysis.get(backend, {}).get("ts_statistics", {})

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
            mean_rmsd = (
                f"{stats.get('mean_rmsd', 0):.4f}"
                if stats.get("count", 0) > 0
                else "N/A"
            )
            max_rmsd = (
                f"{stats.get('max_rmsd', 0):.4f}"
                if stats.get("count", 0) > 0
                else "N/A"
            )

            print(f"{backend:<12} {success_rate:<8} {mean_rmsd:<12} {max_rmsd:<12}")

        return analysis

    def save_results(self, filename: str = "zimmermann93_benchmark_results.json"):
        out = self.output_dir / filename

        # convert numpy etc
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj

        serializable = convert(self.results)
        with open(out, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nResults saved to: {out}")


def main():
    """Main entry point for the benchmark."""
    parser = argparse.ArgumentParser(
        description="QME Zimmermann-93 Benchmark - Two-Ended Transition State Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  conda run -n py312 python zimmermann93_benchmark.py
  conda run -n py312 python zimmermann93_benchmark.py --backends aimnet2,uma
  conda run -n py312 python zimmermann93_benchmark.py --quick
  conda run -n py312 python zimmermann93_benchmark.py --quicker
  conda run -n py312 python zimmermann93_benchmark.py --npoints 15
        """,
    )

    parser.add_argument(
        "--backends", nargs="+", help="QME backends to test (default: all available)"
    )
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
        "--interp-method",
        choices=["linear", "geodesic"],
        default="geodesic",
        help="Interpolation method (default: geodesic)",
    )
    parser.add_argument(
        "--no-optimize-path",
        dest="optimize_path",
        action="store_false",
        help="Skip path optimization step",
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

    benchmark = Zimmermann93Benchmark(dataset_dir=None, output_dir=args.output_dir)
    available_backends = benchmark.get_available_backends()

    if args.backends:
        backends = benchmark.filter_available_backends(args.backends, verbose=True)
        if not backends:
            print("❌ No requested backends are available!")
            return 1
    else:
        backends = available_backends

    benchmark.print_backend_summary(backends, "Benchmarking Backends")

    if args.quicker:
        reactions = benchmark.quicker_reactions
    elif args.quick:
        reactions = benchmark.quick_reactions
    elif args.reactions:
        reactions = args.reactions
    else:
        reactions = benchmark.reactions

    invalid_rxn = [
        r
        for r in reactions
        if (benchmark.dataset_dir / f"{r}_reactant.xyz").exists() is False
    ]
    if invalid_rxn:
        print(f"❌ Invalid reactions: {invalid_rxn}")
        return 1

    print("\nConfiguration:")
    print(f"  NPoints: {args.npoints}")
    print(f"  Interpolation: {args.interp_method}")
    print(f"  Optimize path: {args.optimize_path}")
    print(f"  Force max: {args.fmax}")
    print(f"  Max steps: {args.steps}")

    benchmark.run_benchmark(
        backends,
        reactions,
        npoints=args.npoints,
        interp_method=args.interp_method,
        optimize_path=args.optimize_path,
        fmax=args.fmax,
        steps=args.steps,
    )

    # Analyze performance
    benchmark.analyze_performance(backends)

    # Save results
    benchmark.save_results()
    print("\n✅ Benchmark completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
