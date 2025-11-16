#!/usr/bin/env python3
"""QME Hessian Benchmark - Hessian Method Comparison and Analysis.

This benchmark consolidates functionality from:
- hessian_comparison.py (FD schemes and backend method comparison)
- adaptive_hessian_demo.py (adaptive features demonstration)
- hessian_method_comparison_base.py (support functions)

It provides comprehensive comparison of:
- Finite difference schemes (3-point, 5-point, 7-point, Richardson, adaptive)
- Backend analytical methods (double_backward, vmap, fairchem, etc.)
- Adaptive features (autoselect, adaptive delta, noise estimation)
"""

import sys
import time
import warnings
from typing import Any

import numpy as np
from ase import Atoms

from qme.analysis.frequency import FrequencyAnalysis, HessianCalculator
from qme.analysis.hessian_energy import EnergyBasedHessianCalculator
from qme.analysis.noise_estimation import estimate_force_noise, estimate_optimal_delta
from qme.example_utils import (
    QMEExampleInterface,
    create_methane_molecule,
    create_standard_epilog,
    create_water_molecule,
    get_calculator_for_backend,
    setup_example_environment,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class HarmonicCalculator:
    """Mock calculator for harmonic potential: E = 0.5 * k * Σr²."""

    def __init__(self, k: float = 1.0) -> None:
        """Initialize with force constant k."""
        self.k = k

    def get_forces(self, atoms) -> np.ndarray:
        """Compute harmonic forces: F = -k * r."""
        return -self.k * atoms.positions

    def get_potential_energy(self, atoms) -> float:
        """Compute harmonic potential energy: E = 0.5 * k * Σr²."""
        return 0.5 * self.k * np.sum(atoms.positions**2)

    def get_hessian(self, atoms) -> np.ndarray:
        """Compute analytical harmonic Hessian: H = k * I."""
        n_atoms = len(atoms)
        return self.k * np.eye(3 * n_atoms)


# ============================================================================
# Support functions (merged from hessian_method_comparison_base.py)
# ============================================================================


def compute_finite_difference_hessian(
    atoms: Atoms,
    delta: float = 0.01,
) -> np.ndarray:
    """Compute Hessian using finite differences."""
    hessian_calc = HessianCalculator(
        atoms=atoms,
        calculator=atoms.calc,
        delta=delta,
        method="central",
        verbose=0,
    )
    return hessian_calc.calculate_numerical_hessian()


def compute_frequencies_from_hessian(
    hessian: np.ndarray,
    masses: np.ndarray,
) -> np.ndarray:
    """Compute vibrational frequencies (cm^-1) from Hessian using ASE's convention."""
    import ase.units as units

    # Mass-weighted Hessian
    mass_matrix = np.kron(np.diag(1.0 / np.sqrt(masses)), np.eye(3))
    Hmw = mass_matrix @ hessian @ mass_matrix

    # Eigenvalues of mass-weighted Hessian (omega^2)
    omega2, _ = np.linalg.eigh(Hmw)

    # h * nu in eV
    s = units._hbar * 1e10 / np.sqrt(units._e * units._amu)
    hnu_eV = s * np.sqrt(np.clip(omega2, 0.0, None))

    # Carry sign for imaginary modes
    sign = np.sign(omega2)
    frequencies_cm = (sign * hnu_eV) / units.invcm

    return frequencies_cm


def compute_metrics(
    analytical: np.ndarray,
    reference: np.ndarray,
    verbose: bool = False,
) -> dict[str, float]:
    """Compute comparison metrics between analytical and reference Hessians."""
    diff = analytical - reference
    abs_diff = np.abs(diff)
    rel_diff = abs_diff / (np.abs(reference) + 1e-10)

    metrics = {
        "max_absolute_error": np.max(abs_diff),
        "mean_absolute_error": np.mean(abs_diff),
        "rms_error": np.sqrt(np.mean(diff**2)),
        "max_relative_error": np.max(rel_diff),
        "mean_relative_error": np.mean(rel_diff),
        "elements_within_1e-2": np.sum(abs_diff <= 0.01),
        "elements_within_1e-3": np.sum(abs_diff <= 0.001),
        "total_elements": diff.size,
        "symmetry_error": np.max(np.abs(analytical - analytical.T)),
    }

    if verbose:
        print(f"  Max absolute error: {metrics['max_absolute_error']:.6f} eV/Å²")
        print(f"  Mean absolute error: {metrics['mean_absolute_error']:.6f} eV/Å²")
        print(f"  RMS error: {metrics['rms_error']:.6f} eV/Å²")
        print(f"  Max relative error: {metrics['max_relative_error']:.6e}")
        print(f"  Symmetry error: {metrics['symmetry_error']:.6e} eV/Å²")

    return metrics


def analyze_frequencies(
    hessian: np.ndarray,
    masses: np.ndarray,
    label: str = "",
) -> dict[str, float]:
    """Analyze frequencies from Hessian."""
    frequencies = compute_frequencies_from_hessian(hessian, masses)

    # Skip translational/rotational modes (first 6 for non-linear molecule)
    vib_freqs = frequencies[6:]

    stats = {
        "n_negative_frequencies": np.sum(vib_freqs < 0),
        "n_small_negative": np.sum((vib_freqs < 0) & (vib_freqs > -50)),
        "min_frequency": np.min(vib_freqs),
        "max_frequency": np.max(vib_freqs),
        "mean_abs_frequency": np.mean(np.abs(vib_freqs)),
    }

    return stats


def compare_backend_methods(
    atoms: Atoms,
    verbose: bool = True,
) -> dict[str, dict]:
    """Compare all Hessian computation methods for a backend."""
    if not hasattr(atoms.calc, "get_hessian"):
        raise ValueError("Calculator does not support analytical Hessian")

    if verbose:
        print(f"Comparing Hessian methods for {len(atoms)} atoms")
        print("=" * 80)

    # Ensure omol task invariants are set to avoid backend defaults/warnings
    if "charge" not in atoms.info:
        atoms.info["charge"] = 0
    if "spin" not in atoms.info:
        atoms.info["spin"] = 1

    # Test finite difference convergence first
    if verbose:
        print("\nStep 1: Testing finite difference convergence...")
    step_sizes = [0.05, 0.01, 0.005, 0.001]
    if len(atoms) > 20:
        step_sizes = [0.05, 0.01]  # keep short for big molecules
    fd_results = {}
    for delta in step_sizes:
        if verbose:
            print(f"\n  Computing FD Hessian with delta={delta} Å...")
        start = time.time()
        fd_hessian = compute_finite_difference_hessian(atoms, delta=delta)
        fd_time = time.time() - start
        fd_results[delta] = {"hessian": fd_hessian, "time": fd_time}
        if verbose:
            print(f"    Time: {fd_time:.3f} s")

    # Test Richardson extrapolation
    if verbose and 0.01 in step_sizes and 0.005 in step_sizes:
        print("\n  Computing Richardson extrapolation (delta=0.01, delta2=0.005)...")
        start = time.time()
        H_richardson = HessianCalculator(
            atoms, atoms.calc, delta=0.01, method="central", richardson=True, verbose=0
        ).calculate_numerical_hessian()
        richardson_time = time.time() - start
        fd_results["richardson"] = {"hessian": H_richardson, "time": richardson_time}
        if verbose:
            print(f"    Time: {richardson_time:.3f} s")

    # Use smallest step as reference
    reference_delta = step_sizes[-1]
    reference_hessian = fd_results[reference_delta]["hessian"]

    if verbose:
        print(f"\n  Using FD(delta={reference_delta}) as reference")

    # Test analytical methods
    import inspect

    sig = inspect.signature(atoms.calc.get_hessian)
    params = list(sig.parameters.keys())
    supports_method_param = "method" in params

    if supports_method_param:
        method_candidates = [
            "double_backward",
            "vmap",
            "fairchem",
            "fairchem_loop",
        ]
        methods_to_test = [
            (method_name, symmetrize)
            for method_name in method_candidates
            for symmetrize in (True, False)
        ]
    else:
        # MACE-style: no method/symmetrize parameters
        methods_to_test = [(None, None)]

    results = {}
    masses = atoms.get_masses()

    for method, symmetrize in methods_to_test:
        if method is None:
            method_name = "analytical"
        else:
            method_name = f"{method}_sym={symmetrize}"
        if verbose:
            print(f"\n{method_name}:")
            print("-" * 80)

        # Compute analytical Hessian
        start = time.time()
        try:
            if method is None:
                analytical_hessian = atoms.calc.get_hessian(atoms)
            else:
                analytical_hessian = atoms.calc.get_hessian(
                    atoms, method=method, symmetrize=symmetrize
                )
            analytical_time = time.time() - start
        except Exception as e:
            if verbose:
                print(f"  ERROR: {e}")
            results[method_name] = {"error": str(e)}
            continue

        # Compute metrics against best FD reference
        metrics = compute_metrics(analytical_hessian, reference_hessian, verbose=verbose)

        # Also compute metrics against Richardson if available
        richardson_metrics = None
        if "richardson" in fd_results:
            richardson_metrics = compute_metrics(
                analytical_hessian, fd_results["richardson"]["hessian"], verbose=False
            )

        # Analyze frequencies
        freq_stats = analyze_frequencies(analytical_hessian, masses, method_name)

        results[method_name] = {
            "metrics": metrics,
            "richardson_metrics": richardson_metrics,
            "freq_stats": freq_stats,
            "time": analytical_time,
            "speedup_vs_fd": fd_results[reference_delta]["time"] / analytical_time,
        }

        if verbose:
            print(
                f"\n  Timing: {analytical_time:.3f} s (speedup vs FD: {results[method_name]['speedup_vs_fd']:.2f}x)"
            )

    return results


# ============================================================================
# Benchmark functions
# ============================================================================


def benchmark_fd_schemes(atoms: Atoms, delta: float = 0.05) -> dict[str, dict[str, Any]]:
    """Benchmark finite difference Hessian schemes."""
    calc = HarmonicCalculator(k=1.0)
    hessian_analytical = calc.get_hessian(atoms)

    results = {}

    # Test all FD schemes
    schemes = [
        ("3-point central", "central", False, False),
        ("5-point central", "5point", False, False),
        ("7-point central", "7point", False, False),
        ("3-point + Richardson", "central", True, False),
        ("5-point + Richardson", "5point", True, False),
        ("7-point + Richardson", "7point", True, False),
        ("Adaptive 5-point + Richardson", "5point", True, True),
    ]

    for scheme_name, method, richardson, adaptive in schemes:
        start_time = time.time()
        try:
            hc = HessianCalculator(
                atoms,
                calc,
                delta=delta,
                method=method,
                richardson=richardson,
                adaptive_delta=adaptive,
                max_iterations=3 if adaptive else None,
                verbose=0,
            )
            hessian = hc.calculate_numerical_hessian()
            elapsed = time.time() - start_time
            error = np.max(np.abs(hessian - hessian_analytical))
            results[scheme_name] = {
                "max_error": float(error),
                "time": elapsed,
                "success": True,
            }
        except Exception as e:
            results[scheme_name] = {
                "max_error": float("inf"),
                "time": time.time() - start_time,
                "success": False,
                "error": str(e),
            }

    # Energy-based FD
    start_time = time.time()
    try:
        ebc = EnergyBasedHessianCalculator(atoms, calc, delta=delta, verbose=0)
        hessian_energy = ebc.calculate_energy_hessian()
        elapsed = time.time() - start_time
        error = np.max(np.abs(hessian_energy - hessian_analytical))
        results["Energy-based FD"] = {
            "max_error": float(error),
            "time": elapsed,
            "success": True,
        }
    except Exception as e:
        results["Energy-based FD"] = {
            "max_error": float("inf"),
            "time": time.time() - start_time,
            "success": False,
            "error": str(e),
        }

    return results


def benchmark_adaptive_features(atoms: Atoms, calc, verbose: bool = False) -> dict[str, Any]:
    """Benchmark adaptive Hessian features."""
    results = {}

    # Autoselect
    if verbose:
        print("\nTesting autoselect method...")
    start_time = time.time()
    try:
        freq_analysis = FrequencyAnalysis(atoms, calc, verbose=0)
        hessian = freq_analysis.calculate_hessian(method="autoselect")
        elapsed = time.time() - start_time
        symmetry_error = np.max(np.abs(hessian - hessian.T))
        results["autoselect"] = {
            "success": True,
            "time": elapsed,
            "symmetry_error": float(symmetry_error),
        }
    except Exception as e:
        results["autoselect"] = {"success": False, "error": str(e)}

    # Adaptive delta
    if verbose:
        print("\nTesting adaptive delta selection...")
    start_time = time.time()
    try:
        hc_fixed = HessianCalculator(atoms, calc, delta=0.01, adaptive_delta=False, verbose=0)
        hessian_fixed = hc_fixed.calculate_numerical_hessian()
        asymmetry_fixed = np.max(np.abs(hessian_fixed - hessian_fixed.T))

        hc_adaptive = HessianCalculator(
            atoms, calc, delta=0.01, adaptive_delta=True, max_iterations=3, verbose=0
        )
        hessian_adaptive = hc_adaptive.calculate_numerical_hessian()
        asymmetry_adaptive = np.max(np.abs(hessian_adaptive - hessian_adaptive.T))

        results["adaptive_delta"] = {
            "success": True,
            "fixed_asymmetry": float(asymmetry_fixed),
            "adaptive_asymmetry": float(asymmetry_adaptive),
            "improvement": float(asymmetry_fixed / asymmetry_adaptive)
            if asymmetry_adaptive > 0
            else 0.0,
        }
    except Exception as e:
        results["adaptive_delta"] = {"success": False, "error": str(e)}

    # Noise estimation
    if verbose:
        print("\nTesting noise estimation...")
    try:
        force_noise = estimate_force_noise(atoms, calc, n_samples=5)
        optimal_delta, expected_noise = estimate_optimal_delta(
            atoms, calc, delta_range=(0.001, 0.05), max_iterations=3, verbose=0
        )
        results["noise_estimation"] = {
            "success": True,
            "force_noise": float(force_noise),
            "optimal_delta": float(optimal_delta),
            "expected_noise": float(expected_noise),
        }
    except Exception as e:
        results["noise_estimation"] = {"success": False, "error": str(e)}

    return results


# ============================================================================
# Summary printing functions
# ============================================================================


def print_fd_scheme_summary(results_list: list[dict[str, Any]]) -> None:
    """Print summary table for FD scheme comparison."""
    print(f"\n{'=' * 120}")
    print("FINITE DIFFERENCE SCHEME COMPARISON")
    print(f"{'=' * 120}")

    # Collect all FD results
    fd_results = {}
    for result in results_list:
        if "fd_schemes" in result:
            for scheme, data in result["fd_schemes"].items():
                if scheme not in fd_results:
                    fd_results[scheme] = []
                fd_results[scheme].append(data)

    if not fd_results:
        print("No FD scheme results available.")
        return

    print(f"\n{'Method':<35} {'Max Error':<15} {'Time (s)':<15} {'Status':<10}")
    print("=" * 120)

    for scheme_name in sorted(fd_results.keys()):
        all_data = fd_results[scheme_name]
        successful = [d for d in all_data if d.get("success", False)]
        if successful:
            avg_error = np.mean([d["max_error"] for d in successful])
            avg_time = np.mean([d["time"] for d in successful])
            status = "✅"
        else:
            avg_error = float("inf")
            avg_time = 0.0
            status = "❌"

        error_str = f"{avg_error:.2e}" if avg_error != float("inf") else "N/A"
        time_str = f"{avg_time:.6f}" if avg_time > 0 else "N/A"
        print(f"{scheme_name:<35} {error_str:<15} {time_str:<15} {status:<10}")


def print_backend_method_summary(results_list: list[dict[str, Any]]) -> None:
    """Print summary table for backend method comparison."""
    print(f"\n{'=' * 120}")
    print("BACKEND METHOD COMPARISON")
    print(f"{'=' * 120}")

    # Collect all backend results
    backend_results = {}
    for result in results_list:
        if "backend_methods" in result:
            backend = result.get("backend", "unknown")
            molecule = result.get("molecule", "unknown")
            key = f"{backend}/{molecule}"
            if key not in backend_results:
                backend_results[key] = {}
            backend_results[key].update(result["backend_methods"])

    if not backend_results:
        print("No backend method results available.")
        return

    print(
        f"\n{'Backend/Molecule':<25} {'Method':<30} {'RMS Error':<15} {'Time (s)':<15} {'Status':<10}"
    )
    print("=" * 120)

    for key, methods in backend_results.items():
        for method_name, method_data in sorted(methods.items()):
            if "error" in method_data:
                status = "❌"
                rms_error = "N/A"
                time_val = "N/A"
            else:
                status = "✅"
                rms_error = f"{method_data['metrics']['rms_error']:.6f}"
                time_val = f"{method_data['time']:.3f}"

            print(f"{key:<25} {method_name:<30} {rms_error:<15} {time_val:<15} {status:<10}")


def print_recommendations(results_list: list[dict[str, Any]]) -> None:
    """Print recommendations based on benchmark results."""
    print(f"\n{'=' * 120}")
    print("RECOMMENDATIONS")
    print(f"{'=' * 120}")

    # FD scheme recommendations
    fd_results = {}
    for result in results_list:
        if "fd_schemes" in result:
            for scheme, data in result["fd_schemes"].items():
                if data.get("success", False):
                    if scheme not in fd_results:
                        fd_results[scheme] = []
                    fd_results[scheme].append(data["max_error"])

    if fd_results:
        best_fd = min(fd_results.keys(), key=lambda k: np.mean(fd_results[k]))
        print(f"\n★ Best FD Scheme: {best_fd}")
        print(f"  Average error: {np.mean(fd_results[best_fd]):.2e} eV/Å²")

    # Backend method recommendations
    backend_methods = {}
    for result in results_list:
        if "backend_methods" in result:
            backend = result.get("backend", "unknown")
            for method_name, method_data in result["backend_methods"].items():
                if "error" not in method_data:
                    key = f"{backend}/{method_name}"
                    if key not in backend_methods:
                        backend_methods[key] = []
                    backend_methods[key].append(method_data["metrics"]["rms_error"])

    if backend_methods:
        best_backend_method = min(backend_methods.keys(), key=lambda k: np.mean(backend_methods[k]))
        avg_rms = np.mean(backend_methods[best_backend_method])
        print(f"\n★ Best Backend Method: {best_backend_method}")
        print(f"  Average RMS error: {avg_rms:.6f} eV/Å²")

        # Parse method name for recommendation
        if "_sym=" in best_backend_method:
            parts = best_backend_method.split("/")[1].split("_sym=")
            method = parts[0]
            symmetrize = parts[1] == "True"
            print(f"\nRecommended default: method='{method}', symmetrize={symmetrize}")


# ============================================================================
# Main benchmark function
# ============================================================================


def benchmark_hessian(
    backend: str | None = None,
    molecule: str = "water",
    mode: str = "both",
    device: str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run Hessian benchmark for a specific backend and molecule."""
    result = {
        "backend": backend,
        "molecule": molecule,
        "mode": mode,
        "available": False,
        "error": None,
        "fd_schemes": {},
        "backend_methods": {},
        "adaptive_features": {},
    }

    try:
        # Create test molecule
        if molecule == "water":
            atoms = create_water_molecule()
        elif molecule == "methane":
            atoms = create_methane_molecule()
        else:
            raise ValueError(f"Unknown molecule: {molecule}")

        # FD scheme comparison (if mode is 'fd' or 'both')
        if mode in ["fd", "both"]:
            if verbose:
                print(f"\nBenchmarking FD schemes for {molecule}...")
            result["fd_schemes"] = benchmark_fd_schemes(atoms, delta=0.05)
            result["available"] = True

        # Backend method comparison (if mode is 'backend' or 'both' and backend is available)
        if mode in ["backend", "both"] and backend:
            if not get_calculator_for_backend(backend, device=device):
                result["error"] = f"Backend {backend} not available"
                return result

            # Set up calculator
            if backend == "uma":
                calc = get_calculator_for_backend(backend, device=device, model_name="uma-s-1p1")
            else:
                calc = get_calculator_for_backend(backend, device=device)
            calc.ensure_loaded()
            atoms.calc = calc

            if verbose:
                print(f"\nBenchmarking backend methods for {backend}/{molecule}...")
            result["backend_methods"] = compare_backend_methods(atoms, verbose=verbose)

            # Adaptive features
            if verbose:
                print(f"\nBenchmarking adaptive features for {backend}/{molecule}...")
            result["adaptive_features"] = benchmark_adaptive_features(atoms, calc, verbose=verbose)

            result["available"] = True

    except Exception as e:
        result["error"] = str(e)
        if verbose:
            import traceback

            traceback.print_exc()

    return result


# ============================================================================
# Main entry point
# ============================================================================


@setup_example_environment
def main() -> int:
    """Run the Hessian benchmark."""
    # Create standardized interface
    interface = QMEExampleInterface(
        name="Hessian Benchmark",
        description="Hessian Method Comparison and Analysis",
        epilog=create_standard_epilog("benchmark"),
    )

    parser = interface.create_parser()
    parser.add_argument(
        "--mode",
        choices=["fd", "backend", "both"],
        default="both",
        help="Comparison mode: fd (finite difference schemes only), backend (backend methods only), both (default)",
    )
    parser.add_argument(
        "--molecules",
        type=str,
        help="Comma-separated list of molecules to test (default: water,methane)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with water molecule only",
    )

    args = parser.parse_args()

    interface.print_header()
    interface.setup_logging(args.verbose)

    # Backend handling
    requested = [b.strip() for b in args.backends.split(",")] if args.backends else None
    backend, available_backends = interface.select_backend(
        requested_backends=requested,
        preferred_backends=["uma", "mace"],
        verbose=args.verbose,
    )

    # Get device info
    device = interface.get_device_info(args.device)

    # Determine molecules to test
    if args.quick:
        molecules = ["water"]
    elif args.molecules:
        molecules = [m.strip() for m in args.molecules.split(",")]
    else:
        molecules = ["water", "methane"]

    config = {
        "Backend": backend or "none (FD only)",
        "Device": device,
        "Mode": args.mode,
        "Molecules": ", ".join(molecules),
        "Verbose": args.verbose,
    }
    interface.print_configuration(config)

    # Run benchmarks
    results_list = []
    total_tests = len(molecules) * (len(available_backends) if backend else 1)
    current_test = 0

    try:
        # FD schemes (no backend needed)
        if args.mode in ["fd", "both"]:
            for molecule in molecules:
                current_test += 1
                if args.verbose >= 1:
                    print(f"\n[{current_test}/{total_tests}] Testing FD schemes for {molecule}...")
                result = benchmark_hessian(
                    backend=None,
                    molecule=molecule,
                    mode="fd",
                    device=device,
                    verbose=args.verbose >= 2,
                )
                results_list.append(result)

        # Backend methods
        if args.mode in ["backend", "both"] and backend:
            for molecule in molecules:
                current_test += 1
                if args.verbose >= 1:
                    print(f"\n[{current_test}/{total_tests}] Testing {backend}/{molecule}...")
                result = benchmark_hessian(
                    backend=backend,
                    molecule=molecule,
                    mode="backend",
                    device=device,
                    verbose=args.verbose >= 2,
                )
                results_list.append(result)

        # Print summaries
        if args.mode in ["fd", "both"]:
            print_fd_scheme_summary(results_list)
        if args.mode in ["backend", "both"]:
            print_backend_method_summary(results_list)
        print_recommendations(results_list)

        # Save results
        interface.save_results(results_list, args.output or interface.get_default_output_file())

        interface.print_success()
        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1
    except Exception as e:
        interface.print_error(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
