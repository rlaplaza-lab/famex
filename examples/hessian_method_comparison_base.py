#!/usr/bin/env python3
"""Base functions for Hessian method comparison.

This module contains shared functionality used by both MACE and UMA
Hessian method comparison scripts to eliminate code duplication.
"""

import time
from io import StringIO
from urllib.request import urlopen

import numpy as np
from ase import Atoms

from qme.analysis.frequency import HessianCalculator


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
    """Compute vibrational frequencies (cm^-1) from Hessian using ASE's convention.

    Follows ASE Vibrations:
    - H_mw = M^{-1/2} H M^{-1/2}
    - omega^2 = eig(H_mw)
    - hnu [eV] = s * sqrt(omega^2), s = units._hbar * 1e10 / sqrt(units._e * units._amu)
    - frequencies [cm^-1] = hnu / units.invcm
    Reference: ASE Vibrations source code.
    """
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
        print(
            f"  Elements within 0.01 eV/Å²: {metrics['elements_within_1e-2']}/{metrics['total_elements']} "
            f"({100 * metrics['elements_within_1e-2'] / metrics['total_elements']:.1f}%)"
        )
        print(
            f"  Elements within 0.001 eV/Å²: {metrics['elements_within_1e-3']}/{metrics['total_elements']} "
            f"({100 * metrics['elements_within_1e-3'] / metrics['total_elements']:.1f}%)"
        )

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


def compare_methods(
    atoms: Atoms,
    verbose: bool = True,
) -> dict[str, dict]:
    """Compare all Hessian computation methods."""
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
    # Use lighter FD for larger systems
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
        # Manual Richardson for validation
        H_richardson_manual = (
            4.0 * fd_results[0.005]["hessian"] - fd_results[0.01]["hessian"]
        ) / 3.0
        fd_results["richardson"] = {"hessian": H_richardson, "time": richardson_time}
        if verbose:
            print(f"    Time: {richardson_time:.3f} s")
            print(
                f"    Implementation matches manual: {np.allclose(H_richardson, H_richardson_manual, atol=1e-6)}"
            )
            print(
                f"    Max diff (manual vs implementation): {np.max(np.abs(H_richardson - H_richardson_manual)):.6e}"
            )

    # Find best FD reference (smallest step that converged)
    # Use smallest step as reference
    reference_delta = step_sizes[-1]
    reference_hessian = fd_results[reference_delta]["hessian"]

    if verbose:
        print(f"\n  Using FD(delta={reference_delta}) as reference")
        print(
            f"  Reference symmetry error: {np.max(np.abs(reference_hessian - reference_hessian.T)):.6e}"
        )
        # Diagnostics: norms and extrema for small systems
        if len(atoms) <= 6:
            ref_norm = np.linalg.norm(reference_hessian)
            ref_max = np.max(np.abs(reference_hessian))
            print(f"  Reference Hessian ||H||_F: {ref_norm:.6e}, max|H|: {ref_max:.6e} eV/Å²")

    # Test only the preferred analytical method
    # MACE may not support method/symmetrize parameters, so handle both cases
    if hasattr(atoms.calc, "get_hessian"):
        # Check if get_hessian accepts method/symmetrize parameters (UMA-style)
        import inspect

        sig = inspect.signature(atoms.calc.get_hessian)
        params = list(sig.parameters.keys())
        supports_method_param = "method" in params

        if supports_method_param:
            methods_to_test = [("double_backward", True)]
        else:
            # MACE-style: no method/symmetrize parameters
            methods_to_test = [(None, None)]
    else:
        methods_to_test = []

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
                # MACE-style: no parameters
                analytical_hessian = atoms.calc.get_hessian(atoms)
            else:
                # UMA-style: with method and symmetrize
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
            if verbose:
                print("\n  vs Richardson extrapolation:")
                print(f"    RMS error: {richardson_metrics['rms_error']:.6f} eV/Å²")

        # Analyze frequencies
        freq_stats = analyze_frequencies(analytical_hessian, masses, method_name)
        if verbose:
            print("\n  Frequency analysis:")
            print(f"    Negative frequencies: {freq_stats['n_negative_frequencies']}")
            print(f"    Small negative (< 50 cm^-1): {freq_stats['n_small_negative']}")
            print(
                f"    Frequency range: {freq_stats['min_frequency']:.2f} to {freq_stats['max_frequency']:.2f} cm^-1"
            )
        # Diagnostics: norms and extrema for analytical Hessian
        if verbose and len(atoms) <= 6:
            an_norm = np.linalg.norm(analytical_hessian)
            an_max = np.max(np.abs(analytical_hessian))
            print(f"\n  Analytical Hessian ||H||_F: {an_norm:.6e}, max|H|: {an_max:.6e} eV/Å²")

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

    # Summary
    if verbose:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        # Sort by RMS error (best agreement with FD)
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].get("metrics", {}).get("rms_error", float("inf"))
            if "error" not in x[1]
            else float("inf"),
        )

        print("\nMethods ranked by RMS error (best to worst):")
        for i, (method_name, result) in enumerate(sorted_results, 1):
            if "error" in result:
                print(f"  {i}. {method_name}: ERROR - {result['error']}")
            else:
                print(
                    f"  {i}. {method_name}: "
                    f"RMS={result['metrics']['rms_error']:.6f} eV/Å², "
                    f"Max={result['metrics']['max_absolute_error']:.6f} eV/Å², "
                    f"NegFreq={result['freq_stats']['n_negative_frequencies']}, "
                    f"Time={result['time']:.3f}s"
                )

        # Find best method
        best_method = sorted_results[0][0]
        best_result = sorted_results[0][1]
        if "error" not in best_result:
            print(f"\n★ BEST METHOD: {best_method}")
            print(f"  RMS error: {best_result['metrics']['rms_error']:.6f} eV/Å²")
            print(f"  Max error: {best_result['metrics']['max_absolute_error']:.6f} eV/Å²")
            print(f"  Negative frequencies: {best_result['freq_stats']['n_negative_frequencies']}")
            print(f"  Computation time: {best_result['time']:.3f} s")

    return results


def load_xyz_from_url(url: str) -> Atoms:
    """Load an Atoms object from a URL."""
    with urlopen(url) as resp:
        text = resp.read().decode("utf-8")
    # ASE can read from file-like objects
    from ase.io import read as ase_read

    fh = StringIO(text)
    atoms = ase_read(fh, format="xyz")
    return atoms
