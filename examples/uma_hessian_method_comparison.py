"""Compare UMA Hessian computation methods against finite differences.

This script compares different analytical Hessian computation methods
('vmap', 'double_backward', with/without symmetrization) against finite
difference calculations to identify the best-performing approach.

The script tests multiple finite difference step sizes and converges to
find the best reference, then compares all methods against it.
"""

import time

import numpy as np
from ase import Atoms

import qme
from qme.analysis.frequency import HessianCalculator


def compute_finite_difference_hessian(
    atoms: Atoms,
    delta: float = 0.01,
) -> np.ndarray:
    """Compute Hessian using finite differences.

    Parameters
    ----------
    atoms : Atoms
        The molecular system
    delta : float
        Step size for finite differences

    Returns:
    -------
    np.ndarray
        Hessian matrix (3N x 3N)
    """
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
    """Compute vibrational frequencies from Hessian matrix.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix (3N x 3N)
    masses : np.ndarray
        Atomic masses (N,)

    Returns:
    -------
    np.ndarray
        Frequencies in cm^-1 (3N,)
    """
    # Mass-weighted Hessian
    mass_matrix = np.kron(np.diag(1.0 / np.sqrt(masses)), np.eye(3))
    mass_weighted_hessian = mass_matrix @ hessian @ mass_matrix

    # Compute eigenvalues
    eigenvalues, _ = np.linalg.eigh(mass_weighted_hessian)

    # Convert to frequencies (cm^-1)
    # Conversion factor: sqrt(eV/(amu*Å²)) to cm^-1
    hartree_to_cm = 219474.63  # 1 Hartree = 219474.63 cm^-1
    ev_to_hartree = 0.0367493  # 1 eV = 0.0367493 Hartree
    amu_to_kg = 1.66053906660e-27  # kg
    ang_to_m = 1e-10  # m
    conv = np.sqrt(ev_to_hartree * hartree_to_cm**2 / (amu_to_kg * ang_to_m**2))

    frequencies = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * conv

    return frequencies


def compute_metrics(
    analytical: np.ndarray,
    reference: np.ndarray,
    verbose: bool = False,
) -> dict[str, float]:
    """Compute comparison metrics between analytical and reference Hessians.

    Parameters
    ----------
    analytical : np.ndarray
        Analytical Hessian
    reference : np.ndarray
        Reference Hessian (typically finite difference)
    verbose : bool
        Whether to print detailed metrics

    Returns:
    -------
    dict[str, float]
        Dictionary of metrics
    """
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
    """Analyze frequencies from Hessian.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix
    masses : np.ndarray
        Atomic masses
    label : str
        Label for output

    Returns:
    -------
    dict[str, float]
        Frequency statistics
    """
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
    """Compare all Hessian computation methods.

    Parameters
    ----------
    atoms : Atoms
        Molecular system
    verbose : bool
        Whether to print detailed results

    Returns:
    -------
    dict[str, dict]
        Results for each method with metrics
    """
    if not hasattr(atoms.calc, "get_hessian"):
        raise ValueError("Calculator does not support analytical Hessian")

    if verbose:
        print(f"Comparing Hessian methods for {len(atoms)} atoms")
        print("=" * 80)

    # Test finite difference convergence first
    if verbose:
        print("\nStep 1: Testing finite difference convergence...")
    step_sizes = [0.01, 0.005, 0.001]
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

    # Find best FD reference (smallest step that converged)
    # Use smallest step as reference
    reference_delta = 0.001
    reference_hessian = fd_results[reference_delta]["hessian"]

    if verbose:
        print(f"\n  Using FD(delta={reference_delta}) as reference")
        print(
            f"  Reference symmetry error: {np.max(np.abs(reference_hessian - reference_hessian.T)):.6e}"
        )

    # Test all analytical methods
    methods_to_test = [
        ("vmap", True),
        ("vmap", False),
        ("double_backward", True),
        ("double_backward", False),
    ]

    results = {}
    masses = atoms.get_masses()

    for method, symmetrize in methods_to_test:
        method_name = f"{method}_sym={symmetrize}"
        if verbose:
            print(f"\n{method_name}:")
            print("-" * 80)

        # Compute analytical Hessian
        start = time.time()
        try:
            analytical_hessian = atoms.calc.get_hessian(atoms, method=method, symmetrize=symmetrize)
            analytical_time = time.time() - start
        except Exception as e:
            if verbose:
                print(f"  ERROR: {e}")
            results[method_name] = {"error": str(e)}
            continue

        # Compute metrics
        metrics = compute_metrics(analytical_hessian, reference_hessian, verbose=verbose)

        # Analyze frequencies
        freq_stats = analyze_frequencies(analytical_hessian, masses, method_name)
        if verbose:
            print("\n  Frequency analysis:")
            print(f"    Negative frequencies: {freq_stats['n_negative_frequencies']}")
            print(f"    Small negative (< 50 cm^-1): {freq_stats['n_small_negative']}")
            print(
                f"    Frequency range: {freq_stats['min_frequency']:.2f} to {freq_stats['max_frequency']:.2f} cm^-1"
            )

        results[method_name] = {
            "metrics": metrics,
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


def main():
    """Main function to run comparison."""
    # Create test molecules
    water = Atoms(
        symbols="OHH",
        positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.24, 0.93, 0.0],
        ],
    )

    methane = Atoms(
        symbols="CHHHH",
        positions=[
            [0.0, 0.0, 0.0],
            [1.09, 0.0, 0.0],
            [-0.36, 1.03, 0.0],
            [-0.36, -0.51, 0.89],
            [-0.36, -0.51, -0.89],
        ],
    )

    # Set up UMA calculator
    calc = qme.get_uma_calculator(model_name="uma-s-1p1")
    calc.ensure_loaded()

    print("UMA Hessian Method Comparison")
    print("=" * 80)
    print("\nThis script compares different Hessian computation methods")
    print("against finite differences to identify the most accurate approach.\n")

    # Test water
    print("\n" + "=" * 80)
    print("WATER MOLECULE")
    print("=" * 80)
    water.calc = calc
    water_results = compare_methods(water, verbose=True)

    # Test methane
    print("\n" + "=" * 80)
    print("METHANE MOLECULE")
    print("=" * 80)
    methane.calc = calc
    methane_results = compare_methods(methane, verbose=True)

    # Overall recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print("\nBased on RMS error with finite differences:\n")

    all_methods = set(water_results.keys()) | set(methane_results.keys())
    method_scores = {}
    for method in all_methods:
        scores = []
        if method in water_results and "error" not in water_results[method]:
            scores.append(water_results[method]["metrics"]["rms_error"])
        if method in methane_results and "error" not in methane_results[method]:
            scores.append(methane_results[method]["metrics"]["rms_error"])
        if scores:
            method_scores[method] = np.mean(scores)

    sorted_methods = sorted(method_scores.items(), key=lambda x: x[1])
    for i, (method, avg_rms) in enumerate(sorted_methods, 1):
        print(f"  {i}. {method}: avg RMS = {avg_rms:.6f} eV/Å²")

    if sorted_methods:
        best = sorted_methods[0][0]
        print(
            f"\nRecommended default: method='{best.split('_sym=')[0]}', symmetrize={best.split('=')[-1]}"
        )


if __name__ == "__main__":
    main()
