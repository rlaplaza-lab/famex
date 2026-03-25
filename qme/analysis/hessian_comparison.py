"""Utilities for comparing different Hessian computation methods.

This module provides functions to run multiple Hessian methods and compare
their results, useful for validation and method selection.
"""

from __future__ import annotations

import time
from typing import Any, Protocol

import numpy as np
from ase import Atoms

from qme.analysis.utils import has_calculator_property, validate_indices
from qme.utils.logging import get_qme_logger

logger = get_qme_logger(__name__)


class CalculatorProtocol(Protocol):
    """Protocol for calculator objects compatible with ASE Atoms."""

    def get_forces(self, atoms: Atoms | None = None) -> np.ndarray:
        """Compute forces for the given atoms structure."""
        ...

    def get_potential_energy(
        self, atoms: Atoms | None = None, force_consistent: bool = False
    ) -> float:
        """Compute potential energy for the given atoms structure."""
        ...

    def get_hessian(self, atoms: Atoms | None = None, **kwargs: Any) -> np.ndarray:
        """Compute Hessian matrix (optional)."""
        ...


def compare_hessian_methods(
    atoms: Atoms,
    calculator: CalculatorProtocol,
    methods: list[str] | None = None,
    delta: float = 0.01,
    indices: list[int] | None = None,
    verbose: int = 1,
) -> dict[str, Any]:
    """Compare multiple Hessian computation methods.

    Runs all specified methods and returns comparison results including
    Hessians, timings, and quality metrics.

    Parameters
    ----------
    atoms : Atoms
        Structure to compute Hessian for
    calculator : CalculatorProtocol
        Calculator to use
    methods : list[str], optional
        Methods to compare. If None, tests all available methods.
        Options: 'analytical', 'force_fd', 'energy_fd', 'adaptive'
    delta : float, default 0.01
        Displacement for finite differences (Å)
    indices : list[int], optional
        Indices of atoms to include
    verbose : int, default 1
        Verbosity level

    Returns
    -------
    dict[str, Any]
        Comparison results containing:
        - methods: List of tested method names
        - hessians: Dict mapping method name to Hessian matrix
        - timings: Dict mapping method name to computation time (seconds)
        - metrics: Dict mapping method name to quality metrics
        - recommendations: List of recommended methods

    Examples
    --------
    >>> from ase import Atoms
    >>> atoms = Atoms('H2O', positions=[[0, 0, 0], [0.96, 0, 0], [0, 0.96, 0]])
    >>> calc = SomeCalculator()
    >>> results = compare_hessian_methods(atoms, calc, methods=['analytical', 'force_fd'])
    >>> print(f"Methods tested: {results['methods']}")
    >>> print(f"Analytical time: {results['timings']['analytical']:.3f} s")
    """
    # Determine methods to test
    if methods is None:
        methods = []
        # Check for analytical Hessian
        if has_calculator_property(calculator, "hessian"):
            methods.append("analytical")
        # Add FD methods
        methods.extend(["force_fd", "adaptive"])

    hessians = {}
    timings = {}
    metrics = {}

    for method in methods:
        if verbose >= 1:
            logger.info(f"Testing method: {method}")

        try:
            start_time = time.time()
            hessian = _compute_hessian_method(atoms, calculator, method, delta, indices, verbose)
            elapsed_time = time.time() - start_time

            hessians[method] = hessian
            timings[method] = elapsed_time

            # Compute quality metrics
            metric = _compute_quality_metrics(hessian)
            metrics[method] = metric

            if verbose >= 1:
                logger.info(f"  ✓ Completed in {elapsed_time:.3f} s")
                logger.info(f"  RMS value: {metric['rms_value']:.6e} eV/Å²")
                logger.info(f"  Max asymmetry: {metric['max_asymmetry']:.6e} eV/Å²")

        except Exception as e:
            if verbose >= 1:
                logger.error(f"  ✗ Failed: {e}")
            continue

    # Generate recommendations
    recommendations = _generate_recommendations(hessians, timings, metrics, verbose)

    return {
        "methods": list(hessians.keys()),
        "hessians": hessians,
        "timings": timings,
        "metrics": metrics,
        "recommendations": recommendations,
        "atoms": atoms,  # Store for frequency comparison
        "indices": validate_indices(atoms, indices),
    }


def _compute_hessian_method(
    atoms: Atoms,
    calculator: CalculatorProtocol,
    method: str,
    delta: float,
    indices: list[int] | None,
    verbose: int,
) -> np.ndarray:
    """Compute Hessian using specified method."""
    from qme.analysis.frequency import FrequencyAnalysis

    if method == "analytical":
        # Try analytical method
        freq_analysis = FrequencyAnalysis(
            atoms, calculator, delta=delta, indices=indices, verbose=0
        )
        return freq_analysis.calculate_hessian(method="direct")

    elif method == "force_fd":
        # Standard force-based FD
        from qme.analysis.hessian import HessianCalculator

        hessian_calc = HessianCalculator(
            atoms, calculator, delta=delta, method="5point", indices=indices, verbose=0
        )
        return hessian_calc.calculate_numerical_hessian()

    elif method == "energy_fd":
        # Energy-based FD
        from qme.analysis.hessian_energy import EnergyBasedHessianCalculator

        energy_calc = EnergyBasedHessianCalculator(
            atoms, calculator, delta=delta, indices=indices, verbose=0
        )
        return energy_calc.calculate_energy_hessian()

    elif method == "adaptive":
        # Adaptive FD
        from qme.analysis.hessian import HessianCalculator

        hessian_calc = HessianCalculator(
            atoms,
            calculator,
            delta=delta,
            method="5point",
            richardson=True,
            indices=indices,
            verbose=0,
            adaptive_delta=True,
            max_iterations=3,  # Reduce for comparison
        )
        return hessian_calc.calculate_numerical_hessian()

    else:
        msg = f"Unknown method: {method}"
        raise ValueError(msg)


def _compute_quality_metrics(hessian: np.ndarray) -> dict[str, float]:
    """Compute quality metrics for a Hessian matrix.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix

    Returns
    -------
    dict[str, float]
        Quality metrics including RMS value, max asymmetry, etc.
    """
    # RMS value
    rms_value = float(np.sqrt(np.mean(hessian**2)))

    # Asymmetry
    asymmetry = np.abs(hessian - hessian.T)
    max_asymmetry = float(np.max(asymmetry))

    # Condition number
    eigenvalues = np.linalg.eigvals(hessian)
    eigenvalues = eigenvalues[np.isfinite(eigenvalues)]
    eigenvalues_abs = np.abs(eigenvalues)
    eigenvalues_abs = eigenvalues_abs[eigenvalues_abs > 0]
    if len(eigenvalues_abs) > 0:
        condition_number = float(np.max(eigenvalues_abs) / np.min(eigenvalues_abs))
    else:
        condition_number = float("inf")

    # NaN/Inf check
    has_nan = bool(np.any(np.isnan(hessian)))
    has_inf = bool(np.any(np.isinf(hessian)))

    return {
        "rms_value": rms_value,
        "max_asymmetry": max_asymmetry,
        "condition_number": condition_number,
        "has_nan": has_nan,
        "has_inf": has_inf,
    }


def _generate_recommendations(
    hessians: dict[str, np.ndarray],
    timings: dict[str, float],
    metrics: dict[str, dict[str, float]],
    verbose: int,
) -> list[str]:
    """Generate recommendations based on comparison results.

    Parameters
    ----------
    hessians : dict[str, np.ndarray]
        Computed Hessians by method
    timings : dict[str, float]
        Computation times by method
    metrics : dict[str, dict[str, float]]
        Quality metrics by method
    verbose : int
        Verbosity level

    Returns
    -------
    list[str]
        List of recommended methods in order of preference
    """
    if len(hessians) == 0:
        return []

    # Score each method (higher is better)
    scores: dict[str, float] = {}

    for method, metric in metrics.items():
        score = 0.0

        # Penalize NaN/Inf
        if metric["has_nan"] or metric["has_inf"]:
            score = -1000
            scores[method] = score
            continue

        # Reward low asymmetry
        if metric["max_asymmetry"] < 1e-6:
            score += 10
        elif metric["max_asymmetry"] < 1e-4:
            score += 5

        # Reward reasonable condition number
        if metric["condition_number"] < 1e10:
            score += 5

        # Reward speed
        if method in timings:
            # Normalize by fastest method
            min_time = min(timings.values())
            if timings[method] < min_time * 2:
                score += 5

        scores[method] = score

    # Sort by score
    sorted_methods = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Filter out methods with negative scores
    valid_methods = [method for method, score in sorted_methods if score > 0]

    if verbose >= 1 and len(valid_methods) > 0:
        logger.info("\nMethod recommendations (best to worst):")
        for i, method in enumerate(valid_methods, 1):
            logger.info(f"  {i}. {method} (score: {scores[method]:.1f})")

    return valid_methods


class HessianComparisonReport:
    """Formatted report of Hessian method comparison results."""

    def __init__(self, comparison_results: dict[str, Any]) -> None:
        """Initialize comparison report.

        Parameters
        ----------
        comparison_results : dict[str, Any]
            Results from compare_hessian_methods()
        """
        self.methods = comparison_results["methods"]
        self.hessians = comparison_results["hessians"]
        self.timings = comparison_results["timings"]
        self.metrics = comparison_results["metrics"]
        self.recommendations = comparison_results["recommendations"]
        self.atoms = comparison_results.get("atoms")
        self.indices = comparison_results.get(
            "indices", list(range(len(comparison_results.get("atoms", []))))
        )

    def print_summary(self) -> None:
        """Print formatted summary of comparison."""
        print("\n" + "=" * 80)
        print("HESSIAN METHOD COMPARISON SUMMARY")
        print("=" * 80)

        for method in self.methods:
            print(f"\n{method.upper()}:")
            print(f"  Time: {self.timings[method]:.4f} s")
            print(f"  RMS value: {self.metrics[method]['rms_value']:.6e} eV/Å²")
            print(f"  Max asymmetry: {self.metrics[method]['max_asymmetry']:.6e} eV/Å²")
            print(f"  Condition number: {self.metrics[method]['condition_number']:.2e}")
            if self.metrics[method]["has_nan"]:
                print("  ⚠ Contains NaN values")
            if self.metrics[method]["has_inf"]:
                print("  ⚠ Contains Inf values")

        if len(self.recommendations) > 0:
            print("\n" + "=" * 80)
            print("RECOMMENDATIONS:")
            print("=" * 80)
            for i, method in enumerate(self.recommendations, 1):
                print(f"  {i}. {method}")

        print("\n" + "=" * 80)

    def compare_frequencies(self, atoms: Atoms | None = None) -> None:
        """Compare vibrational frequencies from different Hessians.

        Parameters
        ----------
        atoms : Atoms, optional
            Atoms object to use for frequency calculation.
            If None, uses atoms from comparison_results.
        """
        from qme.analysis.normal_modes import diagonalize_mass_weighted_hessian

        if atoms is None:
            atoms = self.atoms

        if atoms is None:
            logger.warning("Cannot compare frequencies: atoms not available")
            return

        frequencies_dict = {}

        for method, hessian in self.hessians.items():
            try:
                frequencies, _ = diagonalize_mass_weighted_hessian(hessian, atoms, self.indices)
                frequencies_dict[method] = frequencies
            except Exception as e:
                logger.warning(f"Failed to compute frequencies for {method}: {e}")
                continue

        if len(frequencies_dict) < 2:
            logger.warning("Need at least 2 methods to compare frequencies")
            return

        print("\n" + "=" * 80)
        print("FREQUENCY COMPARISON")
        print("=" * 80)

        # Use first method as reference
        ref_method = self.methods[0]
        if ref_method not in frequencies_dict:
            ref_method = list(frequencies_dict.keys())[0]

        ref_freqs = frequencies_dict[ref_method]

        for method in self.methods:
            if method not in frequencies_dict or method == ref_method:
                continue

            freqs = frequencies_dict[method]
            diff = freqs - ref_freqs
            rms_diff = np.sqrt(np.mean(diff**2))

            print(f"\n{method} vs {ref_method}:")
            print(f"  RMS difference: {rms_diff:.2f} cm⁻¹")
            print(f"  Max difference: {np.max(np.abs(diff)):.2f} cm⁻¹")


__all__ = [
    "compare_hessian_methods",
    "HessianComparisonReport",
    "CalculatorProtocol",
]
