#!/usr/bin/env python3
"""Demonstration of enhanced thermochemistry capabilities in QME.

This example shows the new thermochemistry features including quasi-harmonic
corrections, complete statistical thermodynamics, solvation corrections, and
symmetry handling.

Usage:
    python thermochemistry_demo.py [--backends BACKEND1,BACKEND2,...]

Features:
    - Quasi-harmonic corrections (Grimme and Truhlar methods)
    - Complete statistical thermodynamics (translational, rotational, electronic)
    - Solvation corrections
    - Symmetry handling
    - Based on GoodVibes-inspired thermochemistry implementation
"""

import os

# Disable ASE GUI to prevent popup windows
os.environ["DISPLAY"] = ""
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
from ase import Atoms

from qme.analysis import (
    QuasiHarmonicHandler,
    SolvationHandler,
    StatisticalThermodynamics,
    SymmetryHandler,
    ThermodynamicProperties,
)


def print_thermochemistry_results(results: dict) -> None:
    """Pretty print thermochemistry results."""
    print("\n" + "=" * 80)
    print("THERMOCHEMISTRY RESULTS")
    print("=" * 80)

    print(f"\nTemperature: {results['temperature']:.2f} K")
    print(f"Method: {results['method']}")

    print("\n--- ENERGETIC CONTRIBUTIONS (eV) ---")
    print(f"Electronic energy:    {results['energy']:12.6f}")
    print(f"Zero-point energy:    {results['zpe']:12.6f}")
    print(f"Translational H:      {results['enthalpy_trans']:12.6f}")
    print(f"Rotational H:         {results['enthalpy_rot']:12.6f}")
    print(f"Vibrational H:        {results['enthalpy_vib']:12.6f}")
    print(f"TOTAL ENTHALPY:       {results['enthalpy_total']:12.6f}")

    print("\n--- ENTROPIC CONTRIBUTIONS (eV/K) ---")
    print(f"Translational S:      {results['entropy_trans']:12.6f}")
    print(f"Rotational S:         {results['entropy_rot']:12.6f}")
    print(f"Vibrational S:        {results['entropy_vib']:12.6f}")
    print(f"Electronic S:         {results['entropy_elec']:12.6f}")
    print(f"TOTAL ENTROPY:        {results['entropy_total']:12.6f}")

    print("\n--- GIBBS FREE ENERGY (eV) ---")
    print(f"G = H - TS:           {results['gibbs_free_energy']:12.6f}")

    # Convert to kcal/mol for more familiar units
    eV_to_kcal = 23.06035
    print("\n" + "=" * 80)
    print("EQUIVALENT VALUES IN kcal/mol")
    print("=" * 80)
    print(f"ZPE:                  {results['zpe'] * eV_to_kcal:12.2f}")
    print(f"Delta H:              {results['enthalpy_total'] * eV_to_kcal:12.2f}")
    print(
        f"T * Delta S:          {results['entropy_total'] * results['temperature'] * eV_to_kcal:12.2f}"
    )
    print(f"Delta G:              {results['gibbs_free_energy'] * eV_to_kcal:12.2f}")
    print("=" * 80 + "\n")


def main() -> int:
    """Run thermochemistry demonstrations."""
    print("=" * 80)
    print("QME Enhanced Thermochemistry Demo")
    print("=" * 80)

    # Create a simple water molecule for demonstration
    # This would normally come from an actual quantum chemistry calculation
    atoms = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 0.96], [0.82, 0, -0.26]])

    # Example frequencies (in cm^-1) - these would come from frequency analysis
    # Approximate values for water at B3LYP/6-31G* level
    frequencies = np.array([1627.2, 3832.2, 3942.5])

    print("\nExample molecule: H2O")
    print(f"Frequencies: {frequencies} cm^-1")

    # ========================================================================
    # Example 1: Basic RRHO thermodynamics (backward compatible)
    # ========================================================================
    print("\n" + "#" * 80)
    print("# Example 1: Basic RRHO Thermodynamics (Backward Compatible)")
    print("#" * 80)

    # Create ThermodynamicProperties object with default RRHO method
    thermo_rrho = ThermodynamicProperties(
        frequencies,
        atoms,
        temperature=298.15,
        method="rrho",
        multiplicity=1,
        solvent="none",  # Gas phase
    )

    results_rrho = thermo_rrho.calculate_complete_thermodynamics(energy=0.0)
    print_thermochemistry_results(results_rrho)

    # ========================================================================
    # Example 2: Grimme's quasi-harmonic corrections
    # ========================================================================
    print("\n" + "#" * 80)
    print("# Example 2: Grimme's Quasi-Harmonic Corrections")
    print("#" * 80)

    thermo_grimme = ThermodynamicProperties(
        frequencies,
        atoms,
        temperature=298.15,
        method="grimme",
        freq_cutoff=100.0,  # cm^-1
        multiplicity=1,
        solvent="none",
    )

    results_grimme = thermo_grimme.calculate_complete_thermodynamics(energy=0.0)
    print_thermochemistry_results(results_grimme)

    # Compare vibrational entropies
    print("\nQuasi-harmonic effect on vibrational entropy:")
    print(f"RRHO S_vib:    {results_rrho['entropy_vib']:10.6f} eV/K")
    print(f"Grimme S_vib:  {results_grimme['entropy_vib']:10.6f} eV/K")
    print(
        f"Difference:    {results_grimme['entropy_vib'] - results_rrho['entropy_vib']:10.6f} eV/K"
    )

    # ========================================================================
    # Example 3: Truhlar's quasi-harmonic corrections
    # ========================================================================
    print("\n" + "#" * 80)
    print("# Example 3: Truhlar's Quasi-Harmonic Corrections")
    print("#" * 80)

    thermo_truhlar = ThermodynamicProperties(
        frequencies,
        atoms,
        temperature=298.15,
        method="truhlar",
        freq_cutoff=100.0,
        multiplicity=1,
        solvent="none",
    )

    results_truhlar = thermo_truhlar.calculate_complete_thermodynamics(energy=0.0)
    print_thermochemistry_results(results_truhlar)

    # ========================================================================
    # Example 4: Solution-phase thermodynamics
    # ========================================================================
    print("\n" + "#" * 80)
    print("# Example 4: Solution-Phase Thermodynamics (in water)")
    print("#" * 80)

    thermo_solution = ThermodynamicProperties(
        frequencies,
        atoms,
        temperature=298.15,
        method="grimme",
        freq_cutoff=100.0,
        multiplicity=1,
        solvent="H2O",
        concentration=1.0,  # 1 M
    )

    results_solution = thermo_solution.calculate_complete_thermodynamics(energy=0.0)
    print_thermochemistry_results(results_solution)

    print("\nSolvent effects on translational entropy:")
    print(f"Gas phase S_trans:      {results_rrho['entropy_trans']:10.6f} eV/K")
    print(f"Solution S_trans:       {results_solution['entropy_trans']:10.6f} eV/K")
    print(
        f"Difference:             {results_solution['entropy_trans'] - results_rrho['entropy_trans']:10.6f} eV/K"
    )

    # ========================================================================
    # Example 5: Using the individual modules
    # ========================================================================
    print("\n" + "#" * 80)
    print("# Example 5: Using Individual Thermodynamics Modules")
    print("#" * 80)

    print("\n--- Quasi-Harmonic Handler ---")
    qh_handler = QuasiHarmonicHandler(method="grimme", freq_cutoff=100.0)
    entropy, _ = qh_handler.vibrational_entropy(frequencies, 298.15)
    print(f"Total vibrational entropy: {entropy * 1e-3:.4f} J/(mol·K)")

    print("\n--- Solvation Handler ---")
    solvation = SolvationHandler(solvent="toluene", concentration=0.1)
    print(f"Free space: {solvation.free_space_ml_per_l:.2f} mL/L")
    print(f"Effective concentration: {solvation.effective_concentration():.3f} M")

    print("\n--- Symmetry Handler ---")
    # For C2v water molecule
    symmetry = SymmetryHandler(point_group="C2v", warn_on_assumptions=False)
    print(f"Point group: {symmetry.point_group}")
    print(f"Symmetry number: {symmetry.symmetry_number}")
    print(f"Rotational symmetry (linear): {symmetry.get_rotational_symmetry_number(linear=False)}")

    print("\n--- Statistical Thermodynamics ---")
    stat_thermo = StatisticalThermodynamics(
        atoms,
        rotational_constants=np.array([0.9435, 0.9336, 0.5368]),  # GHz for H2O
        symmetry_number=2,
        multiplicity=1,
    )
    rot_entropy = stat_thermo.rotational_entropy(298.15)
    print(f"Rotational entropy: {rot_entropy * 1e-3:.4f} J/(mol·K)")

    print("\n" + "=" * 80)
    print("Demo complete!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
