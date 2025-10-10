#!/usr/bin/env python3
"""
ASE Units Integration Example for QME

This example demonstrates how to use ASE units throughout the QME codebase
for consistent unit handling and conversions.

References:
- ASE Units Documentation: https://ase-lib.org/ase/units.html
"""

import numpy as np
from ase import Atoms
# Basic units; Energy units; Temperature and other; Frequency and vibrational; 
# Mass; Length; Force and pressure; Constants
from ase.units import (
    Ang,
    Bohr,
    GPa,
    Hartree,
    Pascal,
    Ry,
    Rydberg,
    _amu,
    _c,
    _e,
    _hbar,
    _hplanck,
    bar,
    eV,
    fs,
    invcm,
    kB,
    kcal,
    kg,
    kJ,
    nm,
    second,
)

from qme.analysis.frequency import FrequencyAnalysis
from qme.core.validation import MAX_REASONABLE_FORCE, MIN_ATOM_DISTANCE
from qme.potentials.mock_potential import MIN_BOND_CUTOFF, MockCalculator


def demonstrate_basic_units():
    """Demonstrate basic ASE unit conversions."""
    print("=== Basic ASE Units ===")

    # Energy conversions
    energy_ev = 1.0 * eV
    energy_hartree = energy_ev / Hartree
    energy_kcal = energy_ev / (kcal / 1000)  # kcal/mol to kcal
    energy_kj = energy_ev / (kJ / 1000)  # kJ/mol to kJ

    print(f"1.0 eV = {energy_hartree:.6f} Hartree")
    print(f"1.0 eV = {energy_kcal:.6f} kcal")
    print(f"1.0 eV = {energy_kj:.6f} kJ")

    # Length conversions
    distance_ang = 1.0 * Ang
    distance_bohr = distance_ang / Bohr
    distance_nm = distance_ang / nm
    # pm not available in ASE units, use manual conversion
    distance_pm = distance_ang * 100  # 1 Å = 100 pm

    print(f"1.0 Å = {distance_bohr:.6f} Bohr")
    print(f"1.0 Å = {distance_nm:.6f} nm")
    print(f"1.0 Å = {distance_pm:.6f} pm")

    # Force conversions
    force_ev_ang = 0.05 * eV / Ang
    force_hartree_bohr = force_ev_ang * Hartree / Bohr
    force_nn = force_ev_ang * 1.602176634e-10  # eV/Å to nN (manual conversion)

    print(f"0.05 eV/Å = {force_hartree_bohr:.6f} Hartree/Bohr")
    print(f"0.05 eV/Å = {force_nn:.6f} nN")


def demonstrate_frequency_units():
    """Demonstrate frequency unit conversions."""
    print("\n=== Frequency Units ===")

    # Frequency in cm^-1
    freq_cm1 = 1000.0  # cm^-1

    # Convert to different units
    freq_ev = freq_cm1 * invcm  # Convert to eV
    freq_mev = freq_ev * 1000  # Convert to meV
    freq_thz = freq_cm1 * _c * 100 / 1e12  # Convert to THz

    print(f"1000 cm^-1 = {freq_ev:.6f} eV")
    print(f"1000 cm^-1 = {freq_mev:.6f} meV")
    print(f"1000 cm^-1 = {freq_thz:.6f} THz")

    # Zero-point energy calculation
    zpe_cm1 = 0.5 * freq_cm1 * invcm  # ZPE in eV
    print(f"ZPE = {zpe_cm1:.6f} eV")


def demonstrate_qme_units():
    """Demonstrate QME's use of ASE units."""
    print("\n=== QME Units Integration ===")

    # QME validation constants (now using ASE units)
    print(f"MIN_ATOM_DISTANCE = {MIN_ATOM_DISTANCE} Å")
    print(f"MAX_REASONABLE_FORCE = {MAX_REASONABLE_FORCE} eV/Å")
    print(f"MIN_BOND_CUTOFF = {MIN_BOND_CUTOFF} Å")

    # Create a simple molecule
    atoms = Atoms("H2O", positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.24, 0.93, 0.0]])

    # Attach mock calculator
    atoms.calc = MockCalculator()

    # Get energy and forces
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    print(f"Water molecule energy: {energy:.6f} eV")
    print(f"Max force: {np.max(np.abs(forces)):.6f} eV/Å")

    # Demonstrate frequency analysis with ASE units
    try:
        freq_analysis = FrequencyAnalysis(atoms, atoms.calc, delta=0.01 * Ang)
        frequencies = freq_analysis.get_frequencies()

        print("Vibrational frequencies:")
        for i, freq in enumerate(frequencies):
            print(f"  Mode {i+1}: {freq:.2f} cm^-1")

        # Get frequencies in different units
        freq_ev = freq_analysis.get_frequencies(unit="meV") * 1000  # Convert to eV
        print(f"Lowest frequency: {freq_ev[0]:.6f} eV")

    except Exception as e:
        print(f"Frequency analysis failed: {e}")


def demonstrate_thermodynamic_units():
    """Demonstrate thermodynamic unit conversions."""
    print("\n=== Thermodynamic Units ===")

    # Temperature conversions
    temp_k = 298.15  # Kelvin (temperature is dimensionless in ASE units)
    kt_ev = kB * temp_k  # kT in eV

    print(f"Room temperature: {temp_k} K")
    print(f"kT at room temperature: {kt_ev:.6f} eV")

    # Pressure conversions
    pressure_pa = 101325 * Pascal
    pressure_bar = pressure_pa / bar
    pressure_gpa = pressure_pa / GPa

    print(f"Standard pressure: {pressure_pa} Pa")
    print(f"Standard pressure: {pressure_bar:.6f} bar")
    print(f"Standard pressure: {pressure_gpa:.6f} GPa")

    # Time conversions
    time_fs = 100.0 * fs
    time_s = time_fs / second

    print(f"100 fs = {time_s:.2e} s")


def demonstrate_constants():
    """Demonstrate physical constants from ASE units."""
    print("\n=== Physical Constants ===")

    print(f"Planck constant: {_hplanck:.2e} eV·s")
    print(f"Speed of light: {_c:.2e} m/s")
    print(f"Elementary charge: {_e:.2e} C")
    print(f"Atomic mass unit: {_amu:.2e} kg")
    print(f"Reduced Planck constant: {_hbar:.2e} eV·s")

    # Boltzmann constant
    print(f"Boltzmann constant: {kB:.2e} eV/K")


def demonstrate_convergence_criteria():
    """Demonstrate proper convergence criteria with ASE units."""
    print("\n=== Convergence Criteria ===")

    # Common convergence criteria
    criteria = {
        "Loose": 0.1 * eV / Ang,
        "Default": 0.05 * eV / Ang,
        "Tight": 0.01 * eV / Ang,
        "Very Tight": 0.001 * eV / Ang,
    }

    print("Force convergence criteria:")
    for name, criterion in criteria.items():
        print(f"  {name}: {criterion} eV/Å")

    # Energy convergence (typically 10x tighter than force)
    energy_criteria = {name: crit / 10 for name, crit in criteria.items()}

    print("\nEnergy convergence criteria:")
    for name, criterion in energy_criteria.items():
        print(f"  {name}: {criterion:.6f} eV")


def main():
    """Run all demonstrations."""
    print("ASE Units Integration Example for QME")
    print("=" * 50)

    demonstrate_basic_units()
    demonstrate_frequency_units()
    demonstrate_qme_units()
    demonstrate_thermodynamic_units()
    demonstrate_constants()
    demonstrate_convergence_criteria()

    print("\n" + "=" * 50)
    print("Example completed successfully!")
    print("\nKey takeaways:")
    print("1. Always use ASE units for consistency")
    print("2. Import specific units: from ase.units import eV, Ang, Bohr")
    print("3. Use unit multiplication for conversions")
    print("4. QME integrates ASE units throughout the codebase")
    print("5. Reference: https://ase-lib.org/ase/units.html")


if __name__ == "__main__":
    main()
