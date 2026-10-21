#!/usr/bin/env python3
"""Minimize a structure, then evaluate gas-phase thermochemistry at 298.15 K.

Equivalent CLI::

    famex minima --strategy local example_files/A_C_A_B_A_C_reactant.xyz --freq --temperature 298.15
"""

from pathlib import Path

from ase.io import read

import famex
from famex.analysis import ThermodynamicProperties

HERE = Path(__file__).resolve().parent
REACTANT = HERE / "example_files" / "A_C_A_B_A_C_reactant.xyz"


def main() -> None:
    atoms = read(REACTANT)
    explorer = famex.Explorer(atoms=atoms, target="minima", strategy="local")
    result = explorer.run(fmax=0.05, steps=500, calculate_frequencies=True)
    optimized = result["optimized_atoms"]
    freq = result["frequency_analysis"]
    frequencies = freq["frequencies"]
    energy = optimized.get_potential_energy()

    thermo = ThermodynamicProperties(
        frequencies,
        optimized,
        temperature=298.15,
        method="rrho",
        solvent="none",
    )
    props = thermo.calculate_complete_thermodynamics(energy=energy)
    print(f"Electronic energy: {energy:.6f} eV")
    print(f"Gibbs free energy: {props['gibbs_free_energy']:.6f} eV")


if __name__ == "__main__":
    main()
