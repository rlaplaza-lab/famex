#!/usr/bin/env python3
"""Minimize a structure, then evaluate gas-phase thermochemistry at 298.15 K.

Equivalent CLI::

    famex minima --strategy local example_files/A_C_A_B_A_C_reactant.xyz --freq --temperature 298.15
"""

import math
from pathlib import Path
from typing import Any

from ase.io import read

import famex
from famex.analysis import ThermodynamicProperties

HERE = Path(__file__).resolve().parent
REACTANT = HERE / "example_files" / "A_C_A_B_A_C_reactant.xyz"


def _require_frequencies(frequency_analysis: Any) -> list[float]:
    frequencies = (
        frequency_analysis.get("frequencies") if isinstance(frequency_analysis, dict) else None
    )
    if not isinstance(frequencies, list) or not frequencies:
        raise SystemExit("Frequency analysis did not return finite frequencies")
    if not all(isinstance(value, int | float) and math.isfinite(value) for value in frequencies):
        raise SystemExit("Frequency analysis did not return finite frequencies")
    return [float(value) for value in frequencies]


def main() -> None:
    atoms = read(REACTANT)
    explorer = famex.Explorer(atoms=atoms, target="minima", strategy="local")
    result = explorer.run(fmax=0.05, steps=500, calculate_frequencies=True)
    optimized = result["optimized_atoms"]
    frequencies = _require_frequencies(result.get("frequency_analysis"))
    energy = float(optimized.get_potential_energy())
    converged = bool(result.get("converged"))

    thermo = ThermodynamicProperties(
        frequencies,
        optimized,
        temperature=298.15,
        method="rrho",
        solvent="none",
    )
    props = thermo.calculate_complete_thermodynamics(energy=energy)
    gibbs = float(props["gibbs_free_energy"])
    print(f"Electronic energy: {energy:.6f} eV")
    print(f"Gibbs free energy: {gibbs:.6f} eV")
    print(f"Converged: {converged}")
    if not converged or not math.isfinite(energy) or not math.isfinite(gibbs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
