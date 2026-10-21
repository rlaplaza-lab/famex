#!/usr/bin/env python3
"""Minimize from a SMILES string and from a PubChem compound ID.

Equivalent CLI::

    famex minima --strategy local "CCO" --freq
    famex minima --strategy local pubchem:aspirin --freq
    famex minima --strategy local cid:2244 --freq

Requires an embedder for SMILES (and for PubChem when no 3D conformer exists)::

    pip install famex[smiles]
"""

import math
from pathlib import Path
from typing import Any

import famex

HERE = Path(__file__).resolve().parent
SMILES_OUTPUT = HERE / "ethanol_from_smiles.xyz"
PUBCHEM_OUTPUT = HERE / "aspirin_from_pubchem.xyz"


def _require_frequencies(frequency_analysis: Any) -> None:
    frequencies = (
        frequency_analysis.get("frequencies") if isinstance(frequency_analysis, dict) else None
    )
    usable = (
        isinstance(frequencies, list)
        and len(frequencies) > 0
        and all(isinstance(value, int | float) and math.isfinite(value) for value in frequencies)
    )
    if not usable:
        raise SystemExit("Frequency analysis did not return finite frequencies")


def _minimize_and_save(explorer: famex.Explorer, output: Path) -> float:
    result = explorer.run(fmax=0.05, steps=500, calculate_frequencies=True)
    optimized = result["optimized_atoms"]
    explorer.save_structure(optimized, str(output))
    energy = float(optimized.get_potential_energy())
    _require_frequencies(result.get("frequency_analysis"))
    if not result.get("converged") or not math.isfinite(energy):
        raise SystemExit(1)
    return energy


def main() -> None:
    # SMILES to embedded guess to local minimum
    smiles_explorer = famex.Explorer.from_smiles(
        "CCO",
        target="minima",
        strategy="local",
    )
    smiles_energy = _minimize_and_save(smiles_explorer, SMILES_OUTPUT)
    print(f"SMILES CCO energy: {smiles_energy:.6f} eV")
    print(f"Wrote {SMILES_OUTPUT}")

    # PubChem name or CID (uses a 3D conformer when available)
    pubchem_explorer = famex.Explorer.from_pubchem(
        "aspirin",
        target="minima",
        strategy="local",
    )
    pubchem_energy = _minimize_and_save(pubchem_explorer, PUBCHEM_OUTPUT)
    print(f"PubChem aspirin energy: {pubchem_energy:.6f} eV")
    print(f"Wrote {PUBCHEM_OUTPUT}")


if __name__ == "__main__":
    main()
