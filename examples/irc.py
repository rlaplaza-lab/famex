#!/usr/bin/env python3
"""Intrinsic reaction coordinate from a transition-state guess.

Equivalent CLI::

    famex path --strategy irc example_files/A_C_A_B_A_C_ts.xyz --direction both
"""

from pathlib import Path

from ase import Atoms
from ase.io import read

import famex

HERE = Path(__file__).resolve().parent
TS = HERE / "example_files" / "A_C_A_B_A_C_ts.xyz"
OUTPUT = HERE / "irc_path.xyz"


def _stored_energy(atoms: Atoms) -> float:
    energy = atoms.info.get("potential_energy")
    if not isinstance(energy, int | float):
        raise SystemExit("IRC frame is missing a potential energy")
    return float(energy)


def main() -> None:
    ts = read(TS)
    explorer = famex.Explorer(atoms=ts, target="path", strategy="irc")
    result = explorer.run(steps=50, step_size=0.1, fmax=0.05, direction="both")
    trajectory = result["trajectory"]
    if not isinstance(trajectory, list) or len(trajectory) <= 2:
        raise SystemExit(f"IRC trajectory too short: {trajectory!r}")
    backward = result.get("backward_path") or []
    ts_frame = trajectory[len(backward)]
    ts_energy = _stored_energy(ts_frame)
    end_energies = (_stored_energy(trajectory[0]), _stored_energy(trajectory[-1]))
    explorer.save_trajectory(trajectory, str(OUTPUT))
    print(f"IRC images: {len(trajectory)}")
    print(f"TS energy: {ts_energy:.6f} eV")
    print(f"Endpoint energies: {end_energies[0]:.6f} {end_energies[1]:.6f} eV")
    print(f"Wrote {OUTPUT}")
    if end_energies[0] >= ts_energy or end_energies[1] >= ts_energy:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
