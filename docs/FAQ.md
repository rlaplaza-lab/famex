# FAQ and Troubleshooting

Common questions about FAMEX usage, installation, and troubleshooting.

## Installation and Setup

### Which backend should I choose?

CLI/`Explorer` default to UMA. For the simplest torch-only install, use `--backend aimnet2`. Full keyword and checkpoint list: [Default models](USER_GUIDE.md#default-models).

### Can I install multiple backends?

Some backends conflict (UMA vs MACE). Use separate conda environments — see [Dependency Conflicts](USER_GUIDE.md#dependency-conflicts) in the User Guide.

### What Python version do I need?

Python 3.11+ required. The **Orb** backend (`pip install famex[orb]`) requires Python 3.12+.

### Backend not available after installation?

Install backend dependencies. UMA: `pip install famex[uma]` or `pip install "fairchem-core>=2.25.0"`. Other backends: see [README](https://github.com/rlaplaza-lab/famex#readme) and [User Guide](USER_GUIDE.md#backend-guide).

## Using FAMEX

### What's the difference between target and strategy?

See [Core Concepts](USER_GUIDE.md#core-concepts). Target (`minima`, `ts`, `path`) is what you want; strategy (`local`, `interpolate`, `neb`, etc.) is how to get there.

### How is the free energy computed?

See [Frequencies and thermochemistry](USER_GUIDE.md#frequencies-and-thermochemistry). The default is RRHO at 298.15 K and 1 atm.

### NEB finished but the path is not converged?

The band did not reach `--fmax` within `--steps`. The images are still written. Raise `--steps` or loosen `--fmax`.

### How do I choose convergence criteria?

See [Convergence](TUTORIALS.md#convergence) in the Tutorials. Quick reference: `--fmax 0.1 --steps 100` (testing), `--fmax 0.05 --steps 1000` (default), `--fmax 0.01 --steps 2000` (high precision).

### What file formats are supported?

Formats supported by the Atomic Simulation Environment (ASE) I/O layer (XYZ, CIF, PDB, VASP, and others). `famex minima`, `famex ts`, and `famex path` also accept a SMILES string when the argument is not an existing file, or `pubchem:NAME`, `cid:`, and a bare numeric CID to download a conformer. A bare name on those commands is SMILES. `--embedder pysmiles` (default, `pip install famex[smiles]`) is distance geometry. `--embedder rdkit` (`pip install famex[rdkit]`) is ETKDGv3.

### Can SMILES embedding replace a force-field or ML minimization?

No. Both embedders only produce a starting geometry — run `famex minima` (or the Python API) to relax. Details: [Structure inputs](USER_GUIDE.md#structure-inputs-file-smiles-pubchem).

### How do I specify charge and spin?

CLI: `--default-charge` and `--default-spin`. Python: `Explorer(..., default_charge=0, default_spin=1)`. Values are written to `atoms.info` when missing. Required for consistent UMA/MACE/Orb results on charged or open-shell systems. For non-`omol` UMA tasks (`omat`, `oc20`, …), use `--default-spin 0`.

### How do I choose a UMA task or fine-tune?

Defaults stay `uma-s-1p2` + `--task-name omol`. Pass `--task-name` for materials/catalysis heads, or `--model-name uma-s-1p2p1-ft-d4rev` / `uma-s-1p2p1-ft-r2scan` for the 1.2.1 fine-tunes. Use `--dftd3-functional r2scan` with the r²SCAN fine-tune; do **not** add D3 to `ft-d4rev`. See [UMA task heads and fine-tunes](USER_GUIDE.md#uma-task-heads-and-fine-tunes).

### How do I use constraints?

`--constraints` accepts semicolon-separated specs, for example:

```bash
famex minima --strategy local molecule.xyz --constraints "fix 0,1,2"
famex minima --strategy local molecule.xyz --constraints "fix 0,1; harmonic_bond 2,3 k=5.0"
```

Supported types include `fix`, `harmonic_position`, `harmonic_bond`, `harmonic_angle`, and `fixinternals_bond` / `fixinternals_angle` / `fixinternals_dihedral`. See the [global options table](USER_GUIDE.md#global-options) in the User Guide.

## Troubleshooting

### Optimization doesn't converge?

Try:
- Increase steps: `--steps 2000`
- Loosen convergence: `--fmax 0.1`
- Change optimizer: `--local-optimizer bfgs`
- Check input structure quality

### Forces too large or unrealistic energies?

Check:
- Backend compatibility with your elements
- Input structure quality (atoms too close?)
- Charge/spin settings
- System size limits

### CUDA out of memory?

Use CPU (`--device cpu`), reduce system size, or use LBFGS optimizer (`--local-optimizer lbfgs`).

### Transition state validation issues?


- Multiple imaginary frequencies: poor TS guess — try interpolation, growing string, or `rfo` / `sella`
- No imaginary frequencies: structure may be a minimum — verify the TS guess
- Use `--freq` or `calculate_frequencies()`; check `ts_analysis["n_imaginary_frequencies"]`
- Animate that mode with `--write-modes --modes imaginary`, or `save_vibrational_modes(..., which="imaginary")`
- Use `--require-ts` to raise if the result is not a first-order saddle

### UMA and MACE both installed but one fails?

They require incompatible `e3nn` versions. Use separate conda environments (see [Dependency Conflicts](USER_GUIDE.md#dependency-conflicts)).

## Performance

### How do I speed up calculations?

Use a GPU (`--device cuda`) when available, or relax `--fmax` / `--steps` while prototyping.

### Which backend is fastest?

Depends on system size, hardware, and task. AIMNet2 is typically fast for small organic molecules; UMA is the default machine learning potential. Profile your workload with [`examples/timing_benchmark.py`](https://github.com/rlaplaza-lab/famex/blob/main/examples/timing_benchmark.py).

## Getting Help

### Where can I get help?

[User Guide](USER_GUIDE.md), [Tutorials](TUTORIALS.md), [examples](https://github.com/rlaplaza-lab/famex/tree/main/examples), or [GitHub Issues](https://github.com/rlaplaza-lab/famex/issues).

### How do I report a bug?

Include `famex --version`, Python version, OS, backend and model name, full error message, and a minimal reproducing example.

### Where can I find examples?

See [`examples/README.md`](https://github.com/rlaplaza-lab/famex/tree/main/examples). Quick start: `python examples/optimize_minimum.py` from the `examples/` directory.

---

*Last updated: October 2026*
