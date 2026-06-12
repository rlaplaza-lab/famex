# FAMEX FAQ and Troubleshooting

Common questions about FAMEX usage, installation, and troubleshooting.

## Table of Contents

1. [Installation and Setup](#installation-and-setup)
2. [Using FAMEX](#using-famex)
3. [Troubleshooting](#troubleshooting)
4. [Performance](#performance)
5. [Getting Help](#getting-help)

## Installation and Setup

### Q: Which backend should I choose?

**A:** FAMEX defaults to **UMA** (`uma-s-1p2`) via `fairchem-core>=2.21.0`. For the simplest install with no `e3nn` conflicts, use **AIMNet2** (`pip install torch`) and pass `--backend aimnet2`. See the [backend table](USER_GUIDE.md#backend-guide) in the User Guide.

### Q: Can I install multiple backends?

**A:** Some backends conflict (UMA vs MACE). Use separate environments:

```bash
conda create -n famex-uma python=3.12 && conda activate famex-uma && pip install famex[uma]
conda create -n famex-mace python=3.12 && conda activate famex-mace && pip install famex mace-torch
```

### Q: What Python version do I need?

**A:** Python 3.10+ required.

### Q: Backend not available after installation?

**A:** Install backend dependencies. UMA: `pip install famex[uma]` or `pip install "fairchem-core>=2.21.0"`. Other backends: see [README](../README.md) and [User Guide](USER_GUIDE.md#backend-guide).

## Using FAMEX

### Q: What's the difference between target and strategy?

**A:** See [Core Concepts](USER_GUIDE.md#core-concepts). Target (`minima`, `ts`, `path`) is what you want; strategy (`local`, `interpolate`, `neb`, etc.) is how to get there.

### Q: How do I choose convergence criteria?

**A:**
- Quick testing: `--fmax 0.1 --steps 100`
- Standard: `--fmax 0.05 --steps 1000` (default)
- High precision: `--fmax 0.01 --steps 2000`

### Q: What file formats are supported?

**A:** All ASE-compatible formats (XYZ, CIF, PDB, VASP, and others supported by ASE I/O).

### Q: How do I specify charge and spin?

**A:** CLI: `--default-charge` and `--default-spin`. Python: `Explorer(..., default_charge=0, default_spin=1)`. Values are written to `atoms.info` when missing. Required for consistent UMA/MACE/Orb results on charged or open-shell systems.

### Q: How do I use constraints?

**A:** `--constraints` accepts semicolon-separated specs, for example:

```bash
famex minima --strategy local molecule.xyz --constraints "fix 0,1,2"
famex minima --strategy local molecule.xyz --constraints "fix 0,1; harmonic_bond 2,3 k=5.0"
```

Supported types include `fix`, `harmonic_position`, `harmonic_bond`, `harmonic_angle`, and `fixinternals_bond` / `fixinternals_angle` / `fixinternals_dihedral`. See the User Guide global options table.

## Troubleshooting

### Q: Optimization doesn't converge?

**A:** Try:
- Increase steps: `--steps 2000`
- Loosen convergence: `--fmax 0.1`
- Change optimizer: `--local-optimizer bfgs`
- Check input structure quality

### Q: Forces too large or unrealistic energies?

**A:** Check:
- Backend compatibility with your elements
- Input structure quality (atoms too close?)
- Charge/spin settings
- System size limits

### Q: CUDA out of memory?

**A:** Use CPU (`--device cpu`), reduce system size, or use LBFGS optimizer (`--local-optimizer lbfgs`).

### Q: Transition state validation issues?

**A:**
- Multiple imaginary frequencies: poor TS guess — try interpolation, growing string, or `rfo` / `sella`
- No imaginary frequencies: structure may be a minimum — verify the TS guess
- Use `--freq` or `calculate_frequencies()`; check `ts_analysis["n_imaginary_frequencies"]`

### Q: UMA and MACE both installed but one fails?

**A:** They require incompatible `e3nn` versions. Use separate conda environments (see [Dependency Conflicts](USER_GUIDE.md#dependency-conflicts)).

## Performance

### Q: How do I speed up calculations?

**A:** Use GPU (`--device cuda`) when available, or relax `--fmax` / `--steps` while prototyping.

### Q: Which backend is fastest?

**A:** Depends on system size, hardware, and task. AIMNet2 is typically fast for small organic molecules; UMA is the default general-purpose MLIP. Profile your workload with [`examples/timing_benchmark.py`](../examples/timing_benchmark.py).

## Getting Help

### Q: Where can I get help?

**A:** [User Guide](USER_GUIDE.md), [Tutorials](TUTORIALS.md), [examples](../examples/README.md), or [GitHub Issues](https://github.com/rlaplaza-lab/famex/issues).

### Q: How do I report a bug?

**A:** Include `famex --version`, Python version, OS, backend and model name, full error message, and a minimal reproducing example.

### Q: Where can I find examples?

**A:** See [`examples/README.md`](../examples/README.md). Quick start: `python examples/cli_demo.py` from the repo root.

---

*Last updated: June 2026*
