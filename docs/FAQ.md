# QME FAQ and Troubleshooting

Common questions about QME usage, installation, and troubleshooting.

## Table of Contents

1. [Installation and Setup](#installation-and-setup)
2. [Using QME](#using-qme)
3. [Troubleshooting](#troubleshooting)
4. [Performance](#performance)
5. [Getting Help](#getting-help)

## Installation and Setup

### Q: Which backend should I choose?

**A:** See [README](../README.md) for backend recommendations. Start with AIMNet2 for beginners (`pip install torch torch-cluster`).

### Q: Can I install multiple backends?

**A:** Some backends conflict (UMA vs MACE). Use separate environments:

```bash
conda create -n qme-uma python=3.12 && conda activate qme-uma && pip install qme-ml fairchem-core
conda create -n qme-mace python=3.12 && conda activate qme-mace && pip install qme-ml mace-torch
```

### Q: What Python version do I need?

**A:** Python 3.10+ for most backends, 3.11+ for TorchSim.

### Q: Backend not available after installation?

**A:** Install backend dependencies. See [README](../README.md) for installation commands.

## Using QME

### Q: What's the difference between target and strategy?

**A:** See [Core Concepts](USER_GUIDE.md#core-concepts) in the User Guide. Target (`minima`, `ts`, `path`) is what you want, strategy (`local`, `interpolate`, `neb`, etc.) is how to get there.

### Q: How do I choose convergence criteria?

**A:**
- Quick testing: `--fmax 0.1 --steps 100`
- Standard: `--fmax 0.05 --steps 1000` (default)
- High precision: `--fmax 0.01 --steps 2000`

### Q: What file formats are supported?

**A:** All ASE-compatible formats (XYZ, CIF, PDB, VASP).

### Q: How do I specify charge and spin?

**A:** Use `--default-charge` and `--default-spin` options or Python API parameters.

### Q: How do I use constraints?

**A:** Use `--constraints` option: `qme minima --strategy local molecule.xyz --constraints "fix 0,1,2"`

## Troubleshooting

### Q: Optimization doesn't converge?

**A:** Try:
- Increase steps: `--steps 2000`
- Loosen convergence: `--fmax 0.1`
- Change optimizer: `--optimizer bfgs`
- Check input structure quality

### Q: Forces too large or unrealistic energies?

**A:** Check:
- Backend compatibility with your elements
- Input structure quality (atoms too close?)
- Charge/spin settings
- System size limits

### Q: CUDA out of memory?

**A:** Use CPU (`--device cpu`), reduce system size, or use LBFGS optimizer (`--optimizer lbfgs`).

### Q: Transition state validation issues?

**A:**
- Multiple imaginary frequencies: Poor TS guess - try interpolation or different optimizer
- No imaginary frequencies: Structure might be a minimum - verify TS guess

## Performance

### Q: How do I speed up calculations?

**A:** Use GPU (`--device cuda`), TorchSim backends, or reduce convergence criteria for testing.

### Q: Which backend is fastest?

**A:** TorchSim backends (GPU) > AIMNet2 > UMA > MACE > Mock (testing only)

## Getting Help

### Q: Where can I get help?

**A:** Check [User Guide](USER_GUIDE.md), [Tutorials](TUTORIALS.md), or GitHub Issues.

### Q: How do I report a bug?

**A:** Include QME version (`qme --version`), Python version, OS, backend, error message, and minimal reproducing example.

### Q: Where can I find examples?

**A:** See the `examples/` directory. Run: `python examples/cli_demo.py`

---

*Last updated: January 2025*
