# Frequently Asked Questions

Common questions about QME usage, installation, and troubleshooting.

## Installation and Setup

### Q: Which backend should I choose for beginners?

**A:** Start with **AIMNet2** - it's fast, reliable, and has no dependency conflicts:

```bash
pip install qme-ml[aimnet2]
```

### Q: Can I install multiple backends in the same environment?

**A:** Some backends conflict with each other. Use separate environments for conflicting backends:

```bash
# Environment 1: UMA
conda create -n qme-uma python=3.12
conda activate qme-uma
pip install qme-ml[uma]

# Environment 2: MACE  
conda create -n qme-mace python=3.12
conda activate qme-mace
pip install qme-ml[mace]
```

### Q: Why am I getting "Backend 'xyz' not available" errors?

**A:** Install the backend dependencies:

```bash
pip install qme-ml[backend_name]
```

Available backends: `uma`, `aimnet2`, `mace`, `orb`, `so3lr`, `torchsim`

### Q: Do I need Python 3.11+ for TorchSim?

**A:** Yes, TorchSim requires Python 3.11 or higher. For other backends, Python 3.10+ is sufficient.

## Using QME

### Q: What's the difference between target and strategy?

**A:** QME uses a semantic interface:
- **Target**: What you want (`minima`, `ts`, `path`)
- **Strategy**: How to get there (`local`, `interpolate`, `neb`, `cineb`, `irc`)

```python
# Find a minimum using local optimization
explorer.run(target="minima", strategy="local")

# Find a transition state using interpolation
explorer.run(target="ts", strategy="interpolate")
```

### Q: When should I use which strategy?

**A:** Quick guide:

**Minima optimization:**
- `local`: Direct optimization of a single structure
- `interpolate`: Find minima from an interpolated path

**Transition state search:**
- `local`: You have a good TS guess
- `interpolate`: You have reactant and product structures

**Reaction paths:**
- `neb`: Standard NEB for reaction paths
- `cineb`: Climbing image NEB for better TS location
- `irc`: Intrinsic reaction coordinate from a TS

### Q: How do I choose convergence criteria?

**A:** Use these guidelines:

```bash
# Quick testing
qme minima --strategy local molecule.xyz --fmax 0.1 --steps 100

# Standard use (default)
qme minima --strategy local molecule.xyz --fmax 0.05 --steps 1000

# High precision
qme minima --strategy local molecule.xyz --fmax 0.01 --steps 2000
```

### Q: What file formats are supported?

**A:** QME supports all ASE-compatible formats:
- **XYZ**: Most common for molecular systems
- **CIF**: Crystallographic information files
- **PDB**: Protein data bank format
- **VASP**: POSCAR/CONTCAR files

## Troubleshooting

### Q: Optimization doesn't converge - what should I do?

**A:** Try these solutions:

1. **Increase steps:**
   ```bash
   qme minima --strategy local molecule.xyz --steps 2000
   ```

2. **Loosen convergence:**
   ```bash
   qme minima --strategy local molecule.xyz --fmax 0.1
   ```

3. **Change optimizer:**
   ```bash
   qme minima --strategy local molecule.xyz --optimizer bfgs
   ```

### Q: Forces are too large - what's wrong?

**A:** Usually indicates:
- Poor initial geometry (atoms too close)
- Wrong backend for system
- System too large for backend

### Q: I'm getting unrealistic energies - what's happening?

**A:** Check:
- Backend compatibility with your elements
- System size limits
- Input structure quality

### Q: How do I debug optimization failures?

**A:** Use these approaches:

1. **Enable verbose output:**
   ```bash
   qme minima --strategy local molecule.xyz --verbose
   ```

2. **Use mock backend for testing:**
   ```bash
   qme minima --strategy local molecule.xyz --backend mock
   ```

## Performance

### Q: How do I speed up calculations?

**A:** Try these strategies:

1. **Use GPU backends:**
   ```bash
   qme minima --strategy local molecule.xyz --backend torchsim_cpu  # or torchsim_gpu
   ```

2. **Reduce convergence criteria for testing:**
   ```bash
   qme minima --strategy local molecule.xyz --fmax 0.1 --steps 100
   ```

## Getting Help

### Q: Where can I get help with QME?

**A:** Several resources are available:

1. **Documentation**: Check the [User Guide](user_guide/index.md) and [Tutorials](tutorials/index.md)
2. **GitHub Issues**: Report bugs and request features
3. **Troubleshooting Guide**: See [Troubleshooting](troubleshooting.md) for common issues

### Q: How do I report a bug?

**A:** When reporting bugs, include:

- QME version: `qme --version`
- Python version: `python --version`
- Operating system
- Backend used
- Complete error message
- Minimal example that reproduces the issue

---

*Last updated: January 2025*