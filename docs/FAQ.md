# QME FAQ and Troubleshooting

Common questions about QME usage, installation, and troubleshooting.

## Table of Contents

1. [Installation and Setup](#installation-and-setup)
2. [Using QME](#using-qme)
3. [Troubleshooting](#troubleshooting)
4. [Performance](#performance)
5. [Getting Help](#getting-help)

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

### Q: What are the dependency conflicts between backends?

**A:** The main conflict is between UMA and MACE due to incompatible `e3nn` versions:

- **UMA** (fairchem-core) requires `e3nn>=0.5`
- **MACE** requires `e3nn==0.4.4`

**Solution**: Use separate environments or choose one backend per environment.

### Q: How do I install QME for development?

**A:** Clone the repository and install in development mode:

```bash
git clone https://github.com/rlaplaza-lab/qme.git
cd qme
pip install -e .[dev,aimnet2]
```

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

### Q: How do I specify charge and spin multiplicity?

**A:** Use the global options:

```bash
# Command line
qme minima --strategy local molecule.xyz --default-charge 1 --default-spin 2

# Python API
explorer = qme.Explorer.from_file("molecule.xyz", default_charge=1, default_spin=2)
```

### Q: How do I use constraints during optimization?

**A:** Use the `--constraints` option with constraint specifications:

```bash
# Fix specific atoms
qme minima --strategy local molecule.xyz --constraints "fix 0,1,2"

# Harmonic bond constraint
qme minima --strategy local molecule.xyz --constraints "harmonic_bond 0,1 k=5.0"

# Multiple constraints
qme minima --strategy local molecule.xyz --constraints "fix 0,1; harmonic_bond 2,3 k=5.0"
```

### Q: How do I validate a transition state?

**A:** Use the `--freq` flag or calculate frequencies manually:

```bash
# Automatic frequency analysis
qme ts --strategy local ts_guess.xyz --freq

# Manual frequency calculation
qme minima --strategy local ts_structure.xyz --freq
```

A valid transition state should have exactly one imaginary frequency.

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

4. **Check input structure quality**

### Q: Forces are too large - what's wrong?

**A:** Usually indicates:
- Poor initial geometry (atoms too close)
- Wrong backend for system
- System too large for backend
- Unrealistic input structure

### Q: I'm getting unrealistic energies - what's happening?

**A:** Check:
- Backend compatibility with your elements
- System size limits
- Input structure quality
- Charge and spin settings

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

3. **Use dry run to check strategy selection:**
   ```bash
   qme minima --strategy local molecule.xyz --dry-run
   ```

### Q: CUDA out of memory errors

**A:** Try these solutions:

1. **Use CPU instead:**
   ```bash
   qme minima --strategy local molecule.xyz --device cpu
   ```

2. **Reduce system size**

3. **Use smaller model**

4. **Use LBFGS optimizer (less memory):**
   ```bash
   qme minima --strategy local molecule.xyz --optimizer lbfgs
   ```

### Q: Backend not available after installation

**A:** Check installation and dependencies:

```bash
# Check what's installed
qme --help

# Reinstall backend
pip install qme-ml[backend_name]

# Check for conflicts
pip list | grep -E "(torch|e3nn|fairchem)"
```

### Q: Import errors with specific backends

**A:** Common issues and solutions:

**UMA/MACE conflicts:**
```bash
# Use separate environments
conda create -n qme-uma python=3.12
conda activate qme-uma
pip install qme-ml[uma]
```

**TorchSim requires Python 3.11+:**
```bash
# Check Python version
python --version

# Use appropriate Python version
conda create -n qme-torchsim python=3.11
conda activate qme-torchsim
pip install qme-ml[torchsim]
```

### Q: Optimization is very slow

**A:** Try these optimizations:

1. **Use GPU acceleration:**
   ```bash
   qme minima --strategy local molecule.xyz --device cuda
   ```

2. **Use TorchSim backends:**
   ```bash
   qme minima --strategy local molecule.xyz --backend torchsim_mace --device cuda
   ```

3. **Reduce convergence criteria for testing:**
   ```bash
   qme minima --strategy local molecule.xyz --fmax 0.1 --steps 100
   ```

4. **Use appropriate optimizer:**
   ```bash
   qme minima --strategy local molecule.xyz --optimizer lbfgs  # For large systems
   ```

### Q: Transition state has multiple imaginary frequencies

**A:** This indicates the TS guess is poor:

1. **Try interpolation method:**
   ```bash
   qme ts --strategy interpolate reactant.xyz --product product.xyz
   ```

2. **Improve initial TS guess**

3. **Use different optimizer:**
   ```bash
   qme ts --strategy local ts_guess.xyz --optimizer trust-krylov-ts
   ```

### Q: Transition state has no imaginary frequencies

**A:** The structure might be a minimum, not a TS:

1. **Check if you're optimizing the right structure**

2. **Verify your TS guess**

3. **Try interpolation method to find better TS guess**

## Performance

### Q: How do I speed up calculations?

**A:** Try these strategies:

1. **Use GPU backends:**
   ```bash
   qme minima --strategy local molecule.xyz --backend torchsim_mace --device cuda
   ```

2. **Reduce convergence criteria for testing:**
   ```bash
   qme minima --strategy local molecule.xyz --fmax 0.1 --steps 100
   ```

3. **Use appropriate optimizer for system size:**
   ```bash
   qme minima --strategy local molecule.xyz --optimizer lbfgs  # For large systems
   ```

4. **Use TorchSim for maximum performance:**
   ```bash
   pip install qme-ml[torchsim]
   qme minima --strategy local molecule.xyz --backend torchsim_mace --device cuda
   ```

### Q: Which backend is fastest?

**A:** Performance ranking (approximate):

1. **TorchSim backends** (with GPU): 5-20x speedup
2. **AIMNet2**: Fast inference, good for molecules
3. **UMA**: Good balance of speed and accuracy
4. **MACE**: High accuracy but slower
5. **Mock**: Fastest but not chemically meaningful

### Q: How do I optimize memory usage?

**A:** Use these strategies:

1. **Use LBFGS instead of BFGS:**
   ```bash
   qme minima --strategy local molecule.xyz --optimizer lbfgs
   ```

2. **Use CPU instead of GPU for large systems**

3. **Reduce system size when possible**

4. **Use mock backend for testing**

## Getting Help

### Q: Where can I get help with QME?

**A:** Several resources are available:

1. **Documentation**: Check the [User Guide](USER_GUIDE.md) and [Tutorials](TUTORIALS.md)
2. **GitHub Issues**: Report bugs and request features
3. **Examples**: Check the `examples/` directory for working code

### Q: How do I report a bug?

**A:** When reporting bugs, include:

- QME version: `qme --version`
- Python version: `python --version`
- Operating system
- Backend used
- Complete error message
- Minimal example that reproduces the issue

### Q: How do I request a new feature?

**A:** Create a GitHub issue with:

- Clear description of the feature
- Use case and motivation
- Proposed implementation (if you have ideas)
- Any relevant examples

### Q: Where can I find examples?

**A:** Check the `examples/` directory:

```bash
# Run examples
cd examples/
python cli_demo.py
python growing_string_demo.py
python irc_demo.py
```

### Q: How do I contribute to QME?

**A:** See the [Contributing Guidelines](https://github.com/rlaplaza-lab/qme/blob/main/CONTRIBUTING.md) for:

- Development setup
- Code style guidelines
- Testing requirements
- Pull request process

---

*Last updated: January 2025*
