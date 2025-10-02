# QME: Quick Mechanistic Exploration

Quick mechanistic exploration using machine learning potentials (MLPs) for molecular geometry optimization and transition state searches.

QME provides a simple command-line interface for molecular optimization tasks using state-of-the-art neural network potentials including UMA, AIMNet2, MACE, and SO3LR.

## Installation

### Basic Installation

```bash
pip install qme
```

### Per-Backend Installation (Recommended)

Due to dependency conflicts between ML packages, install backends individually:

```bash
# UMA backend (Meta AI, default)
pip install qme[uma]

# AIMNet2 backend (native PyTorch)
pip install qme[aimnet2]

# MACE backend (foundation models)
pip install qme[mace]

# SO3LR backend (requires separate installation)
pip install qme[so3lr]
pip install so3lr  # Install from PyPI or source

# TorchSim acceleration (Python 3.11+ only)
pip install qme[torchsim]
```

### Combined Installation (May Have Conflicts)

⚠️ **Warning**: Installing multiple backends together may cause dependency conflicts, particularly between MACE and UMA due to incompatible e3nn versions.

```bash
# Legacy combined installation (not recommended)
pip install qme[ml]

# Multiple backends (may conflict)
pip install qme[uma,mace]  # Known to conflict due to e3nn versions
```

### Development Installation

```bash
git clone https://github.com/rlaplaza-lab/qme.git
cd qme
pip install -e .[dev]
```

## Quick Start

QME provides three main commands:

### 1. Structure Optimization (`qme opt`)

**Local optimization** - optimize a single structure:
```bash
qme opt molecule.xyz
```

**Two-ended optimization** - interpolate between reactant and product, optimize minima:
```bash
qme opt reactant.xyz --product product.xyz
```

### 2. Transition State Optimization (`qme tsopt`)

**Local TS optimization** - optimize a transition state guess:
```bash
qme tsopt ts_guess.xyz
```

**Two-ended TS search** - find transition state between reactant and product:
```bash
qme tsopt reactant.xyz --product product.xyz
```

**NEB path optimization** - find minimum energy path:
```bash
qme tsopt reactant.xyz --product product.xyz --mode neb
```

### 3. Cache Management (`qme cache`)

```bash
qme cache info    # Show cached models
qme cache clear   # Clear model cache
```

## Command Options

### Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--backend` | ML backend: `uma`, `aimnet2`, `mace`, `so3lr`, `mock` | `uma` |
| `--model-name` | Specific model name for backend | Backend default |
| `--device` | Compute device: `cpu`, `cuda` | Auto-detect |
| `--fmax` | Force convergence threshold (eV/Å) | `0.05` |
| `--steps` | Maximum optimization steps | `1000` |
| `--optimizer` | Optimizer: `sella`, `lbfgs`, `bfgs`, `fire` | `sella` |
| `--output` | Output file path | Auto-generated |

### Two-Ended Options

| Option | Description | Default |
|--------|-------------|---------|
| `--npoints` | Number of interpolation points | `11` |
| `--interp` | Interpolation method: `linear`, `geodesic` | `geodesic` |

### NEB Options

| Option | Description | Default |
|--------|-------------|---------|
| `--mode` | Mode: `interpolate`, `neb` | `interpolate` |
| `--spring-constant` | NEB spring constant (eV/Å²) | `5.0` |

## Examples

### Basic Structure Optimization

```bash
# Optimize with default UMA backend
qme opt molecule.xyz

# Use AIMNet2 backend with custom convergence
qme opt molecule.xyz --backend aimnet2 --fmax 0.01

# Use MACE backend on GPU
qme opt molecule.xyz --backend mace --device cuda
```

### Transition State Searches

```bash
# Optimize TS guess
qme tsopt ts_guess.xyz --fmax 0.03

# Two-ended TS search with geodesic interpolation
qme tsopt reactant.xyz --product product.xyz --npoints 15

# NEB calculation with custom spring constant
qme tsopt reactant.xyz --product product.xyz --mode neb --spring-constant 2.0
```

### Advanced Usage

```bash
# Apply constraints during optimization
qme opt molecule.xyz --constraints "fix 0,1; harmonic_bond 2,3 k=5.0"

# Set molecular charge and spin
qme opt ion.xyz --default-charge 1 --default-spin 2

# Use custom optimizer settings
qme opt molecule.xyz --optimizer lbfgs --optimizer-kw maxstep=0.1
```

## Supported Backends

| Backend | Description | Installation | Conflicts |
|---------|-------------|--------------|-----------|
| `uma` | Universal Materials Accelerator (default) | `pip install qme[uma]` | ⚠️ MACE (e3nn) |
| `aimnet2` | Native PyTorch implementation | `pip install qme[aimnet2]` | ✅ None |
| `mace` | Foundation models for molecules/materials | `pip install qme[mace]` | ⚠️ UMA (e3nn) |
| `so3lr` | SO(3) invariant neural networks | `pip install qme[so3lr]` + separate install | ✅ None |
| `mock` | Harmonic oscillator for testing | Built-in | ✅ None |

### Dependency Conflicts

- **UMA vs MACE**: Both depend on `e3nn` but require incompatible versions:
  - UMA (fairchem-core) requires `e3nn>=0.5`
  - MACE requires `e3nn==0.4.4`
- **PyTorch versions**: Different backends may require different PyTorch versions:
  - UMA may have issues with PyTorch 2.6+ due to `weights_only=True` default
  - MACE works well with recent PyTorch versions
- **Solution**: Install backends in separate environments or use individual installations

### TorchSim Acceleration (Optional)

For significant performance improvements on supported models:

```bash
pip install torch-sim-atomistic  # Requires Python 3.11+
```

Use TorchSim backends:
```bash
qme opt molecule.xyz --backend torchsim_mace --device cuda
qme opt molecule.xyz --backend torchsim_uma --device cuda
```

## File Formats

QME supports all ASE-compatible formats:
- **XYZ**: Standard molecular coordinates
- **CIF**: Crystallographic information files
- **PDB**: Protein data bank format
- **POSCAR**: VASP format
- And many more via ASE

## Requirements

- Python 3.10 or higher
- For TorchSim acceleration: Python 3.11+

Core dependencies (automatically installed):
- ASE ≥ 3.22.0
- NumPy ≥ 1.20.0
- Click ≥ 8.0.0
- SELLA ≥ 2.0.0

## Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_cli.py                    # CLI functionality
pytest tests/test_backends_min_ts.py        # Backend tests
pytest tests/test_comprehensive_optimization.py  # Optimization tests
```

## Performance Tips

1. **Use GPU acceleration** when available: `--device cuda`
2. **Enable TorchSim** for supported models (Python 3.11+)
3. **Adjust convergence criteria** based on your accuracy needs
4. **Use appropriate number of NEB images** (7-15 typically sufficient)
5. **Cache models** are automatically managed for faster startup

## Troubleshooting

### Common Issues

**Backend not available**: Install required dependencies with per-backend installation (e.g., `pip install qme[uma]`)

**Dependency conflicts**: Use individual backend installations instead of `qme[ml]`:
```bash
# Instead of: pip install qme[ml]  # May cause conflicts
# Use: pip install qme[uma]        # Install only UMA
```

**e3nn version conflicts**: UMA and MACE cannot be installed together. Use separate environments:
```bash
# Environment 1: UMA only
conda create -n qme-uma python=3.12
conda activate qme-uma
pip install qme[uma]

# Environment 2: MACE only  
conda create -n qme-mace python=3.12
conda activate qme-mace
pip install qme[mace]
```

**CUDA out of memory**: Use `--device cpu` or reduce system size

**Convergence issues**: Try different optimizers (`--optimizer lbfgs`) or looser convergence (`--fmax 0.1`)

**TorchSim not available**: Requires Python 3.11+, install with `pip install qme[torchsim]`

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@software{qme2025,
  title={QME: Quick Mechanistic Exploration},
  author={QME Development Team},
  year={2025},
  url={https://github.com/rlaplaza-lab/qme}
}
```