# Getting Started with QME

Welcome to QME (Quick Mechanistic Exploration)! This guide will help you install QME and run your first molecular optimization.

## Installation

### Prerequisites

- Python 3.10 or higher (Python 3.11+ required for TorchSim backends)
- pip package manager

### Quick Installation

```bash
# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install QME with a backend
pip install qme-ml[aimnet2]  # Recommended for beginners
```

### Backend Selection

QME supports multiple machine learning backends. Choose one based on your needs:

| Backend | Best For | Installation |
|---------|----------|--------------|
| `aimnet2` | General use, no conflicts | `pip install qme-ml[aimnet2]` |
| `mace` | High accuracy | `pip install qme-ml[mace]` |
| `uma` | Materials science | `pip install qme-ml[uma]` |
| `orb` | Universal forcefield | `pip install qme-ml[orb]` |
| `torchsim` | Maximum performance | `pip install qme-ml[torchsim]` |

> **Important**: Some backends have dependency conflicts (e.g., UMA vs MACE). Use separate environments or choose one backend per environment.

### Alternative Installations

```bash
# Development installation
git clone https://github.com/rlaplaza-lab/qme.git
cd qme
pip install -e .[dev,aimnet2]

# Minimal installation (mock backend only)
pip install qme-ml
```

## Your First Optimization

### 1. Create a Test Structure

Create a simple water molecule:

```bash
# Create water.xyz
cat > water.xyz << EOF
3
Water molecule
O    0.000000    0.000000    0.117283
H    0.000000    0.758602   -0.469132
H    0.000000   -0.758602   -0.469132
EOF
```

### 2. Run Your First Optimization

**Command Line Interface:**
```bash
# Basic optimization
qme minima --strategy local water.xyz

# With specific backend
qme minima --strategy local water.xyz --backend aimnet2

# With custom settings
qme minima --strategy local water.xyz --fmax 0.01 --steps 500
```

**Python API:**
```python
import qme

# Create Explorer and optimize
explorer = qme.Explorer.from_file("water.xyz", backend="aimnet2")
result = explorer.run(target="minima", strategy="local")

# Save results
explorer.save_structure(result['optimized_atoms'], "water_optimized.xyz")
print(f"Final energy: {result['final_energy']:.6f} eV")
```

### 3. Understanding the Output

QME creates output files with descriptive names:
- `water.opt.xyz` - Optimized structure
- `water.opt.log` - Optimization log (if verbose)

The result dictionary contains:
- `optimized_atoms`: The optimized structure
- `final_energy`: Final energy in eV
- `converged`: Whether optimization converged
- `steps_taken`: Number of optimization steps

## Understanding QME's Interface

### Target/Strategy System

QME uses a semantic interface with two key concepts:

- **Target**: What you want to obtain (`minima`, `ts`, `path`)
- **Strategy**: How to get there (`local`, `interpolate`, `neb`, `cineb`, `irc`, `growing_string`)

### Target/Strategy Matrix

| Target | Strategy | Description |
|--------|----------|-------------|
| `minima` | `local` | Direct local optimization |
| `minima` | `interpolate` | Minima from interpolated path |
| `ts` | `local` | Local TS search |
| `ts` | `interpolate` | TS guess from interpolation |
| `ts` | `growing_string` | Growing string method (DE-GSM) |
| `path` | `neb` | NEB path optimization |
| `path` | `cineb` | CI-NEB path optimization |
| `path` | `irc` | IRC path from transition state |
| `path` | `interpolate` | Generate path only (no optimization) |

### Command Line Structure

QME's CLI is organized into three main commands:

```bash
# Minima optimization (outputs single structure)
qme minima --strategy local molecule.xyz                           # Local optimization
qme minima --strategy interpolate reactant.xyz --product product.xyz     # Two-ended minima search

# Transition state optimization (outputs single TS)
qme ts --strategy local ts_guess.xyz                   # Local TS optimization
qme ts --strategy interpolate reactant.xyz --product product.xyz # TS via interpolation
qme ts --strategy growing_string reactant.xyz --product product.xyz         # Growing string method

# Reaction path optimization (outputs trajectories)
qme path --strategy interpolate r.xyz --product p.xyz               # Raw interpolation
qme path --strategy neb r.xyz --product p.xyz                       # NEB path
qme path --strategy cineb r.xyz --product p.xyz                     # CI-NEB path
qme path --strategy irc ts.xyz                            # IRC from transition state
```

## Common Workflows

### Local Minima Optimization

```bash
# Command line
qme minima --strategy local molecule.xyz --backend aimnet2 --fmax 0.01

# Python
explorer = qme.Explorer.from_file("molecule.xyz", backend="aimnet2")
result = explorer.run(target="minima", strategy="local", fmax=0.01)
```

### Transition State Search

```bash
# Command line - local TS optimization
qme ts --strategy local ts_guess.xyz --backend aimnet2

# Command line - TS from interpolation
qme ts --strategy interpolate reactant.xyz --product product.xyz --npoints 15

# Python
explorer = qme.Explorer([reactant, product], target="ts", strategy="interpolate")
result = explorer.run(npoints=15)
```

### Reaction Path Optimization

```bash
# Command line - NEB path
qme path --strategy neb reactant.xyz --product product.xyz --npoints 11

# Command line - IRC from TS
qme path --strategy irc ts.xyz --direction both

# Python
explorer = qme.Explorer([reactant, product], target="path", strategy="neb")
result = explorer.run(npoints=11)
```

## Backend Selection Guide

### For Beginners
Start with **AIMNet2** - fast, reliable, no conflicts:
```bash
pip install qme-ml[aimnet2]
```

### For Production
- **Materials**: Use `uma` backend
- **Molecules**: Use `mace` backend
- **Universal**: Use `orb` backend

### For Maximum Performance
Use **TorchSim** with GPU acceleration:
```bash
pip install qme-ml[torchsim]
qme minima --strategy local molecule.xyz --backend torchsim_mace --device cuda
```

### For Testing
Use **mock** backend (always available):
```bash
qme minima --strategy local molecule.xyz --backend mock
```

## Troubleshooting

### Common Installation Issues

**Dependency Conflicts:**
```bash
# Solution: Use separate environments
conda create -n qme-uma python=3.12
conda activate qme-uma
pip install qme-ml[uma]

conda create -n qme-mace python=3.12
conda activate qme-mace
pip install qme-ml[mace]
```

**Backend Not Available:**
```bash
# Check what's installed
qme --help

# Install specific backend
pip install qme-ml[backend_name]
```

### Common Runtime Issues

**Optimization Not Converging:**
- Increase `--steps` parameter
- Loosen `--fmax` convergence criteria
- Try different optimizer: `--optimizer lbfgs`

**CUDA Out of Memory:**
- Use CPU: `--device cpu`
- Reduce system size
- Use smaller model

**Unrealistic Energies:**
- Verify backend compatibility with your system
- Check charge/spin multiplicity settings
- Try different backend

### Getting Help

```bash
# Command help
qme --help
qme minima --help
qme ts --help
qme path --help
```

## Next Steps

Now that you've completed your first optimization, explore more advanced features:

1. **[Basic Optimization Tutorial](tutorials/basic_optimization.md)** - Learn about different optimizers and settings
2. **[Transition States Tutorial](tutorials/transition_states.md)** - Master TS search techniques
3. **[Backend Guide](user_guide/backends.md)** - Explore different ML backends
4. **[CLI Reference](user_guide/cli_reference.md)** - Complete command reference

## Examples

The `examples/` directory contains many working examples:

```bash
# Run examples
cd examples/
python cli_demo.py
python growing_string_demo.py
python irc_demo.py
```

---

*Last updated: January 2025*
