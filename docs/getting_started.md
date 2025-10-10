# Getting Started with QME

This guide will help you install QME and run your first molecular optimization.

## Installation

### Prerequisites

- Python 3.10 or higher
- For TorchSim acceleration: Python 3.11+

### Recommended Installation (Per-Backend)

Due to dependency conflicts between ML packages, we recommend installing backends individually:

```bash
# UMA backend (Meta AI, default)
pip install qme-ml[uma]

# AIMNet2 backend (native PyTorch)
pip install qme-ml[aimnet2]

# MACE backend (foundation models)
pip install qme-ml[mace]

# SO3LR backend
pip install qme-ml[so3lr]
pip install so3lr  # Install from PyPI or source

# TorchSim acceleration (Python 3.11+ only)
pip install qme-ml[torchsim]
```

### Development Installation

```bash
git clone https://github.com/rlaplaza-lab/qme.git
cd qme
pip install -e .[dev]
```

### Verify Installation

```bash
qme --help
```

## Your First Optimization

### 1. Basic Structure Optimization

Create a simple molecule file (`benzene.xyz`):
```
12
Benzene molecule
C    0.000000    1.396000    0.000000
C    1.209000    0.698000    0.000000
C    1.209000   -0.698000    0.000000
C    0.000000   -1.396000    0.000000
C   -1.209000   -0.698000    0.000000
C   -1.209000    0.698000    0.000000
H    0.000000    2.486000    0.000000
H    2.153000    1.243000    0.000000
H    2.153000   -1.243000    0.000000
H    0.000000   -2.486000    0.000000
H   -2.153000   -1.243000    0.000000
H   -2.153000    1.243000    0.000000
```

Optimize the structure:
```bash
qme opt benzene.xyz
```

This will:
- Use the default UMA backend
- Optimize to a force convergence of 0.05 eV/Å (using ASE units)
- Save the optimized structure to `benzene_opt_uma.xyz`

### 2. Try Different Backends

```bash
# Use AIMNet2 backend
qme opt benzene.xyz --backend aimnet2

# Use MACE backend with GPU acceleration
qme opt benzene.xyz --backend mace --device cuda

# Use mock backend for testing (no ML dependencies required)
qme opt benzene.xyz --backend mock
```

### 3. Customize Optimization

```bash
# Tighter convergence
qme opt benzene.xyz --fmax 0.01

# More optimization steps
qme opt benzene.xyz --steps 2000

# Different optimizer
qme opt benzene.xyz --optimizer lbfgs
```

## Understanding the Output

QME will create output files and display progress:

```
🔧 QME Optimization Starting
Backend: uma (UMA)
Device: cpu
Optimizer: sella
Convergence: 0.050 eV/Å

📁 Loading structure from: benzene.xyz
🧮 Attaching calculator...
⚡ Starting optimization...

Step    Energy (eV)    Max Force (eV/Å)
0       -45.231        0.142
10      -45.298        0.087
20      -45.321        0.034
23      -45.322        0.048  ✅ Converged!

💾 Saved optimized structure to: benzene_opt_uma.xyz
⏱️  Total time: 12.3 seconds
```

### Output Files

- `benzene_opt_uma.xyz`: Optimized structure
- Optimization trajectory (if requested with `--save-trajectory`)
- Log files with detailed optimization information

## Next Steps

Now that you have QME working, explore more features:

1. **[Transition State Searches](user_guide/transition_states.md)** - Find transition states between reactants and products
2. **[Python API](user_guide/python_api.md)** - Use QME programmatically in your Python scripts
3. **[Advanced Features](user_guide/advanced.md)** - Constraints, batch processing, and custom workflows
4. **[Tutorials](tutorials/index.md)** - Step-by-step guides for common tasks

## Common Issues

### Backend Not Available
```
Error: Backend 'uma' not available. Install with: pip install qme-ml[uma]
```
**Solution**: Install the required backend dependencies.

### Dependency Conflicts
```
ERROR: pip's dependency resolver does not currently have the necessary information to solve this problem.
```
**Solution**: Use individual backend installations instead of combined installations.

### GPU Not Available
```
Warning: CUDA not available, falling back to CPU
```
**Solution**: Install PyTorch with CUDA support or use `--device cpu` explicitly.

For more troubleshooting help, see the [Troubleshooting Guide](reference/troubleshooting.md).

## Getting Help

- **Documentation**: Browse the full [User Guide](user_guide/index.md)
- **Examples**: Check out the [Tutorials](tutorials/index.md)
- **Issues**: Report problems on [GitHub Issues](https://github.com/rlaplaza-lab/qme/issues)
- **Questions**: Use [GitHub Discussions](https://github.com/rlaplaza-lab/qme/discussions)
