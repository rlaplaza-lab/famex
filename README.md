# QME: Quick Mechanistic Exploration

**QME** provides a unified interface for molecular geometry optimization using machine learning potentials. It supports minima optimization, transition state searches, and reaction path calculations through both a command-line interface and Python API.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Quick Start

### Installation

```bash
# Install QME from source
git clone https://github.com/rlaplaza-lab/qme.git
cd qme
pip install -e .

# Or install directly from GitHub
pip install git+https://github.com/rlaplaza-lab/qme.git

# Install a backend separately
pip install torch torch-cluster  # AIMNet2 (recommended)
pip install fairchem-core         # UMA
pip install mace-torch            # MACE
pip install orb-models            # Orb
pip install tblite                # TBLite
```

> **Note**: Python version depends on backend choice. TorchSim requires Python 3.11+, others work with Python 3.10+.

### Backend Selection

| Backend | Installation | Notes |
|---------|--------------|-------|
| `aimnet2` | `pip install torch torch-cluster` | Built-in, no conflicts |
| `uma` | `pip install fairchem-core` | Materials science |
| `mace` | `pip install mace-torch` | Conflicts with UMA |
| `torchsim` | `pip install torch-sim-atomistic` | Requires Python 3.11+ |

> **Note**: MACE and UMA have dependency conflicts. Use separate environments.

### Your First Optimization

**Command Line:**
```bash
# Create a test structure
echo "3
Water
O 0.0 0.0 0.0
H 0.0 0.0 1.0
H 0.0 1.0 0.0" > water.xyz

# Optimize it
qme minima --strategy local water.xyz --backend aimnet2
```

**Python API:**
```python
import qme

# Load and optimize
explorer = qme.Explorer.from_file("water.xyz", backend="aimnet2", target="minima", strategy="local")
result = explorer.run(fmax=0.05, steps=1000)

# Save results
explorer.save_structure(result["optimized_atoms"], "water_optimized.xyz")
print(f"Final energy: {result['final_energy']:.6f} eV")
```

## Key Features

- 🧪 **Multiple ML Backends**: UMA, AIMNet2, MACE, SO3LR, TorchSim
- ⚡ **GPU Acceleration**: CUDA support for compatible backends
- 🎯 **Semantic Interface**: Target/strategy system (minima, ts, path)
- 🔄 **Advanced Methods**: NEB, CI-NEB, IRC, growing string method
- 📊 **Frequency Analysis**: Vibrational frequencies and thermodynamics
- 🖥️ **Dual Interface**: Both command-line and Python API
- 📁 **Format Support**: XYZ, CIF, PDB, and more via ASE

## Documentation

- **[User Guide](docs/USER_GUIDE.md)** - Complete reference for CLI, Python API, and backends
- **[Tutorials](docs/TUTORIALS.md)** - Hands-on guides for optimization and transition states
- **[FAQ](docs/FAQ.md)** - Troubleshooting and common questions

## Examples

```bash
# Transition state search
qme ts --strategy interpolate reactant.xyz --product product.xyz

# NEB reaction path
qme path --strategy neb reactant.xyz --product product.xyz --npoints 11

# IRC from transition state
qme path --strategy irc ts.xyz --direction both
```

## Backend Selection

| Backend | Best For | Installation |
|---------|----------|--------------|
| `aimnet2` | Beginners, no conflicts | `pip install torch torch-cluster` |
| `uma` | Materials science | `pip install fairchem-core` |
| `mace` | High accuracy molecules | `pip install mace-torch` |
| `torchsim` | Maximum performance | `pip install torch-sim-atomistic` |

> **Note**: Some backends have dependency conflicts. Use separate environments or choose one backend per environment.

## Community and Support

- **GitHub Repository**: [https://github.com/rlaplaza-lab/qme](https://github.com/rlaplaza-lab/qme)
- **Issues**: [Report bugs and request features](https://github.com/rlaplaza-lab/qme/issues)
- **License**: MIT License

## Citation

```bibtex
@software{qme2025,
  title={QME: Quick Mechanistic Exploration},
  author={QME Development Team},
  year={2025},
  url={https://github.com/rlaplaza-lab/qme}
}
```
