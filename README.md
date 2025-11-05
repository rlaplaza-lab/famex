# QME: Quick Mechanistic Exploration

**QME** provides a unified interface for molecular geometry optimization using machine learning potentials. It supports minima optimization, transition state searches, and reaction path calculations through both a command-line interface and Python API.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Quick Start

### Installation

```bash
pip install qme-ml
# Or from source:
git clone https://github.com/rlaplaza-lab/qme.git && cd qme && pip install -e .
```

Install a backend separately:

| Backend | Installation | Notes |
|---------|--------------|-------|
| `aimnet2` | `pip install torch torch-cluster` | Recommended for beginners, no conflicts |
| `uma` | `pip install fairchem-core` | Materials science |
| `mace` | `pip install mace-torch` | High accuracy, conflicts with UMA |
| `orb` | `pip install orb-models` | Universal forcefield |
| `so3lr` | `pip install so3lr` | Research, custom models |
| `tblite` | `pip install tblite` | Fast semi-empirical |

> **Note**: Python 3.10+ required. MACE and UMA conflict - use separate environments.

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
print(f"Final energy: {result['optimized_atoms'].get_potential_energy():.6f} eV")
```

## Key Features

- Multiple ML backends (UMA, AIMNet2, MACE, SO3LR, Orb, TBLite)
- GPU acceleration with CUDA support
- Semantic target/strategy interface (minima, ts, path)
- Advanced methods (NEB, CI-NEB, IRC, growing string)
- Frequency analysis and thermodynamics
- Command-line and Python API
- Supports XYZ, CIF, PDB via ASE

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
