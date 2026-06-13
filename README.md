# FAMEX: Fast Mechanistic Explorer

**FAMEX** provides a unified interface for molecular geometry optimization using machine learning potentials. It supports minima optimization, transition state searches, and reaction path calculations through both a command-line interface and Python API.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/famex.svg)](https://pypi.org/project/famex/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Quick Start

### Installation

```bash
pip install famex
# Or from source:
git clone https://github.com/rlaplaza-lab/famex.git && cd famex && pip install -e .
```

Install a backend separately:

| Backend | Installation | Notes |
|---------|--------------|-------|
| `aimnet2` | `pip install torch` | Recommended for beginners, no conflicts |
| `uma` | `pip install "fairchem-core>=2.21.0"` or `pip install famex[uma]` | Materials science (default model: uma-s-1p2) |
| `mace` | `pip install mace-torch` | High accuracy, conflicts with UMA |
| `orb` | `pip install orb-models` | Universal forcefield |
| `so3lr` | `pip install so3lr` | Research, custom models |
| `tblite` | `pip install tblite` | Fast semi-empirical |
| `pet` | `pip install upet` or `pip install famex[pet]` | Universal PET-MAD potential (Python 3.11+) |

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

# Optimize it (default backend is uma; use aimnet2 for torch-only install)
famex minima --strategy local water.xyz
# famex minima --strategy local water.xyz --backend aimnet2
```

**Python API:**
```python
import famex

# Load and optimize
explorer = famex.Explorer.from_file("water.xyz", backend="aimnet2", target="minima", strategy="local")
result = explorer.run(fmax=0.05, steps=1000)

# Save results
explorer.save_structure(result["optimized_atoms"], "water_optimized.xyz")
print(f"Final energy: {result['optimized_atoms'].get_potential_energy():.6f} eV")
```

## Key Features

- Multiple ML backends (UMA, AIMNet2, MACE, SO3LR, Orb, TBLite, PET)
- GPU acceleration with CUDA support
- Semantic target/strategy interface (minima, ts, path)
- Advanced methods (NEB, CI-NEB, IRC, growing string)
- Frequency analysis and thermodynamics
- Command-line and Python API
- Supports XYZ, CIF, PDB via ASE

## Documentation

- **[Documentation index](docs/README.md)** - Overview and defaults
- **[User Guide](docs/USER_GUIDE.md)** - Complete reference for CLI, Python API, and backends
- **[Tutorials](docs/TUTORIALS.md)** - Hands-on guides for optimization and transition states
- **[FAQ](docs/FAQ.md)** - Troubleshooting and common questions

## Examples

```bash
# Transition state search
famex ts --strategy interpolate reactant.xyz --product product.xyz

# NEB reaction path
famex path --strategy neb reactant.xyz product.xyz --npoints 11

# IRC from transition state
famex path --strategy irc ts.xyz --direction both
```

## Migrating from qme

The project was renamed from **qme** / **qme-ml** to **famex** in v0.2.0. There are no compatibility shims.

- Uninstall the old package: `pip uninstall qme-ml`
- Install the new package: `pip install famex`
- CLI: `qme` → `famex` (e.g. `famex minima …`)
- Python: `import qme` → `import famex`
- Optional: preserve cached models with `mv ~/.qme ~/.famex`

## Community and Support

- **GitHub Repository**: [https://github.com/rlaplaza-lab/famex](https://github.com/rlaplaza-lab/famex)
- **Issues**: [Report bugs and request features](https://github.com/rlaplaza-lab/famex/issues)
- **Security**: [Report vulnerabilities](https://github.com/rlaplaza-lab/famex/security/policy)
- **License**: MIT License

## Citation

```bibtex
@software{famex2026,
  title={FAMEX: Fast Mechanistic Explorer},
  author={FAMEX Development Team},
  year={2026},
  url={https://github.com/rlaplaza-lab/famex}
}
```
