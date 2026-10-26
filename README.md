# FAMEX: Fast Mechanistic Explorer

**FAMEX** is a command-line and Python API for molecular geometry optimization with machine learning potentials. It supports minima optimization, transition state searches, and reaction path calculations.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/famex.svg)](https://pypi.org/project/famex/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Quick Start

### Installation

```bash
pip install famex
# Or from source:
git clone https://github.com/rlaplaza-lab/famex.git && cd famex && pip install -e .
```

Install a machine learning backend separately (primary use case):

| Backend | Installation | Notes |
|---------|--------------|-------|
| `aimnet2` | `pip install torch` | Simplest install; no dependency conflicts; analytical Hessian |
| `uma` | `pip install "fairchem-core>=2.24.0"` or `pip install famex[uma]` | Default backend (model: uma-s-1p2p1); analytical Hessian |
| `mace` | `pip install mace-torch` | Conflicts with UMA; analytical Hessian |
| `orb` | `pip install "orb-models>=0.7.0"` or `pip install famex[orb]` | OrbMol-v2 and Orb-v3; Python 3.12+; analytical Hessian for conservative models; finite-difference otherwise |
| `so3lr` | `pip install so3lr` | Custom models; analytical Hessian via JAX |
| `pet` | `pip install upet` or `pip install famex[pet]` | PET-MAD potential (Python 3.11+); analytical Hessian |

Optional classical quantum chemistry (same CLI; slower): `tblite` (`pip install tblite`, GFN-xTB, finite-difference Hessians) and `pyscf` (`pip install famex[pyscf]`, default B3LYP-D3BJ / def2-SVP, analytical Hessian).

SMILES and PubChem are optional structure inputs (not separate CLI modes). Install an embedder and pass SMILES or `pubchem:NAME` / `cid:` / a numeric CID to `minima`, `ts`, or `path`:

```bash
pip install famex[smiles]   # UFF distance geometry (default --embedder)
pip install famex[rdkit]    # RDKit ETKDGv3
famex minima --strategy local "CCO" --backend aimnet2
famex minima --strategy local "CCO" --embedder rdkit --backend aimnet2
famex minima --strategy local pubchem:aspirin --backend aimnet2
```

> **Note**: Python 3.11+ required. MACE and UMA conflict - use separate environments.

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

- Multiple machine learning backends (UMA, AIMNet2, MACE, SO3LR, Orb, PET); optional xTB/DFT via TBLite/PySCF
- GPU acceleration with CUDA
- Target/strategy commands (`minima`, `ts`, `path`)
- Path methods: Nudged Elastic Band (NEB), Climbing-Image NEB (CI-NEB), Intrinsic Reaction Coordinate (IRC), growing string
- Frequency analysis, thermodynamics, and vibrational-mode XYZ animations
- Command-line and Python API
- Geometry files via the Atomic Simulation Environment (ASE): XYZ, CIF, PDB, and others
- SMILES / PubChem as structure inputs (`pip install famex[smiles]` or `famex[rdkit]`; use `pubchem:NAME`, `cid:`, or a numeric CID)

## Documentation

- **[Documentation index](docs/README.md)** - Overview and defaults
- **[User Guide](docs/USER_GUIDE.md)** - Complete reference for CLI, Python API, and backends
- **[Tutorials](docs/TUTORIALS.md)** - Hands-on guides for optimization and transition states
- **[FAQ](docs/FAQ.md)** - Troubleshooting and common questions

## Examples

Runnable demos and benchmarks: **[examples/README.md](examples/README.md)**.

```bash
# Transition state search
famex ts --strategy interpolate reactant.xyz --product product.xyz

# NEB reaction path
famex path --strategy neb reactant.xyz product.xyz --npoints 11

# IRC from transition state
famex path --strategy irc ts.xyz --direction both
```

## Community and Support

- **GitHub Repository**: [https://github.com/rlaplaza-lab/famex](https://github.com/rlaplaza-lab/famex)
- **Issues**: [Report bugs and request features](https://github.com/rlaplaza-lab/famex/issues)
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
