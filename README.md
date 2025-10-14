# QME: Quick Mechanistic Exploration

**Quick mechanistic exploration using machine learning potentials for molecular geometry optimization and transition state searches with Sella optimizer.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-tested-green.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](docs/index.md)

QME provides an intuitive interface for molecular optimization using state-of-the-art machine learning potentials including UMA, AIMNet2, MACE, SO3LR, and TorchSim acceleration.

## 🚀 Quick Start

### Installation

```bash
# Recommended: Install with a specific backend
pip install qme-ml[aimnet2]    # Fast and reliable
pip install qme-ml[mace]       # High accuracy
pip install qme-ml[uma]        # General purpose
pip install qme-ml[orb]        # Universal forcefield
pip install qme-ml[torchsim]   # Maximum performance (Python 3.11+)
```

> **Note**: Some backends have dependency conflicts. See [Installation Guide](docs/getting_started.md) for details.

### Your First Optimization

```bash
# Create a water molecule (water.xyz)
echo "3
Water molecule
O    0.000000    0.000000    0.117283
H    0.000000    0.758602   -0.469132
H    0.000000   -0.758602   -0.469132" > water.xyz

# Optimize with QME
qme opt water.xyz
```

### Python API

```python
import qme

# Optimize a structure
explorer = qme.Explorer.from_file("water.xyz", backend="aimnet2")
result = explorer.run(mode="minima")
print(f"Final energy: {result['final_energy']:.6f} eV")
```

## ✨ Features

- 🧪 **Multiple ML Backends**: UMA, AIMNet2, MACE, Orb, SO3LR, TorchSim
- ⚡ **GPU Acceleration**: CUDA support with up to 100x speedup
- 🎯 **Optimization Methods**: Local and two-ended strategies
- 🔄 **Transition States**: Advanced TS search with Sella optimizer
- 🛤️ **Reaction Paths**: NEB and CI-NEB path optimization with trajectory saving
- 📊 **Analysis Tools**: Frequency analysis and thermodynamics
- 🖥️ **Dual Interface**: Command-line and Python API
- 📁 **File Support**: XYZ, CIF, PDB, and more via ASE
- 🛡️ **Robust I/O**: Handles problematic molecular data gracefully
- ✅ **Python 3.12**: Fully tested and compatible

## 📋 Common Commands

```bash
# Minima optimization (outputs single structure, defaults to BFGS)
qme opt molecule.xyz
qme opt reactant.xyz --product product.xyz  # Two-ended minima search

# Transition state optimization (outputs single TS, defaults to Sella)
qme tsopt ts_guess.xyz  # Single-ended TS optimization
qme tsopt reactant.xyz --product product.xyz  # Two-ended TS guess

# Reaction path optimization (outputs trajectories)
qme path interpolate r.xyz p.xyz --npoints 15  # Raw interpolation
qme path neb r.xyz p.xyz --npoints 11 --spring-constant 5.0  # NEB path
qme path cineb r.xyz p.xyz --npoints 11 --spring-constant 5.0  # CI-NEB path

# Different backends and settings
qme opt molecule.xyz --backend mace --device cuda --fmax 0.01
qme tsopt ts_guess.xyz --backend aimnet2 --optimizer sella --fmax 0.01

# Cache management
qme cache info
qme cache clear
```

## 📖 Documentation

Comprehensive documentation is available in the [`docs/`](docs/) directory:

| Section | Description |
|---------|-------------|
| [📚 **Getting Started**](docs/getting_started.md) | Installation and first optimization |
| [👥 **User Guide**](docs/user_guide/index.md) | Complete usage reference |
| [🎓 **Tutorials**](docs/tutorials/index.md) | Step-by-step guides |
| [🏗️ **Developer Guide**](docs/developer_guide/index.md) | Contributing and extending QME |
| [📊 **Benchmarks**](docs/benchmarks/index.md) | Performance and accuracy tests |
| [🔧 **Reference**](docs/reference/index.md) | Configuration and troubleshooting |

### Quick Links

- **Installation Issues**: [Troubleshooting Guide](docs/reference/troubleshooting.md)
- **Backend Selection**: [Supported Backends](docs/user_guide/backends.md)
- **Performance**: [TorchSim Acceleration](docs/user_guide/torchsim_acceleration.md)
- **Contributing**: [Developer Guide](docs/developer_guide/contributing.md)

## 🏁 Supported Backends

| Backend | Description | Best For | Installation |
|---------|-------------|----------|--------------|
| `aimnet2` | Native PyTorch implementation | General use, no conflicts | `pip install qme-ml[aimnet2]` |
| `mace` | Foundation models | High accuracy | `pip install qme-ml[mace]` |
| `uma` | Universal Materials Accelerator | Materials science | `pip install qme-ml[uma]` |
| `torchsim_*` | TorchSim acceleration | Maximum performance | `pip install qme-ml[torchsim]` |
| `so3lr` | SO(3) neural networks | Research applications | `pip install qme-ml[so3lr]` |
| `mock` | Harmonic oscillator | Testing and development | Built-in |

> **Note**: UMA and MACE cannot be installed together due to dependency conflicts. Use separate environments or choose one.

## ⚙️ Supported Optimizers

| Optimizer | Description | Best For | Transition States |
|-----------|-------------|----------|-------------------|
| `sella` | Modern saddle point optimizer | General TS searches | ✅ Yes |
| `lbfgs` | Limited-memory BFGS | Fast minima optimization | ❌ No |
| `bfgs` | Broyden-Fletcher-Goldfarb-Shanno | Standard optimization | ❌ No |
| `fire` | Fast Inertial Relaxation Engine | Quick relaxation | ❌ No |

> **Note**: Sella is the recommended optimizer for transition state searches.

## 🧪 Testing and Development

```bash
# Run tests
pytest tests/

# Development installation
git clone https://github.com/rlaplaza-lab/qme.git
cd qme
pip install -e .[dev]
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/developer_guide/contributing.md) for:

- 🐛 Bug reports and feature requests
- 💻 Code contributions and new backends
- 📝 Documentation improvements
- 🧪 Test coverage and benchmarks

## 🆘 Getting Help

- **📖 Documentation**: Browse the comprehensive [docs/](docs/) directory
- **🐛 Issues**: [GitHub Issues](https://github.com/rlaplaza-lab/qme/issues) for bugs and features
- **💬 Discussions**: [GitHub Discussions](https://github.com/rlaplaza-lab/qme/discussions) for questions
- **🔍 Troubleshooting**: See the [Troubleshooting Guide](docs/reference/troubleshooting.md)

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
