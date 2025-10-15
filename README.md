# QME: Quick Mechanistic Exploration

**Quick mechanistic exploration using machine learning potentials for molecular geometry optimization and transition state searches with Sella or Trust-Krylov TS optimizers.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-tested-green.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](docs/index.md)

QME provides an intuitive interface for molecular optimization using state-of-the-art machine learning potentials including UMA, AIMNet2, MACE, SO3LR, and TorchSim acceleration. Built on top of ASE, QME couples reproducible command-line workflows with a flexible Python API so you can prototype on a laptop and scale on a cluster without rewriting code.

## 🚀 Quick Start

### Installation

```bash
# Recommended: install inside a fresh environment and pick a backend extra
python -m venv .venv && source .venv/bin/activate  # or use conda/mamba
pip install --upgrade pip

# Choose one backend extra per environment to avoid dependency conflicts
pip install "qme-ml[aimnet2]"    # Fast and reliable
pip install "qme-ml[mace]"       # High accuracy
pip install "qme-ml[uma]"        # General-purpose chemistry
pip install "qme-ml[orb]"        # Universal forcefield
pip install "qme-ml[torchsim]"   # Maximum performance (Python 3.11+)
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

explorer = qme.Explorer.from_file("water.xyz", backend="aimnet2")
result = explorer.run(mode="minima", local_optimizer_name="BFGS")
print(f"Final energy: {result['final_energy']:.6f} eV")
```

The Explorer object supports the same strategies available via the CLI, so you can experiment interactively and then encode a workflow in `qme` commands for automation.

## ✨ Features

- 🧪 **Multiple ML Backends**: UMA, AIMNet2, MACE, Orb, SO3LR, TorchSim
- ⚡ **GPU Acceleration**: CUDA support with up to 100x speedup
- 🎯 **Optimization Methods**: Local (minima/TS) and two-ended strategies (NEB, CI-NEB, GSM)
- 🔄 **Transition States**: Advanced TS search with Sella or the Trust-Krylov TS variant plus Hessian-based trust-region solvers
- 🛤️ **Reaction Paths**: NEB/CI-NEB path optimization and IRC (Intrinsic Reaction Coordinate) from transition states
- 📊 **Analysis Tools**: Frequency analysis, zero-point energy, and reaction energetics workflows
- 🖥️ **Dual Interface**: Command-line and Python API share the same strategy registry
- 📁 **File Support**: XYZ, CIF, PDB, and more via ASE import/export
- 🛡️ **Robust I/O**: Built-in validation catches problematic molecular input before optimisation starts
- ✅ **Python 3.10–3.12**: Fully tested across the entire supported range

## 📋 Common Commands

```bash
# Minima optimization (outputs single structure, defaults to BFGS)
qme opt molecule.xyz
qme opt reactant.xyz --product product.xyz  # Two-ended minima search

# Transition state optimization (outputs single TS, defaults to Sella)
qme tsopt ts_guess.xyz  # Single-ended TS optimization
qme tsopt reactant.xyz --product product.xyz  # Two-ended TS guess
qme tsopt ts_guess.xyz --optimizer trust-krylov-ts --fmax 0.02

# Reaction path optimization (outputs trajectories)
qme path interpolate r.xyz p.xyz --npoints 15  # Raw interpolation
qme path neb r.xyz p.xyz --npoints 11 --spring-constant 5.0  # NEB path
qme path cineb r.xyz p.xyz --npoints 11 --spring-constant 5.0  # CI-NEB path
qme path irc ts.xyz --direction both --steps 100  # IRC from transition state

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

### First-Order Optimizers (Gradient-Based)

| Optimizer | Description | Best For | Transition States |
|-----------|-------------|----------|-------------------|
| `lbfgs` | Limited-memory BFGS | Fast minima optimization | ❌ No |
| `bfgs` | Broyden-Fletcher-Goldfarb-Shanno | Standard optimization | ❌ No |
| `fire` | Fast Inertial Relaxation Engine | Quick relaxation | ❌ No |

### Second-Order Optimizers (Hessian-Based)

| Optimizer | Description | Best For | Transition States |
|-----------|-------------|----------|-------------------|
| `sella` | Modern saddle point optimizer | General TS searches | ✅ Yes |
| `trust-krylov` | Trust-region Krylov subspace | Challenging landscapes | ⚠️ Not Yet* |
| `trust-krylov-ts` | Trust-Krylov with min-mode reflection | Sella alternative when Hessians are cheap | ✅ Yes |
| `trust-ncg` | Trust-region Newton-CG | High-accuracy minima | ⚠️ Not Yet* |
| `trust-exact` | Trust-region exact solver | Maximum accuracy | ⚠️ Not Yet* |
| `newton-cg` | Newton Conjugate Gradient | Second-order minima | ❌ No |

> **Note**: Second-order optimizers use Hessian information for better convergence but are computationally more expensive. The trust-region methods (trust-krylov, trust-ncg, trust-exact) compute Hessians efficiently using ML potentials. By default, they compute the Hessian once at the start; use `hessian_update_freq` parameter for periodic updates.
>
> *The dedicated `trust-krylov-ts` variant performs min-mode following for transition states. Other trust-region optimizers remain focused on minima searches in QME.*

## 🧪 Testing and Development

```bash
# Run tests
pytest tests/

# Development installation
git clone https://github.com/rlaplaza-lab/qme.git
cd qme
pip install -e .[dev]
```

### Test Suite Overview

- `pytest tests/unit` — fast unit coverage for validation, constraints, optimizers, and helpers
- `pytest tests/integration` — lightweight smoke tests for Explorer flows, CLI commands, and TorchSim (skips if real backends are unavailable)
- `pytest tests/integration -k torchsim` — exercise TorchSim-dependent paths when the optional dependencies are installed
- `pytest -m "not slow"` — convenient selector if you add custom markers for heavier workflows

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
