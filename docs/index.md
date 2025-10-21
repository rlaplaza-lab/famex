# QME Documentation

**Quick Mechanistic Exploration using Machine Learning Potentials**

Welcome to the QME documentation! QME provides a unified interface for molecular geometry optimization and transition state searches using state-of-the-art machine learning potentials.

## Documentation Overview

### 🚀 [Getting Started](getting_started.md)
- Installation and first optimization
- Quick start guide with new Explorer API

### 📚 [User Guide](user_guide/index.md)
- [Supported Backends](user_guide/backends.md)
- [CLI Reference](user_guide/cli_reference.md)

### 🔬 [Tutorials](tutorials/index.md)
- [Basic Optimization](tutorials/basic_optimization.md)
- [Transition States](tutorials/transition_states.md)

### 🏗️ [Developer Guide](developer_guide/index.md)
- [Contributing Guidelines](developer_guide/contributing.md)
- [Adding New Backends](developer_guide/adding_backends.md)
- [Creating Custom Strategies](developer_guide/creating_custom_strategies.md)

### 📊 [Benchmarks and Examples](benchmarks/index.md)
- [BH28 Benchmark](benchmarks/bh28.md)

### 🔧 [Reference](reference/index.md)
- [Troubleshooting](reference/troubleshooting.md)
- [FAQ](reference/faq.md)

## Quick Links

- **Installation**: `pip install qme-ml[aimnet2]` (see [Getting Started](getting_started.md) for details)
- **GitHub Repository**: [https://github.com/rlaplaza-lab/qme](https://github.com/rlaplaza-lab/qme)
- **Issue Tracker**: [Report bugs and request features](https://github.com/rlaplaza-lab/qme/issues)
- **License**: MIT License

## Key Features

- 🧪 **Multiple ML Backends**: UMA, AIMNet2, MACE, SO3LR, TorchSim
- ⚡ **GPU Acceleration**: CUDA support for compatible backends
- 🎯 **Optimization Methods**: Target/strategy system (minima, ts, path)
- 🔄 **Transition States**: Advanced TS search algorithms including NEB
- 📊 **Frequency Analysis**: Vibrational frequency calculations and thermodynamics
- 🖥️ **CLI and Python API**: Both command-line and programmatic interfaces
- 📁 **File Format Support**: XYZ, CIF, PDB, and more via ASE

## Community and Support

- **Documentation Issues**: Help us improve this documentation by [opening an issue](https://github.com/rlaplaza-lab/qme/issues)
- **Questions**: Use GitHub Discussions for general questions
- **Bug Reports**: Please use the issue tracker for bug reports
- **Feature Requests**: We welcome feature requests via GitHub issues

---

*Last updated: {current_date}*
