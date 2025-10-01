# QME: Quick Mechanistic Exploration

Quick mechanistic exploration using multiple neural network potentials (MLPs/NNPs).

QME combines the power of ASE and SELLA optimizers with state-of-the-art neural network potentials including **SO3LR** (SO(3) invariant networks), **UMA** (Universal Model for Atoms), and **AIMNET2** (Accurate Neural Network Potential) to perform efficient molecular geometry optimization and transition state searches.

## Features

- **Multiple Neural Network Backends**: Support for UMA, AIMNet2, MACE, and SO3LR potentials with easy switching
- **TorchSim Integration**: Optional TorchSim acceleration for up to 100x speedup on supported models
- **Local and Two-Ended Optimization**: Find stable molecular geometries and transition states
- **NEB Path Optimization**: Nudged Elastic Band method for reaction pathway exploration
- **Command Line Interface**: Easy-to-use CLI with `qme opt` and `qme tsopt` commands
- **Multiple File Formats**: Support for XYZ, CIF, PDB and other ASE-compatible molecular formats
- **Flexible Constraints**: Apply geometric constraints during optimization
- **Automatic Fallback**: Graceful degradation to mock calculators when ML backends unavailable
- **Comprehensive Testing**: Full test suite with benchmark examples

## Supported Neural Network Backends

### UMA (Default)
UMA machine learning potentials from the FAIR Chemistry team provide state-of-the-art accuracy for diverse chemical systems. UMA is the **default backend** as the most advanced MLP available.

### AIMNet2
AIMNet2 provides fast and reliable energy, force, and property calculations for molecules containing a diverse range of elements. It excels at modeling neutral, charged, organic, and elemental-organic systems with flexible long-range interactions.

### MACE
MACE (Machine learning Accelerated Computational Environment) provides foundation models for molecules, transition metals, and cations with excellent accuracy and charge/spin embedding capabilities.

### SO3LR
SO3LR provides SO(3) invariant neural network potentials with excellent accuracy for molecular systems. Requires separate installation.

### TorchSim Acceleration (Optional)
TorchSim provides significant performance improvements for supported models through automatic batching and GPU acceleration. Install with `pip install torch-sim-atomistic` to enable up to 100x speedup for MACE and Fairchem models.

## Installation

### Prerequisites

QME requires Python 3.12 or higher.

### Basic Installation

For testing and development (mock calculator by default, no heavy ML dependencies):

```bash
pip install qme
```

Or install from source:

```bash
git clone https://github.com/rlaplaza-lab/qme.git
cd qme
pip install -e .
```

### Full Installation with ML Backends

For production use with machine learning potentials:

```bash
pip install qme[ml]
```

Or from source:

```bash
git clone https://github.com/rlaplaza-lab/qme.git
cd qme
pip install -e .[ml]
```

This installs:
- **UMA**: Universal Materials Accelerator (Meta AI) - default backend
- **AIMNet2**: Native implementation with PyTorch
- **MACE**: Foundation models for molecules and materials
- **JAX**: For SO3LR backend (if installed separately)

### Transition-State Optimizer (SELLA)

SELLA is optional and only needed for transition state optimization:

```bash
pip install qme[opt]
```

### Complete Installation

For full functionality with all backends and optimizers:

```bash
pip install qme[ml,opt]
```

### Development Installation

For contributors:

```bash
git clone https://github.com/rlaplaza-lab/qme.git
cd qme
pip install -e .[dev]  # Includes testing and linting tools
```

### SO3LR Backend Installation (optional)

To use the SO3LR backend (requires separate installation):

```bash
# Install JAX (CPU version recommended for most users)
pip install "jax[cpu]>=0.4.20"

# Install SO3LR from source
git clone https://github.com/general-molecular-simulations/so3lr.git
cd so3lr
pip install .
```

### TorchSim Acceleration (optional)

For significant performance improvements, install TorchSim:

```bash
pip install torch-sim-atomistic
```

This enables up to 100x speedup for supported models (MACE, Fairchem) through automatic batching and GPU acceleration.

**Note**: If any ML backend is not installed, QME will automatically fall back to mock implementations for testing and development.

## Quick Start

### Python API - Basic Usage

```python
from qme import QMEOptimizer

# Initialize optimizer (automatically uses mock if UMA unavailable)
qme = QMEOptimizer(model_name="uma-m-1p1")

# Load structure
atoms = qme.load_structure("molecule.xyz")

# Optimize to minimum energy
results = qme.optimize_minimum(
    optimizer="BFGS",
    fmax=0.01,
    steps=200
)

# Save result
if results['converged']:
    qme.save_structure(results['optimized_atoms'], "optimized.xyz")
    print(f"Optimization completed in {results['steps_taken']} steps")
else:
    print("Optimization did not converge")
```

### Python API - TorchSim Acceleration

```python
from qme import QMEOptimizer

# Initialize with TorchSim MACE for maximum performance
qme = QMEOptimizer(
    backend="torchsim_mace",
    model_name="mace-omol-0",
    device="cuda"  # Use GPU for best performance
)

# Load structure
atoms = qme.load_structure("molecule.xyz")

# Optimize (with TorchSim acceleration)
results = qme.optimize_minimum(
    optimizer="BFGS",
    fmax=0.01,
    steps=200
)

# TorchSim provides significant speedup for large systems
print(f"Optimization completed in {results['steps_taken']} steps")
```

### Command Line Interface

```bash
# Basic structure optimization (local minima)
qme opt molecule.xyz --backend uma --fmax 0.01

# Two-ended minima optimization (interpolate R->P and optimize minima along path)
qme opt reactant.xyz --product product.xyz --interp geodesic --npoints 21 --backend aimnet2

# Transition state search (local, requires SELLA)
qme tsopt ts_guess.xyz --optimizer sella --fmax 0.05 --backend uma

# Two-ended TS optimization (interpolate R->P and find TS along path)
qme tsopt reactant.xyz --product product.xyz --interp geodesic --npoints 15 --backend mace

# NEB path optimization (find minimum energy path between R and P)
qme tsopt reactant.xyz --product product.xyz --mode neb --backend uma --npoints 11 --spring-constant 5.0

# NEB with custom parameters for challenging systems
qme tsopt reactant.xyz --product product.xyz --mode neb --npoints 21 --spring-constant 2.0 --fmax 0.01

# Using different backends
qme opt molecule.xyz --backend aimnet2 --model-name aimnet2_wb97m
qme opt molecule.xyz --backend mace --model-name mace-omol-0
qme opt molecule.xyz --backend so3lr  # Requires SO3LR installation

# Using TorchSim acceleration (requires torch-sim-atomistic)
qme opt molecule.xyz --backend torchsim_mace --model-name mace-omol-0 --device cuda
qme opt molecule.xyz --backend torchsim_fairchem --model-name equiformer_v2_31M_s2ef_all_md
```

### Test Installation

```bash
pytest -q
```

## Features in Detail

### Multiple Neural Network Backends
- **UMA**: Universal Materials Accelerator potentials from Meta AI (default backend)
- **SO3LR**: SO(3) invariant neural networks
- **AIMNET2**: Accurate neural network potentials for diverse molecular systems
- **Mock Calculator**: Harmonic oscillator model for testing

### Optimization Algorithms
- **L-BFGS**: Limited-memory quasi-Newton method (default for minima)
- **BFGS**: Quasi-Newton method for fast convergence
- **FIRE**: Fast Inertial Relaxation Engine
- **SELLA**: Saddle point optimization for transition states
- **NEB**: Nudged Elastic Band for reaction pathway optimization

### NEB Parameters
- **Spring Constant**: Controls path smoothness (default: 5.0 eV/Å²)
  - Higher values: Smoother paths, may miss sharp transitions
  - Lower values: More flexible paths, may have kinks
  - Typical range: 1.0-10.0 eV/Å² depending on system
- **Number of Images**: Path resolution (default: 11)
  - More images: Better resolution, higher computational cost
  - Fewer images: Faster computation, may miss details

### Supported File Formats
All ASE-compatible formats including XYZ, CIF, PDB, POSCAR, and more.

## Examples

See the `examples/` directory for complete demonstrations:

- `cli_demo.py`: Comprehensive CLI demonstration with all backends
- `bh28_benchmark/`: Benchmark suite for barrier height calculations
- `zimmermann93_benchmark/`: Additional benchmark examples
- `example_files/`: Sample molecular structures for testing

### Running the CLI Demo

```bash
# Run the comprehensive CLI demo
python examples/cli_demo.py

# This will test all available backends with various optimization tasks
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_backends_min_ts.py    # Backend functionality tests
pytest tests/test_cli.py                # CLI command tests
pytest tests/test_comprehensive_optimization.py  # Full optimization tests
pytest tests/test_constraints.py        # Constraint handling tests
pytest tests/test_frequencies.py        # Frequency analysis tests
pytest tests/test_reaction.py           # Reaction pathway tests

# Run with coverage
pytest --cov=qme --cov-report=term-missing
```

### Troubleshooting

#### Common Issues

**Missing dependencies**: Install with `pip install qme[ml]` for full functionality.

**SELLA missing**: Install with `pip install qme[opt]`.

**JAX backend warnings**: These are informational and can be safely ignored for CPU usage.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Install development dependencies (`pip install -e .[dev]`)
4. Run tests and linting (`pytest`, `black qme/ tests/`, `isort qme/ tests/`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use QME in your research, please cite:

```bibtex
@software{qme2025,
  title={QME: Quick Mechanistic Exploration},
  author={QME Development Team},
  year={2025},
  url={https://github.com/rlaplaza-lab/qme}
}
```

## Acknowledgments

- Inspired by [pysisyphus](https://github.com/eljost/pysisyphus) for reaction pathway methods
- Built on [ASE](https://wiki.fysik.dtu.dk/ase/) for molecular structure handling
- Integrates [SELLA](https://github.com/zadorlab/sella) for transition state optimization




