# QME: Quick Mechanistic Exploration

Quick mechanistic exploration using multiple neural network potentials (MLPs/NNPs).

QME combines the power of ASE and SELLA optimizers with state-of-the-art neural network potentials including **SO3LR** (SO(3) invariant networks), **UMA** (Universal Model for Atoms), and **AIMNET2** (Accurate Neural Network Potential) to perform efficient molecular geometry optimization and transition state searches.

## Features

- **Multiple Neural Network Backends**: Support for SO3LR, UMA, and AIMNET2 potentials with easy switching
- **Minimum Energy Optimization**: Find stable molecular geometries using ASE optimizers (BFGS, LBFGS, FIRE)
- **Transition State Search**: Locate saddle points using the SELLA optimizer
- **Command Line Interface**: Easy-to-use CLI for batch processing and automation
- **Multiple File Formats**: Support for XYZ, CIF, PDB and other ASE-compatible molecular formats
- **Flexible Constraints**: Apply geometric constraints during optimization
- **Robust Testing**: Comprehensive test suite with CI/CD automation

## Supported Neural Network Backends

### SO3LR
SO3LR provides SO(3) invariant neural network potentials with excellent accuracy for molecular systems.

### UMA
UMA machine learning potentials from the FAIR Chemistry team provide state-of-the-art accuracy for diverse chemical systems. UMA is the **default backend** as the most advanced MLP available.

### AIMNET2
AIMNET2 provides fast and reliable energy, force, and property calculations for molecules containing a diverse range of elements. It excels at modeling neutral, charged, organic, and elemental-organic systems with flexible long-range interactions.

## Installation

### Prerequisites

QME requires Python 3.12 or higher.

### Basic Installation

For testing and development with mock calculators:

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
pip install qme[ml]  # Includes torch, ase, sella, and fairchem-core
```

Or from source:

```bash
git clone https://github.com/rlaplaza-lab/qme.git
cd qme
pip install -e .[ml]
```

### Development Installation

For contributors:

```bash
git clone https://github.com/rlaplaza-lab/qme.git
cd qme
pip install -e .[dev]  # Includes testing and linting tools
```

### SO3LR Backend Installation

To use the SO3LR backend (optional):

```bash
# Install JAX (CPU version)
pip install jax==0.4.23

# Install SO3LR
git clone https://github.com/general-molecular-simulations/so3lr.git
cd so3lr
pip install .
```

**Note**: If SO3LR is not installed, QME will automatically fall back to mock implementations for testing and development.

## Quick Start

### Python API - Basic Usage

```python
from qme.core import QMEOptimizer

# Initialize optimizer (automatically uses mock if UMA unavailable)
optimizer = QMEOptimizer(model_name="uma-m-1p1")

# Load structure
atoms = optimizer.load_structure("molecule.xyz")

# Optimize to minimum energy
results = optimizer.optimize_minimum(
    optimizer="BFGS",
    fmax=0.01,
    steps=200,
)

# Save result
if results["converged"]:
    optimizer.save_structure(results["optimized_atoms"], "optimized.xyz")
    print(f"Optimization completed in {results['steps_taken']} steps")
else:
    print("Optimization did not converge")
```

### Command Line Interface

```bash
# Basic optimization  
qme opt molecule.xyz

# With custom parameters and constraints
qme opt molecule.xyz \
    --output optimized.xyz \
    --optimizer LBFGS \
    --fmax 0.005 \
    --steps 300 \
    --fix-atoms "0,1,2" \
    --harmonic-constraints "3,4" \
    --spring-constant 15.0 \
    --verbose

# Transition state search (requires SELLA)
qme tsopt ts_guess.xyz --trajectory ts_optimization.traj

# Find transition state using NEB method
qme neb reactant.xyz product.xyz --output ts.xyz
```

### Test Installation

```bash
# Test QME setup and backends
qme test-setup --backend so3lr
qme test-setup --backend uma  
qme test-setup --backend aimnet2

# Show system information
qme info

# Show configuration
qme config --show
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

### Supported File Formats
All ASE-compatible formats including XYZ, CIF, PDB, POSCAR, and more.

## Examples

See the `examples/` directory for complete demonstrations:

- `demo.py`: Basic molecular optimization
- `h2_dissociation_demo.py`: H₂ dissociation pathway
- `sn2_reaction_demo.py`: SN2 reaction mechanism
- `geodesic_demo.py`: Advanced interpolation methods

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_qme.py          # Basic functionality
pytest tests/test_so3lr.py        # SO3LR backend tests
pytest tests/test_aimnet2.py      # AIMNET2 backend tests
pytest tests/test_calculators.py  # Calculator interfaces

# Run with coverage
pytest --cov=qme --cov-report=term-missing
```

### Troubleshooting

#### Common Issues

**Missing dependencies**: Install with `pip install qme[ml]` for full functionality.

**SELLA import errors**: Install with `pip install sella` or `conda install sella`.

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




