# QME: Quick Mechanistic Exploration

Quick mechanistic exploration using multiple neural network potentials (MLPs/NNPs).

QME combines the power of ASE and SELLA optimizers with state-of-the-art neural network potentials including **SO3LR** (SO(3) invariant networks) and **UMA** (Universal Model for Atoms) to perform efficient molecular geometry optimization and transition state searches.

## Features

- **Multiple Neural Network Backends**: Support for SO3LR and UMA potentials with easy switching
- **Minimum Energy Optimization**: Find stable molecular geometries using ASE optimizers (BFGS, LBFGS, FIRE)
- **Transition State Search**: Locate saddle points using the SELLA optimizer
- **Command Line Interface**: Easy-to-use CLI for batch processing and automation
- **Multiple File Formats**: Support for XYZ, CIF, PDB and other ASE-compatible molecular formats
- **Flexible Constraints**: Apply geometric constraints during optimization
- **Robust Testing**: Comprehensive test suite with CI/CD automation

## Supported Neural Network Backends

### SO3LR (Default)
SO3LR provides SO(3) invariant neural network potentials with excellent accuracy for molecular systems. It's now the **default backend** for better testing and performance.

### UMA (Universal Model for Atoms)
UMA machine learning potentials from the FAIR Chemistry team provide state-of-the-art accuracy for diverse chemical systems.

## Installation

### Prerequisites

QME requires Python 3.8 or higher.

### Basic Installation (Mock Calculator)

For testing and development without ML dependencies:

```bash
pip install ase click numpy matplotlib
cd /path/to/qme
pip install -e .
```

### Full Installation (with UMA)

For production use with UMA machine learning potentials:

```bash
pip install -e .[ml]  # Includes torch, ase, fairchem-core
```

## Quick Start

### Python API - Basic Usage

```python
from qme import QMEOptimizer

# Initialize optimizer (automatically uses mock if UMA unavailable)
qme = QMEOptimizer(model_name="uma-4m")

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

### Command Line Interface

```bash
# Basic optimization
qme minimize molecule.xyz

# With custom parameters
qme minimize molecule.xyz \
    --output optimized.xyz \
    --optimizer BFGS \
    --fmax 0.005 \
    --steps 300 \
    --verbose

# Transition state search (requires SELLA)
qme transition-state ts_guess.xyz --trajectory ts_optimization.traj
```

### Test Installation

```bash
qme test-setup
```

# Generate reaction pathway
path = reaction.interpolate(npoints=10)
for i, geom in enumerate(path):
    calculator.calculate(geom)
    print(f"Point {i}: Energy = {geom.energy:.6f} Hartree")
```

## Organic Reaction Examples

The package includes comprehensive tests for classic organic reactions:

### SN2 Reaction
```python
# CH3Cl + OH- → CH3OH + Cl-
from qme.tests.test_sn2_reaction import TestSN2Reaction
```

### Diels-Alder Reaction  
```python
# 1,3-butadiene + ethylene → cyclohexene
from qme.tests.test_diels_alder_reaction import TestDielsAlderReaction
```

### Proton Transfer
```python
# HCl + NH3 → NH4+ + Cl-
from qme.tests.test_proton_transfer_reaction import TestProtonTransferReaction
```

## Testing

Run the comprehensive test suite:

```bash
pytest tests/ -v
```

Current status: **51/57 tests passing** (89% success rate)

## Architecture

- `qme.geometry`: Molecular geometry handling and manipulation
- `qme.reactions`: Reaction pathway creation and analysis  
- `qme.calculators`: Interfaces for MLP/NNP energy and force calculations
- `tests/`: Comprehensive test suite for organic reactions

## Contributing

This project is inspired by the excellent [pysisyphus](https://github.com/eljost/pysisyphus) package by @eljost. Contributions are welcome!

## License

MIT License
