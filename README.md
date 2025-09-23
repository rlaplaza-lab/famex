# QME: Quick Mechanistic Exploration

Quick mechanistic exploration using machine learning potentials (MLPs/NNPs).

QME combines the power of ASE and SELLA optimizers with UMA (Universal Model for Atoms) machine learning potentials to perform efficient molecular geometry optimization and transition state searches.

## Features

- **Minimum Energy Optimization**: Find stable molecular geometries using ASE optimizers (BFGS, LBFGS, FIRE)
- **Transition State Search**: Locate saddle points using the SELLA optimizer
- **UMA Integration**: Leverage state-of-the-art Universal Model for Atoms machine learning potentials
- **Command Line Interface**: Easy-to-use CLI for batch processing and automation
- **Multiple File Formats**: Support for XYZ, CIF, PDB and other common molecular formats
- **Flexible Constraints**: Apply geometric constraints during optimization

## Installation

### Prerequisites

QME requires Python 3.8 or higher. Install the package and its dependencies:

```bash
pip install -e .
```

## Quick Start

```python
from qme import Geometry, Reaction, MLPCalculator
import numpy as np

# Create molecular geometries
atoms = ["H", "H"]
reactant_coords = np.array([0.0, 0.0, 0.0, 0.74, 0.0, 0.0])  # H2
product_coords = np.array([0.0, 0.0, 0.0, 3.0, 0.0, 0.0])    # H + H

reactant = Geometry(atoms=atoms, coords=reactant_coords, charge=0, mult=1)
product = Geometry(atoms=atoms, coords=product_coords, charge=0, mult=3)

# Create reaction
reaction = Reaction(reactant, product, name="H2_dissociation")

# Set up calculator (mock for demonstration)
calculator = MLPCalculator(model_type="mock")

# Calculate energies
calculator.calculate(reactant)
calculator.calculate(product)

print(f"Reaction energy: {reaction.reaction_energy:.6f} Hartree")

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
=======
### Dependencies

The main dependencies will be installed automatically:

- `ase>=3.22.0` - Atomic Simulation Environment
- `sella>=2.0.0` - Transition state optimization
- `torch>=2.0.0` - PyTorch for ML models
- `fairchem-core>=1.0.0` - UMA machine learning potentials
- `click>=8.0.0` - Command line interface

## Quick Start

### Test Installation

First, verify your installation:

```bash
qme test-setup
```

### Minimum Energy Optimization

Optimize a molecular structure to find its minimum energy configuration:

```bash
# Basic optimization
qme minimize examples/water.xyz

# With custom parameters
qme minimize examples/methane.xyz \
    --optimizer BFGS \
    --fmax 0.005 \
    --steps 300 \
    --output optimized_methane.xyz \
    --verbose
```

### Transition State Search

Search for transition states (saddle points):

```bash
# Basic TS search  
qme transition-state examples/reaction_guess.xyz

# With trajectory logging
qme transition-state examples/ts_guess.xyz \
    --output ts_result.xyz \
    --trajectory ts_optimization.traj \
    --logfile ts_search.log \
    --verbose
```

## Command Line Reference

### `qme minimize`

Find minimum energy geometry using specified optimizer.

**Options:**
- `--output, -o`: Output file for optimized structure
- `--optimizer, -opt`: Optimizer to use (`BFGS`, `LBFGS`, `FIRE`)  
- `--fmax, -f`: Force convergence criterion (eV/Å) [default: 0.01]
- `--steps, -s`: Maximum optimization steps [default: 200]
- `--model, -m`: UMA model name [default: uma-4m]
- `--device, -d`: Computation device (`cpu`, `cuda`)
- `--logfile`: Log file for optimization output
- `--trajectory`: Trajectory file to save steps
- `--constraint-atoms`: Comma-separated atom indices to fix
- `--verbose, -v`: Verbose output

### `qme transition-state`

Find transition state using SELLA optimizer.

**Options:**
- Similar to `minimize` but uses SELLA for saddle point optimization
- Requires good initial guess geometry near transition state

### `qme test-setup`

Test QME installation and UMA model loading.

## Python API

### Basic Usage

```python
from qme import QMEOptimizer

# Initialize optimizer with UMA potential
qme = QMEOptimizer(model_name="uma-4m")

# Load structure
atoms = qme.load_structure("molecule.xyz")

# Optimize to minimum
results = qme.optimize_minimum(
    optimizer="BFGS",
    fmax=0.01,
    steps=200
)

# Save result
qme.save_structure(results['optimized_atoms'], "optimized.xyz")
```

### Advanced Usage

```python
from qme import QMEOptimizer
from qme.uma_potential import get_uma_calculator
from ase.constraints import FixAtoms

# Custom calculator setup
calculator = get_uma_calculator(model_name="uma-4m", device="cuda")
qme = QMEOptimizer(calculator=calculator)

# Load and optimize with constraints
atoms = qme.load_structure("complex.xyz")
constraints = [FixAtoms(indices=[0, 1, 2])]  # Fix first 3 atoms

results = qme.optimize_minimum(
    atoms=atoms,
    optimizer="LBFGS", 
    constraints=constraints,
    fmax=0.005,
    trajectory="optimization.traj"
)

# Transition state search
ts_results = qme.find_transition_state(
    atoms=atoms,
    fmax=0.01,
    steps=500
)

print(qme.get_optimization_summary())
```

## UMA Models

QME supports various UMA model variants. The default `uma-4m` model provides good balance of accuracy and speed for most organic molecules.

## File Format Support

QME supports all file formats handled by ASE, including:

- **XYZ**: Simple Cartesian coordinates
- **CIF**: Crystallographic Information File  
- **PDB**: Protein Data Bank format
- **VASP**: POSCAR/CONTCAR files
- **Gaussian**: Input/output files

## Examples

The `examples/` directory contains sample molecular structures:

- `water.xyz`: Simple water molecule
- `methane.xyz`: Methane for basic testing

## Contributing

Contributions are welcome! Please see our contributing guidelines and submit pull requests.

## License

This project is licensed under the MIT License.

## Citation

If you use QME in your research, please cite:

```
QME: Quick Mechanistic Exploration using Machine Learning Potentials
https://github.com/rlaplaza-lab/qme
```

## Support

For questions and support, please open an issue on GitHub.

