# QME - Quick Mechanistic Exploration

A Python package for exploring chemical reaction mechanisms using Machine Learning Potentials (MLP) and Neural Network Potentials (NNPs). Inspired by [pysisyphus](https://github.com/eljost/pysisyphus) but optimized for ML-driven mechanistic studies.

## Features

- **Molecular Geometry Handling**: Easy manipulation of molecular structures with XYZ I/O
- **Reaction Pathway Analysis**: Create and analyze reaction pathways between reactants and products
- **MLP/NNP Integration**: Interface for machine learning potential energy calculations
- **Organic Reaction Tests**: Comprehensive test suite for common organic reactions
- **Visualization Tools**: Generate trajectories and analyze reaction mechanisms

## Installation

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