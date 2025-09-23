# Installation and Usage Guide for QME

## Quick Start Guide

This guide shows how to get QME up and running quickly for testing and development.

### Option 1: Mock Calculator (for testing/development)

If you want to test QME functionality without installing the full ML dependencies:

1. **Install basic dependencies:**
   ```bash
   pip install ase click numpy matplotlib
   ```

2. **Clone and test:**
   ```bash
   cd /path/to/qme
   python qme_simple.py test
   ```

3. **Run optimization:**
   ```bash
   python qme_simple.py minimize examples/water.xyz --verbose
   ```

### Option 2: Full Installation with UMA Models

For production use with actual ML potentials:

1. **Install PyTorch:** (choose appropriate version for your system)
   ```bash
   pip install torch torchvision torchaudio
   ```

2. **Install fairchem-core for UMA models:**
   ```bash
   pip install fairchem-core
   ```

3. **Install additional dependencies:**
   ```bash
   pip install ase sella click numpy
   ```

4. **Install QME package:**
   ```bash
   pip install -e .
   ```

5. **Test with real UMA models:**
   ```bash
   qme test-setup
   ```

## Usage Examples

### Command Line Interface

**Basic optimization:**
```bash
python qme_simple.py minimize molecule.xyz
```

**With custom parameters:**
```bash
python qme_simple.py minimize molecule.xyz \
    --output optimized.xyz \
    --optimizer BFGS \
    --fmax 0.005 \
    --steps 300 \
    --verbose
```

**Full installation CLI (with UMA):**
```bash
qme minimize molecule.xyz --model uma-4m --device cuda
qme transition-state ts_guess.xyz --trajectory ts.traj
```

### Python API

**Basic usage with mock calculator:**
```python
from qme import QMEOptimizer

# Use mock for testing
qme = QMEOptimizer(use_mock=True)
atoms = qme.load_structure("molecule.xyz")

results = qme.optimize_minimum(
    optimizer="BFGS",
    fmax=0.01,
    steps=200
)

qme.save_structure(results['optimized_atoms'], "optimized.xyz")
print(qme.get_optimization_summary())
```

**Production usage with UMA:**
```python
from qme import QMEOptimizer

# Use real UMA model
qme = QMEOptimizer(model_name="uma-4m", device="cuda")
atoms = qme.load_structure("complex_molecule.xyz")

# Minimum optimization
min_results = qme.optimize_minimum(optimizer="LBFGS")

# Transition state search (requires SELLA)
ts_results = qme.find_transition_state(fmax=0.005)
```

## File Formats

QME supports all ASE-compatible formats:
- **XYZ**: Simple Cartesian coordinates (`.xyz`)
- **CIF**: Crystallographic files (`.cif`) 
- **PDB**: Protein Data Bank (`.pdb`)
- **VASP**: POSCAR/CONTCAR files
- **Gaussian**: Input/output files (`.com`, `.log`)

## Testing

Run the test suite:
```bash
pip install pytest
PYTHONPATH=. python -m pytest tests/ -v
```

## Dependencies Summary

**Core (minimal):**
- `ase` - Atomic Simulation Environment
- `click` - Command line interface
- `numpy` - Numerical computing

**For ML potentials:**
- `torch` - PyTorch ML framework
- `fairchem-core` - UMA and other ML potentials

**For transition states:**
- `sella` - Saddle point optimization

**Development:**
- `pytest` - Testing framework
- `black` - Code formatting
- `isort` - Import sorting

## Troubleshooting

**Import errors for ML dependencies:**
- QME automatically falls back to mock calculator for testing
- Install missing dependencies as needed

**Optimization not converging:**
- Try different optimizers (BFGS, LBFGS, FIRE)
- Adjust `fmax` convergence criterion
- Increase `steps` limit
- Check initial geometry quality

**Memory issues:**
- Use CPU instead of GPU for large systems
- Reduce batch size if applicable

## Development Notes

The package is structured as:
```
qme/
├── __init__.py          # Main package interface
├── core.py              # Core optimization logic
├── uma_potential.py     # UMA ML potential wrapper
├── mock_calculator.py   # Mock calculator for testing
├── cli.py               # Full CLI implementation
└── qme_simple.py        # Simple CLI for testing
```

For development, the mock calculator provides a simple harmonic oscillator model that allows testing all optimization workflows without requiring heavy ML dependencies.