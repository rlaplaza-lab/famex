# QME: Quick Mechanistic Exploration

**QME** provides a unified interface for molecular geometry optimization using machine learning potentials. It supports minima optimization, transition state searches, and reaction path calculations through both a command-line interface and Python API.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Installation

```bash
pip install qme-ml
```

For backend-specific installations (uma, aimnet2, mace, orb, so3lr, torchsim, tblite), use extras:
```bash
pip install "qme-ml[aimnet2]"  # or [uma], [mace], etc.
```

## CLI Reference

QME provides three main commands: `minima`, `ts`, and `path`.

### Common Options

These options are available for all commands:

```
--backend TEXT              Backend: uma|so3lr|aimnet2|mace|orb|torchsim|torchsim_mace|
                           torchsim_fairchem|tblite|mock [default: uma]
--model-name TEXT          Model name for backend
--model-path TEXT          Path to model file (if applicable)
--device TEXT              Device: cpu|cuda
--default-charge INTEGER   Default molecular charge [default: 0]
--default-spin INTEGER     Default spin multiplicity [default: 1]
--local-optimizer TEXT     Local optimizer: default|lbfgs|bfgs|fire|sella|trust-krylov|
                           trust-krylov-ts|trust-ncg|trust-exact|newton-cg [default: default]
--optimizer-kw TEXT        Optimizer kwargs as key=value, repeatable
--ts-kw TEXT              TS optimizer kwargs as key=value, repeatable
--constraints TEXT         Constraints spec string; e.g., 'fix 0,1; harmonic_bond 2,3 k=5.0'
-v, --verbose             Verbosity level: -v=quiet, -vv=normal, -vvv=debug [default: 1]
--dry-run                 Validate inputs and show strategy selection without running
--freq, --frequencies     Perform frequency analysis after optimization
--temperature FLOAT        Temperature in Kelvin for thermodynamic calculations [default: 298.15]
```

### qme minima

Optimize molecular structures to local minima.

```bash
qme minima [OPTIONS] INPUT
```

**Options:**

```
--strategy [local|interpolate]  Optimization strategy [default: local]
--product PATH                  Product structure (required for interpolate strategy)
--output PATH                   Output optimized XYZ path
--fmax FLOAT                    Convergence threshold [default: 0.05]
--steps INTEGER                 Max optimization steps [default: 1000]
--npoints INTEGER              Number of interpolation points (interpolate strategy) [default: 11]
--interp [linear|geodesic|idpp|quadratic|spline]  Interpolation method [default: geodesic]
```

**Examples:**

```bash
# Local minima optimization
qme minima --strategy local molecule.xyz

# Minima from interpolated path
qme minima --strategy interpolate reactant.xyz --product product.xyz --npoints 21
```

### qme ts

Optimize transition state structures.

```bash
qme ts [OPTIONS] INPUT
```

**Options:**

```
--strategy [local|interpolate|growing_string]  Optimization strategy [default: local]
--product PATH                                 Product structure (required for interpolate/growing_string)
--output PATH                                  Output TS XYZ path
--fmax FLOAT                                   Convergence threshold [default: 0.05]
--steps INTEGER                                Max optimization steps [default: 1000]
--npoints INTEGER                             Number of interpolation points [default: 11]
--interp [linear|geodesic|idpp|quadratic|spline]  Interpolation method [default: geodesic]
--max-images INTEGER                          Maximum images (growing_string) [default: 100]
--distance-threshold FLOAT                    Distance threshold (growing_string) [default: 0.1]
--step-size FLOAT                             Step size (growing_string) [default: 0.1]
```

**Examples:**

```bash
# Local TS optimization
qme ts --strategy local ts_guess.xyz --local-optimizer sella

# TS from interpolation
qme ts --strategy interpolate reactant.xyz --product product.xyz

# Growing string method
qme ts --strategy growing_string reactant.xyz --product product.xyz --npoints 20
```

### qme path

Generate and optimize reaction pathways.

```bash
qme path [OPTIONS] INPUT
```

**Options:**

```
--strategy [interpolate|neb|cineb|irc]  Path optimization strategy [default: neb]
--product PATH                          Product structure (required for interpolate/neb/cineb)
--output PATH                           Output trajectory XYZ path
--fmax FLOAT                            Convergence threshold [default: 0.05]
--steps INTEGER                         Max optimization steps [default: 1000]
--npoints INTEGER                      Number of images in path [default: 11]
--interp [linear|geodesic|idpp|quadratic|spline]  Initial interpolation method [default: geodesic]
--spring-constant FLOAT                Spring constant for NEB/CI-NEB [default: 0.5]
--step-size FLOAT                      IRC step size (amu^1/2 * Angstrom) [default: 0.1]
--direction [forward|backward|both]    Direction to follow from TS (IRC) [default: both]
```

**Examples:**

```bash
# Raw interpolation (no optimization)
qme path --strategy interpolate reactant.xyz --product product.xyz --npoints 15

# NEB path optimization
qme path --strategy neb reactant.xyz --product product.xyz --npoints 11 --spring-constant 5.0

# CI-NEB path optimization
qme path --strategy cineb reactant.xyz --product product.xyz --npoints 11

# IRC from transition state
qme path --strategy irc ts.xyz --direction both --steps 100
```

### qme cache

Manage model cache.

```bash
qme cache info    # Show cache information
qme cache clear   # Clear model cache
```

## Python API Reference

### Explorer Class

The `Explorer` class is the main entry point for the Python API.

```python
from qme import Explorer

explorer = Explorer(
    atoms,                    # Atoms or Sequence[Atoms]
    backend="uma",            # Backend name
    model_name=None,          # Model name (optional)
    model_path=None,          # Path to model file (optional)
    device=None,              # Device: cpu|cuda (auto-detected if None)
    default_charge=0,         # Default molecular charge
    default_spin=1,           # Default spin multiplicity
    local_optimizer="default", # Optimizer name
    optimizer_kwargs=None,    # Dict of optimizer kwargs
    strategy="local",         # Strategy: local|neb|cineb|interpolate|growing_string|irc
    target="minima",          # Target: minima|ts|path
    ts_kwargs=None,           # Dict of TS optimizer kwargs
    constraints=None,         # Constraint specification
    initial_hessian=None,     # Initial Hessian matrix (optional)
    auto_register=True,       # Auto-register strategies
    verbose=1,                # Verbosity level: 0=quiet, 1=normal, 2=verbose
    profile=False             # Enable profiling
)
```

**Available Backends:**
- `uma` - Universal Materials Accelerator
- `aimnet2` - AIMNet2 neural network potential
- `mace` - MACE foundation models
- `so3lr` - SO3LR equivariant neural network
- `orb` - Orb universal forcefield
- `torchsim_mace` - TorchSim-accelerated MACE
- `torchsim_uma` - TorchSim-accelerated UMA
- `tblite` - TBLite semi-empirical
- `mock` - Mock calculator for testing

**Available Targets:**
- `minima` - Find local minimum
- `ts` - Find transition state
- `path` - Find reaction pathway

**Available Strategies by Target:**

| Target  | Strategy         | Description                           |
|---------|-----------------|---------------------------------------|
| minima  | local           | Direct local optimization             |
| minima  | interpolate     | Minima from interpolated path         |
| ts      | local           | Local TS search                       |
| ts      | interpolate     | TS guess from interpolation           |
| ts      | growing_string  | Growing string method (DE-GSM)        |
| path    | neb             | NEB path optimization                 |
| path    | cineb           | CI-NEB path optimization              |
| path    | irc             | IRC path from transition state        |
| path    | interpolate     | Generate path only (no optimization)  |

**Available Optimizers:**
- First-order (gradient-based): `lbfgs`, `bfgs`, `fire`
- Second-order (Hessian-based): `sella`, `trust-krylov`, `trust-krylov-ts`, `trust-ncg`, `trust-exact`, `newton-cg`

### Key Methods

```python
# Run optimization
result = explorer.run(
    fmax=0.05,                    # Convergence threshold
    steps=1000,                   # Max steps
    calculate_frequencies=False,  # Perform frequency analysis
    temperature=298.15,           # Temperature for thermodynamics
    **kwargs                      # Strategy-specific kwargs
)

# Load from file
explorer = Explorer.from_file(
    "molecule.xyz",
    backend="aimnet2",
    target="minima",
    strategy="local"
)

# Calculate frequencies
freq_result = explorer.calculate_frequencies(
    atoms=None,              # Uses atoms_list[0] if None
    delta=0.01,              # Finite difference step size
    method="auto",           # Method: auto|numerical
    temperature=298.15,      # Temperature (K)
    save_hessian=True        # Include Hessian in results
)

# Save results
explorer.save_structure(result["optimized_atoms"], "output.xyz")
explorer.save_trajectory(trajectory_list, "path.xyz")

# Introspection
strategies = explorer.list_strategies()       # List all strategies
explanation = explorer.explain_run()          # Explain strategy selection
```

## Quick Start Examples

### Command Line

```bash
# Optimize water molecule
echo "3
Water
O 0.0 0.0 0.0
H 0.0 0.0 1.0
H 0.0 1.0 0.0" > water.xyz

qme minima --strategy local water.xyz --backend aimnet2

# Find transition state between reactant and product
qme ts --strategy interpolate reactant.xyz --product product.xyz

# Generate NEB path
qme path --strategy neb reactant.xyz --product product.xyz --npoints 11
```

### Python API

```python
from qme import Explorer

# Minima optimization
explorer = Explorer.from_file("molecule.xyz", backend="aimnet2", target="minima")
result = explorer.run(fmax=0.05, steps=1000)
explorer.save_structure(result["optimized_atoms"], "optimized.xyz")

# Transition state search
explorer = Explorer.from_file("ts_guess.xyz", backend="mace", target="ts", strategy="local")
result = explorer.run(fmax=0.01)

# NEB path with two structures
from ase.io import read
reactant = read("reactant.xyz")
product = read("product.xyz")

explorer = Explorer(
    atoms=[reactant, product],
    backend="uma",
    target="path",
    strategy="neb"
)
result = explorer.run(npoints=11, fmax=0.05, spring_constant=5.0)
explorer.save_trajectory(result["trajectory"], "neb_path.xyz")

# With frequency analysis
result = explorer.run(
    fmax=0.05,
    calculate_frequencies=True,
    temperature=298.15
)
print(f"Frequencies: {result['frequency_analysis']['frequencies']}")
print(f"ZPE: {result['frequency_analysis']['zero_point_energy']} eV")
```

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

