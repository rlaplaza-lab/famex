# QME User Guide

Complete guide to using QME for molecular geometry optimization and transition state searches.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Installation](#installation)
3. [Command Line Interface](#command-line-interface)
4. [Python API](#python-api)
5. [Backend Guide](#backend-guide)
6. [Examples](#examples)

## Core Concepts

QME uses a semantic interface with two key concepts:

- **Target**: What you want to obtain (`minima`, `ts`, `path`)
- **Strategy**: How to get there (`local`, `interpolate`, `neb`, `cineb`, `irc`, `growing_string`)

### Target/Strategy Matrix

| Target | Strategy | Description |
|--------|----------|-------------|
| `minima` | `local` | Direct local optimization |
| `minima` | `interpolate` | Minima from interpolated path |
| `ts` | `local` | Local TS search |
| `ts` | `interpolate` | TS guess from interpolation |
| `ts` | `growing_string` | Growing string method (DE-GSM) |
| `path` | `neb` | NEB path optimization |
| `path` | `cineb` | CI-NEB path optimization |
| `path` | `irc` | IRC path from transition state |
| `path` | `interpolate` | Generate path only (no optimization) |

## Installation

### Prerequisites

- Python 3.10 or higher (Python 3.11+ required for TorchSim backends)
- pip package manager

### Quick Installation

```bash
# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install QME with a backend
pip install qme-ml[aimnet2]  # Recommended for beginners
```

### Backend Selection

QME supports multiple machine learning backends. Choose one based on your needs:

| Backend | Best For | Installation |
|---------|----------|--------------|
| `aimnet2` | General use, no conflicts | `pip install qme-ml[aimnet2]` |
| `mace` | High accuracy | `pip install qme-ml[mace]` |
| `uma` | Materials science | `pip install qme-ml[uma]` |
| `orb` | Universal forcefield | `pip install qme-ml[orb]` |
| `torchsim` | Maximum performance | `pip install qme-ml[torchsim]` |

> **Important**: Some backends have dependency conflicts (e.g., UMA vs MACE). Use separate environments or choose one backend per environment.

### Alternative Installations

```bash
# Development installation
git clone https://github.com/rlaplaza-lab/qme.git
cd qme
pip install -e .[dev,aimnet2]

# Minimal installation (mock backend only)
pip install qme-ml
```

## Command Line Interface

QME provides a structured command-line interface organized into three main commands:

- `qme minima` - Minima optimization (outputs single structure)
- `qme ts` - Transition state optimization (outputs single TS)
- `qme path` - Reaction path optimization (outputs trajectories)
- `qme cache` - Cache management

### Global Options

All commands support these common options:

| Option | Default | Description |
|--------|---------|-------------|
| `--backend` | `uma` | Backend: uma\|aimnet2\|mace\|orb\|so3lr\|torchsim\|torchsim_mace\|torchsim_fairchem\|mock |
| `--model-name` | `None` | Model name for backend |
| `--model-path` | `None` | Path to model file (if applicable) |
| `--device` | `None` | Device: cpu\|cuda |
| `--default-charge` | `0` | Default molecular charge |
| `--default-spin` | `1` | Default spin multiplicity |
| `--local-optimizer` | `default` | Local optimizer: default\|lbfgs\|bfgs\|fire\|sella\|trust-krylov\|trust-krylov-ts\|trust-ncg\|trust-exact\|newton-cg |
| `--optimizer-kw` | `None` | Optimizer kwargs as key=value, repeatable |
| `--ts-kw` | `None` | TS optimizer kwargs as key=value, repeatable |
| `--constraints` | `None` | Constraints spec string; e.g., 'fix 0,1; harmonic_bond 2,3 k=5.0' |
| `--verbose`, `-v` | `1` | Verbosity level: -v=quiet, -vv=normal, -vvv=debug |
| `--dry-run` | `False` | Validate inputs and show strategy selection without running |
| `--validate-ts` | `False` | Validate TS structure via frequency analysis after optimization |

### qme minima - Minima Optimization

Optimize molecular structures to find energy minima.

#### Usage

```bash
qme minima --strategy {local,interpolate} INPUT [OPTIONS]
```

#### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `INPUT` | Path | Input XYZ file (required) |

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--strategy` | `local` | Optimization strategy: local\|interpolate |
| `--product` | `None` | Product XYZ for interpolate strategy |
| `--output` | Auto | Output optimized XYZ path |
| `--fmax` | `0.05` | Convergence threshold |
| `--steps` | `1000` | Max optimization steps |
| `--npoints` | `11` | Number of interpolation points (interpolate strategy only) |
| `--interp` | `geodesic` | Interpolation method: linear\|geodesic\|idpp\|quadratic\|spline |

#### Examples

```bash
# Basic local optimization
qme minima --strategy local molecule.xyz

# With specific backend and convergence
qme minima --strategy local molecule.xyz --backend aimnet2 --fmax 0.01

# Two-ended minima search via interpolation
qme minima --strategy interpolate reactant.xyz --product product.xyz --npoints 21

# With constraints
qme minima --strategy local molecule.xyz --constraints "fix 0,1,2"

# Dry run to check strategy selection
qme minima --strategy local molecule.xyz --dry-run
```

#### Output Files

- Local optimization: `{input}.opt.local.xyz`
- Interpolate optimization: `{input}.opt.interpolate.xyz`

### qme ts - Transition State Optimization

Find and optimize transition state structures.

#### Usage

```bash
qme ts --strategy {local,interpolate,growing_string} INPUT [OPTIONS]
```

#### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `INPUT` | Path | Input XYZ file (required) |

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--strategy` | `local` | Optimization strategy: local\|interpolate\|growing_string |
| `--product` | `None` | Product XYZ for interpolate/growing_string strategies |
| `--output` | Auto | Output TS XYZ path |
| `--fmax` | `0.05` | Convergence threshold |
| `--steps` | `1000` | Max optimization steps |
| `--npoints` | `11` | Number of interpolation points (interpolate/growing_string strategies only) |
| `--interp` | `geodesic` | Interpolation method (interpolate strategy only) |
| `--max-images` | `100` | Maximum number of images (growing_string strategy only) |
| `--distance-threshold` | `0.1` | Distance threshold for convergence (growing_string strategy only) |
| `--step-size` | `0.1` | Step size for growing string method (growing_string strategy only) |

#### Examples

```bash
# Basic local TS optimization
qme ts --strategy local ts_guess.xyz

# With validation
qme ts --strategy local ts_guess.xyz --validate-ts

# With custom optimizer
qme ts --strategy local ts_guess.xyz --optimizer trust-krylov-ts --fmax 0.02

# TS from interpolation
qme ts --strategy interpolate reactant.xyz --product product.xyz

# With custom settings
qme ts --strategy interpolate reactant.xyz --product product.xyz --npoints 15 --interp idpp

# Growing string method
qme ts --strategy growing_string reactant.xyz --product product.xyz --npoints 20 --step-size 0.1

# With validation
qme ts --strategy interpolate reactant.xyz --product product.xyz --validate-ts
```

#### Output Files

- Local TS: `{input}.ts.local.xyz`
- Interpolated TS: `{input}.ts.interpolate.xyz`
- Growing string TS: `{input}.ts.gsm.xyz`

### qme path - Reaction Path Optimization

Generate and optimize reaction pathways.

#### Usage

```bash
qme path --strategy {interpolate,neb,cineb,irc} INPUT [OPTIONS]
```

#### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `INPUT` | Path | Input XYZ file (required) |

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--strategy` | `neb` | Path optimization strategy: interpolate\|neb\|cineb\|irc |
| `--product` | `None` | Product XYZ for interpolate/neb/cineb strategies |
| `--output` | Auto | Output trajectory XYZ path |
| `--fmax` | `0.05` | Convergence threshold |
| `--steps` | `1000` | Max optimization steps |
| `--npoints` | `11` | Number of images in path |
| `--interp` | `geodesic` | Initial interpolation method |
| `--spring-constant` | `0.5` | Spring constant for NEB/CI-NEB |
| `--step-size` | `0.1` | IRC step size (IRC strategy only) |
| `--direction` | `both` | Direction: forward\|backward\|both (IRC strategy only) |

#### Examples

```bash
# Raw interpolation
qme path --strategy interpolate reactant.xyz --product product.xyz

# NEB path optimization
qme path --strategy neb reactant.xyz --product product.xyz

# CI-NEB path optimization
qme path --strategy cineb reactant.xyz --product product.xyz

# IRC from transition state
qme path --strategy irc ts.xyz --direction both
```

#### Output Files

- Interpolation: `{input}.interpolate.xyz`
- NEB: `{input}.neb.xyz`
- CI-NEB: `{input}.cineb.xyz`
- IRC: `{input}.irc.xyz`

### qme cache - Cache Management

Manage QME's model cache.

#### Usage

```bash
qme cache {info,clear} [OPTIONS]
```

#### Subcommands

##### info - Cache Information

Display cache information.

```bash
qme cache info
```

##### clear - Clear Cache

Clear the model cache.

```bash
qme cache clear
```

## Python API

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

## Backend Guide

### Backend Overview

| Backend | Description | Installation | Best For |
|---------|-------------|--------------|----------|
| `uma` | Universal Materials Accelerator (Meta AI) | `pip install qme-ml[uma]` | General purpose, materials |
| `aimnet2` | Native PyTorch implementation | `pip install qme-ml[aimnet2]` | Molecules, fast inference |
| `mace` | Foundation models for chemistry | `pip install qme-ml[mace]` | High accuracy, diverse systems |
| `orb` | Orbital Materials universal forcefield | `pip install qme-ml[orb]` | Universal, molecules and materials |
| `so3lr` | SO(3) invariant neural networks | `pip install qme-ml[so3lr]` | Research, custom models |
| `torchsim_*` | TorchSim accelerated backends | `pip install qme-ml[torchsim]` | High performance, GPU |
| `mock` | Harmonic oscillator for testing | Built-in | Testing, development |

### Backend Selection Guide

#### For Beginners
Start with **AIMNet2** - no conflicts, fast, reliable:
```bash
pip install qme-ml[aimnet2]
```

#### For Production Use
Use **UMA** for materials, **MACE** for molecules, or **Orb** for universal coverage:
```bash
# Materials and general purpose
pip install qme-ml[uma]

# High accuracy molecules
pip install qme-ml[mace]

# Universal forcefield (molecules and materials)
pip install qme-ml[orb]
```

#### For Maximum Performance
Use **TorchSim** backends with GPU acceleration:
```bash
pip install qme-ml[torchsim]
qme minima --strategy local molecule.xyz --backend torchsim_mace --device cuda
```

#### For Development/Testing
Use **Mock** backend:
```bash
qme minima --strategy local molecule.xyz --backend mock
```

### Dependency Conflicts

#### Known Conflicts

**UMA vs MACE**: Both depend on `e3nn` but require incompatible versions:
- UMA (fairchem-core) requires `e3nn>=0.5`
- MACE requires `e3nn==0.4.4`

**Solution**: Use separate environments:
```bash
# Environment 1: UMA only
conda create -n qme-uma python=3.12
conda activate qme-uma
pip install qme-ml[uma]

# Environment 2: MACE only
conda create -n qme-mace python=3.12
conda activate qme-mace
pip install qme-ml[mace]
```

### Interpolation Methods

QME supports multiple interpolation strategies for generating reaction pathways between molecular structures.

| Method | Description | Best For |
|--------|-------------|----------|
| `linear` | Simple linear interpolation between coordinates | Quick initial guesses, simple systems |
| `geodesic` | Distance-preserving interpolation with bond length refinement | Chemically reasonable intermediates (default) |
| `idpp` | Image-Dependent Pair Potential interpolation | Large geometry changes, robust pathways |
| `quadratic` | Quadratic curve fitting through start, midpoint, and end | When approximate transition region is known |
| `spline` | Cubic spline interpolation for smooth pathways | Smooth, continuous reaction coordinates |

#### Usage

```bash
# Command line - specify interpolation method
qme minima --strategy interpolate reactant.xyz --product product.xyz --interp idpp
qme path --strategy neb reactant.xyz --product product.xyz --interp spline
qme ts --strategy interpolate reactant.xyz --product product.xyz --interp quadratic

# Python API
explorer = qme.Explorer(atoms=[reactant, product], target="path", strategy="interpolate")
result = explorer.run(method="idpp", npoints=15)
```

## Examples

### Basic Workflows

```bash
# 1. Local minima optimization
qme minima --strategy local molecule.xyz --backend aimnet2

# 2. Transition state search
qme ts --strategy interpolate reactant.xyz --product product.xyz --npoints 15

# 3. Reaction path optimization
qme path --strategy neb reactant.xyz --product product.xyz --npoints 11

# 4. IRC from transition state
qme path --strategy irc ts.xyz --direction both
```

### Python API Examples

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

---

*Last updated: January 2025*
