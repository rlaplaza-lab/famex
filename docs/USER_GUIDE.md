# QME User Guide

Reference for CLI, Python API, and backends.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Command Line Interface](#command-line-interface)
3. [Python API](#python-api)
4. [Backend Guide](#backend-guide)

## Core Concepts

- **Target**: What you want (`minima`, `ts`, `path`)
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

See [README.md](../README.md) for installation instructions.

> **Note**: Python 3.10+ required. Some backends conflict (e.g., UMA vs MACE) - use separate environments.

## Command Line Interface

- `qme minima` - Minima optimization (outputs single structure)
- `qme ts` - Transition state optimization (outputs single TS)
- `qme path` - Reaction path optimization (outputs trajectories)
- `qme cache` - Cache management

### Global Options

All commands support these common options:

| Option | Default | Description |
|--------|---------|-------------|
| `--backend` | `uma` | Backend: uma\|aimnet2\|mace\|orb\|so3lr\|tblite\|mock |
| `--model-name` | `None` | Model name for backend |
| `--model-path` | `None` | Path to model file (if applicable) |
| `--device` | `None` | Device: cpu\|cuda |
| `--default-charge` | `0` | Default molecular charge |
| `--default-spin` | `1` | Default spin multiplicity |
| `--local-optimizer` | `default` | Local optimizer: default\|lbfgs\|bfgs\|fire\|sella\|trust-krylov\|trust-ncg\|trust-exact\|newton-cg\|rfo (default=auto-select based on target) |
| `--optimizer-kw` | `None` | Optimizer kwargs as key=value, repeatable |
| `--ts-kw` | `None` | TS optimizer kwargs as key=value, repeatable |
| `--constraints` | `None` | Constraints spec string; e.g., 'fix 0,1; harmonic_bond 2,3 k=5.0' |
| `--verbose`, `-v` | `1` | Verbosity level: -v=quiet, -vv=normal, -vvv=debug |
| `--temperature` | `298.15` | Temperature in Kelvin for thermodynamic calculations |
| `--dry-run` | `False` | Validate inputs and show strategy selection without running |
| `--freq`, `--frequencies` | `False` | Perform frequency analysis after optimization (includes thermodynamic properties) |
| `--force-finite-diff-hessian` | `False` | Force use of finite difference hessians for TS optimizers and frequency calculations |

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
| `--require-ts/--allow-ts` | `--allow-ts` | Require a validated first-order saddle (raises an error if GSM/refinement fails) |

#### Examples

```bash
# Basic local TS optimization
qme ts --strategy local ts_guess.xyz

# With frequency analysis
qme ts --strategy local ts_guess.xyz --freq

# With custom optimizer
qme ts --strategy local ts_guess.xyz --local-optimizer rfo --fmax 0.02

# TS from interpolation
qme ts --strategy interpolate reactant.xyz --product product.xyz

# With custom settings
qme ts --strategy interpolate reactant.xyz --product product.xyz --npoints 15 --interp idpp

# Growing string method
qme ts --strategy growing_string reactant.xyz --product product.xyz --npoints 20 --step-size 0.1

# Strict validation (raises if the TS is not first-order)
qme ts --strategy growing_string reactant.xyz --product product.xyz --require-ts

# With frequency analysis
qme ts --strategy interpolate reactant.xyz --product product.xyz --freq
```

#### Output Files

- Local TS: `{input}.ts.local.xyz`
- Interpolated TS: `{input}.ts.interpolate.xyz`
- Growing string TS: `{input}.ts.gsm.xyz`

### qme path - Reaction Path Optimization

Generate and optimize reaction pathways.

#### Usage

```bash
qme path --strategy {interpolate,neb,cineb,irc} STRUCTURES... [OPTIONS]
```

#### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `STRUCTURES...` | Path(s) | Structure file(s) (required). Can be:<br>- Multiple files: `reactant.xyz product.xyz [intermediate.xyz ...]`<br>- Single multi-frame XYZ: all frames used as path guess<br>- Single single-frame XYZ: for IRC strategy |

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--strategy` | `neb` | Path optimization strategy: interpolate\|neb\|cineb\|irc |
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
# Raw interpolation (two structures)
qme path --strategy interpolate reactant.xyz product.xyz

# NEB path optimization (two structures)
qme path --strategy neb reactant.xyz product.xyz

# NEB with multiple intermediate structures
qme path --strategy neb reactant.xyz intermediate.xyz product.xyz --npoints 11

# CI-NEB path optimization
qme path --strategy cineb reactant.xyz product.xyz

# IRC from transition state (single structure)
qme path --strategy irc ts.xyz --direction both
```

#### Output Files

- Interpolation: `{input}.path.interpolate.xyz`
- NEB: `{input}.path.neb.xyz`
- CI-NEB: `{input}.path.cineb.xyz`
- IRC: `{input}.path.irc.xyz`

### qme cache - Cache Management

```bash
qme cache info   # Display cache information
qme cache clear  # Clear the model cache
```

## Python API

### Explorer Class

```python
from qme import Explorer

explorer = Explorer(
    atoms,                    # Atoms or Sequence[Atoms]
    backend="uma",            # Backend name
    target="minima",          # Target: minima|ts|path
    strategy="local",         # Strategy: local|neb|cineb|interpolate|growing_string|irc
    device=None,              # Device: cpu|cuda (auto-detected if None)
    local_optimizer="default", # Optimizer (auto-selects based on target)
    # ... see Python API documentation for full parameter list
)
```

**Targets:** `minima`, `ts`, `path`

**Strategies:** `local`, `interpolate`, `neb`, `cineb`, `irc`, `growing_string` (see [Target/Strategy Matrix](#targetstrategy-matrix))

**Optimizers:** `default` (auto-selects), first-order (`lbfgs`, `bfgs`, `fire`), second-order (`sella`, `trust-krylov`, `trust-ncg`, `trust-exact`, `newton-cg`, `rfo`)

### Key Methods

```python
# Run optimization
result = explorer.run(fmax=0.05, steps=1000, calculate_frequencies=False)

# Load from file
explorer = Explorer.from_file("molecule.xyz", backend="aimnet2", target="minima")

# Calculate frequencies
freq_result = explorer.calculate_frequencies(atoms=None, delta=0.01, method="auto")

# Save results
explorer.save_structure(result["optimized_atoms"], "output.xyz")
explorer.save_trajectory(trajectory_list, "path.xyz")
```

## Backend Guide

| Backend | Installation | Best For | Notes |
|---------|--------------|----------|-------|
| `aimnet2` | `pip install torch torch-cluster` | Beginners, molecules | No conflicts, fast |
| `uma` | `pip install fairchem-core` | Materials science | Conflicts with MACE |
| `mace` | `pip install mace-torch` | High accuracy molecules | Conflicts with UMA |
| `orb` | `pip install orb-models` | Universal coverage | Molecules and materials |
| `tblite` | `pip install tblite` | Fast semi-empirical | Quick calculations |
| `so3lr` | `pip install so3lr` | Research | Custom models |
| `mock` | Built-in | Testing | Development only |

### Dependency Conflicts

UMA and MACE conflict due to incompatible `e3nn` versions. Use separate environments:

```bash
# UMA environment
conda create -n qme-uma python=3.12
conda activate qme-uma && pip install qme-ml fairchem-core

# MACE environment
conda create -n qme-mace python=3.12
conda activate qme-mace && pip install qme-ml mace-torch
```

### Interpolation Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `geodesic` | Distance-preserving with bond refinement | Default, chemically reasonable |
| `idpp` | Image-Dependent Pair Potential | Large geometry changes |
| `linear` | Simple linear interpolation | Quick initial guesses |
| `quadratic` | Quadratic curve fitting | Known transition region |
| `spline` | Cubic spline interpolation | Smooth pathways |

Usage: `qme path --strategy neb reactant.xyz product.xyz --interp idpp`

---

*Last updated: January 2025*
