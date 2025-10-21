# CLI Reference

Complete reference for all QME command-line interface commands and options.

## Overview

QME provides a structured command-line interface organized into three main commands:

- `qme minima` - Minima optimization (outputs single structure)
- `qme ts` - Transition state optimization (outputs single TS)
- `qme path` - Reaction path optimization (outputs trajectories)
- `qme cache` - Cache management

## Global Options

All commands support these common options:

| Option | Default | Description |
|--------|---------|-------------|
| `--backend` | `uma` | Backend: uma\|aimnet2\|mace\|orb\|so3lr\|torchsim\|torchsim_mace\|torchsim_fairchem\|mock |
| `--model-name` | `None` | Model name for backend |
| `--model-path` | `None` | Path to model file (if applicable) |
| `--device` | `None` | Device: cpu\|cuda |
| `--default-charge` | `0` | Default molecular charge |
| `--default-spin` | `1` | Default spin multiplicity |
| `--optimizer` | `bfgs` | Local optimizer: lbfgs\|bfgs\|fire\|sella\|trust-krylov\|trust-krylov-ts\|trust-ncg\|trust-exact\|newton-cg |
| `--optimizer-kw` | `None` | Optimizer kwargs as key=value, repeatable |
| `--ts-kw` | `None` | TS optimizer kwargs as key=value, repeatable |
| `--constraints` | `None` | Constraints spec string; e.g., 'fix 0,1; harmonic_bond 2,3 k=5.0' |
| `--verbose`, `-v` | `1` | Verbosity level: -v=quiet, -vv=normal, -vvv=debug |
| `--dry-run` | `False` | Validate inputs and show strategy selection without running |
| `--validate-ts` | `False` | Validate TS structure via frequency analysis after optimization |

## qme minima - Minima Optimization

Optimize molecular structures to find energy minima.

### Usage

```bash
qme minima --strategy {local,interpolate} INPUT [OPTIONS]
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `INPUT` | Path | Input XYZ file (required) |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--strategy` | `local` | Optimization strategy: local\|interpolate |
| `--product` | `None` | Product XYZ for interpolate strategy |
| `--output` | Auto | Output optimized XYZ path |
| `--fmax` | `0.05` | Convergence threshold |
| `--steps` | `1000` | Max optimization steps |
| `--npoints` | `11` | Number of interpolation points (interpolate strategy only) |
| `--interp` | `geodesic` | Interpolation method: linear\|geodesic\|idpp\|quadratic\|spline |

### Examples

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

### Output Files

- Local optimization: `{input}.opt.local.xyz`
- Interpolate optimization: `{input}.opt.interpolate.xyz`

## qme ts - Transition State Optimization

Find and optimize transition state structures.

### Usage

```bash
qme ts --strategy {local,interpolate,growing_string} INPUT [OPTIONS]
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `INPUT` | Path | Input XYZ file (required) |

### Options

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

### Examples

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

### Output Files

- Local TS: `{input}.ts.local.xyz`
- Interpolated TS: `{input}.ts.interpolate.xyz`
- Growing string TS: `{input}.ts.gsm.xyz`

## qme path - Reaction Path Optimization

Generate and optimize reaction pathways.

### Usage

```bash
qme path --strategy {interpolate,neb,cineb,irc} INPUT [OPTIONS]
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `INPUT` | Path | Input XYZ file (required) |

### Options

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

### Examples

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

### Output Files

- Interpolation: `{input}.interpolate.xyz`
- NEB: `{input}.neb.xyz`
- CI-NEB: `{input}.cineb.xyz`
- IRC: `{input}.irc.xyz`

## qme cache - Cache Management

Manage QME's model cache.

### Usage

```bash
qme cache {info,clear} [OPTIONS]
```

### Subcommands

#### info - Cache Information

Display cache information.

```bash
qme cache info
```

**Examples:**
```bash
# Show cache information
qme cache info
```

#### clear - Clear Cache

Clear the model cache.

```bash
qme cache clear
```

**Examples:**
```bash
# Clear all cached models
qme cache clear
```

## Command Examples

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

### Advanced Workflows

```bash
# Two-ended minima search with custom interpolation
qme minima --strategy interpolate reactant.xyz --product product.xyz --interp idpp --npoints 21

# Growing string method with validation
qme ts --strategy growing_string reactant.xyz --product product.xyz --npoints 20 --validate-ts

# CI-NEB with custom spring constant
qme path --strategy cineb reactant.xyz --product product.xyz --spring-constant 1.0 --fmax 0.01

# Constrained optimization
qme minima --strategy local molecule.xyz --constraints "fix 0,1,2; harmonic_bond 3,4 k=5.0"
```

### Backend-Specific Examples

```bash
# AIMNet2 backend
qme minima --strategy local molecule.xyz --backend aimnet2

# MACE backend with GPU
qme minima --strategy local molecule.xyz --backend mace --device cuda

# TorchSim backend for maximum performance
qme minima --strategy local molecule.xyz --backend torchsim_mace --device cuda

# Mock backend for testing
qme minima --strategy local molecule.xyz --backend mock
```

## Output File Naming

QME uses descriptive output file names:

| Command | Input | Output Pattern |
|---------|-------|----------------|
| `qme minima --strategy local` | `molecule.xyz` | `molecule.opt.local.xyz` |
| `qme minima --strategy interpolate` | `reactant.xyz` | `reactant.opt.interpolate.xyz` |
| `qme ts --strategy local` | `ts_guess.xyz` | `ts_guess.ts.local.xyz` |
| `qme ts --strategy interpolate` | `reactant.xyz` | `reactant.ts.interpolate.xyz` |
| `qme ts --strategy growing_string` | `reactant.xyz` | `reactant.ts.gsm.xyz` |
| `qme path --strategy interpolate` | `reactant.xyz` | `reactant.interpolate.xyz` |
| `qme path --strategy neb` | `reactant.xyz` | `reactant.neb.xyz` |
| `qme path --strategy cineb` | `reactant.xyz` | `reactant.cineb.xyz` |
| `qme path --strategy irc` | `ts.xyz` | `ts.irc.xyz` |

## Verbosity Levels

Control output verbosity with `-v` flags:

- `qme command` - Normal output (default)
- `qme -v command` - Quiet output (minimal)
- `qme -vv command` - Normal output (same as default)
- `qme -vvv command` - Debug output (verbose)

## Dry Run Mode

Use `--dry-run` to validate inputs and show strategy selection without running:

```bash
# Check what strategy would be used
qme minima --strategy local molecule.xyz --dry-run

# Check strategy with custom parameters
qme ts --strategy interpolate reactant.xyz --product product.xyz --dry-run
```

## Help and Documentation

Get help for any command:

```bash
# General help
qme --help

# Command help
qme minima --help
qme ts --help
qme path --help
```

## Common Issues and Solutions

### Backend Not Available

```bash
# Error: Backend 'xyz' not available
# Solution: Install backend dependencies
pip install qme-ml[backend_name]
```

### Dependency Conflicts

```bash
# Error: pip dependency conflicts
# Solution: Use separate environments
conda create -n qme-uma python=3.12
conda activate qme-uma
pip install qme-ml[uma]
```

### Optimization Not Converging

```bash
# Solution: Increase steps or loosen convergence
qme minima --strategy local molecule.xyz --steps 2000 --fmax 0.1
```

### CUDA Out of Memory

```bash
# Solution: Use CPU or reduce system size
qme minima --strategy local molecule.xyz --device cpu
```

## Related Documentation

- [Getting Started Guide](../getting_started.md) - Installation and first steps
- [Tutorials](../tutorials/index.md) - Step-by-step guides
- [Backend Guide](backends.md) - Backend selection and usage

---

*Last updated: January 2025*
