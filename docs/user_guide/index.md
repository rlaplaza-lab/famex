# User Guide

This section provides comprehensive documentation for using QME's new target/strategy interface effectively.

## Overview

QME (Quick Mechanistic Exploration) uses a semantic interface where you specify what you want to achieve (target) and how to achieve it (strategy). This guide covers all aspects of using QME for your computational chemistry workflows.

## Sections

### [Supported Backends](backends.md)
Detailed information about machine learning backends: UMA, AIMNet2, MACE, SO3LR, and TorchSim.


### [CLI Reference](cli_reference.md)
Complete reference for all command-line interface commands and options.

## Understanding QME's Interface

### Target/Strategy System

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

## Quick Reference

### Common Commands
```bash
# Minima optimization (outputs single structure)
qme minima --strategy local molecule.xyz
qme minima --strategy interpolate reactant.xyz --product product.xyz  # Two-ended minima search

# Transition state optimization (outputs single TS)
qme ts --strategy local ts_guess.xyz                # Local TS optimization
qme ts --strategy interpolate reactant.xyz --product product.xyz  # TS via interpolation
qme ts --strategy growing_string reactant.xyz --product product.xyz      # Growing string method

# Reaction path optimization (outputs trajectories)
qme path --strategy interpolate r.xyz --product p.xyz            # Raw interpolation
qme path --strategy neb r.xyz --product p.xyz                    # NEB path
qme path --strategy cineb r.xyz --product p.xyz                  # CI-NEB path
qme path --strategy irc ts.xyz                         # IRC from transition state

# Cache management
qme cache info
```

### Python Quick Start
```python
import qme

# Local minima optimization
explorer = qme.Explorer.from_file("molecule.xyz", backend="aimnet2")
result = explorer.run(target="minima", strategy="local")

# Transition state from interpolation
explorer = qme.Explorer([reactant, product], target="ts", strategy="interpolate")
result = explorer.run(npoints=15)

# NEB path optimization
explorer = qme.Explorer([reactant, product], target="path", strategy="neb")
result = explorer.run(npoints=11)

# Save results
explorer.save_structure(result['optimized_atoms'], "optimized.xyz")
```

### Backend Selection
```bash
# Available backends
qme minima --strategy local molecule.xyz --backend aimnet2    # AIMNet2 (recommended)
qme minima --strategy local molecule.xyz --backend uma        # UMA (default)
qme minima --strategy local molecule.xyz --backend mace       # MACE
qme minima --strategy local molecule.xyz --backend orb        # Orb
qme minima --strategy local molecule.xyz --backend so3lr      # SO3LR
qme minima --strategy local molecule.xyz --backend mock       # Mock (testing)
```

## Common Workflows

### Local Minima Optimization
```python
# Command line
qme minima --strategy local molecule.xyz --backend aimnet2 --fmax 0.01

# Python
explorer = qme.Explorer.from_file("molecule.xyz", backend="aimnet2")
result = explorer.run(target="minima", strategy="local", fmax=0.01)
```

### Transition State Search
```python
# Command line - local TS optimization
qme ts --strategy local ts_guess.xyz --backend aimnet2

# Command line - TS from interpolation
qme ts --strategy interpolate reactant.xyz --product product.xyz --npoints 15

# Python
explorer = qme.Explorer([reactant, product], target="ts", strategy="interpolate")
result = explorer.run(npoints=15)
```

### Reaction Path Optimization
```python
# Command line - NEB path
qme path --strategy neb reactant.xyz --product product.xyz --npoints 11

# Command line - IRC from TS
qme path --strategy irc ts.xyz --direction both

# Python
explorer = qme.Explorer([reactant, product], target="path", strategy="neb")
result = explorer.run(npoints=11)
```

## Getting Help

Each section provides detailed explanations with examples. For specific issues:

- Check the [Troubleshooting Guide](../reference/troubleshooting.md)
- Review the [FAQ](../reference/faq.md)
