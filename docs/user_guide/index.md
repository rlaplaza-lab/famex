# User Guide

This section provides comprehensive documentation for using QME effectively.

## Overview

QME (Quick Mechanistic Exploration) is designed to make molecular optimization and transition state searches accessible through both command-line and Python interfaces. This guide covers all aspects of using QME for your computational chemistry workflows.

## Sections

### [Command Line Interface](cli.md)
Complete reference for the `qme` command-line tool, including all options and examples.

### [Python API](python_api.md)
Programmatic interface for integrating QME into your Python workflows and scripts.

### [Supported Backends](backends.md)
Detailed information about machine learning backends: UMA, AIMNet2, MACE, SO3LR, and TorchSim.

### [File Formats](file_formats.md)
Supported input/output formats and file handling conventions.

### [Optimization Strategies](optimization.md)
Local and two-ended optimization methods, optimizers, and convergence criteria.

### [Transition State Searches](transition_states.md)
Advanced transition state finding methods including NEB and interpolation strategies.

### [Advanced Features](advanced.md)
Constraints, batch processing, custom workflows, and performance optimization.

## Quick Reference

### Common Commands
```bash
# Basic optimization
qme opt molecule.xyz

# Transition state optimization
qme tsopt ts_guess.xyz

# Two-ended optimization
qme opt reactant.xyz --product product.xyz

# Cache management
qme cache info
```

### Python Quick Start
```python
import qme

# Load and optimize a structure
explorer = qme.Explorer.from_file("molecule.xyz", backend="uma")
result = explorer.optimize_minimum()
explorer.save_structure(result['optimized_atoms'], "optimized.xyz")
```

### Backend Selection
```bash
# Available backends
qme opt molecule.xyz --backend uma        # UMA (default)
qme opt molecule.xyz --backend aimnet2    # AIMNet2
qme opt molecule.xyz --backend mace       # MACE
qme opt molecule.xyz --backend so3lr      # SO3LR
qme opt molecule.xyz --backend mock       # Mock (testing)
```

## Getting Help

Each section provides detailed explanations with examples. For specific issues:

- Check the [Troubleshooting Guide](../reference/troubleshooting.md)
- Browse the [FAQ](../reference/faq.md)
- See [Error Reference](../reference/errors.md) for error explanations
