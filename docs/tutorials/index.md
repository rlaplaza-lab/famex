# Tutorials

Step-by-step tutorials for common QME workflows using the new target/strategy interface.

## Tutorial Overview

Our tutorials are organized by complexity and use case, each building on previous knowledge:

### Core Tutorials

1. **[Basic Optimization](basic_optimization.md)** - Learn QME's target/strategy system and local optimization
2. **[Transition States](transition_states.md)** - Master TS search techniques with Sella and interpolation

## Tutorial Format

Each tutorial includes:

- **Learning objectives** - What you'll accomplish
- **Prerequisites** - Required knowledge and setup
- **Step-by-step instructions** - Detailed walkthrough with examples
- **Code examples** - Copy-paste ready commands and Python code
- **Expected output** - What results to expect
- **Troubleshooting** - Common issues and solutions
- **Best practices** - Tips for success
- **Next steps** - How to build on the tutorial

## Getting Started

### Prerequisites

Before starting the tutorials, ensure you have:

1. **QME installed** with at least one backend:
   ```bash
   pip install qme-ml[aimnet2]  # Recommended for beginners
   ```

2. **Example files** available:
   ```bash
   # Download tutorial files
   git clone https://github.com/rlaplaza-lab/qme.git
   cd qme/examples/example_files/
   ```

3. **Working environment**:
   ```bash
   # Test your installation
   qme --help
   qme minima --help
   ```

### Quick Start

If you're new to QME, start with the **[Getting Started Guide](../getting_started.md)** for installation and basic usage, then proceed to the tutorials.

### Estimated Time

- **Basic Optimization**: 20-30 minutes
- **Transition States**: 30-45 minutes

## Tutorial Highlights

### New Target/Strategy Interface

All tutorials use QME's new semantic interface:

```python
# Clear, intuitive interface
explorer = qme.Explorer(atoms, target="minima", strategy="local")
result = explorer.run()

# Explicit mode specification
result = explorer.run(mode="ts:interpolate")
```

### CLI Command Structure

Learn the new organized CLI commands:

```bash
# Minima optimization
qme minima --strategy local molecule.xyz
qme minima --strategy interpolate reactant.xyz --product product.xyz

# Transition state optimization
qme ts --strategy local ts_guess.xyz
qme ts --strategy interpolate reactant.xyz --product product.xyz
```

## Tutorial Examples

### Basic Optimization
```python
# Local minima optimization
explorer = qme.Explorer.from_file("molecule.xyz", backend="aimnet2")
result = explorer.run(target="minima", strategy="local", fmax=0.01)
```

### Transition States
```python
# TS from interpolation
explorer = qme.Explorer([reactant, product], target="ts", strategy="interpolate")
result = explorer.run(npoints=15)
```

### Advanced Workflows
```python
# Constrained optimization with analysis
explorer = qme.Explorer.from_file("molecule.xyz", constraints="fix 0,1,2")
result = explorer.run(target="minima", strategy="local")
freq_result = explorer.calculate_frequencies(result['optimized_atoms'])
```

## Support and Feedback

### Getting Help

If you encounter issues with the tutorials:

1. **Check prerequisites** - Ensure your environment is set up correctly
2. **Review troubleshooting sections** - Each tutorial includes common issues
3. **Search documentation** - Use the search function to find relevant info
4. **Ask for help** - Use GitHub Discussions for questions

### Improving Tutorials

We welcome feedback on tutorials:

- **Report issues** - If instructions don't work as expected
- **Suggest improvements** - Ideas for clarity or additional content
- **Contribute examples** - Share your own use cases
- **Request topics** - What tutorials would be helpful?

## Related Resources

### Documentation

- **[Getting Started Guide](../getting_started.md)** - Installation and first steps
- **[User Guide](../user_guide/index.md)** - Comprehensive reference
- **[Backend Guide](../user_guide/backends.md)** - Choose optimal backends
- **[CLI Reference](../user_guide/cli_reference.md)** - Complete command reference

### External Resources

- **ASE Documentation** - [https://wiki.fysik.dtu.dk/ase/](https://wiki.fysik.dtu.dk/ase/)
- **SELLA Optimizer** - [https://github.com/zadorlab/sella](https://github.com/zadorlab/sella)
- **Machine Learning Potentials** - Background reading on ML methods

## Quick Reference

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

### Common Commands
```bash
# Basic optimization
qme minima --strategy local molecule.xyz --backend aimnet2

# Transition state optimization
qme ts --strategy interpolate reactant.xyz --product product.xyz

# Help and options
qme --help
qme minima --help
qme ts --help
qme path --help
```

### Python Quick Start
```python
import qme

# Basic optimization
explorer = qme.Explorer.from_file("molecule.xyz", backend="aimnet2")
result = explorer.run(target="minima", strategy="local")

# Save results
explorer.save_structure(result['optimized_atoms'], "optimized.xyz")
```

Ready to get started? Begin with **[Basic Optimization](basic_optimization.md)**!

---

*Last updated: January 2025*
