# QME Tutorials

Hands-on tutorials for molecular geometry optimization and transition state searches using QME.

## Table of Contents

1. [Basic Optimization](#basic-optimization)
2. [Transition State Search](#transition-state-search)
3. [Quick Reference](#quick-reference)

## Basic Optimization

QME uses a target/strategy interface: **target** (`minima`, `ts`, `path`) specifies what you want, **strategy** (`local`, `interpolate`, `neb`, etc.) specifies how to get there.

### Local Optimization

**Command line:**
```bash
qme minima --strategy local molecule.xyz --backend aimnet2 --fmax 0.05
```

**Python API:**
```python
import qme
explorer = qme.Explorer.from_file("molecule.xyz", backend="aimnet2", target="minima")
result = explorer.run(fmax=0.05, steps=1000)
explorer.save_structure(result['optimized_atoms'], "optimized.xyz")
```

### Optimizers

| Optimizer | Type | Best For |
|-----------|------|----------|
| `lbfgs` | First-order | Large systems, default for minima |
| `bfgs` | First-order | General purpose |
| `fire` | First-order | Fast initial relaxation |
| `sella` | Second-order | Transition states, robust |
| `trust-krylov` | Second-order | When Hessians are cheap |

### Convergence

```bash
# Quick testing
qme minima --strategy local molecule.xyz --fmax 0.1 --steps 100

# Standard (default)
qme minima --strategy local molecule.xyz --fmax 0.05 --steps 1000

# High precision
qme minima --strategy local molecule.xyz --fmax 0.01 --steps 2000
```

### Two-Ended Optimization

```bash
qme minima --strategy interpolate reactant.xyz --product product.xyz --npoints 11
```

## Transition State Search

### Local TS Optimization

```bash
qme ts --strategy local ts_guess.xyz --backend aimnet2 --freq
```

```python
explorer = qme.Explorer.from_file("ts_guess.xyz", backend="aimnet2", target="ts")
result = explorer.run(fmax=0.01)
```

### TS from Interpolation

```bash
qme ts --strategy interpolate reactant.xyz --product product.xyz --npoints 15 --freq
```

```python
explorer = qme.Explorer([reactant, product], target="ts", strategy="interpolate")
result = explorer.run(npoints=15)
```

### Validation

Always validate TS structures with frequency analysis - a valid TS has exactly one imaginary frequency:

```bash
qme ts --strategy local ts_guess.xyz --freq
```

```python
freq_result = explorer.calculate_frequencies(result['optimized_atoms'])
if len(freq_result['imaginary_frequencies']) == 1:
    print("✓ Valid transition state")
```

## Quick Reference

### Common Commands

```bash
# Minima optimization
qme minima --strategy local molecule.xyz --backend aimnet2 --fmax 0.05

# Transition state search
qme ts --strategy interpolate reactant.xyz --product product.xyz --npoints 15 --freq

# NEB path
qme path --strategy neb reactant.xyz --product product.xyz --npoints 11

# IRC from TS
qme path --strategy irc ts.xyz --direction both
```

### Python API

```python
import qme

# Minima
explorer = qme.Explorer.from_file("molecule.xyz", backend="aimnet2", target="minima")
result = explorer.run(fmax=0.05)
explorer.save_structure(result['optimized_atoms'], "optimized.xyz")

# TS
explorer = qme.Explorer([reactant, product], backend="aimnet2", target="ts")
result = explorer.run(npoints=15)

# NEB path
explorer = qme.Explorer([reactant, product], target="path", strategy="neb")
result = explorer.run(npoints=11)
explorer.save_trajectory(result["trajectory"], "neb_path.xyz")
```

---

*Last updated: January 2025*
