# FAMEX Tutorials

Hands-on tutorials for molecular geometry optimization and transition state searches using FAMEX.

> **Defaults:** CLI and `Explorer` use `backend="uma"` and model `uma-s-1p2`. Tutorials below often show `--backend aimnet2` for a minimal `torch`-only install; omit it when using UMA.

## Table of Contents

1. [Basic Optimization](#basic-optimization)
2. [Transition State Search](#transition-state-search)
3. [Reaction Paths](#reaction-paths)
4. [Quick Reference](#quick-reference)

## Basic Optimization

**target** (`minima`, `ts`, `path`) specifies what you want; **strategy** (`local`, `interpolate`, `neb`, etc.) specifies how to get there.

### Local Optimization

**Command line:**

```bash
famex minima --strategy local molecule.xyz --fmax 0.05
# conflict-free backend:
famex minima --strategy local molecule.xyz --backend aimnet2 --fmax 0.05
```

**Python API:**

```python
import famex

explorer = famex.Explorer.from_file("molecule.xyz", target="minima", strategy="local")
result = explorer.run(fmax=0.05, steps=1000)
explorer.save_structure(result["optimized_atoms"], "optimized.xyz")
```

### Optimizers

| Optimizer | Type | Best For |
|-----------|------|----------|
| `default` | Auto-select | `lbfgs` for minima/path, `sella` for TS |
| `lbfgs` | First-order | Large systems, default for minima |
| `bfgs` | First-order | General purpose |
| `fire` | First-order | Fast initial relaxation |
| `sella` | Second-order | Transition states, robust |
| `trust-krylov` | Second-order | When Hessians are cheap |
| `trust-ncg` | Second-order | Trust-region with nonlinear CG |
| `trust-exact` | Second-order | Trust-region with exact Hessian |
| `newton-cg` | Second-order | Newton-CG method |
| `rfo` | Second-order | Rational Function Optimization for TS |

### Convergence

```bash
# Quick testing
famex minima --strategy local molecule.xyz --fmax 0.1 --steps 100

# Standard (default)
famex minima --strategy local molecule.xyz --fmax 0.05 --steps 1000

# High precision
famex minima --strategy local molecule.xyz --fmax 0.01 --steps 2000
```

### Two-Ended Optimization

```bash
famex minima --strategy interpolate reactant.xyz --product product.xyz --npoints 11
```

## Transition State Search

### Local TS Optimization

```bash
famex ts --strategy local ts_guess.xyz --freq
```

```python
explorer = famex.Explorer.from_file("ts_guess.xyz", target="ts", strategy="local")
result = explorer.run(fmax=0.01, calculate_frequencies=True)
```

### TS from Interpolation

```bash
famex ts --strategy interpolate reactant.xyz --product product.xyz --npoints 15 --freq
```

```python
from ase.io import read

reactant = read("reactant.xyz")
product = read("product.xyz")
explorer = famex.Explorer([reactant, product], target="ts", strategy="interpolate")
result = explorer.run(npoints=15)
```

### Growing String (DE-GSM)

```bash
famex ts --strategy growing_string reactant.xyz --product product.xyz --npoints 20 --step-size 0.1
famex ts --strategy growing_string reactant.xyz --product product.xyz --require-ts --freq
```

See also `examples/growing_string_demo.py`.

### Validation

A valid TS has exactly one significant imaginary frequency:

```bash
famex ts --strategy local ts_guess.xyz --freq
```

```python
freq_result = explorer.calculate_frequencies(result["optimized_atoms"])
if freq_result["is_ts"]:
    print("Valid transition state")
# or explicitly:
n_imag = freq_result["ts_analysis"]["n_imaginary_frequencies"]
```

## Reaction Paths

### NEB / CI-NEB

```bash
famex path --strategy neb reactant.xyz product.xyz --npoints 11
famex path --strategy cineb reactant.xyz product.xyz --spring-constant 5.0
```

```python
explorer = famex.Explorer([reactant, product], target="path", strategy="neb")
result = explorer.run(npoints=11, fmax=0.05)
path = result.get("trajectory", result["optimized_atoms"])
explorer.save_trajectory(path, "neb_path.xyz")
```

### IRC from a transition state

```bash
famex path --strategy irc ts.xyz --direction both
```

See `examples/irc_demo.py`.

### Interpolation only (no optimization)

```bash
famex path --strategy interpolate reactant.xyz product.xyz --npoints 15 --interp idpp
```

## Quick Reference

### Common Commands

```bash
# Minima
famex minima --strategy local molecule.xyz --fmax 0.05

# Transition state
famex ts --strategy interpolate reactant.xyz --product product.xyz --npoints 15 --freq

# NEB path
famex path --strategy neb reactant.xyz product.xyz --npoints 11

# IRC from TS
famex path --strategy irc ts.xyz --direction both

# Cache (AIMNet2 model downloads)
famex cache info
```

### Python API

```python
import famex
from ase.io import read

# Minima
explorer = famex.Explorer.from_file("molecule.xyz", target="minima")
result = explorer.run(fmax=0.05)
explorer.save_structure(result["optimized_atoms"], "optimized.xyz")

# TS from two structures
r, p = read("reactant.xyz"), read("product.xyz")
explorer = famex.Explorer([r, p], target="ts", strategy="interpolate")
result = explorer.run(npoints=15)

# NEB path
explorer = famex.Explorer([r, p], target="path", strategy="neb")
result = explorer.run(npoints=11)
explorer.save_trajectory(result.get("trajectory", result["optimized_atoms"]), "neb_path.xyz")
```

More examples: [`examples/README.md`](../examples/README.md).

---

*Last updated: June 2026*
