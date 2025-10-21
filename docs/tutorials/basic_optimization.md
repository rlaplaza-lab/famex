# Basic Optimization Tutorial

This tutorial covers the fundamentals of molecular geometry optimization using QME's new target/strategy interface.

## Learning Objectives

By the end of this tutorial, you will:
- Understand QME's target/strategy system
- Perform local minima optimization using both CLI and Python API
- Compare different optimizers (BFGS, LBFGS, FIRE)
- Control convergence criteria and optimization settings
- Save and analyze optimization results

## Prerequisites

- QME installed with at least one backend
- Basic understanding of molecular structures
- Familiarity with command line or Python

## Step 1: Understanding Targets and Strategies

QME uses a semantic interface where you specify:
- **Target**: What you want to achieve (`minima`, `ts`, `path`)
- **Strategy**: How to achieve it (`local`, `interpolate`, `neb`, etc.)

For basic optimization, we use:
- Target: `minima` (find energy minimum)
- Strategy: `local` (direct local optimization)

## Step 2: Prepare Your Structure

Let's create a simple test structure - a water molecule:

```bash
# Create water.xyz
cat > water.xyz << EOF
3
Water molecule
O    0.000000    0.000000    0.117283
H    0.000000    0.758602   -0.469132
H    0.000000   -0.758602   -0.469132
EOF
```

## Step 3: Basic Local Optimization

### Command Line Interface

```bash
# Basic optimization (uses default settings)
qme minima --strategy local water.xyz

# With specific backend
qme minima --strategy local water.xyz --backend aimnet2

# With custom convergence criteria
qme minima --strategy local water.xyz --fmax 0.01 --steps 1000
```

### Python API

```python
import qme

# Create Explorer instance
explorer = qme.Explorer.from_file("water.xyz", backend="aimnet2")

# Run optimization
result = explorer.run(target="minima", strategy="local")

# Save results
explorer.save_structure(result['optimized_atoms'], "water_optimized.xyz")

# Print results
print(f"Optimization converged: {result['converged']}")
print(f"Final energy: {result['final_energy']:.6f} eV")
print(f"Steps taken: {result['steps_taken']}")
```

## Step 4: Comparing Optimizers

QME supports multiple optimization algorithms. Let's compare them:

### BFGS (Broyden-Fletcher-Goldfarb-Shanno)

```bash
# Command line
qme minima --strategy local water.xyz --optimizer bfgs

# Python
result = explorer.run(target="minima", strategy="local",
                     local_optimizer="bfgs")
```

**Characteristics:**
- Good general-purpose optimizer
- Uses gradient and Hessian approximation
- Memory intensive for large systems

### LBFGS (Limited-memory BFGS)

```bash
# Command line
qme minima --strategy local water.xyz --optimizer lbfgs

# Python
result = explorer.run(target="minima", strategy="local",
                     local_optimizer="lbfgs")
```

**Characteristics:**
- Memory-efficient version of BFGS
- Good for large systems
- Default optimizer for minima optimization

### FIRE (Fast Inertial Relaxation Engine)

```bash
# Command line
qme minima --strategy local water.xyz --optimizer fire

# Python
result = explorer.run(target="minima", strategy="local",
                     local_optimizer="fire")
```

**Characteristics:**
- Fast relaxation for initial structure preparation
- Good for removing bad contacts
- May not find true minimum

### Sella (Transition State Optimizer)

```bash
# Command line
qme minima --strategy local water.xyz --optimizer sella

# Python
result = explorer.run(target="minima", strategy="local",
                     local_optimizer="sella")
```

**Characteristics:**
- Designed for transition states but works for minima
- Uses second-order information
- More robust but slower

## Step 5: Controlling Convergence

### Force Convergence (`fmax`)

Controls how small forces must be for convergence:

```bash
# Loose convergence (faster)
qme minima --strategy local water.xyz --fmax 0.1

# Standard convergence (default)
qme minima --strategy local water.xyz --fmax 0.05

# Tight convergence (slower, more accurate)
qme minima --strategy local water.xyz --fmax 0.01
```

### Maximum Steps

```bash
# Allow more steps for difficult optimizations
qme minima --strategy local water.xyz --steps 2000

# Quick optimization for testing
qme minima --strategy local water.xyz --steps 100
```

### Python API with Custom Settings

```python
# Create Explorer with custom settings
explorer = qme.Explorer.from_file(
    "water.xyz",
    backend="aimnet2",
    local_optimizer="lbfgs"
)

# Run with custom convergence criteria
result = explorer.run(
    target="minima",
    strategy="local",
    fmax=0.01,
    steps=1000
)
```

## Step 6: Two-Ended Minima Optimization

For finding minima along a reaction path between two structures:

```bash
# Command line - interpolate between reactant and product
qme minima --strategy interpolate reactant.xyz --product product.xyz --npoints 11

# Python
explorer = qme.Explorer([reactant, product], target="minima", strategy="interpolate")
result = explorer.run(npoints=11, method="geodesic")
```

## Step 7: Analyzing Results

### Understanding Output Files

QME creates descriptive output files:
- `water.opt.local.xyz` - Local optimization result
- `water.opt.interpolate.xyz` - For two-ended optimizations

### Result Dictionary Structure

```python
result = {
    'optimized_atoms': <ASE Atoms object>,
    'final_energy': -76.123456,  # eV
    'converged': True,
    'steps_taken': 45,
    'strategy': 'minima:local',
    'max_force': 0.008,  # Maximum force at convergence
    'optimizer_info': {...}  # Optimizer-specific information
}
```

### Saving and Loading Results

```python
# Save optimized structure
explorer.save_structure(result['optimized_atoms'], "optimized.xyz")

# Save with specific format
explorer.save_structure(result['optimized_atoms'], "optimized.cif", format="cif")

# Load structure for further analysis
new_explorer = qme.Explorer.from_file("optimized.xyz")
```

## Step 8: Practical Examples

### Example 1: Small Molecule Optimization

```python
import qme

# Optimize methane
explorer = qme.Explorer.from_file("methane.xyz", backend="aimnet2")
result = explorer.run(target="minima", strategy="local", fmax=0.01)

print(f"Methane optimized energy: {result['final_energy']:.6f} eV")
```

### Example 2: Batch Optimization

```python
import qme
from pathlib import Path

# Optimize all XYZ files in directory
xyz_files = list(Path(".").glob("*.xyz"))

for xyz_file in xyz_files:
    explorer = qme.Explorer.from_file(xyz_file, backend="aimnet2")
    result = explorer.run(target="minima", strategy="local")

    # Save with descriptive name
    output_name = f"{xyz_file.stem}_optimized.xyz"
    explorer.save_structure(result['optimized_atoms'], output_name)

    print(f"{xyz_file.name} -> {output_name} (E = {result['final_energy']:.6f} eV)")
```

### Example 3: Optimization with Constraints

```bash
# Fix specific atoms during optimization
qme minima --strategy local molecule.xyz --constraints "fix 0,1,2"

# Harmonic constraints
qme minima --strategy local molecule.xyz --constraints "harmonic_bond 0,1 k=5.0"
```

```python
# Python API with constraints
explorer = qme.Explorer.from_file("molecule.xyz", constraints="fix 0,1,2")
result = explorer.run(target="minima", strategy="local")
```

## Troubleshooting

### Optimization Not Converging

**Symptoms:**
- `converged: False` in results
- Large maximum forces
- Optimization stops at step limit

**Solutions:**
1. Increase maximum steps: `--steps 2000`
2. Loosen convergence: `--fmax 0.1`
3. Try different optimizer: `--optimizer sella`
4. Check input structure quality

### Unrealistic Energies

**Symptoms:**
- Very high or very low energies
- Energies that don't make chemical sense

**Solutions:**
1. Verify backend compatibility with your system
2. Check charge/spin settings: `--default-charge 0 --default-spin 1`
3. Try different backend
4. Verify input structure

### Memory Issues

**Symptoms:**
- Out of memory errors
- Slow optimization

**Solutions:**
1. Use LBFGS instead of BFGS: `--optimizer lbfgs`
2. Use CPU instead of GPU: `--device cpu`
3. Reduce system size
4. Use mock backend for testing: `--backend mock`

## Best Practices

1. **Start with loose convergence** (`fmax=0.1`) for initial exploration
2. **Use LBFGS** for large systems to save memory
3. **Check input structures** for unrealistic geometries
4. **Use appropriate backends** for your chemical system
5. **Save intermediate results** for long optimizations
6. **Monitor convergence** and adjust settings as needed

## Next Steps

Now that you understand basic optimization, explore:

1. **[Transition States Tutorial](transition_states.md)** - Learn TS search techniques
2. **[Backend Guide](../user_guide/backends.md)** - Explore different ML backends
3. **[CLI Reference](../user_guide/cli_reference.md)** - Complete command reference

## Quick Reference

### Common Commands
```bash
# Basic optimization
qme minima --strategy local molecule.xyz --backend aimnet2 --fmax 0.01

# Two-ended optimization
qme minima --strategy interpolate reactant.xyz --product product.xyz --npoints 11

# With constraints
qme minima --strategy local molecule.xyz --constraints "fix 0,1,2"
```

### Python Quick Start
```python
explorer = qme.Explorer.from_file("molecule.xyz", backend="aimnet2")
result = explorer.run(target="minima", strategy="local", fmax=0.01)
explorer.save_structure(result['optimized_atoms'], "optimized.xyz")
```

---

*Last updated: January 2025*
