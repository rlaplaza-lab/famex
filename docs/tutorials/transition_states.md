# Transition State Search Tutorial

This tutorial covers transition state (TS) optimization using QME's new interface. You'll learn the main TS search strategies and how to validate your results.

## Learning Objectives

By the end of this tutorial, you will:
- Understand different TS search strategies
- Perform local TS optimization
- Generate TS guesses via interpolation
- Validate TS structures with frequency analysis

## Prerequisites

- Completed [Basic Optimization Tutorial](basic_optimization.md)
- QME installed with at least one backend

## Step 1: Understanding TS Search Strategies

QME provides two main approaches for finding transition states:

| Strategy | Use Case | Description |
|----------|----------|-------------|
| `local` | TS guess available | Direct local TS optimization |
| `interpolate` | Reactant + product | TS guess from interpolation |

## Step 2: Local TS Optimization

When you have a good guess for the transition state structure:

### Command Line Interface

```bash
# Basic local TS optimization
qme ts --strategy local ts_guess.xyz

# With specific backend and optimizer
qme ts --strategy local ts_guess.xyz --backend aimnet2 --optimizer sella

# With validation
qme ts --strategy local ts_guess.xyz --validate-ts
```

### Python API

```python
import qme

# Create Explorer for TS optimization
explorer = qme.Explorer.from_file("ts_guess.xyz", backend="aimnet2")

# Run local TS optimization
result = explorer.run(target="ts", strategy="local")

# Save TS structure
explorer.save_structure(result['optimized_atoms'], "ts_optimized.xyz")

print(f"TS optimization converged: {result['converged']}")
print(f"TS energy: {result['final_energy']:.6f} eV")
```

### TS Optimizers

#### Sella (Default)
- Modern saddle point optimizer
- Uses second-order information
- Good for most TS searches

#### Trust-Krylov TS
- Trust-region method with min-mode following
- Efficient Hessian computation
- Alternative to Sella when Hessians are cheap

## Step 3: TS from Interpolation

When you have reactant and product structures but no TS guess:

### Command Line Interface

```bash
# TS via interpolation
qme ts --strategy interpolate reactant.xyz --product product.xyz

# With specific number of points
qme ts --strategy interpolate reactant.xyz --product product.xyz --npoints 15

# With validation
qme ts --strategy interpolate reactant.xyz --product product.xyz --validate-ts
```

### Python API

```python
import qme

# Create Explorer with reactant and product
explorer = qme.Explorer([reactant, product], target="ts", strategy="interpolate")

# Run TS optimization
result = explorer.run(npoints=15)

# Save TS structure
explorer.save_structure(result['optimized_atoms'], "ts_from_interpolation.xyz")

print(f"TS optimization converged: {result['converged']}")
print(f"TS energy: {result['final_energy']:.6f} eV")
```

## Step 4: Validating Transition States

Always validate your TS structures:

### Frequency Analysis

```python
# Calculate frequencies for the TS
freq_result = explorer.calculate_frequencies(result['optimized_atoms'])

print(f"Frequencies: {freq_result['frequencies']}")
print(f"Imaginary frequencies: {freq_result['imaginary_frequencies']}")

# A TS should have exactly one imaginary frequency
if len(freq_result['imaginary_frequencies']) == 1:
    print("✓ Valid transition state (one imaginary frequency)")
else:
    print("✗ Invalid transition state (wrong number of imaginary frequencies)")
```

### Command Line Validation

```bash
# Automatic validation
qme ts --strategy local ts_guess.xyz --validate-ts

# Manual frequency calculation
qme minima --strategy local ts_structure.xyz --frequencies
```

## Step 5: Complete Example

Here's a complete workflow for finding a transition state:

```python
import qme

# Step 1: Load reactant and product
explorer = qme.Explorer([reactant, product], target="ts", strategy="interpolate")

# Step 2: Find TS via interpolation
result = explorer.run(npoints=15)

# Step 3: Validate TS
freq_result = explorer.calculate_frequencies(result['optimized_atoms'])

# Step 4: Check results
if result['converged'] and len(freq_result['imaginary_frequencies']) == 1:
    print("✓ Successfully found and validated transition state")
    explorer.save_structure(result['optimized_atoms'], "validated_ts.xyz")
else:
    print("✗ TS optimization failed or invalid TS structure")
```

## Troubleshooting

### Common Issues

1. **TS optimization doesn't converge:**
   - Try different optimizer: `--optimizer trust-krylov-ts`
   - Increase steps: `--steps 2000`
   - Check initial TS guess quality

2. **Multiple imaginary frequencies:**
   - TS guess might be too far from saddle point
   - Try local optimization with better initial guess
   - Consider using interpolation method

3. **No imaginary frequencies:**
   - Structure might be a minimum, not a TS
   - Check if you're optimizing the right structure
   - Verify your TS guess

### Best Practices

1. **Start with interpolation** when you have reactant and product
2. **Validate all TS structures** with frequency analysis
3. **Use appropriate backends** for your system type
4. **Check convergence** before analyzing results

## Next Steps

- Try the [Basic Optimization Tutorial](basic_optimization.md) for minima optimization
- Explore different backends in the [Backend Guide](user_guide/backends.md)
- Learn about reaction paths in advanced workflows

---

*Last updated: January 2025*