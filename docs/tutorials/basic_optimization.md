# Basic Optimization Tutorial

Learn the fundamentals of molecular geometry optimization using QME.

## Learning Objectives

By the end of this tutorial, you will:
- Understand how to optimize molecular structures with QME
- Know how to choose appropriate backends and settings
- Be able to interpret optimization results
- Understand the difference between local and two-ended optimization

## Prerequisites

- QME installed with at least one backend: `pip install qme-ml-ml[aimnet2]`
- Basic understanding of molecular structure
- Familiarity with XYZ file format

## Part 1: Your First Optimization

### Step 1: Create a Test Molecule

Create a file called `water.xyz`:

```
3
Water molecule
O    0.000000    0.000000    0.117283
H    0.000000    0.758602   -0.469132
H    0.000000   -0.758602   -0.469132
```

This represents a water molecule with slightly distorted geometry that we'll optimize.

### Step 2: Basic Optimization

Run your first optimization:

```bash
qme opt water.xyz
```

**Expected output:**
```
🔧 QME Optimization Starting
Backend: aimnet2 (AIMNet2)
Device: cpu
Optimizer: sella
Convergence: 0.050 eV/Å

📁 Loading structure from: water.xyz
🧮 Attaching calculator...
⚡ Starting optimization...

Step    Energy (eV)    Max Force (eV/Å)
0       -14.123        0.089
5       -14.127        0.034
8       -14.128        0.042  ✅ Converged!

💾 Saved optimized structure to: water_opt_aimnet2.xyz
⏱️  Total time: 3.2 seconds
```

### Step 3: Examine the Results

Check the output file:
```bash
cat water_opt_aimnet2.xyz
```

You'll see the optimized geometry with lower energy and forces.

## Part 2: Understanding Backends

### Step 4: Try Different Backends

Compare results with different ML potentials:

```bash
# AIMNet2 (default, fast and reliable)
qme opt water.xyz --backend aimnet2

# UMA (good for materials and molecules)
qme opt water.xyz --backend uma

# MACE (high accuracy)
qme opt water.xyz --backend mace

# Mock (for testing, harmonic potential)
qme opt water.xyz --backend mock
```

**Note**: You may need to install additional backends:
```bash
pip install qme-ml-ml[uma]    # For UMA
pip install qme-ml-ml[mace]   # For MACE
```

### Step 5: Backend Comparison

Create a comparison script `compare_backends.py`:

```python
import qme
from ase.io import read

# Load the initial structure
atoms = read("water.xyz")

backends = ["aimnet2", "mock"]  # Add others if available
results = {}

for backend in backends:
    try:
        print(f"\n--- Testing {backend} ---")
        explorer = qme.Explorer.from_atoms(atoms.copy(), backend=backend)
<<<<<<< HEAD
        result = explorer.run(mode="minima")
=======
        result = explorer.optimize_minima()
>>>>>>> 20afbbd (feat: Implement hardcoded TS optimization restrictions and clean API)
        
        results[backend] = {
            'final_energy': result['final_energy'],
            'n_steps': result['n_steps'],
            'converged': result['converged']
        }
        
        print(f"Energy: {result['final_energy']:.6f} eV")
        print(f"Steps: {result['n_steps']}")
        print(f"Converged: {result['converged']}")
        
    except Exception as e:
        print(f"Error with {backend}: {e}")

print("\n--- Summary ---")
for backend, data in results.items():
    print(f"{backend}: {data['final_energy']:.6f} eV in {data['n_steps']} steps")
```

Run it:
```bash
python compare_backends.py
```

## Part 3: Optimization Settings

### Step 6: Convergence Criteria

Try different convergence thresholds:

```bash
# Loose convergence (faster)
qme opt water.xyz --fmax 0.1

# Default convergence
qme opt water.xyz --fmax 0.05

# Tight convergence (slower but more accurate)
qme opt water.xyz --fmax 0.01
```

### Step 7: Different Optimizers

QME supports multiple optimization algorithms:

```bash
# SELLA optimizer (default, good for TS)
qme opt water.xyz --optimizer sella

# L-BFGS (memory efficient)
qme opt water.xyz --optimizer lbfgs

# BFGS (robust)
qme opt water.xyz --optimizer bfgs

# FIRE (simple and fast)
qme opt water.xyz --optimizer fire
```

### Step 8: Maximum Steps

Control the optimization length:

```bash
# Short optimization
qme opt water.xyz --steps 50

# Long optimization
qme opt water.xyz --steps 2000
```

## Part 4: Larger Molecules

### Step 9: Optimize Benzene

Create `benzene.xyz`:
```
12
Benzene molecule
C    0.000000    1.396000    0.000000
C    1.209000    0.698000    0.000000
C    1.209000   -0.698000    0.000000
C    0.000000   -1.396000    0.000000
C   -1.209000   -0.698000    0.000000
C   -1.209000    0.698000    0.000000
H    0.000000    2.486000    0.000000
H    2.153000    1.243000    0.000000
H    2.153000   -1.243000    0.000000
H    0.000000   -2.486000    0.000000
H   -2.153000   -1.243000    0.000000
H   -2.153000    1.243000    0.000000
```

Optimize it:
```bash
qme opt benzene.xyz --backend aimnet2
```

Notice how larger molecules take more time and steps to converge.

### Step 10: GPU Acceleration

If you have a CUDA-capable GPU:

```bash
# Use GPU acceleration
qme opt benzene.xyz --backend mace --device cuda

# Compare with CPU
qme opt benzene.xyz --backend mace --device cpu
```

## Part 5: Two-Ended Optimization

### Step 11: Optimize Between Two Structures

Create a distorted benzene `benzene_distorted.xyz`:
```
12
Distorted benzene
C    0.100000    1.496000    0.100000
C    1.309000    0.798000   -0.100000
C    1.109000   -0.598000    0.100000
C   -0.100000   -1.296000   -0.100000
C   -1.309000   -0.598000    0.100000
C   -1.109000    0.798000   -0.100000
H    0.100000    2.586000    0.100000
H    2.253000    1.343000   -0.100000
H    2.053000   -1.143000    0.100000
H   -0.100000   -2.386000   -0.100000
H   -2.253000   -1.143000    0.100000
H   -2.053000    1.343000   -0.100000
```

Now optimize from distorted to regular benzene:
```bash
qme opt benzene_distorted.xyz --product benzene.xyz
```

This creates an interpolated path between the two structures and optimizes it.

## Part 6: Python API

### Step 12: Using QME Programmatically

Create `python_optimization.py`:

```python
import qme
from ase.io import read, write
from ase.build import molecule

# Method 1: From file
print("=== Optimizing from file ===")
explorer = qme.Explorer.from_file("water.xyz", backend="aimnet2")
<<<<<<< HEAD
result = explorer.run(mode="minima")
=======
result = explorer.optimize_minima()
>>>>>>> 20afbbd (feat: Implement hardcoded TS optimization restrictions and clean API)

print(f"Initial energy: {result['initial_energy']:.6f} eV")
print(f"Final energy: {result['final_energy']:.6f} eV")
print(f"Energy change: {result['final_energy'] - result['initial_energy']:.6f} eV")
print(f"Optimization steps: {result['n_steps']}")
print(f"Converged: {result['converged']}")

# Save the result
explorer.save_structure(result['optimized_atoms'], "water_python.xyz")

# Method 2: From ASE atoms
print("\n=== Optimizing from ASE atoms ===")
atoms = molecule('NH3')  # Ammonia
explorer2 = qme.Explorer.from_atoms(atoms, backend="aimnet2")
result2 = explorer2.run(mode="minima")

print(f"Ammonia final energy: {result2['final_energy']:.6f} eV")

# Method 3: With custom settings
print("\n=== Custom settings ===")
explorer3 = qme.Explorer.from_file(
    "water.xyz", 
    backend="aimnet2",
    device="cpu"
)

result3 = explorer3.run(
    mode="minima",
    fmax=0.01,          # Tight convergence
    steps=500,          # Maximum steps
    optimizer='lbfgs'   # Different optimizer
)

print(f"Tight optimization energy: {result3['final_energy']:.6f} eV")
```

Run it:
```bash
python python_optimization.py
```

### Step 13: Error Handling

Create `error_handling.py`:

```python
import qme

def safe_optimization(filename, backend="aimnet2"):
    """Safely optimize a structure with error handling."""
    try:
        explorer = qme.Explorer.from_file(filename, backend=backend)
<<<<<<< HEAD
        result = explorer.run(mode="minima")
=======
        result = explorer.optimize_minima()
>>>>>>> 20afbbd (feat: Implement hardcoded TS optimization restrictions and clean API)
        
        if result['converged']:
            print(f"✅ Optimization converged in {result['n_steps']} steps")
            print(f"Final energy: {result['final_energy']:.6f} eV")
            return result
        else:
            print("⚠️ Optimization did not converge")
            return None
            
    except FileNotFoundError:
        print(f"❌ File {filename} not found")
        return None
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        return None

# Test with valid file
result = safe_optimization("water.xyz")

# Test with invalid file
result = safe_optimization("nonexistent.xyz")

# Test with invalid backend
try:
    explorer = qme.Explorer.from_file("water.xyz", backend="invalid_backend")
except Exception as e:
    print(f"❌ Invalid backend: {e}")
```

## Understanding Results

### Energy Units
- QME reports energies in **electronvolts (eV)**
- Forces are in **eV/Å**
- Typical optimization reduces energy by 0.01-0.1 eV for small molecules

### Convergence Criteria
- **fmax**: Maximum force on any atom (default: 0.05 eV/Å)
- Lower fmax = tighter convergence = more accurate but slower
- Typical values: 0.1 (loose), 0.05 (default), 0.01 (tight)

### Optimization Steps
- Simple molecules: 5-20 steps
- Complex molecules: 20-100+ steps
- If >500 steps, consider different optimizer or looser convergence

## Common Issues and Solutions

### Problem: "Backend not available"
```
Error: Backend 'uma' not available
```
**Solution**: Install the backend dependencies:
```bash
pip install qme-ml-ml[uma]
```

### Problem: Slow optimization
**Solutions**:
- Use looser convergence: `--fmax 0.1`
- Try different optimizer: `--optimizer lbfgs`
- Use faster backend: `--backend aimnet2`
- Use GPU if available: `--device cuda`

### Problem: Optimization doesn't converge
**Solutions**:
- Increase max steps: `--steps 1000`
- Try different optimizer: `--optimizer bfgs`
- Check input structure for errors
- Use looser convergence temporarily

### Problem: Unrealistic results
**Solutions**:
- Check input file format
- Verify atom coordinates are reasonable
- Try different backend for comparison
- Compare with known reference structures

## Next Steps

Now that you understand basic optimization:

1. **[Transition State Finding](transition_states.md)** - Learn to find transition states
2. **[Frequency Analysis](frequency_analysis.md)** - Calculate vibrational properties
3. **[Batch Processing](batch_processing.md)** - Process multiple structures
4. **[User Guide](../user_guide/index.md)** - Comprehensive reference

## Summary

You've learned:
- ✅ How to run basic optimizations with QME
- ✅ How to choose and compare different backends
- ✅ How to adjust convergence criteria and optimization settings
- ✅ How to use both CLI and Python API
- ✅ How to handle common issues

**Key takeaways**:
- Start with AIMNet2 backend for reliability
- Use default settings first, then adjust as needed
- Always check convergence before trusting results
- GPU acceleration can significantly speed up calculations
- Error handling is important for automated workflows

Continue to the next tutorial to learn about transition state searches!
