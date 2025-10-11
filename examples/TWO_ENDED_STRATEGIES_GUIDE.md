# Two-Ended Strategies Guide

This guide explains all the two-ended strategies available in QME, how they work, and what they return. All strategies are fully compatible with 2+ input atom objects.

## Overview

Two-ended strategies operate on two or more ASE Atoms objects to:
1. Generate interpolated reaction paths
2. Optimize these paths using various algorithms
3. Return optimized geometries along the reaction coordinate

The strategies handle multiple input atoms by:
- **2 atoms**: Direct interpolation between reactant and product
- **3+ atoms**: Interpolation between consecutive pairs, then stitching segments together

## Available Strategies

### 1. `twoended:minima` - Two-Ended Minima Optimization

**What it does:**
- Generates an interpolated path between input atoms
- Computes energies along the path
- Identifies local minima (or global minimum if none found)
- Runs local minima optimization on those frames

**How it works:**
1. Creates interpolated path using geodesic interpolation
2. Calculates energies for all path points
3. Finds local minima by comparing each point to its neighbors
4. Optimizes the identified minima using the specified local optimizer
5. Returns the optimized minima geometries

**Returns:**
- **2 atoms input**: Single optimized geometry (the lowest energy minimum)
- **3+ atoms input**: List of optimized geometries (all local minima found)

**Parameters:**
- `npoints`: Number of interpolation points (default: 11)
- `method`: Interpolation method (default: "geodesic")
- `fmax`: Force convergence threshold (default: 0.05)
- `steps`: Maximum optimization steps (default: 1000)
- `local_optimizer_name`: Local optimizer to use (default: "sella")

**Example:**
```python
from qme import Explorer

explorer = Explorer(
    atoms=[reactant, product],
    backend="uma",
    strategy="two-ended",
    target="minima"
)

# Find local minima along the reaction path
result = explorer.run(mode="minima", npoints=11)
```

---

### 2. `twoended:ts` - Two-Ended Transition State Guess

**What it does:**
- Generates an interpolated path between input atoms
- Locates the highest energy frame(s) along the path
- Runs local transition state optimization on those frames
- Returns optimized transition state geometries

**How it works:**
1. Creates interpolated path using geodesic interpolation
2. Calculates energies for all path points
3. Identifies the highest energy frame(s) (potential transition states)
4. Runs local TS optimization using SELLA or other TS optimizer
5. Returns the optimized transition state geometries

**Returns:**
- **2 atoms input**: Single optimized geometry (the highest energy TS)
- **3+ atoms input**: List of optimized geometries (all TS candidates found)

**Parameters:**
- `npoints`: Number of interpolation points (default: 11)
- `method`: Interpolation method (default: "geodesic")
- `fmax`: Force convergence threshold (default: 0.05)
- `steps`: Maximum optimization steps (default: 1000)
- `local_optimizer_name`: Local optimizer to use (default: "sella")

**Example:**
```python
explorer = Explorer(
    atoms=[reactant, product],
    backend="uma",
    strategy="two-ended",
    target="ts"
)

# Find transition state along the reaction path
ts_result = explorer.run(mode="ts", npoints=11)
```

---

### 3. `twoended:neb` - Nudged Elastic Band (NEB)

**What it does:**
- Generates an initial path using geodesic interpolation
- Applies spring forces between adjacent images
- Projects forces perpendicular to the path (nudging)
- Optimizes the entire path using NEB algorithm
- Supports batch evaluation for improved performance

**How it works:**
1. Creates initial interpolated path
2. For each optimization step:
   - Calculates energies and forces for all images
   - Applies spring forces between adjacent images
   - Projects forces perpendicular to reaction path (nudging)
   - Updates positions of all images
3. Continues until convergence or maximum steps reached

**Returns:**
- **Always**: List of optimized Atoms objects representing the NEB path

**Parameters:**
- `npoints`: Number of images in the NEB path (default: 11)
- `method`: Interpolation method for initial path (default: "geodesic")
- `fmax`: Force convergence threshold (default: 0.05)
- `steps`: Maximum optimization steps (default: 1000)
- `spring_constant`: Spring constant for NEB forces (default: 5.0)
- `local_optimizer_name`: Local optimizer to use (default: "sella")

**Example:**
```python
explorer = Explorer(
    atoms=[reactant, product],
    backend="uma",
    strategy="two-ended",
    target="neb"
)

# Run NEB optimization
neb_path = explorer.run(mode="neb", npoints=11, spring_constant=5.0)
```

---

### 4. `twoended:cineb` - Climbing Image NEB (CI-NEB)

**What it does:**
- Enhanced version of NEB with climbing image behavior
- One image actively "climbs" uphill along the reaction coordinate
- More accurate transition state location than regular NEB
- Supports batch evaluation for improved performance

**How it works:**
1. Creates initial interpolated path
2. For each optimization step:
   - Calculates energies and forces for all images
   - Identifies the highest energy image (climbing image)
   - Inverts the parallel component of forces for climbing image
   - Applies regular NEB forces to other images
   - Updates positions of all images
3. Continues until convergence or maximum steps reached

**Key Features:**
- **Climbing Image Selection**: Automatically chooses highest energy image
- **Force Inversion**: Makes climbing image move uphill
- **Energy-Weighted Tangents**: Uses proper tangent calculations
- **Spring Forces**: Maintains path connectivity

**Returns:**
- **Always**: List of optimized Atoms objects representing the CI-NEB path

**Parameters:**
- `npoints`: Number of images in the CI-NEB path (default: 11)
- `method`: Interpolation method for initial path (default: "geodesic")
- `fmax`: Force convergence threshold (default: 0.05)
- `steps`: Maximum optimization steps (default: 1000)
- `spring_constant`: Spring constant for NEB forces (default: 5.0)
- `climb`: Whether to enable climbing behavior (default: True)
- `local_optimizer_name`: Local optimizer to use (default: "sella")

**Example:**
```python
explorer = Explorer(
    atoms=[reactant, product],
    backend="uma",
    strategy="two-ended",
    target="cineb"
)

# Run CI-NEB optimization
cineb_path = explorer.run(
    mode="cineb", 
    npoints=11, 
    climb=True, 
    spring_constant=5.0
)
```

## Multi-Atom Input Handling

All two-ended strategies support multiple input atoms by stitching together interpolated segments:

### 2 Atoms (Reactant → Product)
```
Reactant ----interpolate---- Product
```

### 3 Atoms (Reactant → Intermediate → Product)
```
Reactant ----interpolate---- Intermediate ----interpolate---- Product
                ↓                    ↓
              Segment 1           Segment 2
                ↓                    ↓
              Stitched together into single path
```

### 4+ Atoms (Multiple Intermediates)
```
Reactant → Int1 → Int2 → ... → Product
    ↓        ↓      ↓              ↓
  Seg1    Seg2   Seg3           SegN
    ↓        ↓      ↓              ↓
  All segments stitched into single continuous path
```

## Return Value Consistency

The strategies maintain consistent return patterns:

| Input Atoms | Return Type | Description |
|-------------|-------------|-------------|
| 2 atoms | Single Geometry | One optimized structure |
| 3+ atoms | List of Geometries | Multiple optimized structures |

**Exception**: NEB and CI-NEB always return lists (the full path), regardless of input size.

## Batch Evaluation Support

NEB and CI-NEB strategies support batch evaluation when using compatible calculators (e.g., TorchSim):

**Benefits:**
- Calculate energies and forces for all images simultaneously
- Significant performance improvement for multi-image calculations
- Graceful fallback to individual calculations if batch evaluation fails

**Automatic Detection:**
- Strategies automatically detect if calculator supports batch evaluation
- No user intervention required

## Error Handling

All strategies include robust error handling:

- **Single atom input**: Raises `ValueError` with clear message
- **Empty input**: Raises `ValueError` 
- **Invalid parameters**: Raises appropriate exceptions with helpful messages
- **Calculator failures**: Graceful fallback or clear error reporting

## Performance Tips

1. **Use appropriate `npoints`**: More points = higher accuracy but slower computation
2. **Choose suitable `fmax`**: Tighter convergence = more accurate but slower
3. **Leverage batch evaluation**: Use TorchSim backends for NEB/CI-NEB when possible
4. **Start with fewer steps**: Use smaller `steps` for initial testing, increase for production

## Comparison Summary

| Strategy | Best For | Returns | Key Feature |
|----------|----------|---------|-------------|
| `minima` | Finding stable intermediates | Optimized minima | Local minima identification |
| `ts` | Quick TS estimation | Optimized TS | Highest energy frame optimization |
| `neb` | Smooth reaction paths | Full path | Spring forces + nudging |
| `cineb` | Accurate TS location | Full path | Climbing image behavior |

## Example: Complete Workflow

```python
from qme import Explorer
from ase.build import molecule

# Create reactant and product
h2 = molecule('H2')
h2.set_cell([10, 10, 10])
h2.center()

h_atoms = Atoms('H2', positions=[[0, 0, 0], [2, 0, 0]])
h_atoms.set_cell([10, 10, 10])
h_atoms.center()

# 1. Find minima along path
explorer = Explorer(atoms=[h2, h_atoms], backend="mock", strategy="two-ended")
minima = explorer.run(target="minima", npoints=7)

# 2. Get quick TS estimate
ts_guess = explorer.run(target="ts", npoints=7)

# 3. Run full NEB optimization
neb_path = explorer.run(target="neb", npoints=11, spring_constant=5.0)

# 4. Run CI-NEB for accurate TS
cineb_path = explorer.run(target="cineb", npoints=11, climb=True)

print(f"Found {len(minima)} minima")
print(f"TS guess: {type(ts_guess)}")
print(f"NEB path: {len(neb_path)} images")
print(f"CI-NEB path: {len(cineb_path)} images")
```

This guide covers all aspects of the two-ended strategies in QME. All strategies are fully tested and compatible with 2+ input atom objects.
