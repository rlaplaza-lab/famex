# Growing String Method (GSM) for Transition State Search

## Overview

The Growing String Method (GSM) is a two-ended strategy for finding transition states between reactant and product structures. Unlike interpolation-based methods (e.g., NEB) that create a fixed path between endpoints, GSM iteratively "grows" strings from both ends until they meet near the transition state.

## Implementation

The implementation in QME follows the double-ended growing string method (DE-GSM) style as seen in pysisyphus:

1. **Initialize**: Start with reactant and product structures
2. **Optimize endpoints** (optional): Relax reactant/product to local minima
3. **Grow strings**: Iteratively add nodes from both ends
   - New nodes are placed along steepest descent (negative gradient) direction
   - Each new node is optimized perpendicular to the path
4. **Identify TS**: Highest energy image is selected as TS guess
5. **Refine TS** (optional): Local TS optimization of the guess

## Usage

### Python API

```python
import qme
from ase.io import read

# Load reactant and product
reactant = read("reactant.xyz")
product = read("product.xyz")

# Create explorer
explorer = qme.Explorer([reactant, product], backend="aimnet2")

# Run growing string method
from qme.core.twoended_strategies import twoended_growing_string_runner

result = twoended_growing_string_runner(
    [reactant, product],
    npoints=15,              # Maximum number of images
    explorer=explorer,
    fmax=0.05,              # Force convergence threshold
    steps=100,              # Maximum growing iterations
    step_size=0.1,          # Step size for adding nodes (Å)
    optimize_endpoints=True, # Optimize R and P first
    refine_ts=True,         # Refine TS with local optimization
)

# Access results
ts_structure = result["optimized_atoms"]
full_trajectory = result["trajectory"]
forward_string = result["forward_string"]
backward_string = result["backward_string"]
converged = result["strings_met"]
```

### Using the Demo Script

```bash
# Basic usage with mock backend
python examples/growing_string_demo.py

# With custom structures
python examples/growing_string_demo.py \
    --reactant reactant.xyz \
    --product product.xyz \
    --backend aimnet2 \
    --npoints 20 \
    --optimize-endpoints \
    --refine-ts

# Help
python examples/growing_string_demo.py --help
```

### Strategy Registration

The growing string method is registered with multiple aliases:

- `"ts:growing_string"` - Full name with target prefix
- `"growing_string"` - Short name
- `"gsm"` - Abbreviation

Check available strategies:

```python
explorer = qme.Explorer(atoms, backend="aimnet2")
strategies = explorer.list_strategies()
print(strategies["ts:growing_string"])
# {'type': 'two-ended', 'description': 'Growing string method for TS search (DE-GSM style)'}
```

## Parameters

### Core Parameters

- `atoms_list`: Two Atoms objects (reactant and product)
- `npoints` (default=15): Maximum number of images in final path
- `fmax` (default=0.05): Force convergence threshold (eV/Å)
- `steps` (default=100): Maximum growing iterations

### Growing Parameters

- `step_size` (default=0.1): Step size for adding new nodes (Å)
- `distance_threshold` (default=0.5): Distance for strings to be considered "met" (Å)

### Optimization Parameters

- `optimize_endpoints` (default=True): Optimize R/P to local minima before growing
- `refine_ts` (default=True): Refine TS with local optimization after identification
- `local_optimizer_name` (default="sella"): Optimizer for TS refinement

### Explorer Parameters

- `explorer`: Explorer instance for calculator and constraint management
- `**kwargs`: Additional arguments passed through

## Return Value

Dictionary with:

- `optimized_atoms`: TS structure (Atoms object)
- `trajectory`: Full path from reactant to product (list of Atoms)
- `forward_string`: Images grown from reactant (list)
- `backward_string`: Images grown from product (list)
- `converged`: Whether TS refinement converged (bool)
- `strings_met`: Whether strings successfully met (bool)
- `strategy`: Strategy name ("twoended_growing_string_runner")

## Algorithm Details

The implementation follows these key principles:

1. **Node placement**: New nodes are added by moving along the negative gradient (downhill) from the previous node
2. **Perpendicular optimization**: Each new node is optimized with forces perpendicular to the path (simplified implementation uses limited LBFGS steps)
3. **Convergence criteria**: Strings are considered to have met when the distance between their tips is below `distance_threshold`
4. **TS identification**: The highest energy image in the combined path is selected as the TS guess

## Comparison with Other Methods

| Method | Fixed Path | Requires Hessian | String Growth | Best For |
|--------|-----------|------------------|---------------|----------|
| Interpolation | ✓ | ✗ | ✗ | Quick TS guess |
| NEB | ✓ | ✗ | ✗ | Path refinement |
| CI-NEB | ✓ | ✗ | ✗ | Accurate TS |
| Growing String | ✗ | ✗ | ✓ | Exploratory TS search |
| Local TS | ✗ | ✓ | ✗ | TS refinement |

## Limitations

1. **Simplified perpendicular optimization**: Current implementation uses a simplified approach with limited optimization steps
2. **Mock backend**: The mock backend may not provide meaningful forces, limiting demo effectiveness
3. **No reparametrization**: Unlike full GSM implementations, nodes are not reparametrized to maintain equal spacing
4. **Calculator dependency**: Requires properly attached calculators on all nodes

## References

- Peters, B., et al. "A growing string method for determining transition states: Comparison to the nudged elastic band and string methods." J. Chem. Phys. 120, 7877 (2004).
- Pysisyphus documentation: https://pysisyphus.readthedocs.io/

## Testing

Unit tests are available in `tests/unit/test_growing_string.py`:

```bash
pytest tests/unit/test_growing_string.py -v
```

All tests validate:
- Basic functionality with mock backend
- Input validation (requires exactly 2 structures)
- Endpoint optimization option
- TS refinement option
- Strategy registration
- Maximum images limit
- Distance threshold convergence
