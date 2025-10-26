<!-- c54ca036-1b86-4e7c-9690-5c2a69ce1d8b 02367fd4-a970-4edc-bb81-4f233b882f9a -->
# Refactor UMA Hessian Implementation to Match MACE

## Overview

Update UMA's analytical Hessian calculation to follow MACE's approach using vector-Jacobian products (VJP) with `torch.autograd.grad` and `torch.vmap`, then strengthen test coverage.

## Implementation Steps

### 1. Update UMA Hessian Implementation

**File**: `qme/potentials/uma_potential.py`

Current approach uses `torch.autograd.functional.hessian` for direct double differentiation. Replace with MACE's VJP approach:

- Create helper function `_compute_hessian_vmap` similar to MACE's implementation
- First compute forces with gradients enabled
- Use `torch.vmap` to vectorize VJP computation over identity matrix
- Add fallback to loop-based implementation if vmap fails
- Maintain same sign convention: Hessian = -∂forces/∂positions = ∂²E/∂positions²

Key changes in `get_hessian()` method:

1. Enable gradients on positions before force calculation
2. Compute forces using predictor
3. Call `_compute_hessian_vmap(forces, positions)`
4. Handle shape and symmetry as before

### 2. Enhance Test Suite

**File**: `tests/unit/test_hessian_consistency.py`

Strengthen the existing tests:

- Tighten tolerances for analytical vs finite difference comparison (currently rtol=1e-1, atol=1e-1)
- Target: rtol=1e-3, atol=1e-3 for better numerical agreement
- Add element-wise comparison statistics (max error, mean error, RMS error)
- Test multiple finite difference step sizes (0.001, 0.005, 0.01) to validate convergence
- Test on multiple molecular geometries (water, ammonia, methane)
- Add test for Hessian symmetry check (analytical should be perfectly symmetric)
- Add performance comparison (analytical should be much faster than FD)
- Add eigenvalue/frequency comparison for physically meaningful validation

### 3. Run Full Test Suite

Execute tests in py312 conda environment:

```bash
conda activate py312
pytest tests/unit/test_hessian_consistency.py -v
pytest tests/ -v  # Full test suite
```

## Key Technical Details

**MACE's VJP approach** (from their codebase):

```python
def compute_hessians_vmap(forces, positions):
    forces_flatten = forces.view(-1)
    def get_vjp(v):
        return torch.autograd.grad(-1 * forces_flatten, positions, v,
                                   retain_graph=True, create_graph=False)
    I_N = torch.eye(num_elements).to(forces.device)
    gradient = torch.vmap(get_vjp, chunk_size=chunk_size)(I_N)[0]
    return gradient
```

**Benefits**:

- More memory efficient with chunking for large molecules
- Matches MACE's proven implementation
- Potentially better numerical stability
- Aligns with established ML potential practices

## Files Modified

- `qme/potentials/uma_potential.py` - Hessian implementation
- `tests/unit/test_hessian_consistency.py` - Enhanced tests
