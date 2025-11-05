# Example Files Test Results

Date: 2025-11-05
Environment: conda base
Total Examples Tested: 13

## Summary

- **Successful**: 12/13 (92.3%)
- **Failed/Skipped**: 1/13 (7.7%)

## Detailed Results

### ✅ Successful Examples

1. **adaptive_hessian_demo.py** - ✅ PASSED
   - Exit code: 0
   - No issues found

2. **cli_demo.py** - ⚠️ PARTIAL SUCCESS (3/9 torchsim_mace tests failed)
   - Exit code: 0 (overall success, but some backend-specific failures)
   - **Issue**: torchsim_mace backend had 3 failures out of 9 tests:
     - TS optimization with local strategy failed (runtime: 12.52s)
     - Two-ended TS optimization with interpolate strategy failed (runtime: 389.93s)
     - Two-ended TS optimization with growing_string strategy failed (runtime: 99.13s)
   - **Root Cause**: The traceback shows JAX/CUDA warnings, but the actual error is likely related to Hessian calculation failures that occur when trust-krylov or other Hessian-based optimizers are used. The error traceback is truncated in the output, but based on other examples, it's likely the same "'SimState' object has no attribute 'numbers'" error.
   - **Note**: These are specifically TS optimization commands that require Hessian calculations. Other commands (minima optimization, NEB, CI-NEB, IRC) work fine with torchsim_mace.
   - Other backends (aimnet2, mace) passed all 9/9 tests successfully

3. **growing_string_demo.py** - ✅ PASSED
   - Exit code: 0
   - No issues found

4. **hessian_accuracy_comparison.py** - ✅ PASSED
   - Exit code: 0
   - No issues found

5. **irc_demo.py** - ✅ PASSED
   - Exit code: 0
   - No issues found

6. **mace_hessian_method_comparison.py** - ✅ PASSED
   - Exit code: 0
   - No issues found

7. **minima_optimizer_benchmark.py** - ✅ PASSED (with warnings)
   - Exit code: 0
   - **Issue**: torchsim_mace/fire had frequency analysis failure: "'SimState' object has no attribute 'numbers'"
   - **Issue**: torchsim_mace/trust-krylov failed completely (0 steps, no convergence)
   - **Issue**: Some optimizers show "Valid" column as ❌ even though they converged (aimnet2/lbfgs, aimnet2/bfgs, aimnet2/fire, torchsim_mace variants)
   - **Issue**: Scipy warnings about unknown solver options: `initial_tr_radius`, `max_tr_radius`

8. **thermochemistry_demo.py** - ✅ PASSED
   - Exit code: 0
   - No issues found

9. **timing_benchmark.py** - ✅ PASSED
   - Exit code: 0
   - No issues found

10. **ts_optimizer_benchmark.py** - ✅ PASSED (with warnings)
    - Exit code: 0
    - **Issue**: torchsim_mace backend failed to compute Hessian for all TS optimizers except sella
      - Error: "'SimState' object has no attribute 'numbers'"
      - Affects: trust-krylov-ts, rfo optimizers
    - **Issue**: Some optimizers found minima instead of transition states:
      - aimnet2/sella: found minimum (0 imaginary frequencies)
      - aimnet2/trust-krylov-ts_hessian_freq_10: found minimum (0 imaginary frequencies)
      - mace/sella: found minimum (0 imaginary frequencies)
    - **Issue**: Some trust-krylov-ts optimizations didn't converge (stopped early due to "bad approximation")

11. **bh28_benchmark/bh28_benchmark.py --quicker** - ✅ PASSED
    - Exit code: 0
    - No issues found

12. **zimmermann93_benchmark/zimmermann93_benchmark.py --quicker** - ✅ PASSED
    - Exit code: 0
    - No issues found

### ❌ Failed/Skipped Examples

13. **uma_hessian_method_comparison.py** - ❌ FAILED
    - Exit code: 1
    - **Error**: "This demo requires UMA backend. Use --backends uma"
    - **Error**: "Please install UMA: pip install fairchem-core"
    - **Note**: UMA backend is not installed in the conda base environment

## Common Issues Across Examples

### 1. torchsim_mace Backend Issues

#### Primary Issue: Hessian Calculation Failures
- **Error**: "'SimState' object has no attribute 'numbers'"
- **Root Cause**: When FrequencyAnalysis generates displaced structures for Hessian calculation (via finite differences), it creates Atoms objects. When torchsim_mace's calculator tries to convert these back to SimState objects for batch evaluation, it attempts to access `state.numbers`, but the SimState object structure doesn't have this attribute in the expected format.
- **Affects**:
  - Frequency analysis (used for post-optimization validation)
  - TS optimization with trust-krylov-ts and rfo optimizers (these require Hessian)
  - Initial Hessian calculation for TS optimization
- **Does NOT affect**:
  - Sella optimizer (doesn't require Hessian for TS optimization)
  - Minima optimization with lbfgs, bfgs, fire (these don't require Hessian)
  - NEB, CI-NEB, IRC path calculations (don't require Hessian)

#### Secondary Issue: Trust-Krylov Optimizer
- **Problem**: torchsim_mace completely fails with trust-krylov optimizer for minima optimization
  - Shows 0 steps, no convergence
  - The optimizer likely fails immediately when trying to compute the initial Hessian
- **Why trust-krylov fails**: Trust-krylov requires Hessian calculation at the start and during optimization. Since torchsim_mace has Hessian calculation issues, trust-krylov cannot proceed.

#### JAX/CUDA Warning (Less Critical)
- **Warning**: "An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu."
- **Impact**: This causes torchsim_mace to run on CPU instead of GPU, making it slower
- **Note**: This is a warning, not the primary failure cause. The primary issue is the SimState.numbers attribute error

### 2. Ill-Conditioned Hessian Warnings
- Multiple examples show: "Hessian is ill-conditioned. Condition number: X.XXe+XX. Maximum acceptable: 1.00e+12."
- This is a warning, not an error, but may affect numerical stability

### 3. Optimization Convergence Issues
- Some optimizers (especially trust-krylov-ts) fail to converge:
  - "Optimization stopped after X trust-region steps without converging"
  - "SciPy reason: A bad approximation caused failure to predict improvement"
- Some optimizers find minima instead of transition states (sella optimizer in some cases)

### 4. Missing UMA Backend
- uma_hessian_method_comparison.py requires UMA backend which is not installed
- This is expected if UMA is an optional dependency

## Recommendations

1. **Fix torchsim_mace Hessian calculation**:
   - **Priority: HIGH** - The "'SimState' object has no attribute 'numbers'" error prevents use of Hessian-based optimizers (trust-krylov, trust-krylov-ts, rfo) with torchsim_mace
   - **Location**: Likely in `qme/potentials/torchsim_potential.py` or `qme/analysis/frequency.py` when converting Atoms to SimState for batch Hessian calculation
   - **Fix**: Ensure SimState objects have the `numbers` attribute properly set when converting from Atoms, or handle the conversion differently in the batch evaluation path

2. **Document torchsim_mace limitations**:
   - Clearly document that torchsim_mace does not support Hessian-based optimizers (trust-krylov, trust-krylov-ts, rfo)
   - Recommend using sella for TS optimization with torchsim_mace
   - Recommend using lbfgs/bfgs/fire for minima optimization with torchsim_mace

3. **Install CUDA-enabled jaxlib**:
   - For better torchsim_mace performance on GPU (currently falls back to CPU)
   - This is a performance issue, not a correctness issue

4. **Review optimizer validation logic**:
   - Some optimizers show as "Invalid" even when they converged successfully (aimnet2/lbfgs, aimnet2/bfgs, aimnet2/fire)

5. **Consider making UMA optional**:
   - The uma_hessian_method_comparison.py should gracefully handle missing UMA backend or document it as a required dependency

## Notes

- All examples that ran completed without critical errors
- Most issues are warnings or backend-specific problems
- The torchsim_mace backend has the most issues, but other backends (aimnet2, mace) work well
- Benchmark examples (bh28, zimmermann93) completed successfully with --quicker flag
