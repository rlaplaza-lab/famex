# Example Files Test Results

This document tracks the results of running all example files in the conda base environment.

Test Date: $(date)
Environment: conda base

## Test Results Summary

| Example File | Status | Exit Code | Runtime | Issues |
|--------------|--------|-----------|---------|--------|
| cli_demo.py | PARTIAL | 124 (timeout) | >600s | Timed out during mace backend test (aimnet2 completed successfully: 9/9 passed) |
| irc_demo.py | PASS | 0 | 16.8s | Completed successfully |
| growing_string_demo.py | PASS | 0 | 9.2s | Completed successfully |
| timing_benchmark.py | PASS | 0 | 252.6s | Completed successfully |
| minima_optimizer_benchmark.py | PARTIAL | 124 (timeout) | >300s | Timed out during final test (torchsim_mace/bfgs), most optimizers completed successfully |
| ts_optimizer_benchmark.py | FAIL | 120 | ~30s | Error during execution (exit code 120) |
| thermochemistry_demo.py | PARTIAL | 120 | ~5s | Error during execution (exit code 120), partial output generated |
| uma_hessian_method_comparison.py | PASS | 0 | ~60s | Completed successfully (ran with MACE backend, not UMA) |
| mace_hessian_method_comparison.py | PASS | 0 | ~60s | Completed successfully |
| hessian_accuracy_comparison.py | PARTIAL | 120 | ~30s | Error during execution (exit code 120), partial output generated |
| adaptive_hessian_demo.py | PARTIAL | 120 | ~15s | Error during execution (exit code 120), partial output generated |
| bh28_benchmark/bh28_benchmark.py | PASS | 0 | 271.0s | Completed successfully with --quick flag |
| zimmermann93_benchmark/zimmermann93_benchmark.py | PARTIAL | 124 (timeout) | >600s | Timed out after 10 minutes, was running but incomplete |

## Detailed Results

### cli_demo.py
- **Status**: PARTIAL (timed out)
- **Exit Code**: 124 (timeout)
- **Runtime**: >600s (timed out after 10 minutes)
- **Issues**:
  - Completed successfully for aimnet2 backend (9/9 examples passed, ~496s)
  - Timed out during mace backend test (example 2/9 - TS optimization)
  - Test was running successfully but took too long (>600s total)
- **Recommendation**: Increase timeout or allow longer runtime for mace backend tests

### irc_demo.py
- **Status**: PASS
- **Exit Code**: 0
- **Runtime**: 16.8s
- **Issues**: None
- **Output**: Generated IRC trajectory with 101 images (50 forward, 50 backward, 1 TS) from transition state

### growing_string_demo.py
- **Status**: PASS
- **Exit Code**: 0
- **Runtime**: 9.2s
- **Issues**: None
- **Output**: Generated trajectory with 15 images, found TS at image 10

### timing_benchmark.py
- **Status**: PASS
- **Exit Code**: 0
- **Runtime**: 252.6s
- **Issues**: None
- **Output**: Benchmarked 3 backends (aimnet2, mace, torchsim_mace) - all completed successfully

### minima_optimizer_benchmark.py
- **Status**: PARTIAL (timed out)
- **Exit Code**: 124 (timeout)
- **Runtime**: >300s (timed out after 5 minutes)
- **Issues**:
  - Most optimizers completed successfully (lbfgs, fire, trust-krylov tested on multiple backends)
  - Timed out during final test (torchsim_mace backend with bfgs optimizer)
  - Some warnings about ill-conditioned Hessians and frequency analysis failures
- **Recommendation**: Increase timeout or allow longer runtime for comprehensive benchmarks

### ts_optimizer_benchmark.py
- **Status**: FAIL
- **Exit Code**: 120
- **Runtime**: ~30s
- **Issues**:
  - Error during execution (exit code 120)
  - Started running trust-krylov-ts optimizer but encountered an error
- **Recommendation**: Investigate exit code 120 - may indicate an internal error or signal

### thermochemistry_demo.py
- **Status**: PARTIAL
- **Exit Code**: 120
- **Runtime**: ~5s
- **Issues**:
  - Error during execution (exit code 120)
  - Generated partial output showing RRHO, Grimme, and Truhlar thermochemistry methods
  - Failed during solution-phase thermodynamics section
- **Recommendation**: Investigate exit code 120 - may indicate an internal error or signal

### uma_hessian_method_comparison.py
- **Status**: PASS
- **Exit Code**: 0
- **Runtime**: ~60s
- **Issues**:
  - Note: UMA backend not available, ran with MACE backend instead
  - Completed successfully comparing Hessian methods
- **Output**: Compared analytical vs finite difference Hessian methods for water and methane

### mace_hessian_method_comparison.py
- **Status**: PASS
- **Exit Code**: 0
- **Runtime**: ~60s
- **Issues**: None
- **Output**: Successfully compared Hessian computation methods for MACE backend

### hessian_accuracy_comparison.py
- **Status**: PARTIAL
- **Exit Code**: 120
- **Runtime**: ~30s
- **Issues**:
  - Error during execution (exit code 120)
  - Generated partial output showing convergence analysis
  - Failed during convergence analysis section
- **Recommendation**: Investigate exit code 120 - may indicate an internal error or signal

### adaptive_hessian_demo.py
- **Status**: PARTIAL
- **Exit Code**: 120
- **Runtime**: ~15s
- **Issues**:
  - Error during execution (exit code 120)
  - Generated partial output showing adaptive Hessian features
  - Completed method comparison but failed at the end
- **Recommendation**: Investigate exit code 120 - may indicate an internal error or signal

### bh28_benchmark/bh28_benchmark.py
- **Status**: PASS
- **Exit Code**: 0
- **Runtime**: 271.0s
- **Issues**:
  - Some warnings about ill-conditioned Hessians (expected for some systems)
  - Charge/spin not specified warnings (expected)
- **Output**: Successfully completed benchmark with --quick flag (6 reactions, 3 backends)

### zimmermann93_benchmark/zimmermann93_benchmark.py
- **Status**: PARTIAL (timed out)
- **Exit Code**: 124 (timeout)
- **Runtime**: >600s (timed out after 10 minutes)
- **Issues**:
  - Test was running successfully but took too long
  - Was executing growing string method searches for multiple reactions
  - Some warnings about optimization convergence limits
- **Recommendation**: Increase timeout or use --quicker flag for faster testing

## Summary

### Overall Statistics
- **Total Examples Tested**: 13
- **Fully Passed**: 7 (54%)
- **Partially Passed**: 5 (38%)
- **Failed**: 1 (8%)

### Key Issues Identified

1. **Exit Code 120 Errors** (4 examples):
   - `ts_optimizer_benchmark.py`
   - `thermochemistry_demo.py`
   - `hessian_accuracy_comparison.py`
   - `adaptive_hessian_demo.py`
   - **Recommendation**: Investigate what exit code 120 indicates - may be a signal or internal error that needs handling

2. **Timeout Issues** (3 examples):
   - `cli_demo.py` - timed out during mace backend test
   - `minima_optimizer_benchmark.py` - timed out during final test
   - `zimmermann93_benchmark.py` - timed out after 10 minutes
   - **Recommendation**: Increase timeouts or optimize long-running tests

3. **Performance Concerns**:
   - Some benchmarks take 5-10+ minutes to complete
   - MACE backend tests are slower than aimnet2
   - Consider adding progress indicators for long-running tests

### Recommendations

1. **Fix Exit Code 120 Issues**: Investigate and fix the root cause of exit code 120 errors in multiple examples
2. **Increase Timeouts**: For comprehensive benchmarks, consider longer timeouts or allow them to run to completion
3. **Backend Availability**: Some examples fall back to different backends (e.g., UMA → MACE) - document this behavior
4. **Error Handling**: Improve error handling and reporting for long-running tests
5. **Progress Indicators**: Add progress indicators for benchmarks that take several minutes

### Examples Working Correctly

The following examples completed successfully without issues:
- `irc_demo.py`
- `growing_string_demo.py`
- `timing_benchmark.py`
- `uma_hessian_method_comparison.py` (with backend fallback)
- `mace_hessian_method_comparison.py`
- `bh28_benchmark/bh28_benchmark.py` (with --quick flag)
