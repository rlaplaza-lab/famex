# Proposal: Unified Hessian Benchmark

## Overview

Merge all Hessian comparison tools into a single benchmark-style script (`hessian_benchmark.py`) following the pattern established by other benchmarks (`timing_benchmark.py`, `minima_optimizer_benchmark.py`, `ts_optimizer_benchmark.py`).

## Current State

**Files to be consolidated:**
1. `hessian_comparison.py` - Unified comparison tool (FD schemes + backend methods)
2. `adaptive_hessian_demo.py` - User-facing demo of adaptive features
3. `hessian_method_comparison_base.py` - Support module (only used by `hessian_comparison.py`)

**Total:** 3 files → 1 file

## Proposed Structure

### New File: `hessian_benchmark.py`

**Features:**
- Uses `QMEExampleInterface` pattern (like other benchmarks)
- Standardized output with summary tables
- JSON result saving via `interface.save_results()`
- Combines all current functionality:
  - FD scheme comparison (3-point, 5-point, 7-point, Richardson, adaptive)
  - Backend method comparison (analytical methods with/without symmetrization)
  - Adaptive delta selection demonstration
  - Noise estimation
  - Method recommendations

**Command-line interface:**
```bash
# Compare FD schemes only
python hessian_benchmark.py --mode fd --backends uma

# Compare backend methods only
python hessian_benchmark.py --mode backend --backends uma,mace

# Compare both (default)
python hessian_benchmark.py --mode both --backends uma,mace

# Quick test with single molecule
python hessian_benchmark.py --quick --backends uma

# Test multiple molecules
python hessian_benchmark.py --molecules water,methane --backends uma
```

**Output format:**
- Summary tables (like other benchmarks)
- Comparison metrics (RMS error, max error, timing, etc.)
- Recommendations (best method per backend)
- JSON results file

## Design Pattern

Following the established benchmark pattern:

```python
@setup_example_environment
def main() -> int:
    """Run the Hessian benchmark."""
    interface = QMEExampleInterface(
        name="Hessian Benchmark",
        description="Hessian Method Comparison",
        epilog=create_standard_epilog("benchmark"),
    )

    parser = interface.create_parser()
    parser.add_argument("--mode", choices=["fd", "backend", "both"], default="both")
    parser.add_argument("--molecules", help="Comma-separated list: water,methane")
    parser.add_argument("--quick", action="store_true")

    args = parser.parse_args()
    # ... setup ...

    results_list = []
    for backend in available_backends:
        for molecule in test_molecules:
            results = benchmark_hessian_methods(
                backend=backend,
                molecule=molecule,
                mode=args.mode,
            )
            results_list.append(results)

    # Print summaries
    print_fd_scheme_summary(results_list)
    print_backend_method_summary(results_list)
    print_recommendations(results_list)

    # Save results
    interface.save_results(results_list, args.output)

    return 0
```

## Functionality to Include

### 1. FD Scheme Comparison
- 3-point, 5-point, 7-point central differences
- Richardson extrapolation variants
- Adaptive delta selection
- Energy-based FD
- Convergence analysis (varying step sizes)

### 2. Backend Method Comparison
- Analytical methods (double_backward, vmap, fairchem, fairchem_loop)
- With/without symmetrization
- Comparison against FD reference
- Frequency analysis validation

### 3. Adaptive Features
- Autoselect demonstration
- Adaptive delta selection
- Noise estimation
- Optimal delta recommendation

### 4. Summary Output
- **FD Scheme Summary Table:**
  ```
  Method                    | Max Error | RMS Error | Time (s) | Speedup
  ---------------------------------------------------------------------
  3-point central           | 1.23e-15 | 5.67e-16 | 0.123    | 1.0x
  5-point central           | 8.88e-16 | 3.33e-16 | 0.234    | 0.5x
  5-point + Richardson      | 2.22e-16 | 1.11e-16 | 0.456    | 0.3x
  Adaptive 5-point + Rich.  | 1.11e-16 | 5.55e-17 | 0.567    | 0.2x
  ```

- **Backend Method Summary Table:**
  ```
  Backend | Method              | RMS Error | Max Error | Time (s) | Neg Freq
  ---------------------------------------------------------------------------
  uma     | double_backward     | 0.000304  | 0.002908  | 1.326    | 0
  uma     | fairchem_loop       | 0.000304  | 0.002908  | 1.359    | 0
  uma     | vmap                | 0.000304  | 0.002912  | 1.739    | 0
  ```

- **Recommendations Section:**
  ```
  RECOMMENDATIONS
  ===============
  Best FD method: Adaptive 5-point + Richardson
  Best backend method (UMA): double_backward with symmetrize=True
  Best backend method (MACE): analytical (default)
  ```

## Benefits

1. **Consistency:** Matches pattern of other benchmarks
2. **Simplicity:** One file instead of three
3. **Standardization:** Uses `QMEExampleInterface` for consistent UX
4. **Results:** JSON output for programmatic analysis
5. **Maintainability:** Single codebase to maintain
6. **Documentation:** Clearer purpose (benchmark vs. demo vs. diagnostic)

## Migration Path

1. Create `hessian_benchmark.py` with merged functionality
2. Test thoroughly with UMA and MACE backends
3. Update `examples/README.md` to reference new benchmark
4. Deprecate old files (add deprecation notices)
5. Remove old files after confirmation period

## Example Usage

```bash
# Quick test
python hessian_benchmark.py --quick --backends uma

# Full comparison
python hessian_benchmark.py --mode both --backends uma,mace --molecules water,methane

# FD schemes only (no backend needed)
python hessian_benchmark.py --mode fd

# Backend methods only
python hessian_benchmark.py --mode backend --backends uma,mace
```

## Implementation Notes

- Merge `hessian_method_comparison_base.py` functions directly into `hessian_benchmark.py`
- Keep adaptive features but present as benchmark results
- Use same molecule creation utilities (`create_water_molecule`, `create_methane_molecule`)
- Follow benchmark output formatting conventions
- Include timing information for performance comparison

## Questions to Consider

1. **Should we keep `adaptive_hessian_demo.py` as a separate user-facing demo?**
   - **Option A:** Merge everything into benchmark (recommended)
   - **Option B:** Keep demo separate, merge only comparison tools

2. **Naming:**
   - `hessian_benchmark.py` (matches other benchmarks)
   - `hessian_method_benchmark.py` (more specific)
   - `hessian_comparison_benchmark.py` (descriptive)

3. **Default behavior:**
   - Quick mode by default?
   - Full comparison by default?
   - Require `--mode` flag?

## Recommendation

**Merge all three files into `hessian_benchmark.py`** following the benchmark pattern. This provides:
- Consistency with other benchmarks
- Single source of truth
- Better maintainability
- Standardized output format
- JSON results for analysis

The benchmark can serve both diagnostic and educational purposes, with clear documentation about when to use it.
