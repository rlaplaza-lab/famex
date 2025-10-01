Zimmermann-93 NEB / Two-ended TS Benchmark

This folder contains a simple benchmark driver `zimmermann93_benchmark.py` that
runs two-ended (reactant->product) transition-state searches across ML
backends supported by QME.

What it does:
- For each reaction in `zimmermann93_dataset/` (reactant/product/ts XYZ files),
  it interpolates a reaction path (geodesic by default).
- Optionally optimizes the path with a simplified NEB-like routine (if a model
  calculator is available).
- Selects the highest-energy interpolated image as a TS guess and runs a TS
  optimization using QME's `find_transition_state` (SElLA required for true TS
  optimizations; otherwise a single-point energy is recorded).
- Computes RMSD between the located TS and the reference TS geometry.

Usage:
    python zimmermann93_benchmark/zimmermann93_benchmark.py --quick
    python zimmermann93_benchmark/zimmermann93_benchmark.py --backends uma mace

Notes:
- This script mirrors the behaviour of `examples/bh28_benchmark/bh28_benchmark.py`
  but focuses on two-ended TS finding and geometry comparison.
- The benchmark will try to use available ML backends (UMA, SO3LR, AIMNet2,
  MACE). If none are available the benchmark will fail early.

Output:
- Results are saved to `examples/zimmermann93_benchmark/benchmark_results/zimmermann93_benchmark_results.json`.
