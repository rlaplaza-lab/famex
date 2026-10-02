# Paper tooling

Helpers used when assembling manuscript tables and figures from
`benchmark_runs/`. Figure outputs and the manuscript tree stay local
(gitignored).

```bash
# After benchmarks have written results under benchmark_runs/
python examples/paper/finalize_paper_from_benchmarks.py
```

Parallel multi-env runners live next to their datasets:

- [`../maiti30_benchmark/run_maiti30_parallel.py`](../maiti30_benchmark/run_maiti30_parallel.py)
- [`../zimmermann93_benchmark/run_zimmermann93_parallel.py`](../zimmermann93_benchmark/run_zimmermann93_parallel.py)
