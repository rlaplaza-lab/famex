# Maiti-30 Benchmark

Two-ended growing-string transition-state search on the 30 closed-shell
metal-catalyzed reactions of Maiti, Buttar, and Duarte
(*J. Chem. Theory Comput.*, 2026; DOI: 10.1021/acs.jctc.6c00992).

Geometries and ORCA inputs are deposited at
[ioChem-BD 10.19061/iochem-bd-6-566](http://dx.doi.org/10.19061/iochem-bd-6-566)
(BSC browse node). Reactant and product are the IRC-endpoint minima
(`minim_F` / RCT and `minim_B` / PROD); the reference saddle is the
PBE0/def2-SVP optimized TS.

## Layout

```
maiti30_dataset/
  reaction_001_reactant.xyz
  reaction_001_product.xyz
  reaction_001_ts.xyz
  ...
  charges.json
```

`charges.json` stores total charge and spin multiplicity from each ORCA
`*xyzfile` line. The benchmark stamps these onto `atoms.info` before every
Explorer call.

## Run

```bash
# Single backend, full set
conda run -n famex-uma python maiti30_benchmark.py --backends uma --device cuda

# Parallel across OMol backends (from this directory or repo root)
python run_maiti30_parallel.py
# python examples/maiti30_benchmark/run_maiti30_parallel.py
```

AIMNet2 is not included: its ωB97M checkpoint has no transition metals.
