# FAMEX User Guide

Reference for CLI, Python API, and backends.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Installation](#installation)
3. [Command Line Interface](#command-line-interface)
4. [Python API](#python-api)
5. [Backend Guide](#backend-guide)
6. [Examples](#examples)

## Core Concepts

- **Target**: What you want (`minima`, `ts`, `path`)
- **Strategy**: How to get there (`local`, `interpolate`, `neb`, `cineb`, `irc`, `growing_string`, `dhs`)

### Target/Strategy Matrix

| Target | Strategy | Description |
|--------|----------|-------------|
| `minima` | `local` | Direct local optimization |
| `minima` | `interpolate` | Minima from interpolated path |
| `ts` | `local` | Local transition-state (TS) search |
| `ts` | `interpolate` | TS guess from Image-Dependent Pair Potential (IDPP) interpolation + local refinement |
| `ts` | `cineb` | IDPP path, energy-weighted Climbing-Image Nudged Elastic Band (CI-NEB; looser band `fmax`), then local TS refinement |
| `ts` | `dhs` | Dewar–Healy–Stewart (DHS) two-image bracket + local TS refinement |
| `ts` | `growing_string` | Double-ended growing string method (DE-GSM) |
| `path` | `neb` | Nudged Elastic Band (NEB) path optimization |
| `path` | `cineb` | CI-NEB path optimization |
| `path` | `irc` | Intrinsic Reaction Coordinate (IRC) path from transition state |
| `path` | `interpolate` | Generate path only (no optimization) |

## Installation

```bash
pip install famex
# Or from source:
git clone https://github.com/rlaplaza-lab/famex.git && cd famex && pip install -e .

# UMA backend (recommended default):
pip install famex[uma]
# equivalent to: pip install "fairchem-core>=2.25.0"

# Development / testing:
pip install -e ".[dev,uma]"
```

See [README.md](../README.md) for the install table (with default checkpoints) and [Default models](#default-models) below.

> **Note**: Python 3.11+ required. UMA and MACE conflict on `e3nn` versions — use separate conda environments.

> **Default backend**: CLI and `Explorer` default to `uma` with checkpoint `uma-s-1p2` and task head `omol`. For a conflict-free quick start, pass `--backend aimnet2` (loads `aimnet2_wb97m_0`) or install only `torch`. See [Default models](#default-models) for every backend keyword.

## Command Line Interface

- `famex minima` - Minima optimization (outputs single structure)
- `famex ts` - Transition state optimization (outputs single TS)
- `famex path` - Reaction path optimization (outputs trajectories)
- `famex cache` - Cache management

Structure inputs for `minima`, `ts`, and `path` can be a geometry file, a SMILES string (a text encoding of molecular connectivity), `pubchem:NAME`, `cid:`, or a bare numeric CID. A bare non-numeric name on those commands is treated as SMILES. Use `--embedder pysmiles` (default, `pip install famex[smiles]`) or `--embedder rdkit` (`pip install famex[rdkit]`) when coordinates are built from SMILES.

### Global Options

`minima`, `ts`, and `path` share these options. `famex cache` does not take them. `--freq`, `--temperature`, `--cleanup-frequencies`, and the vibrational-mode animation flags apply to `minima` and `ts` only.

| Option | Default | Description |
|--------|---------|-------------|
| `--backend` | `uma` | Backend: uma\|aimnet2\|mace\|orb\|so3lr\|tblite\|pet\|pyscf\|mock |
| `--model-name` | backend default | Override model (see [Default models](#default-models) when omitted) |
| `--model-path` | `None` | Path to model file (if applicable) |
| `--device` | `None` | Device: cpu\|cuda |
| `--task-name` | `omol` (UMA) | UMA task head: `omol`\|`omat`\|`omc`\|`oc20`\|`oc25`\|`odac`\|`oc22` (ignored for other backends) |
| `--dftd3-functional` | `None` | Optional FairChem DFT-D3(BJ) wrap for UMA: `r2scan` or `pbe` |
| `--default-charge` | `0` | Default molecular charge |
| `--default-spin` | `1` | Default spin multiplicity (use `0` for non-`omol` UMA tasks) |
| `--local-optimizer` | `default` | Local optimizer: default\|lbfgs\|bfgs\|fire\|sella\|sella-analytical\|trust-krylov\|trust-ncg\|trust-exact\|newton-cg\|rfo. `default` selects `lbfgs` for minima and path, and `rfo` for TS. `newton-cg` is minima-only. |
| `--optimizer-kw` | `None` | Optimizer kwargs as key=value, repeatable |
| `--ts-kw` | `None` | TS optimizer kwargs as key=value, repeatable |
| `--constraints` | `None` | Constraints spec string; e.g., `'fix 0,1; harmonic_bond 2,3 k=5.0; fixinternals_bond 4,5 value=1.25'` |
| `--verbose`, `-v` | quiet (no flag) | `0` quiet / `1` normal / `2+` debug. Same scale as `Explorer(verbose=…)`; CLI default is `0`, Python default is `1`. |
| `--temperature` | `298.15` | `minima` and `ts` only. Temperature in Kelvin for thermodynamic calculations |
| `--dry-run` | `False` | Validate inputs and show strategy selection without running |
| `--freq`, `--frequencies` | `False` | `minima` and `ts` only. Perform frequency analysis after optimization (includes thermodynamic properties) |
| `--write-modes` | `False` | `minima` and `ts` only. Write each vibrational mode as a multi-frame XYZ animation. Implies `--freq` |
| `--modes` | `all` | `all` or `imaginary`. Which modes `--write-modes` animates. `imaginary` keeps frequencies below -50 cm^-1. Ignored when `--mode-index` is set |
| `--mode-index` | none | Vibrational mode index to write. Repeatable. Overrides `--modes` |
| `--mode-frames` | `20` | Frames in each mode animation |
| `--mode-amplitude` | `0.3` | Largest atomic displacement in each animation, in angstroms |
| `--cleanup-frequencies` | `False` | `minima` and `ts` only. After optimization, clean up using the Hessian (matrix of second energy derivatives) until there are 0 (minimum) or 1 (TS) relevant imaginary frequencies |
| `--force-finite-diff-hessian` | `False` | Force finite-difference Hessians for TS optimizers and frequency calculations. When disabled (default), backends with analytical Hessian support (`uma`, `mace`, `aimnet2`, `orb` conservative models, `pet`, `so3lr`, `pyscf`) use the analytical Hessian automatically. `tblite` and non-conservative Orb models fall back to finite differences. |

### famex minima - Minima Optimization

Optimize molecular structures to find energy minima.

#### Usage

```bash
famex minima --strategy {local,interpolate} INPUT [OPTIONS]
```

#### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `INPUT` | Path, SMILES, or PubChem | Geometry file, SMILES, `pubchem:NAME`, `cid:2244`, or a bare numeric CID. A missing path that looks like a file is an error. A bare name is SMILES, not a PubChem lookup. |

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--strategy` | `local` | Optimization strategy: local\|interpolate |
| `--product` | `None` | Product geometry, SMILES, or PubChem (`pubchem:NAME`, `cid:`, or a bare CID) for interpolate |
| `--embedder` | `pysmiles` | SMILES 3D method: `pysmiles` or `rdkit` |
| `--output` | Auto | Output optimized XYZ path. Replaces only the default file |
| `--history` | off | Also write the optimizer step history |
| `--fmax` | `0.05` | Convergence threshold |
| `--steps` | `1000` | Max optimization steps |
| `--npoints` | `11` | Number of interpolation points (interpolate strategy only) |
| `--interp` | `geodesic` | Interpolation method: linear\|geodesic\|idpp |

#### Examples

```bash
# Basic local optimization
famex minima --strategy local molecule.xyz

# With specific backend and convergence
famex minima --strategy local molecule.xyz --backend aimnet2 --fmax 0.01

# Two-ended minima search via interpolation
famex minima --strategy interpolate reactant.xyz --product product.xyz --npoints 21

# With constraints
famex minima --strategy local molecule.xyz --constraints "fix 0,1,2"

# Dry run to check strategy selection
famex minima --strategy local molecule.xyz --dry-run

# Animate every vibrational mode (also runs the frequency analysis)
famex minima --strategy local molecule.xyz --write-modes
```

#### Output Files

The default file follows the target. `local` omits the strategy name.

- Local optimization: `{input}.minima.xyz` (one structure)
- Interpolate optimization: `{input}.minima.interpolate.xyz` (one optimized minimum per image)

`--history` also writes optimizer steps: `{input}.minima_trajectory.xyz` or `{input}.minima.interpolate_trajectory.xyz`. Frames from an interpolated minima search are tagged with `image` and `step` in the XYZ comment.

`--write-modes` writes one multi-frame XYZ per mode under `{output_stem}_modes/` (e.g. `water.minima_modes/mode_000_1234.5cm-1.xyz`). Frame 0 is the optimized geometry; peak displacement is `--mode-amplitude`. Interpolated minima use an `image{i}_` prefix and skip images without frequency analysis.

### famex ts - Transition State Optimization

Find and optimize transition state structures.

#### Usage

```bash
famex ts --strategy {local,interpolate,cineb,growing_string,dhs} INPUT [OPTIONS]
```

#### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `INPUT` | Path, SMILES, or PubChem | Geometry file, SMILES, `pubchem:NAME`, `cid:2244`, or a bare numeric CID. A missing path that looks like a file is an error. A bare name is SMILES, not a PubChem lookup. |

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--strategy` | `local` | Optimization strategy: local\|interpolate\|cineb\|growing_string\|dhs |
| `--product` | `None` | Product geometry, SMILES, or PubChem (`pubchem:NAME`, `cid:`, or a bare CID) for interpolate/cineb/growing_string/dhs |
| `--embedder` | `pysmiles` | SMILES 3D method: `pysmiles` or `rdkit` |
| `--output` | Auto | Output TS XYZ path. Replaces only the default file |
| `--history` | off | Also write the search path and the local TS optimizer history |
| `--fmax` | `0.05` | Convergence threshold for local TS refinement |
| `--steps` | strategy default | Max optimization steps. `1000` for local, interpolate, and cineb; `200` for growing_string and dhs |
| `--npoints` | strategy default | Images: `11` for interpolate and cineb; `15` for growing_string |
| `--interp` | `idpp` | Interpolation method (interpolate/cineb; default IDPP) |
| `--spring-constant` | `5.0` | Upper spring constant for energy-weighted CI-NEB (`cineb`) |
| `--max-images` | unset | Growing-string image count. Overrides `--npoints`. When both are omitted, growing string uses 15 |
| `--distance-threshold` | strategy default | growing_string: 0.1; dhs `dist_tol`: 0.5 |
| `--step-size` | strategy default | growing_string: 0.1; dhs `large_step`: 0.2 |
| `--dhs-small-step` | `0.05` | DHS small step when images are close |
| `--dhs-switch-thresh` | `1.5` | DHS distance below which small_step is used |
| `--require-ts/--allow-ts` | `--allow-ts` | For all `famex ts` strategies: raise unless the result is vibrationally a first-order saddle (exactly one imaginary mode). For growing string, whether the two sides formally met does not affect this check. |

#### Examples

```bash
# Basic local TS optimization
famex ts --strategy local ts_guess.xyz

# With frequency analysis
famex ts --strategy local ts_guess.xyz --freq

# Animate the imaginary mode
famex ts --strategy local ts_guess.xyz --write-modes --modes imaginary

# With custom optimizer
famex ts --strategy local ts_guess.xyz --local-optimizer rfo --fmax 0.02

# TS from IDPP interpolation
famex ts --strategy interpolate reactant.xyz --product product.xyz

# With custom settings
famex ts --strategy interpolate reactant.xyz --product product.xyz --npoints 15 --interp idpp

# CI-NEB TS guess (IDPP + energy-weighted springs, looser band fmax)
famex ts --strategy cineb reactant.xyz --product product.xyz

# Dewar–Healy–Stewart bracket
famex ts --strategy dhs reactant.xyz --product product.xyz

# Growing string method
famex ts --strategy growing_string reactant.xyz --product product.xyz --npoints 20 --step-size 0.1

# Strict validation (raises unless vibrationally a first-order saddle)
famex ts --strategy local ts_guess.xyz --require-ts
famex ts --strategy growing_string reactant.xyz --product product.xyz --require-ts

# With frequency analysis
famex ts --strategy interpolate reactant.xyz --product product.xyz --freq
```

#### Output Files

The default file is always the single optimized saddle.

- Local TS: `{input}.ts.xyz`
- Interpolated TS: `{input}.ts.interpolate.xyz`
- CI-NEB TS: `{input}.ts.cineb.xyz`
- DHS TS: `{input}.ts.dhs.xyz`
- Growing string TS: `{input}.ts.growing_string.xyz`

`--history` also writes:

- Local optimizer steps: `{input}.ts_trajectory.xyz`, or `{input}.ts.{strategy}_trajectory.xyz` for the refinement inside another TS method
- The search geometry that is not the default file: `{input}.ts.{strategy}_path.xyz` (interpolation guess, CI-NEB band, DHS bracket images, or the growing-string path from reactant to product)

`--write-modes` writes the same mode animations as `minima`, in a directory beside the TS geometry file (`{output_stem}_modes/`).

### famex path - Reaction Path Optimization

Generate and optimize reaction pathways.

#### Usage

```bash
famex path --strategy {interpolate,neb,cineb,irc} STRUCTURES... [OPTIONS]
```

#### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `STRUCTURES...` | Path, SMILES, or PubChem | One or more endpoints. Each may be a geometry file, a SMILES string, `pubchem:NAME`, `cid:2244`, or a bare numeric CID. A bare name is SMILES. A single multi-frame XYZ is used as the path guess. Endpoints must have the same number of atoms. |

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--strategy` | `neb` | Path optimization strategy: interpolate\|neb\|cineb\|irc |
| `--embedder` | `pysmiles` | SMILES 3D method: `pysmiles` or `rdkit` |
| `--output` | Auto | Output trajectory XYZ path. Replaces only the default file |
| `--history` | off | Also write the NEB or CI-NEB iteration history, or IRC forward and reverse branches |
| `--fmax` | `0.05` | Convergence threshold |
| `--steps` | strategy default | Max optimization steps. `1000` for neb and cineb; `100` for irc. interpolate does not optimize |
| `--npoints` | `11` | Number of images in path |
| `--interp` | `geodesic` | Initial interpolation method |
| `--spring-constant` | `5.0` | Spring constant for NEB/CI-NEB (same default as the Python strategies) |
| `--step-size` | `0.1` | IRC step size (IRC strategy only) |
| `--direction` | `both` | Direction: forward\|backward\|both (IRC strategy only) |

#### Examples

```bash
# Raw interpolation (two structures)
famex path --strategy interpolate reactant.xyz product.xyz

# NEB path optimization (two structures)
famex path --strategy neb reactant.xyz product.xyz

# NEB with multiple intermediate structures
famex path --strategy neb reactant.xyz intermediate.xyz product.xyz --npoints 11

# CI-NEB path optimization
famex path --strategy cineb reactant.xyz product.xyz

# IRC from transition state (single structure)
famex path --strategy irc ts.xyz --direction both
```

#### Output Files

The default file is the path from one end to the other. NEB, CI-NEB, and interpolation run from reactant to product. IRC with `--direction both` is the reversed backward branch, then the transition state, then the forward branch. A one-sided IRC writes that side plus the transition state under the same default name.

- Interpolation: `{input}.path.interpolate.xyz`
- NEB: `{input}.path.neb.xyz`
- CI-NEB: `{input}.path.cineb.xyz`
- IRC: `{input}.path.irc.xyz`

`--history` also writes `{input}.path.neb_trajectory.xyz` or `{input}.path.cineb_trajectory.xyz` (one frame per image per iteration, with `iteration` and `image` in the XYZ comment), and `{input}.path.irc_forward.xyz` / `{input}.path.irc_reverse.xyz` for the IRC branches that were followed.

### famex cache - Cache Management

Manages the on-disk model cache (downloaded checkpoints for any backend that uses it). Calculator instances are cached separately in memory for the session.

```bash
famex cache info              # Show cache directory, size, and cached models
famex cache verify            # Checksum-verify cached model files
famex cache clear             # Clear entire cache (prompts for confirmation)
famex cache clear --model M   # Clear one model entry
famex cache clear --yes       # Skip confirmation prompt
```

### Structure inputs (file, SMILES, PubChem)

`famex minima`, `famex ts`, and `famex path` accept a geometry file, a SMILES string, `pubchem:NAME`, `cid:2244`, or a bare numeric CID anywhere a structure is expected (including `--product` and path endpoints). A bare non-numeric name is SMILES, not a PubChem lookup. Endpoints of one calculation must have the same number of atoms.

When the input is SMILES (or PubChem has no 3D conformer), coordinates are built internally. `--embedder` selects the method:

| Embedder | Install | What it does |
|----------|---------|--------------|
| `pysmiles` (default) | `pip install famex[smiles]` | Universal Force Field (UFF) distance geometry, then a short UFF-like cleanup |
| `rdkit` | `pip install famex[rdkit]` | RDKit ETKDGv3 distance-geometry embedder |

Either result is a starting geometry: pass it straight into optimization (for example `famex minima`).

```bash
famex minima --strategy local "CCO" --backend aimnet2 --embedder pysmiles
famex minima --strategy local "CCO" --embedder rdkit --backend aimnet2
famex minima --strategy local pubchem:aspirin --backend aimnet2
famex minima --strategy local cid:2244 --backend aimnet2
```

```python
from famex import Explorer, smiles_to_atoms
from famex.io.geometry import Geometry

geom = smiles_to_atoms("CCO")
rdkit_geom = smiles_to_atoms("CCO", embedder="rdkit")
explorer = Explorer.from_smiles(
    "CCO", backend="uma", target="minima", strategy="local", embedder="pysmiles"
)
pubchem_geom = Geometry.from_pubchem("aspirin")
explorer = Explorer.from_pubchem("aspirin", backend="uma", target="minima", strategy="local")
```

Limits of the pysmiles embedder:

- Four-coordinate centers are embedded as tetrahedral. Square-planar versus tetrahedral, and octahedral chirality, are not in ordinary SMILES.
- Write metal charges and ligand bonds explicitly. Dative `A->B` is not OpenSMILES; use a normal bond, with charge separation when you need it (`[C-]#[O+]`).
- No periodic crystals. Dot-disconnected fragments (salts) are embedded separately and placed at a van der Waals contact.
- 1–4 distances use generic cis/gauche/trans windows, not element-specific torsion terms. `@` / `@@` and `/` `\` stereo are enforced when the SMILES contains them.

RDKit ETKDGv3 follows RDKit's own valence and UFF atom types. It can reject metal SMILES that the pysmiles path embeds, including `[Ni](C#[O])4`. Disconnected salts can be left on top of each other. The PubChem conformer is a computed model, not a crystal structure.

## Python API

### Explorer Class

```python
from famex import Explorer

explorer = Explorer(
    atoms,                    # Atoms or Sequence[Atoms]
    backend="uma",            # Backend name
    target="minima",          # Target: minima|ts|path
    strategy="local",         # Strategy: local|neb|cineb|interpolate|growing_string|dhs|irc
    device=None,              # Device: cpu|cuda (auto-detected if None)
    local_optimizer="default", # Optimizer (auto-selects based on target)
    default_charge=0,
    default_spin=1,
)
```

**Targets:** `minima`, `ts`, `path`

**Strategies:** `local`, `interpolate`, `neb`, `cineb`, `irc`, `growing_string`, `dhs` (see [Target/Strategy Matrix](#targetstrategy-matrix))

**Optimizers:** `default` selects `lbfgs` for minima and path, and `rfo` for TS. First-order: `lbfgs`, `bfgs`, `fire`. Second-order: `sella`, `sella-analytical`, `trust-krylov`, `trust-ncg`, `trust-exact`, `newton-cg`, `rfo`. `newton-cg` is minima-only.

### Key Methods

```python
# Run optimization
result = explorer.run(fmax=0.05, steps=1000, calculate_frequencies=False)

# Keep optimizer steps and search paths on the result. Write them with
# famex.io.run_output.write_run_outputs(..., history=True).
result = explorer.run(record_history=True)

# Load from file
explorer = Explorer.from_file("molecule.xyz", backend="uma", target="minima")
explorer = Explorer.from_smiles("CCO", backend="uma", target="minima", strategy="local")

# Calculate frequencies (after optimization or on any structure)
freq_result = explorer.calculate_frequencies(atoms=None, delta=0.01, method="auto")
# TS check: freq_result["is_ts"] or freq_result["ts_analysis"]["n_imaginary_frequencies"] == 1

# Save results
explorer.save_structure(result["optimized_atoms"], "output.xyz")
path = result.get("trajectory", result["optimized_atoms"])
explorer.save_trajectory(path, "path.xyz")

# One multi-frame XYZ per mode. which="imaginary" keeps frequencies below -50 cm^-1.
# famex.io.write_vibrational_modes writes the same files without an Explorer.
explorer.save_vibrational_modes(freq_result, "water.minima_modes")
```

### Run result keys

Strategy results always include `optimized_atoms` and `strategy`. Common optional keys:

| Key | When present |
|-----|----------------|
| `optimized_atoms` | Single `Atoms` (minima/TS) or `list[Atoms]` (path, and minima interpolation) |
| `trajectory` | End-to-end path for `neb`, `cineb`, `irc`, `interpolate`, and growing string |
| `optimization_trajectory` | Optimizer steps, or NEB/CI-NEB iteration frames, when `record_history=True` |
| `search_path` | TS search band or string when `record_history=True`. Growing string also keeps the string on `trajectory` |
| `forward_path`, `backward_path` | IRC branches |
| `converged`, `steps_taken` | After optimization. Path NEB/CI-NEB: the band reached `fmax` (images are still returned). `ts` CI-NEB and growing string: the local refinement |
| `frequency_analysis` | When `calculate_frequencies=True` or `--freq` (also computed when `--require-ts` needs frequencies) |
| `ts_validation` | When `validate_ts=True` (soft check; does not raise). CLI `--require-ts` / `run(require_ts=True)` raises instead of only attaching this key |

### Frequencies and thermochemistry

`--freq` and `calculate_frequencies=True` attach `frequency_analysis`. Frequencies are in cm⁻¹, with imaginary modes negative. `zero_point_energy` is the harmonic value in eV. Imaginary and zero modes are omitted from the vibrational sums.

```python
from famex.analysis import ThermodynamicProperties

thermo = ThermodynamicProperties(
    freq["frequencies"],
    optimized,
    temperature=298.15,
    pressure=101325.0,
    method="rrho",
    solvent="none",
)
props = thermo.calculate_complete_thermodynamics(energy=energy)
```

`pressure` is in pascal (default 1 atm). `method` is `rrho`, `grimme`, or `truhlar`. `freq_scale_factor` (default 1) is applied before the zero-point energy, vibrational entropy, and heat capacity. `gibbs_free_energy` is the enthalpy minus temperature times entropy, in eV. `free_energy_correction` is that value minus the electronic energy.

| `method` | Vibrational treatment |
|----------|------------------------|
| `rrho` | Harmonic oscillator. `zpe` is the scaled zero-point energy; `enthalpy_vib` is the thermal part |
| `truhlar` | Same energy as `rrho`. Modes at or below `freq_cutoff` (default 100 cm⁻¹) use that cutoff for entropy, after scaling |
| `grimme` | Low modes are damped toward a free rotor (cutoff 100 cm⁻¹). `zpe` here is 0; that energy is already in `enthalpy_vib`. The frequency report keeps the harmonic zero-point energy |

Gas phase uses the ideal-gas concentration at `pressure`. Solution defaults to 1.0 mol/L. A set `concentration` overrides both. Solvents: `none`, `H2O`, `toluene`, `DMF`, `AcOH`, `chloroform`. The symmetry number comes from the geometry unless `symmetry_number` is set. A linear molecule uses one rotational temperature. See `examples/thermochemistry.py`.

### Default models

When `--model-name` / `model_name` is omitted, each backend keyword loads this checkpoint (or method / DFT level):

| Backend keyword | Default `--model-name` | Checkpoint / method loaded |
|-----------------|------------------------|----------------------------|
| `uma` | `uma-s-1p2` | fairchem UMA-S 1.2 (task head default `omol`) |
| `aimnet2` | `aimnet2` | `aimnet2_wb97m_0` (aliases: `aimnet2_wb97m`; also `aimnet2_b973c`, `aimnet2-qr`) |
| `mace` | `mace-omol-0` | MACE omol pretrained (molecules / TMs) |
| `pet` | `pet-mad-s` | PET-MAD small |
| `orb` | `orb-v3-conservative-omol` | Orb-v3 conservative molecular |
| `so3lr` | `so3lr-small` | SO3LR small pretrained |
| `tblite` | `GFN2-xTB` | xTB method (also e.g. `GFN1-xTB`) |
| `pyscf` | `b3lyp-d3bj/def2-svp` | DFT as `xc/basis` |
| `mock` | `mock-model` | Built-in harmonic mock (testing) |

OrbMol-v2 is selected with `--model-name orbmol-v2`. That checkpoint ships in `orb-models>=0.7.0`, which requires Python 3.12.

Charge and spin default to `0` and `1` via `--default-charge` / `--default-spin` (or `Explorer` kwargs). UMA and related backends read `atoms.info["charge"]` and `atoms.info["spin"]` when set. For non-`omol` UMA tasks, prefer `--default-spin 0`.

#### UMA task heads and fine-tunes

UMA routes through a Mixture-of-Linear-Experts head selected by `--task-name` / `task_name` (default `omol`). Requires `fairchem-core>=2.25.0`.

| Task | Typical use | Notes |
|------|-------------|-------|
| `omol` | Molecules, organics, homogeneous catalysis | Needs charge/spin; FAMEX default |
| `omat` | Inorganic materials | Prefer spin `0`; no dispersion in base labels |
| `omc` | Molecular crystals / organic electronics | Prefer spin `0` |
| `oc20` | Heterogeneous catalysis | Prefer spin `0` |
| `oc25` | (Electro)catalysis (UMA-1.2+) | Prefer spin `0` |
| `oc22` | Oxide catalysis (UMA-1.2+) | Prefer spin `0`; validated by the checkpoint |
| `odac` | MOFs / direct air capture | Prefer spin `0` |

Optional newer checkpoints (defaults stay `uma-s-1p2` + `omol`):

```bash
# Patch release of UMA-S 1.2 (multi-task)
famex minima --strategy local mol.xyz --model-name uma-s-1p2p1 --task-name omol

# Fine-tune: ωB97M-D4rev molecules (dispersion included — do not add D3)
famex minima --strategy local mol.xyz \
  --model-name uma-s-1p2p1-ft-d4rev --task-name omol

# Fine-tune: r²SCAN materials + recommended D3(BJ)
famex minima --strategy local crystal.xyz \
  --model-name uma-s-1p2p1-ft-r2scan --task-name omat \
  --dftd3-functional r2scan --default-spin 0
```

Python equivalents: `Explorer(..., task_name="omat", dftd3_functional="r2scan", default_spin=0)` or `get_uma_calculator(task_name=..., dftd3_functional=...)`.

## Backend Guide

FAMEX is built around machine learning potentials. Classical quantum-chemistry backends are optional for checks or small systems. Default checkpoints match [Default models](#default-models).

| Backend | Installation | Default checkpoint | Use when | Notes |
|---------|--------------|--------------------|----------|-------|
| `aimnet2` | `pip install torch` | `aimnet2` → `aimnet2_wb97m_0` | Molecules; simplest install | No `e3nn` conflict with other backends; analytical Hessian |
| `uma` | `pip install "fairchem-core>=2.25.0"` or `pip install famex[uma]` | `uma-s-1p2` | Materials and molecules (CLI default; task `omol`) | Conflicts with MACE; analytical Hessian; see [UMA task heads](#uma-task-heads-and-fine-tunes) |
| `mace` | `pip install mace-torch` | `mace-omol-0` | Molecules | Conflicts with UMA; analytical Hessian |
| `orb` | `pip install "orb-models>=0.7.0"` or `pip install famex[orb]` | `orb-v3-conservative-omol` | Molecules (`orbmol-v2`) and materials | Python 3.12+; analytical Hessian for conservative models; finite-difference otherwise |
| `so3lr` | `pip install so3lr` | `so3lr-small` | Custom / research models | Analytical Hessian via JAX |
| `pet` | `pip install upet` or `pip install famex[pet]` | `pet-mad-s` | PET-MAD potential | Python 3.11+; analytical Hessian |
| `tblite` | `pip install tblite` | `GFN2-xTB` | Semi-empirical xTB | `--model-name` selects the xTB method; finite-difference Hessian |
| `pyscf` | `pip install famex[pyscf]` | `b3lyp-d3bj/def2-svp` | Small-system DFT | `--model-name` as `xc/basis`; analytical Hessian |
| `mock` | Built-in | `mock-model` | Testing | Development only |

### Dependency Conflicts

UMA and MACE conflict due to incompatible `e3nn` versions. Use separate environments:

```bash
# UMA environment
conda create -n famex-uma python=3.12
conda activate famex-uma && pip install famex[uma]

# MACE environment
conda create -n famex-mace python=3.12
conda activate famex-mace && pip install famex mace-torch
```

### Interpolation Methods

| Method | Description | Use when |
|--------|-------------|----------|
| `geodesic` | Distance-preserving with bond refinement | Default path guesses |
| `idpp` | Image-Dependent Pair Potential | Large geometry changes |
| `linear` | Straight-line interpolation in Cartesian coordinates | Quick initial guesses |

Usage: `famex path --strategy neb reactant.xyz product.xyz --interp idpp`

## Examples

See [`examples/README.md`](../examples/README.md) for the catalog of workflow scripts and benchmarks (`optimize_minimum.py`, `thermochemistry.py`, `irc.py`, `timing_benchmark.py`, `bh28_benchmark/`, `zimmermann93_benchmark/`, and others).

---

*Last updated: October 2026*
