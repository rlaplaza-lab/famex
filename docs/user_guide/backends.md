# Supported Backends

QME supports multiple machine learning potential backends, each with different strengths and use cases.

## Backend Overview

| Backend | Description | Installation | Best For |
|---------|-------------|--------------|----------|
| `uma` | Universal Materials Accelerator (Meta AI) | `pip install qme-ml[uma]` | General purpose, materials |
| `aimnet2` | Native PyTorch implementation | `pip install qme-ml[aimnet2]` | Molecules, fast inference |
| `mace` | Foundation models for chemistry | `pip install qme-ml[mace]` | High accuracy, diverse systems |
| `orb` | Orbital Materials universal forcefield | `pip install qme-ml[orb]` | Universal, molecules and materials |
| `so3lr` | SO(3) invariant neural networks | `pip install qme-ml[so3lr]` | Research, custom models |
| `torchsim_*` | TorchSim accelerated backends | `pip install qme-ml[torchsim]` | High performance, GPU |
| `mock` | Harmonic oscillator for testing | Built-in | Testing, development |

## Interpolation Methods

QME supports multiple interpolation strategies for generating reaction pathways between molecular structures. These methods are used in two-ended optimizations, NEB calculations, and transition state searches.

| Method | Description | Best For |
|--------|-------------|----------|
| `linear` | Simple linear interpolation between coordinates | Quick initial guesses, simple systems |
| `geodesic` | Distance-preserving interpolation with bond length refinement | Chemically reasonable intermediates (default) |
| `idpp` | Image-Dependent Pair Potential interpolation | Large geometry changes, robust pathways |
| `quadratic` | Quadratic curve fitting through start, midpoint, and end | When approximate transition region is known |
| `spline` | Cubic spline interpolation for smooth pathways | Smooth, continuous reaction coordinates |

### Usage

```bash
# Command line - specify interpolation method
qme minima --strategy interpolate reactant.xyz --product product.xyz --interp idpp
qme path --strategy neb reactant.xyz --product product.xyz --interp spline
qme ts --strategy interpolate reactant.xyz --product product.xyz --interp quadratic

# Python API
explorer = qme.Explorer(atoms=[reactant, product], target="path", strategy="interpolate")
result = explorer.run(method="idpp", npoints=15)
```

### Method Selection Guide

- **linear**: Fastest, good for initial exploration
- **geodesic**: Default choice, balances speed and chemical reasonableness
- **idpp**: Most robust for large structural changes, slower but more reliable
- **quadratic**: Good when you have a rough idea of the transition region
- **spline**: Smoothest pathways, good for visualization and analysis

## UMA Backend

**Universal Materials Accelerator** from Meta AI's FAIR research.

### Installation
```bash
pip install qme-ml[uma]
```

### Models
- `equiformer_v2_31M_s2ef_all_md`: General purpose model (default)
- Various specialized models available

### Usage
```bash
# Command line
qme minima --strategy local molecule.xyz --backend uma

# Python API
explorer = qme.Explorer.from_file("molecule.xyz", backend="uma")
```

### Strengths
- General purpose materials and molecules
- Good balance of speed and accuracy
- Well-tested and stable
- Default backend for QME

### Limitations
- Conflicts with MACE due to e3nn version requirements
- May have issues with PyTorch 2.6+ due to `weights_only=True` default

## AIMNet2 Backend

**Accurate Interatomic Machine-learned Network 2** - native PyTorch implementation.

### Installation
```bash
pip install qme-ml[aimnet2]
```

### Models
- `aimnet2_wb97m_0`: Multi-purpose model for organic chemistry

### Usage
```bash
# Command line
qme minima --strategy local molecule.xyz --backend aimnet2

# Python API
explorer = qme.Explorer.from_file("molecule.xyz", backend="aimnet2")
```

### Strengths
- Fast inference
- No dependency conflicts
- Good for organic molecules
- Excellent for rapid prototyping

### Limitations
- Limited to specific chemical domains
- Fewer model options compared to other backends

## MACE Backend

**Multi-Atomic Cluster Expansion** foundation models.

### Installation
```bash
pip install qme-ml[mace]
```

### Models
- `mace-mp-0-small`: Materials Project small model
- `mace-mp-0-medium`: Materials Project medium model
- `mace-mp-0-large`: Materials Project large model
- `mace-off-small`: Organic chemistry small model
- `mace-off-medium`: Organic chemistry medium model
- `mace-off-large`: Organic chemistry large model

### Usage
```bash
# Command line
qme minima --strategy local molecule.xyz --backend mace --model-name mace-mp-0-medium

# Python API
explorer = qme.Explorer.from_file("molecule.xyz",
                                  backend="mace",
                                  model_name="mace-mp-0-medium")
```

### Strengths
- Foundation models with broad applicability
- High accuracy for diverse chemical systems
- Multiple model sizes for speed/accuracy trade-offs
- Excellent for materials and organic chemistry

### Limitations
- Conflicts with UMA due to e3nn version requirements
- Larger models require more memory
- Slower than some alternatives

## Orb Backend

**Orbital Materials** universal neural network forcefields for molecules and materials.

### Installation
```bash
pip install qme-ml[orb]
```

### Models
- `orb-v3-conservative-omol`: Conservative molecular model (default)
- `orb-v3-conservative-inf-omat`: Inference materials model
- `orb-v2`: Orb v2 model

### Usage
```bash
# Command line (charge and spin are required for OrbMol models)
qme minima --strategy local molecule.xyz --backend orb --charge 0 --spin 1

# Python API
explorer = qme.Explorer.from_file("molecule.xyz",
                                  backend="orb",
                                  charge=0,
                                  spin=1)
```

### Strengths
- Universal forcefield for both molecules and materials
- High accuracy across diverse chemical systems
- Conservative models for reliable predictions
- Supports both molecular (OrbMol) and materials (OrbMat) variants

### Limitations
- Large package size (orb-models is a substantial download)
- Requires charge and spin multiplicity specification
- May have compatibility issues with other backends
- Newer package with evolving ecosystem

## SO3LR Backend

**SO(3) Linear Regression** models with rotation equivariance.

### Installation
```bash
pip install qme-ml[so3lr]
pip install so3lr  # Additional package required
```

### Usage
```bash
# Command line
qme minima --strategy local molecule.xyz --backend so3lr

# Python API
explorer = qme.Explorer.from_file("molecule.xyz", backend="so3lr")
```

### Strengths
- Research-focused implementation
- Custom model support
- Mathematically elegant approach

### Limitations
- Requires separate SO3LR package installation
- Limited pre-trained models
- More complex setup

## TorchSim Backends

**TorchSim** provides significant performance improvements through automatic batching and GPU optimization.

### Installation
```bash
pip install qme-ml[torchsim]  # Requires Python 3.11+
```

### Available TorchSim Backends
- `torchsim`: Generic TorchSim backend (defaults to MACE)
- `torchsim_mace`: TorchSim MACE models
- `torchsim_fairchem`: TorchSim Fairchem models

### TorchSim MACE Models
- `mace-omol-0`: Large model for molecules/transition metals/cations
- `mace-mp-*`: Materials Project models (small, medium, large)
- `mace-off-*`: Organic chemistry models (small, medium, large)

### TorchSim Fairchem Models
- `equiformer_v2_31M_s2ef_all_md`: Equiformer v2 model

### Usage
```bash
# TorchSim MACE
qme minima --strategy local molecule.xyz --backend torchsim_mace --model-name mace-omol-0 --device cuda

# TorchSim Fairchem
qme minima --strategy local molecule.xyz --backend torchsim_fairchem --model-name equiformer_v2_31M_s2ef_all_md

# Python API
explorer = qme.Explorer.from_file("molecule.xyz",
                                  backend="torchsim_mace",
                                  model_name="mace-omol-0",
                                  device="cuda")
```

### Performance Benefits
- **Single calculations**: 2-5x speedup
- **Optimization**: 5-20x speedup
- **Batch calculations**: 10-100x speedup
- **GPU acceleration**: Additional 2-10x speedup

### Limitations
- Requires Python 3.11+
- Falls back to mock calculator if TorchSim unavailable
- Limited to supported model architectures

## Mock Backend

**Harmonic oscillator** calculator for testing and development.

### Usage
```bash
# Command line
qme minima --strategy local molecule.xyz --backend mock

# Python API
explorer = qme.Explorer.from_file("molecule.xyz", backend="mock")
```

### Strengths
- Always available (no dependencies)
- Fast execution
- Predictable behavior
- Perfect for testing

### Limitations
- Not chemically meaningful
- Simple harmonic potential only

## Dependency Conflicts

### Known Conflicts

**UMA vs MACE**: Both depend on `e3nn` but require incompatible versions:
- UMA (fairchem-core) requires `e3nn>=0.5`
- MACE requires `e3nn==0.4.4`

**Solution**: Use separate environments:
```bash
# Environment 1: UMA only
conda create -n qme-uma python=3.12
conda activate qme-uma
pip install qme-ml[uma]

# Environment 2: MACE only
conda create -n qme-mace python=3.12
conda activate qme-mace
pip install qme-ml[mace]
```

### Compatibility Matrix

| Backend | UMA | AIMNet2 | MACE | Orb | SO3LR | TorchSim |
|---------|-----|---------|------|-----|-------|----------|
| UMA | ✅ | ✅ | ❌ | ⚠️ | ✅ | ✅ |
| AIMNet2 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| MACE | ❌ | ✅ | ✅ | ⚠️ | ✅ | ✅ |
| Orb | ⚠️ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| SO3LR | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| TorchSim | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

⚠️ = Potential compatibility issues due to package size or dependency conflicts

## Backend Selection Guide

### For Beginners
Start with **AIMNet2** - no conflicts, fast, reliable:
```bash
pip install qme-ml[aimnet2]
```

### For Production Use
Use **UMA** for materials, **MACE** for molecules, or **Orb** for universal coverage:
```bash
# Materials and general purpose
pip install qme-ml[uma]

# High accuracy molecules
pip install qme-ml[mace]

# Universal forcefield (molecules and materials)
pip install qme-ml[orb]
```

### For Maximum Performance
Use **TorchSim** backends with GPU acceleration:
```bash
pip install qme-ml[torchsim]
qme minima --strategy local molecule.xyz --backend torchsim_mace --device cuda
```

### For Development/Testing
Use **Mock** backend:
```bash
qme minima --strategy local molecule.xyz --backend mock
```

## Backend-Specific Options
Each backend supports specific model names and parameters. See the [Developer Guide](../developer_guide/index.md) for implementation details.

## Troubleshooting

### Backend Not Available
```
Error: Backend 'uma' not available
```
**Solution**: Install backend dependencies:
```bash
pip install qme-ml[uma]
```

### Dependency Conflicts
```
ERROR: pip's dependency resolver does not currently have the necessary information...
```
**Solution**: Use separate environments or individual backend installations.

### Model Loading Errors
```
Error: Could not load model 'unknown_model'
```
**Solution**: Check available models for your backend in this documentation.

For more troubleshooting help, see the [Troubleshooting Guide](../reference/troubleshooting.md).
