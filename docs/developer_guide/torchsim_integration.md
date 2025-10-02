# TorchSim Integration (Developer Documentation)

> **Note**: This document contains technical implementation details for developers. For user-facing TorchSim documentation, see [TorchSim Acceleration](user_guide/torchsim_acceleration.md).

This document describes the technical implementation of TorchSim integration in QME, providing significant performance improvements for machine learning potential calculations.

## Overview

TorchSim is a next-generation atomistic simulation engine that provides up to 100x speedup over ASE for machine learning potentials through:
- Automatic batching and GPU memory management
- Native PyTorch implementation
- Efficient parallel processing of multiple structures
- Optimized ML potential calculations

## Supported Models

The TorchSim integration supports the following models:

### MACE Models
- `mace-omol-0`: Large model for molecules/transition metals/cations
- `mace-mp-*`: Materials Project models (small, medium, large)
- `mace-off-*`: Organic chemistry models (small, medium, large)

### Fairchem Models
- `equiformer_v2_31M_s2ef_all_md`: Equiformer v2 model
- Other Fairchem models supported by TorchSim

## Installation

To enable TorchSim acceleration, install the required dependencies:

```bash
pip install torch-sim-atomistic
```

TorchSim requires PyTorch, which should already be installed for QME's ML potentials.

## Usage

### Basic Usage

```python
import qme
from ase.build import molecule

# Create a molecule
benzene = molecule("C6H6")

# Create QME optimizer with TorchSim MACE
qme_opt = qme.Explorer.from_file("molecule.xyz",
    backend="torchsim_mace",
    model_name="mace-omol-0",
    device="cuda"  # Use GPU for best performance
)

# Load structure and optimize
qme_opt.load_structure(benzene)
result = qme_opt.optimize()

print(f"Optimized energy: {result['final_energy']:.6f} eV")
```

### Available Backends

QME now supports three TorchSim backends:

1. **`torchsim`**: Generic TorchSim backend (defaults to MACE)
2. **`torchsim_mace`**: TorchSim MACE models
3. **`torchsim_fairchem`**: TorchSim Fairchem models

### Backend Selection

```python
# TorchSim MACE
qme_opt = qme.Explorer.from_file("molecule.xyz",backend="torchsim_mace", model_name="mace-omol-0")

# TorchSim Fairchem
qme_opt = qme.Explorer.from_file("molecule.xyz",backend="torchsim_fairchem", model_name="equiformer_v2_31M_s2ef_all_md")

# Generic TorchSim (defaults to MACE)
qme_opt = qme.Explorer.from_file("molecule.xyz",backend="torchsim", model_name="mace-mp-medium")
```

### Direct Calculator Usage

You can also use TorchSim calculators directly:

```python
from qme.potentials import get_torchsim_mace_calculator

# Create calculator
calc = get_torchsim_mace_calculator(
    model_name="mace-omol-0",
    device="cuda"
)

# Attach to atoms
benzene.calc = calc

# Calculate properties
energy = benzene.get_potential_energy()
forces = benzene.get_forces()
```

## Performance Benefits

### Expected Speedups

Based on TorchSim benchmarks:
- **Single calculations**: 2-5x speedup
- **Optimization**: 5-20x speedup (depending on system size)
- **Batch calculations**: 10-100x speedup
- **GPU acceleration**: Additional 2-10x speedup

### When to Use TorchSim

TorchSim is most beneficial for:
- Large-scale optimizations
- Multiple structure calculations
- GPU-accelerated computations
- Production runs where performance matters

### When to Use Standard Backends

Standard ASE-based backends are still useful for:
- Small test calculations
- When TorchSim is not available
- Compatibility with existing workflows
- Debugging and development

## Fallback Behavior

The integration is designed to be robust:

1. **TorchSim Available**: Uses TorchSim for maximum performance
2. **TorchSim Not Available**: Falls back to mock calculators with clear warnings
3. **Model Not Supported**: Falls back to standard ASE backends when possible

## Implementation Details

### Lazy Loading

TorchSim is loaded lazily through the `dependencies.py` system:
- Only imported when actually needed
- Graceful fallback when not available
- No performance impact when not used

### Calculator Registry

New TorchSim backends are registered in the calculator registry:
- `torchsim`: Generic TorchSim backend
- `torchsim_mace`: MACE-specific backend
- `torchsim_fairchem`: Fairchem-specific backend

### ASE Compatibility

TorchSim calculators implement the standard ASE Calculator interface:
- `get_potential_energy()`
- `get_forces()`
- `calculate()` method
- Full compatibility with ASE optimizers

## Testing

Test the integration:

```bash
python test_torchsim_integration.py
```

This will:
- Check TorchSim availability
- Test backend registration
- Verify calculator creation
- Test fallback behavior
- Run sample calculations (if TorchSim is available)

## Troubleshooting

### Common Issues

1. **TorchSim not found**: Install with `pip install torch-sim-atomistic`
2. **CUDA errors**: Ensure PyTorch CUDA is properly installed
3. **Model loading errors**: Check model names and availability
4. **Memory errors**: Reduce batch size or use CPU

### Debug Mode

Enable verbose output to debug issues:

```python
import qme
qme.deps.warn_fallback("torchsim", "debug mode")
```

## Future Enhancements

Planned improvements:
- Support for more TorchSim models
- Batch optimization capabilities
- Advanced GPU memory management
- Integration with QME's frequency analysis
- Performance monitoring and profiling

## Contributing

To add support for new TorchSim models:

1. Add model loading logic to `torchsim_potential.py`
2. Update the calculator registry
3. Add tests for the new model
4. Update documentation

## References

- [TorchSim GitHub](https://github.com/TorchSim/torch-sim)
- [TorchSim Documentation](https://torchsim.github.io/torch-sim/)
- [QME Documentation](https://github.com/your-org/qme)
