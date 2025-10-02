# TorchSim Acceleration

QME supports TorchSim acceleration for significant performance improvements with machine learning potentials.

## Overview

TorchSim is a next-generation atomistic simulation engine that provides up to 100x speedup over traditional ASE implementations through:

- **Automatic batching** and GPU memory management
- **Native PyTorch** implementation optimized for ML potentials
- **Efficient parallel processing** of multiple structures
- **Advanced memory management** for large-scale calculations

## Supported Models

TorchSim acceleration is available for several popular ML potential architectures:

### MACE Models
- `mace-omol-0`: Large model for molecules/transition metals/cations
- `mace-mp-small`: Materials Project small model
- `mace-mp-medium`: Materials Project medium model
- `mace-mp-large`: Materials Project large model
- `mace-off-small`: Organic chemistry small model
- `mace-off-medium`: Organic chemistry medium model
- `mace-off-large`: Organic chemistry large model

### Fairchem Models
- `equiformer_v2_31M_s2ef_all_md`: Equiformer v2 model
- Other Fairchem models supported by TorchSim

## Installation

### Requirements
- Python 3.11+ (TorchSim requirement)
- PyTorch (already required for QME ML backends)
- CUDA-capable GPU (recommended for best performance)

### Install TorchSim
```bash
pip install torch-sim-atomistic
```

### Verify Installation
```bash
python -c "import torchsim; print('TorchSim available')"
```

## Usage

### Available TorchSim Backends

QME provides three TorchSim-accelerated backends:

1. **`torchsim`**: Generic TorchSim backend (defaults to MACE)
2. **`torchsim_mace`**: TorchSim with MACE models
3. **`torchsim_fairchem`**: TorchSim with Fairchem models

### Command Line Interface

```bash
# TorchSim MACE with GPU acceleration
qme opt molecule.xyz --backend torchsim_mace --model-name mace-omol-0 --device cuda

# TorchSim Fairchem
qme opt molecule.xyz --backend torchsim_fairchem \
    --model-name equiformer_v2_31M_s2ef_all_md --device cuda

# Generic TorchSim (defaults to MACE)
qme opt molecule.xyz --backend torchsim --model-name mace-mp-medium --device cuda

# CPU usage (not recommended but possible)
qme opt molecule.xyz --backend torchsim_mace --model-name mace-omol-0 --device cpu
```

### Python API

```python
import qme

# TorchSim MACE
explorer = qme.Explorer.from_file(
    "molecule.xyz",
    backend="torchsim_mace",
    model_name="mace-omol-0",
    device="cuda"
)

result = explorer.optimize_minimum()
print(f"Optimized energy: {result['final_energy']:.6f} eV")

# TorchSim Fairchem
explorer = qme.Explorer.from_file(
    "molecule.xyz",
    backend="torchsim_fairchem", 
    model_name="equiformer_v2_31M_s2ef_all_md",
    device="cuda"
)
```

### Direct Calculator Usage

```python
from qme.potentials import get_torchsim_mace_calculator

# Create TorchSim calculator directly
calc = get_torchsim_mace_calculator(
    model_name="mace-omol-0",
    device="cuda"
)

# Use with ASE
from ase.build import molecule
benzene = molecule("C6H6")
benzene.calc = calc

energy = benzene.get_potential_energy()
forces = benzene.get_forces()
```

## Performance Benefits

### Expected Speedups

Based on TorchSim benchmarks and QME integration:

| Operation Type | Speedup Range | Best Use Cases |
|----------------|---------------|----------------|
| Single energy/force calculation | 2-5x | Quick property evaluation |
| Geometry optimization | 5-20x | Routine structure optimization |
| Batch calculations | 10-100x | Multiple structures, NEB paths |
| Frequency analysis | 10-50x | Hessian calculations |

### Performance Factors

**System Size**:
- Small molecules (< 20 atoms): 2-5x speedup
- Medium molecules (20-100 atoms): 5-15x speedup
- Large systems (> 100 atoms): 10-50x speedup

**GPU vs CPU**:
- GPU acceleration provides additional 2-10x speedup
- Memory bandwidth is crucial for performance
- Batch processing shows largest GPU benefits

**Operation Type**:
- **Highest speedup**: Batch evaluations, frequency analysis
- **Good speedup**: Standard optimizations, NEB calculations
- **Moderate speedup**: Single point calculations

## When to Use TorchSim

### Recommended For:
- **Large-scale studies** with many structures
- **Optimization of large molecules** (> 50 atoms)
- **Frequency calculations** and Hessian analysis
- **NEB pathway calculations** with many images
- **Production workflows** where performance matters
- **GPU-accelerated environments**

### Standard Backends May Be Better For:
- **Small test calculations** (< 20 atoms)
- **Quick prototyping** and development
- **CPU-only environments** 
- **Compatibility testing** with other tools
- **Systems without TorchSim dependencies**

## Configuration and Optimization

### Memory Management

TorchSim automatically manages GPU memory, but you can optimize:

```python
# For memory-constrained systems
explorer = qme.Explorer.from_file(
    "large_molecule.xyz",
    backend="torchsim_mace",
    model_name="mace-omol-0",
    device="cuda",
    # TorchSim-specific options
    batch_size=16,  # Smaller batches for limited memory
    precision="half"  # Use half-precision for memory savings
)
```

### Performance Tuning

```bash
# Monitor GPU utilization
nvidia-smi -l 1

# For maximum performance
qme opt molecule.xyz \
    --backend torchsim_mace \
    --model-name mace-omol-0 \
    --device cuda \
    --optimizer sella \
    --steps 1000
```

## Fallback Behavior

QME handles TorchSim availability gracefully:

### TorchSim Available
- Uses TorchSim for maximum performance
- Automatic GPU memory management
- Optimized batch processing

### TorchSim Not Available
- Falls back to mock calculators with clear warnings
- Continues execution without crashing
- Provides installation instructions

### Example Fallback
```python
import qme

# This will work whether TorchSim is available or not
explorer = qme.Explorer.from_file("molecule.xyz", backend="torchsim_mace")

# If TorchSim unavailable:
# Warning: TorchSim not available, using mock calculator
# Install with: pip install torch-sim-atomistic
```

## Troubleshooting

### Common Issues

**TorchSim Not Found**
```
ImportError: No module named 'torchsim'
```
**Solution**: Install TorchSim: `pip install torch-sim-atomistic`

**Python Version Incompatibility**
```
TorchSim requires Python 3.11+
```
**Solution**: Use Python 3.11 or higher: `conda create -n qme-torchsim python=3.11`

**CUDA Errors**
```
RuntimeError: CUDA out of memory
```
**Solutions**:
- Use smaller batch sizes
- Switch to CPU: `--device cpu`
- Use half precision
- Free GPU memory from other processes

**Model Loading Errors**
```
Error: Could not load model 'mace-omol-0'
```
**Solutions**:
- Check model name spelling
- Verify internet connection for model download
- Clear model cache: `qme cache clear`

### Performance Issues

**Slower Than Expected**
```
TorchSim not showing speedup
```
**Check**:
- Using GPU acceleration: `--device cuda`
- System size is large enough to benefit
- Batch operations vs single calculations
- No other GPU processes competing

**Memory Issues**
```
CUDA out of memory during optimization
```
**Solutions**:
- Reduce batch size in configuration
- Use CPU for very large systems
- Monitor memory with `nvidia-smi`

### Debug Mode

Enable verbose output for debugging:

```python
import qme

# Enable debug logging
qme.config.set_log_level("DEBUG")

# This will show detailed TorchSim operations
explorer = qme.Explorer.from_file("molecule.xyz", backend="torchsim_mace")
result = explorer.optimize_minimum()
```

## Advanced Features

### Batch Optimization

TorchSim excels at processing multiple structures simultaneously:

```python
import qme
from ase.io import read

# Load multiple structures
molecules = [read(f"molecule_{i}.xyz") for i in range(10)]

# Batch processing (future feature)
# results = qme.batch_optimize(molecules, backend="torchsim_mace")
```

### Custom Model Integration

TorchSim supports custom models that follow the interface:

```python
# Future: Custom model support
# explorer = qme.Explorer.from_file(
#     "molecule.xyz",
#     backend="torchsim",
#     custom_model="path/to/my_model.pt"
# )
```

## Benchmarking TorchSim

### Performance Comparison

Run the QME timing benchmark to compare TorchSim performance:

```bash
cd examples
python timing_benchmark.py --backends aimnet2 torchsim_mace --device cuda
```

### Custom Benchmarks

```python
import time
import qme
from ase.build import molecule

# Compare backends
backends = ["aimnet2", "torchsim_mace"]
molecule_name = "C6H6"

for backend in backends:
    atoms = molecule(molecule_name)
    
    start_time = time.time()
    explorer = qme.Explorer.from_atoms(atoms, backend=backend, device="cuda")
    result = explorer.optimize_minimum()
    end_time = time.time()
    
    print(f"{backend}: {end_time - start_time:.2f}s, "
          f"Energy: {result['final_energy']:.6f} eV")
```

## Future Enhancements

TorchSim integration in QME is actively developed. Planned features include:

- **Batch frequency analysis** for multiple structures
- **Advanced memory management** options
- **Support for more model architectures**
- **Integration with NEB optimization**
- **Performance monitoring and profiling tools**

## Resources

- **TorchSim Documentation**: [https://torchsim.github.io/torch-sim/](https://torchsim.github.io/torch-sim/)
- **TorchSim GitHub**: [https://github.com/TorchSim/torch-sim](https://github.com/TorchSim/torch-sim)
- **QME Performance Benchmarks**: [../benchmarks/performance.md](../benchmarks/performance.md)
- **GPU Optimization Guide**: [../tutorials/performance_optimization.md](../tutorials/performance_optimization.md)

## Summary

TorchSim acceleration in QME provides:

- ✅ **Significant speedups** for ML potential calculations
- ✅ **Easy integration** with existing QME workflows  
- ✅ **Automatic fallback** when TorchSim unavailable
- ✅ **GPU optimization** for maximum performance
- ✅ **Memory management** for large-scale calculations

For maximum performance with QME, especially for large systems or batch calculations, TorchSim acceleration is highly recommended.
