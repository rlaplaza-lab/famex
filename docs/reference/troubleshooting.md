# Troubleshooting Guide

This guide helps you diagnose and solve common issues with QME.

## Quick Diagnosis

### Check Your Setup
```bash
# Verify QME installation
qme --version

# Check available backends
python -c "
import qme
from qme.calculator_registry import calculator_registry
print('Available backends:', list(calculator_registry.list_backends().keys()))
"

# Test with mock backend
qme opt --help
```

### Common Error Patterns

| Error Pattern | Likely Cause | Quick Fix |
|---------------|--------------|-----------|
| `Backend 'X' not available` | Missing dependencies | Install backend: `pip install qme-ml-ml[X]` |
| `CUDA out of memory` | GPU memory full | Use `--device cpu` |
| `File not found` | Wrong file path | Check file exists and path is correct |
| `Optimization did not converge` | Difficult system or wrong settings | Try `--fmax 0.1 --steps 2000` |
| `Import Error` | Missing dependencies | Follow installation instructions |

## Installation Issues

### Backend Not Available

**Problem**: 
```
Error: Backend 'uma' not available. Available backends: ['mock']
```

**Cause**: Backend dependencies not installed

**Solutions**:
```bash
# Install specific backend
pip install qme-ml-ml[uma]        # UMA backend
pip install qme-ml-ml[aimnet2]    # AIMNet2 backend  
pip install qme-ml-ml[mace]       # MACE backend
pip install qme-ml-ml[so3lr]      # SO3LR backend
pip install qme-ml-ml[torchsim]   # TorchSim backends

# Check installation
python -c "import torch; print('PyTorch version:', torch.__version__)"
```

### Dependency Conflicts

**Problem**:
```
ERROR: pip's dependency resolver does not currently have the necessary information to solve this problem.
```

**Cause**: Conflicting package versions (especially UMA vs MACE)

**Solutions**:
```bash
# Use separate environments
conda create -n qme-uma python=3.12
conda activate qme-uma
pip install qme-ml-ml[uma]

conda create -n qme-mace python=3.12  
conda activate qme-mace
pip install qme-ml-ml[mace]

# Or install individually
pip install qme-ml  # Base package only
pip install qme-ml-ml[aimnet2]  # Add specific backends
```

### Python Version Issues

**Problem**:
```
TorchSim requires Python 3.11+
```

**Solutions**:
```bash
# Check Python version
python --version

# Upgrade Python
conda create -n qme-py311 python=3.11
conda activate qme-py311
pip install qme-ml-ml[torchsim]

# Or use older backends with Python 3.10
pip install qme-ml-ml[aimnet2,uma]
```

### SELLA Optimizer Missing

**Problem**:
```
ImportError: SELLA optimizer not available
```

**Solution**:
```bash
pip install sella
```

## Runtime Issues

### GPU/CUDA Problems

**CUDA Out of Memory**:
```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**Solutions**:
```bash
# Use CPU instead
qme opt molecule.xyz --device cpu

# Free GPU memory
nvidia-smi  # Check GPU usage
# Kill other GPU processes if needed

# Reduce system size or use smaller model
qme opt molecule.xyz --backend aimnet2  # Smaller memory footprint
```

**CUDA Not Available**:
```
CUDA is not available, using CPU
```

**Check GPU Setup**:
```bash
# Verify CUDA installation
nvidia-smi

# Check PyTorch CUDA support
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Reinstall PyTorch with CUDA if needed
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Optimization Problems

**Convergence Issues**:
```
Optimization did not converge after 1000 steps
```

**Solutions**:
```bash
# Increase maximum steps
qme opt molecule.xyz --steps 2000

# Loosen convergence criteria
qme opt molecule.xyz --fmax 0.1

# Try different optimizer
qme opt molecule.xyz --optimizer lbfgs
qme opt molecule.xyz --optimizer bfgs

# Check input structure quality
# Bad geometries can prevent convergence
```

**Unrealistic Results**:
```
Warning: Energy seems unusually high/low
```

**Check**:
- Input file format correctness
- Coordinate units (should be Angstrom)
- Atomic numbers vs symbols
- Backend appropriate for your system

**Solutions**:
```bash
# Try different backend for comparison
qme opt molecule.xyz --backend aimnet2
qme opt molecule.xyz --backend mace

# Use mock backend to test workflow
qme opt molecule.xyz --backend mock
```

### File and Path Issues

**File Not Found**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'molecule.xyz'
```

**Solutions**:
```bash
# Check file exists
ls -la molecule.xyz

# Use absolute path
qme opt /full/path/to/molecule.xyz

# Check working directory
pwd
```

**Permission Errors**:
```
PermissionError: [Errno 13] Permission denied
```

**Solutions**:
```bash
# Check file permissions
ls -la molecule.xyz

# Fix permissions
chmod 644 molecule.xyz

# Check directory permissions
chmod 755 .
```

## Backend-Specific Issues

### UMA Backend

**Fairchem Import Error**:
```
ImportError: No module named 'fairchem'
```

**Solution**:
```bash
pip install fairchem-core
```

**e3nn Version Conflict**:
```
Conflicting e3nn versions
```

**Solution**:
```bash
# Use UMA in separate environment
conda create -n qme-uma python=3.12
conda activate qme-uma
pip install qme-ml-ml[uma]
```

### MACE Backend

**Model Download Issues**:
```
Error downloading MACE model
```

**Solutions**:
```bash
# Check internet connection
ping github.com

# Clear model cache
qme cache clear

# Try different model
qme opt molecule.xyz --backend mace --model-name mace-mp-0-small
```

**e3nn Version Error**:
```
MACE requires e3nn==0.4.4
```

**Solution**: Use separate environment (MACE conflicts with UMA)

### AIMNet2 Backend

**Model Loading Error**:
```
Could not load AIMNet2 model
```

**Solutions**:
```bash
# Clear cache and retry
qme cache clear
qme opt molecule.xyz --backend aimnet2

# Check internet connection for model download
# Models are downloaded on first use
```

### TorchSim Backends

**TorchSim Not Available**:
```
TorchSim backend not available, falling back to mock
```

**Solutions**:
```bash
# Install TorchSim (requires Python 3.11+)
pip install torch-sim-atomistic

# Check Python version
python --version  # Should be 3.11+

# Verify installation
python -c "import torchsim; print('TorchSim available')"
```

**Model Compatibility**:
```
Model not supported by TorchSim
```

**Solution**: Check [supported models](../user_guide/torchsim_acceleration.md#supported-models)

## Performance Issues

### Slow Performance

**Diagnosis**:
```bash
# Time your calculation
time qme opt molecule.xyz

# Check CPU/GPU usage
htop  # CPU usage
nvidia-smi -l 1  # GPU usage
```

**Solutions**:
```bash
# Use GPU acceleration
qme opt molecule.xyz --device cuda

# Try TorchSim backend (if available)
qme opt molecule.xyz --backend torchsim_mace --device cuda

# Use faster backend
qme opt molecule.xyz --backend aimnet2  # Usually fastest

# Reduce precision for testing
qme opt molecule.xyz --fmax 0.1 --steps 100
```

### Memory Issues

**High Memory Usage**:
```bash
# Monitor memory usage
top -p $(pgrep -f qme)

# Or use system monitor
htop
```

**Solutions**:
```bash
# Use CPU instead of GPU
qme opt molecule.xyz --device cpu

# Try smaller model
qme opt molecule.xyz --backend aimnet2  # Lower memory

# Reduce system size or split calculation
```

## CLI-Specific Issues

### Command Not Found

**Problem**:
```
bash: qme: command not found
```

**Solutions**:
```bash
# Check installation
pip list | grep qme

# Reinstall if needed
pip install qme-ml

# Check PATH (for conda/virtual environments)
which python
which pip

# Use module syntax if needed
python -m qme.cli opt molecule.xyz
```

### Option Errors

**Unknown Option**:
```
Error: No such option: --unknown-option
```

**Solution**:
```bash
# Check available options
qme opt --help
qme tsopt --help

# Check spelling of option names
```

**Type Errors**:
```
Error: Invalid value for '--fmax': 'abc' is not a valid float
```

**Solution**: Ensure numeric options receive numeric values:
```bash
# Correct
qme opt molecule.xyz --fmax 0.05

# Incorrect  
qme opt molecule.xyz --fmax abc
```

## Debug Mode

### Enable Verbose Output

```bash
# CLI debug mode
qme opt molecule.xyz --verbose

# Python debug mode
import logging
logging.basicConfig(level=logging.DEBUG)

import qme
explorer = qme.Explorer.from_file("molecule.xyz")
result = explorer.run(mode="minima")
```

### Log Analysis

```python
# Enable detailed logging
import qme
qme.config.set_log_level("DEBUG")

# This will show:
# - Backend loading details
# - Model initialization
# - Optimization progress
# - Error stack traces
```

## Getting More Help

### Information to Collect

When asking for help, provide:

```bash
# System information
qme --version
python --version
uname -a  # Linux/macOS
# Or systeminfo on Windows

# Available backends
python -c "
import qme
from qme.calculator_registry import calculator_registry
print('Available backends:', list(calculator_registry.list_backends().keys()))
"

# Package versions
pip list | grep -E "(qme|torch|ase|numpy)"

# Error output
qme opt molecule.xyz --backend aimnet2 2>&1 | tee error.log
```

### Example Files

Test with minimal examples:

```bash
# Create simple test molecule
cat > test.xyz << EOF
3
Water molecule
O    0.000000    0.000000    0.117283
H    0.000000    0.758602   -0.469132
H    0.000000   -0.758602   -0.469132
EOF

# Test basic functionality
qme opt test.xyz --backend mock
```

### Where to Ask

1. **GitHub Issues**: For bugs and feature requests
   - [QME Issues](https://github.com/rlaplaza-lab/qme/issues)

2. **GitHub Discussions**: For usage questions
   - [QME Discussions](https://github.com/rlaplaza-lab/qme/discussions)

3. **Documentation**: Search this documentation
   - Use the search function
   - Check [FAQ](faq.md)

### Emergency Workarounds

If QME is completely broken:

```bash
# Reset to clean state
pip uninstall qme
pip cache purge
rm -rf ~/.cache/qme
pip install qme-ml-ml[aimnet2]

# Test with minimal setup
qme opt --help
```

## Preventing Issues

### Best Practices

1. **Use virtual environments** for different projects
2. **Test with mock backend** before using ML backends
3. **Start with simple systems** before complex ones
4. **Keep dependencies updated** but test changes
5. **Monitor system resources** during calculations
6. **Save intermediate results** for long calculations

### Regular Maintenance

```bash
# Update QME occasionally
pip install --upgrade qme

# Clear old cached models
qme cache info
qme cache clear  # If needed

# Check for new backend versions
pip list --outdated | grep -E "(torch|fairchem|mace)"
```

This troubleshooting guide covers the most common issues. For additional help, please refer to the [FAQ](faq.md) or ask in [GitHub Discussions](https://github.com/rlaplaza-lab/qme/discussions).
