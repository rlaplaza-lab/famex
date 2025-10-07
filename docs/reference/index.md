# Reference

Complete reference documentation for QME configuration, error handling, and troubleshooting.

## Sections

### [Configuration Options](configuration.md)
Complete reference for QME configuration settings, environment variables, and customization options.

### [Error Handling](errors.md)
Comprehensive guide to QME error messages, their meanings, and solutions.

### [Troubleshooting](troubleshooting.md)
Common issues and their solutions, organized by category and backend.

### [FAQ](faq.md)
Frequently asked questions about QME usage, installation, and best practices.

## Quick Reference

### Common Error Patterns

**Backend Issues**:
- `Backend 'xyz' not available` → Install backend dependencies
- `Could not load model 'xyz'` → Check model name and network connection
- `CUDA out of memory` → Use CPU or reduce system size

**Installation Issues**:
- `pip dependency conflicts` → Use individual backend installations
- `Python version incompatible` → Use Python 3.10+ (3.11+ for TorchSim)
- `Missing SELLA optimizer` → Install with `pip install sella`

**Optimization Issues**:
- `Optimization did not converge` → Increase steps, change optimizer, or loosen convergence
- `Forces too large` → Check input structure quality
- `Unrealistic energies` → Verify backend compatibility with your system

### Configuration Locations

**User Configuration**:
- Linux/macOS: `~/.config/qme/config.yaml`
- Windows: `%APPDATA%/qme/config.yaml`

**Cache Directories**:
- Linux/macOS: `~/.cache/qme/`
- Windows: `%LOCALAPPDATA%/qme/`

**Environment Variables**:
- `QME_CACHE_DIR`: Override default cache directory
- `QME_CONFIG_FILE`: Override default config file location
- `QME_DEFAULT_BACKEND`: Set default backend globally

### Performance Guidelines

**Backend Selection**:
- **Testing**: Use `mock` backend
- **General use**: Start with `aimnet2`
- **High accuracy**: Try `mace` models
- **Maximum speed**: Use `torchsim_*` with GPU

**System Size Recommendations**:
- Small molecules (< 20 atoms): Any backend
- Medium molecules (20-100 atoms): Prefer GPU backends
- Large systems (> 100 atoms): Use TorchSim with GPU

**Convergence Settings**:
- **Quick testing**: `--fmax 0.1 --steps 100`
- **Standard use**: `--fmax 0.05 --steps 1000` (default)
- **High precision**: `--fmax 0.01 --steps 2000`

## Support Resources

### Getting Help

1. **Search Documentation**: Use the search function to find relevant information
2. **Check FAQ**: Common questions are answered in the FAQ section
3. **Review Error Guide**: Specific error messages have detailed explanations
4. **GitHub Issues**: Report bugs and request features
5. **GitHub Discussions**: Ask questions and share experiences

### Reporting Issues

When reporting problems, include:

- **QME version**: `qme --version`
- **Python version**: `python --version`
- **Operating system**: Linux/macOS/Windows version
- **Backend used**: Which ML backend you're using
- **Complete error message**: Full traceback if available
- **Minimal example**: Smallest input that reproduces the issue

### Contributing

Help improve QME documentation and code:

- **Documentation**: Fix typos, improve clarity, add examples
- **Bug fixes**: Identify and fix issues
- **New features**: Implement requested functionality
- **Testing**: Add test coverage and benchmarks

See the [Contributing Guide](../developer_guide/contributing.md) for details.

## Version Information

This documentation is for QME v0.1.0. Check the [changelog](https://github.com/rlaplaza-lab/qme/releases) for updates and new features.

For the latest development version, see the [main branch documentation](https://github.com/rlaplaza-lab/qme/tree/main/docs).
