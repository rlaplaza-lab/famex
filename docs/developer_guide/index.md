# Developer Guide

Welcome to the QME developer documentation. This guide covers everything you need to know to contribute to QME or extend it for your own needs.

## Overview

QME is designed with extensibility in mind. The modular architecture allows you to:

- Add new machine learning backends
- Implement custom optimization strategies
- Create new analysis tools
- Extend the command-line interface
- Add benchmarks and tests

## Quick Start for Developers

### 1. Development Setup

```bash
# Clone the repository
git clone https://github.com/rlaplaza-lab/qme.git
cd qme

# Create development environment
conda create -n qme-dev python=3.12
conda activate qme-dev

# Install in development mode
pip install -e .[dev]

# Install additional backends for testing
pip install -e .[uma,aimnet2,mace]  # Note: may have conflicts
```

### 2. Run Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_cli.py                    # CLI functionality
pytest tests/test_backends_min_ts.py        # Backend tests
pytest tests/test_comprehensive_optimization.py  # Integration tests

# Run with coverage
pytest --cov=qme --cov-report=html
```

### 3. Code Quality

```bash
# Format code
black qme/ tests/

# Sort imports
isort qme/ tests/

# Check style
flake8 qme/ tests/
```

## Documentation Sections

### [Contributing Guidelines](contributing.md)
- How to contribute code, documentation, and bug reports
- Pull request process
- Code style guidelines
- Review process


### [Adding New Backends](adding_backends.md)
- Step-by-step guide to adding ML potential backends
- Backend interface requirements
- Testing new backends

### [Architecture Overview](architecture.md)
- System design and component relationships
- Plugin architecture
- Extension points

### [Testing](testing.md)
- Testing philosophy and practices
- Writing tests for new features
- Continuous integration setup

## Architecture Overview

### Core Components

```
qme/
├── core/                   # Core optimization and exploration logic
│   ├── explorer.py        # Main Explorer class
│   ├── geometry.py        # Geometry handling
│   ├── strategy.py        # Strategy base classes and registry
│   ├── local_strategies.py # Local optimization strategies
│   ├── twoended_strategies.py # Multi-structure strategies
│   └── validation.py      # Input validation
├── potentials/            # ML potential backends
│   ├── base_potential.py  # Base calculator interface
│   ├── uma_potential.py   # UMA backend
│   ├── aimnet2_potential.py # AIMNet2 backend
│   └── ...
├── cli/                   # Command-line interface
├── analysis/              # Analysis tools (frequencies, etc.)
└── dependencies.py        # Lazy loading system
```

### Plugin Architecture

QME uses a registry-based plugin system:

```python
# Register a new backend
from qme.calculator_registry import calculator_registry

calculator_registry.register(
    "my_backend",
    get_my_calculator,
    description="My custom ML potential"
)
```

### Extension Points

1. **New Backends**: Implement `BasePotential` interface
2. **Optimization Strategies**: Create new strategy classes inheriting from `BaseStrategy`
3. **Analysis Tools**: Extend `analysis/` module
4. **CLI Commands**: Add to `cli/` module

## Development Workflow

### 1. Feature Development

```bash
# Create feature branch
git checkout -b feature/my-new-feature

# Make changes
# ... code changes ...

# Test changes
pytest tests/

# Commit changes
git add .
git commit -m "Add my new feature"

# Push and create PR
git push origin feature/my-new-feature
```

### 2. Adding a New Backend

See [Adding New Backends](adding_backends.md) for detailed instructions.

### 3. Writing Tests

```python
# tests/test_my_feature.py
import pytest
from qme import Explorer

def test_my_feature():
    """Test my new feature."""
    explorer = Explorer.from_file("test.xyz", backend="mock")
    result = explorer.run(target="minima", strategy="local")
    assert result is not None
```

### 4. Documentation

- Update relevant documentation files
- Add docstrings to new functions/classes
- Include examples in docstrings
- Update API reference if needed

## Key Design Principles

### 1. Lazy Loading

QME uses lazy loading to avoid importing heavy dependencies until needed:

```python
# dependencies.py handles conditional imports
from qme.dependencies import deps

# Only loads torch if available
torch = deps.get_dependency("torch")
```

### 2. Graceful Fallbacks

All backends should handle missing dependencies gracefully:

```python
def get_my_calculator():
    try:
        import my_ml_package
        return MyCalculator()
    except ImportError:
        from qme.potentials import MockCalculator
        return MockCalculator()
```

### 3. ASE Compatibility

All calculators must implement the ASE Calculator interface:

```python
class MyCalculator(BasePotential):
    def calculate(self, atoms, properties=None, system_changes=None):
        # Implement ASE calculator interface
        pass
```

### 4. Strategy Registry

Use the strategy registry system for new components:

```python
from qme.core.strategy import BaseStrategy, StrategyMetadata, REGISTRY

class MyCustomStrategy(BaseStrategy):
    metadata = StrategyMetadata(
        name="minima:my_custom",
        target="minima",
        strategy="my_custom",
        description="My custom optimization method",
        aliases=["my_custom"],
        requires_multiple_structures=False,
    )

    def run(self, atoms_list, **kwargs):
        # Implementation here
        pass

# Register the strategy
REGISTRY.register(MyCustomStrategy)
```

## Common Development Tasks

### Adding a Command-Line Option

```python
# cli/cli.py
@click.option('--my-option', help='My new option')
def opt_command(my_option, **kwargs):
    # Use the option
    pass
```


### Adding an Analysis Tool

```python
# analysis/my_analysis.py
class MyAnalysis:
    def __init__(self, atoms, calculator):
        self.atoms = atoms
        self.calculator = calculator

    def run(self):
        # Implement analysis
        pass
```

## Testing Philosophy

### Test Categories

1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test component interactions
3. **Backend Tests**: Test with mock and real backends
4. **CLI Tests**: Test command-line interface
5. **Benchmarks**: Performance and accuracy tests

### Test Requirements

- All new features must have tests
- Tests should use the mock backend when possible
- Real ML backends used only for integration tests
- Tests should be fast (< 1 second each)

### Continuous Integration

We use GitHub Actions for CI:
- Tests run on multiple Python versions
- Tests run with and without optional dependencies
- Code coverage is tracked
- Style checks are enforced

## Release Process

### Version Numbering

QME uses semantic versioning:
- Major: Breaking changes
- Minor: New features, backward compatible
- Patch: Bug fixes

### Release Checklist

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release branch
4. Run full test suite
5. Create GitHub release
6. Publish to PyPI

## Getting Help

### Development Questions

- **GitHub Discussions**: For general development questions
- **GitHub Issues**: For bug reports and feature requests
- **Code Review**: All changes go through pull request review

### Code Review Process

1. Create pull request with clear description
2. Ensure all tests pass
3. Request review from maintainers
4. Address review feedback
5. Merge after approval

## Contributing Areas

We welcome contributions in:

- **New ML Backends**: Support for new potential types
- **Optimization Methods**: New optimization strategies
- **Analysis Tools**: Frequency analysis, thermodynamics, etc.
- **Performance**: Speed and memory optimizations
- **Documentation**: Improve existing docs or add examples
- **Testing**: Expand test coverage
- **Benchmarks**: New benchmark datasets

See [Contributing Guidelines](contributing.md) for detailed information on how to contribute.

## Resources

- **ASE Documentation**: [https://wiki.fysik.dtu.dk/ase/](https://wiki.fysik.dtu.dk/ase/)
- **PyTorch Documentation**: [https://pytorch.org/docs/](https://pytorch.org/docs/)
- **SELLA Optimizer**: [https://github.com/zadorlab/sella](https://github.com/zadorlab/sella)
- **QME Repository**: [https://github.com/rlaplaza-lab/qme](https://github.com/rlaplaza-lab/qme)
