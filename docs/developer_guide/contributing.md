# Contributing to QME

Thank you for your interest in contributing to QME! This document provides guidelines for contributing code, documentation, bug reports, and feature requests.

## Quick Start

1. **Fork** the repository on GitHub
2. **Clone** your fork locally
3. **Create** a feature branch
4. **Make** your changes
5. **Test** your changes
6. **Submit** a pull request

## Development Setup

### Prerequisites

- Python 3.10 or higher (3.11+ recommended for TorchSim)
- Git
- GitHub account

### Environment Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/qme.git
cd qme

# Add upstream remote
git remote add upstream https://github.com/rlaplaza-lab/qme.git

# Create development environment
conda create -n qme-dev python=3.12
conda activate qme-dev

# Install development dependencies
pip install -e .[dev]

# Install backends for testing (optional, may have conflicts)
pip install -e .[uma]     # UMA backend
pip install -e .[aimnet2] # AIMNet2 backend
```

### Verify Setup

```bash
# Run tests to ensure everything works
pytest tests/test_package_deps.py

# Check code style tools
black --version
isort --version
flake8 --version
```

## Types of Contributions

### 🐛 Bug Reports

Good bug reports are extremely helpful! When reporting bugs:

1. **Use the GitHub issue tracker**
2. **Check if the bug has already been reported**
3. **Provide a clear, descriptive title**
4. **Include steps to reproduce the bug**
5. **Provide system information**

#### Bug Report Template

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Run command '...'
2. With input file '...'
3. See error

**Expected behavior**
What you expected to happen.

**System Information**
- QME version: [e.g., 0.1.0]
- Python version: [e.g., 3.12]
- OS: [e.g., Ubuntu 22.04]
- Backend: [e.g., uma, aimnet2]

**Additional context**
Any other context about the problem.
```

### 🚀 Feature Requests

We welcome feature requests! To suggest a new feature:

1. **Check if it's already been requested**
2. **Describe the problem you're trying to solve**
3. **Explain your proposed solution**
4. **Consider the scope and complexity**

#### Feature Request Template

```markdown
**Is your feature request related to a problem?**
A clear description of what the problem is.

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
Alternative solutions or features you've considered.

**Additional context**
Any other context about the feature request.
```

### 📝 Documentation

Documentation improvements are always welcome:

- Fix typos and grammatical errors
- Improve clarity and organization
- Add missing examples
- Update outdated information
- Translate documentation

### 💻 Code Contributions

We welcome code contributions including:

- Bug fixes
- New features
- Performance improvements
- Test coverage improvements
- Code quality improvements

## Development Workflow

### 1. Before You Start

- **Discuss large changes** in an issue first
- **Check the existing codebase** for similar functionality
- **Review the architecture** documentation

### 2. Making Changes

```bash
# Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/my-feature-name

# Make your changes
# ... edit files ...

# Run tests frequently
pytest tests/

# Check code style
black qme/ tests/
isort qme/ tests/
flake8 qme/ tests/
```

### 3. Commit Guidelines

Write clear, descriptive commit messages:

```bash
# Good commit messages
git commit -m "Add support for custom optimizers in Explorer"
git commit -m "Fix memory leak in UMA calculator initialization"
git commit -m "Update installation docs for Python 3.11+"

# Avoid vague messages
git commit -m "Fix bug"
git commit -m "Update docs"
git commit -m "Changes"
```

#### Commit Message Format

```
<type>: <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

### 4. Testing

All contributions must include appropriate tests:

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_cli.py
pytest tests/test_backends_min_ts.py

# Run with coverage
pytest --cov=qme --cov-report=html

# Test with different backends
pytest tests/test_backends_min_ts.py -k "mock"
pytest tests/test_backends_min_ts.py -k "aimnet2"  # if available
```

#### Test Requirements

- **All new features** must have tests
- **Bug fixes** should include regression tests
- **Use mock backend** for unit tests when possible
- **Keep tests fast** (< 1 second each)
- **Test edge cases** and error conditions

### 5. Documentation

Update documentation for any user-facing changes:

- **API changes**: Update docstrings and API reference
- **New features**: Add to user guide and examples
- **CLI changes**: Update CLI documentation
- **Breaking changes**: Update migration guide

## Code Style Guidelines

### Python Code Style

We follow [PEP 8](https://pep8.org/) with some modifications:

```python
# Line length: 88 characters (Black default)
# Use double quotes for strings
example = "This is a string"

# Function and variable names: snake_case
def calculate_energy(atoms):
    total_energy = 0.0
    return total_energy

# Class names: PascalCase
class MyCalculator:
    def __init__(self):
        pass

# Constants: UPPER_CASE
DEFAULT_FMAX = 0.05
```

### Import Organization

```python
# Standard library imports first
import os
import sys
from typing import List, Optional

# Third-party imports
import numpy as np
from ase import Atoms
import torch

# Local imports
from qme.core.explorer import Explorer
from qme.potentials.base_potential import BasePotential
```

### Docstring Style

Use Google-style docstrings:

```python
def optimize_structure(atoms, fmax=0.05, steps=1000):
    """Optimize molecular structure to minimum energy.

    Args:
        atoms: ASE Atoms object to optimize
        fmax: Force convergence threshold in eV/Å
        steps: Maximum number of optimization steps

    Returns:
        dict: Optimization results with keys:
            - 'optimized_atoms': Optimized ASE Atoms object
            - 'final_energy': Final energy in eV
            - 'converged': Whether optimization converged

    Raises:
        ValueError: If atoms object is invalid
        RuntimeError: If optimization fails to converge

    Example:
        >>> from ase.build import molecule
        >>> atoms = molecule('H2O')
        >>> result = optimize_structure(atoms, fmax=0.01)
        >>> print(f"Final energy: {result['final_energy']:.3f} eV")
    """
```

## Pull Request Process

### 1. Before Submitting

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] Changes are focused and atomic

### 2. Creating the Pull Request

```bash
# Push your branch
git push origin feature/my-feature-name

# Create PR on GitHub
# Include clear title and description
```

#### PR Template

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change)
- [ ] New feature (non-breaking change)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Tested with multiple backends (if applicable)

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review of code completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] Changelog updated (for significant changes)
```

### 3. Review Process

1. **Automated checks** must pass (CI, style, tests)
2. **Code review** by maintainers
3. **Address feedback** if requested
4. **Approval** from maintainer
5. **Merge** by maintainer

### 4. After Merge

```bash
# Sync your fork
git checkout main
git pull upstream main
git push origin main

# Delete feature branch
git branch -d feature/my-feature-name
git push origin --delete feature/my-feature-name
```

## Specific Contribution Areas

### Adding New Backends

See [Adding New Backends](adding_backends.md) for detailed instructions.

Key requirements:
- Implement `BasePotential` interface
- Handle missing dependencies gracefully
- Include comprehensive tests
- Document installation and usage

### Performance Optimizations

When contributing performance improvements:

- **Profile first**: Use tools like `cProfile` to identify bottlenecks
- **Measure impact**: Include benchmark results
- **Consider memory**: Not just speed
- **Test thoroughly**: Ensure correctness is maintained

### Documentation Improvements

Documentation contributions are highly valued:

- **User guides**: Improve clarity and add examples
- **API documentation**: Ensure all public APIs are documented
- **Tutorials**: Step-by-step guides for common tasks
- **Architecture docs**: Help developers understand the codebase

## Community Guidelines

### Code of Conduct

We follow the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). Please be respectful and inclusive in all interactions.

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and community discussions
- **Pull Requests**: Code review and technical discussions

### Getting Help

If you need help contributing:

1. **Check existing documentation**
2. **Search closed issues** for similar problems
3. **Ask in GitHub Discussions**
4. **Tag maintainers** in issues for urgent problems

## Recognition

We value all contributions and will:

- **Credit contributors** in release notes
- **Maintain contributor list** in repository
- **Recognize significant contributions** in documentation

## Release Process

For maintainers, the release process is:

1. Update version numbers
2. Update CHANGELOG.md
3. Create release branch
4. Final testing
5. Create GitHub release
6. Publish to PyPI
7. Update documentation

## Questions?

If you have questions about contributing:

- **General questions**: Use GitHub Discussions
- **Specific issues**: Comment on relevant GitHub issues
- **Documentation feedback**: Create an issue with the "documentation" label

Thank you for contributing to QME! 🎉
