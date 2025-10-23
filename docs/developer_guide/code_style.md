# Code Style Guidelines

This document outlines the code style standards for the QME project to ensure consistency and maintainability across the codebase.

## Overview

QME follows Python best practices with a focus on:
- **Consistency**: Uniform formatting and style across all files
- **Readability**: Clear, self-documenting code
- **Type Safety**: Comprehensive type annotations
- **Documentation**: Complete docstrings for all public APIs

## Tools and Configuration

### Primary Tools

- **Black**: Code formatting (line length: 100 characters)
- **isort**: Import sorting (profile: black)
- **ruff**: Fast linting and code analysis
- **mypy**: Static type checking

### Configuration Files

- `pyproject.toml`: Main configuration for all tools
- `.pre-commit-config.yaml`: Pre-commit hooks for automated checks

## Code Formatting

### Black Configuration

```toml
[tool.black]
line-length = 100
target-version = ['py312']
```

- **Line length**: 100 characters maximum
- **Target version**: Python 3.12+
- **String quotes**: Double quotes preferred

### Import Organization (isort)

```toml
[tool.isort]
profile = "black"
line_length = 100
```

Imports are organized in this order:
1. Standard library imports
2. Third-party imports
3. Local application imports

## Linting (ruff)

### Enabled Rules

```toml
[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "N",   # pep8-naming
    "UP",  # pyupgrade
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "PIE", # flake8-pie
    "SIM", # flake8-simplify
    "D",   # pydocstyle
]
```

### Ignored Rules

```toml
ignore = [
    "E501",  # line too long (handled by black)
    "E203",  # whitespace before ':' (conflicts with black)
    "T201",  # print statements (allow in tests and examples)
    "B904",  # raise without from (allow for now)
    "N806",  # variable names like kT, OptClass are intentional
    "SIM102", # nested ifs are sometimes clearer
    "SIM105", # contextlib.suppress is not always clearer
    "SIM108", # ternary operators are not always clearer
    "D100",  # Missing docstring in public module
    "D101",  # Missing docstring in public class
    "D102",  # Missing docstring in public method
    "D103",  # Missing docstring in public function
    "D104",  # Missing docstring in public package
    "D105",  # Missing docstring in magic method
    "D107",  # Missing docstring in __init__
]
```

## Type Annotations

### mypy Configuration

```toml
[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
strict = true
```

### Type Annotation Standards

- **All public functions** must have complete type annotations
- **Use `from __future__ import annotations`** for forward references
- **Prefer `|` union syntax** over `Union` (Python 3.10+)
- **Use `Any` sparingly** and document why it's necessary

### Example

```python
from __future__ import annotations

from typing import Any
from ase import Atoms

def optimize_geometry(
    atoms: Atoms,
    backend: str = "uma",
    max_steps: int = 100,
    **kwargs: Any,
) -> dict[str, Any]:
    """Optimize molecular geometry.
    
    Parameters
    ----------
    atoms : Atoms
        Molecular structure to optimize
    backend : str, default "uma"
        ML potential backend to use
    max_steps : int, default 100
        Maximum optimization steps
    **kwargs : Any
        Additional optimizer parameters
        
    Returns
    -------
    dict[str, Any]
        Optimization results including optimized structure
    """
    # Implementation here
    pass
```

## Docstring Standards

### Style: Google Format

All docstrings follow the Google style format:

```python
def function_name(param1: str, param2: int = 10) -> bool:
    """Brief description of the function.
    
    Longer description if needed, explaining the purpose,
    behavior, and any important details.
    
    Parameters
    ----------
    param1 : str
        Description of param1
    param2 : int, default 10
        Description of param2
        
    Returns
    -------
    bool
        Description of return value
        
    Raises
    ------
    ValueError
        When param1 is invalid
        
    Examples
    --------
    >>> result = function_name("test", 5)
    >>> print(result)
    True
    """
    pass
```

### Required Docstrings

- **All public modules**: Module-level docstring
- **All public classes**: Class docstring
- **All public methods**: Method docstring
- **All public functions**: Function docstring
- **All `__init__` methods**: Constructor docstring

### Docstring Sections

- **Summary**: One-line description
- **Description**: Detailed explanation (if needed)
- **Parameters**: All parameters with types and descriptions
- **Returns**: Return value type and description
- **Raises**: Exceptions that may be raised
- **Examples**: Usage examples (when helpful)
- **Notes**: Additional information (when needed)

## Naming Conventions

### Variables and Functions
- **snake_case**: `optimize_geometry`, `max_steps`
- **Descriptive names**: `optimized_atoms` not `opt_atoms`
- **Constants**: `UPPER_CASE`: `DEFAULT_BACKEND`

### Classes
- **PascalCase**: `GeometryOptimizer`, `BackendRegistry`
- **Descriptive names**: `TransitionStateStrategy` not `TSStrategy`

### Modules and Packages
- **snake_case**: `geometry_optimizer.py`, `backend_registry.py`
- **Short, clear names**: `io/`, `strategies/`, `potentials/`

## Code Organization

### File Structure
```
qme/
├── __init__.py          # Package initialization
├── core/                # Core functionality
├── strategies/          # Optimization strategies
├── potentials/          # ML potential backends
├── backends/            # Backend management
├── io/                  # Input/output utilities
├── utils/               # Utility functions
└── cli/                 # Command-line interface
```

### Import Organization
```python
"""Module docstring."""

from __future__ import annotations

# Standard library
import os
from pathlib import Path
from typing import Any

# Third-party
import numpy as np
from ase import Atoms

# Local imports
from qme.core.base_strategy import BaseStrategy
from qme.utils.validation import QMEError
```

## Error Handling

### Exception Classes
```python
class QMEError(Exception):
    """Base exception for QME errors."""
    
    def __init__(self, message: str, suggestion: str | None = None) -> None:
        """Initialize QME error with message and optional suggestion."""
        super().__init__(message)
        self.message = message
        self.suggestion = suggestion
```

### Error Messages
- **Clear and actionable**: Include suggestions when possible
- **Context-aware**: Include relevant information
- **User-friendly**: Avoid technical jargon in user-facing messages

## Testing Standards

### Test Organization
```
tests/
├── __init__.py
├── conftest.py          # Pytest configuration
├── unit/                # Unit tests
│   ├── test_strategies.py
│   └── test_potentials.py
└── integration/         # Integration tests
    ├── test_cli.py
    └── test_workflows.py
```

### Test Naming
- **Descriptive names**: `test_optimize_geometry_with_uma_backend`
- **Grouped by functionality**: `test_*_strategy`, `test_*_potential`
- **Clear assertions**: Use specific assertion messages

## Pre-commit Hooks

### Setup
```bash
pip install pre-commit
pre-commit install
```

### Hooks Configuration
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.3.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/psf/black
    rev: 24.1.1
    hooks:
      - id: black
        args: [--line-length=100]

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: [--profile=black, --line-length=100]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        args: [--config-file=pyproject.toml, --no-incremental]
```

## Development Workflow

### Before Committing
1. **Run linting**: `ruff check qme/`
2. **Format code**: `black qme/`
3. **Sort imports**: `isort qme/`
4. **Type check**: `mypy qme/`
5. **Run tests**: `pytest tests/`

### Automated Checks
- **Pre-commit hooks**: Run automatically on commit
- **CI/CD**: Automated checks on pull requests
- **Code review**: Manual review for complex changes

## Best Practices

### Code Quality
- **Keep functions small**: Single responsibility principle
- **Avoid deep nesting**: Use early returns and guard clauses
- **Use type hints**: Improve code clarity and catch errors
- **Write tests**: Ensure code reliability

### Performance
- **Lazy imports**: Import heavy dependencies only when needed
- **Caching**: Cache expensive operations when appropriate
- **Batch operations**: Group similar operations together

### Documentation
- **Update docstrings**: Keep documentation current
- **Add examples**: Show how to use the code
- **Explain complex logic**: Add comments for non-obvious code

## Tools Usage

### Manual Checks
```bash
# Format code
black qme/

# Sort imports
isort qme/

# Lint code
ruff check qme/

# Type check
mypy qme/

# Run all checks
pre-commit run --all-files
```

### IDE Integration
- **VS Code**: Install Python, Black, isort, and mypy extensions
- **PyCharm**: Configure external tools for Black and isort
- **Vim/Neovim**: Use ALE or similar plugins

## Contributing

When contributing to QME:

1. **Follow these guidelines**: Ensure code meets standards
2. **Run pre-commit hooks**: Fix issues before submitting
3. **Add tests**: Include tests for new functionality
4. **Update documentation**: Keep docs current
5. **Request review**: Get feedback from maintainers

## Resources

- [PEP 8](https://peps.python.org/pep-0008/): Python style guide
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Black Documentation](https://black.readthedocs.io/)
- [ruff Documentation](https://docs.astral.sh/ruff/)
- [mypy Documentation](https://mypy.readthedocs.io/)
