# Creating Custom Strategies

This guide shows how to create custom optimization strategies for QME Explorer using the new class-based architecture.

## Overview

The new strategy system uses a class-based approach where each strategy is a subclass of `BaseStrategy`. This provides:

- Clear interfaces and type hints
- Built-in validation and result standardization
- Metadata-driven registration
- Easy extensibility

## Basic Strategy Structure

Here's a minimal example of a custom strategy:

```python
from qme.core.strategy import BaseStrategy, StrategyMetadata, REGISTRY
from ase import Atoms
from typing import Any

class MyCustomStrategy(BaseStrategy):
    """My custom optimization strategy."""

    metadata = StrategyMetadata(
        name="minima:my_custom",
        target="minima",
        strategy="my_custom",
        description="My custom optimization method",
        aliases=["my_custom", "custom"],
        requires_multiple_structures=False,
    )

    def run(self, atoms_list: list[Atoms], **kwargs) -> dict[str, Any]:
        """Run the custom optimization.

        Parameters
        ----------
        atoms_list : list[Atoms]
            List of structures to optimize
        **kwargs
            Additional keyword arguments

        Returns
        -------
        dict[str, Any]
            Standardized result dictionary
        """
        self.validate_inputs(atoms_list)

        # Your custom optimization logic here
        # ...

        return self.prepare_result(
            optimized_atoms=atoms_list[0],  # or your optimized structure
            converged=True,
            # Add any additional metadata
        )

# Register the strategy
REGISTRY.register(MyCustomStrategy)
```

## Strategy Metadata

The `StrategyMetadata` class defines how your strategy is registered and discovered:

- **name**: Full strategy name in format "target:strategy" (e.g., "minima:local")
- **target**: What the strategy optimizes ("minima", "ts", "path")
- **strategy**: The method name ("local", "neb", "custom", etc.)
- **description**: Human-readable description
- **aliases**: Alternative names for the strategy
- **requires_multiple_structures**: Whether the strategy needs 2+ structures

## Required Methods

### `run(atoms_list, **kwargs) -> dict[str, Any]`

The main method that performs the optimization. Must return a standardized result dictionary with at least:

- `optimized_atoms`: The optimized structure(s)
- `strategy`: Strategy name (automatically added by `prepare_result()`)
- `converged`: Whether optimization converged

### Optional Methods

- `validate_inputs(atoms_list)`: Override for custom input validation
- `prepare_result(optimized_atoms, **metadata)`: Override for custom result formatting

## Using the Strategy

Once registered, your strategy can be used like any other:

```python
from qme.core.explorer import Explorer
from ase import Atoms

# Create Explorer with your custom strategy
atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
explorer = Explorer(atoms, target="minima", strategy="my_custom")

# Run the optimization
result = explorer.run()

# Or use explicit mode
result = explorer.run(mode="minima:my_custom")
```

## Advanced Example: Two-Ended Strategy

Here's an example of a two-ended strategy that works with multiple structures:

```python
class MyPathStrategy(BaseStrategy):
    """Custom path optimization strategy."""

    metadata = StrategyMetadata(
        name="path:my_method",
        target="path",
        strategy="my_method",
        description="My custom path optimization method",
        aliases=["my_method", "my-path"],
        requires_multiple_structures=True,
    )

    def run(self, atoms_list: list[Atoms], npoints: int = 11, **kwargs) -> dict[str, Any]:
        """Run custom path optimization."""
        self.validate_inputs(atoms_list)

        # Ensure we have at least 2 structures
        if len(atoms_list) < 2:
            raise ValueError("Path strategy requires at least 2 structures")

        # Your custom path optimization logic here
        # For example, interpolate between structures
        path = self._interpolate_path(atoms_list, npoints)

        # Optimize the path
        optimized_path = self._optimize_path(path)

        return self.prepare_result(
            optimized_atoms=optimized_path,
            converged=True,
            trajectory=optimized_path,
        )

    def _interpolate_path(self, atoms_list, npoints):
        """Interpolate between structures."""
        # Implementation here
        pass

    def _optimize_path(self, path):
        """Optimize the interpolated path."""
        # Implementation here
        pass

# Register the strategy
REGISTRY.register(MyPathStrategy)
```

## Integration with Explorer

The strategy system integrates seamlessly with Explorer:

1. **Automatic Registration**: Strategies are registered when their modules are imported
2. **Alias Support**: Short names like "neb" automatically resolve to "path:neb"
3. **Validation**: Built-in input validation based on strategy requirements
4. **Error Handling**: Clear error messages for missing or invalid strategies

## Best Practices

1. **Use Descriptive Names**: Choose clear, descriptive names for your strategy
2. **Provide Good Documentation**: Include docstrings and type hints
3. **Handle Errors Gracefully**: Use try/except blocks and provide meaningful error messages
4. **Follow the Interface**: Always call `validate_inputs()` and use `prepare_result()`
5. **Test Thoroughly**: Write unit tests for your strategy

## Example: Complete Custom Strategy

Here's a complete example that demonstrates all the features:

```python
import numpy as np
from ase import Atoms
from qme.core.strategy import BaseStrategy, StrategyMetadata, REGISTRY
from qme.core.strategy_utils import StrategyUtils
from qme.logging_utils import get_qme_logger

logger = get_qme_logger(__name__)

class SimpleMinimizationStrategy(BaseStrategy):
    """A simple minimization strategy for demonstration."""

    metadata = StrategyMetadata(
        name="minima:simple",
        target="minima",
        strategy="simple",
        description="Simple minimization using basic optimization",
        aliases=["simple", "basic"],
        requires_multiple_structures=False,
    )

    def run(self, atoms_list: list[Atoms], fmax: float = 0.05, steps: int = 100, **kwargs) -> dict[str, Any]:
        """Run simple minimization.

        Parameters
        ----------
        atoms_list : list[Atoms]
            List of structures to optimize
        fmax : float, default=0.05
            Force convergence threshold
        steps : int, default=100
            Maximum optimization steps
        **kwargs
            Additional keyword arguments

        Returns
        -------
        dict[str, Any]
            Standardized result dictionary
        """
        self.validate_inputs(atoms_list)

        logger.info(f"Starting simple minimization with fmax={fmax}, steps={steps}")

        # Get the first structure
        atoms = atoms_list[0].copy()

        # Attach calculator and constraints
        self.explorer._create_and_attach_calculator(atoms)
        self.explorer._apply_constraints(atoms)

        # Simple optimization loop
        for step in range(steps):
            forces = atoms.get_forces()
            max_force = np.max(np.abs(forces))

            if max_force < fmax:
                logger.info(f"Converged after {step + 1} steps")
                break

            # Simple steepest descent step
            positions = atoms.get_positions()
            step_size = 0.01
            new_positions = positions - step_size * forces
            atoms.set_positions(new_positions)

        converged = max_force < fmax

        return self.prepare_result(
            optimized_atoms=atoms,
            converged=converged,
            steps_taken=step + 1,
            max_force=max_force,
        )

# Register the strategy
REGISTRY.register(SimpleMinimizationStrategy)
```

This strategy can then be used with:

```python
explorer = Explorer(atoms, target="minima", strategy="simple")
result = explorer.run(fmax=0.01, steps=50)
```

## Conclusion

The new class-based strategy system makes it easy to create custom optimization methods while maintaining consistency and providing clear interfaces. The system handles registration, validation, and result formatting automatically, allowing you to focus on implementing your optimization logic.
